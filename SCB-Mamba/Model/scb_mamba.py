import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from gsde_module import GrayScaleDifferenceEnhancement
from irbe_module import InjuredRegionBoundaryEnhancement
from multi_path_mamba import MultiPathMambaWithSelection, PathConsistencyLoss


class SCBMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gsde_module = GrayScaleDifferenceEnhancement(
            in_channels=3,
            hidden_dims=[64, 128, 256],
            complexity_thresholds=config.superpixel_config['complexity_thresholds'],
            base_clusters=config.superpixel_config['base_clusters'],
            cluster_increments=config.superpixel_config['cluster_increments']
        )

        self.irbe_module = InjuredRegionBoundaryEnhancement(
            in_channels=3,
            feature_channels=64,
            gate_channels=config.boundary_config['gate_channels']
        )

        self.multi_path_backbone = MultiPathMambaWithSelection(config)

        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        self.attention_mechanism = CrossModalAttention(
            hidden_dim=config.hidden_dim,
            num_heads=8
        )

        self.gradient_flow = GradientFlowModule(config.hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, start_position: str = 'top_left') -> Dict[str, Any]:
        batch_size, channels, height, width = x.shape

        enhanced_saliency, multi_scale_saliency, complexity = self.gsde_module(x)

        boundary_map = self.irbe_module(x, enhanced_saliency)

        mamba_outputs = self.multi_path_backbone(
            x, enhanced_saliency, boundary_map, start_position
        )

        global_features = mamba_outputs['global_features']
        path_features = mamba_outputs['path_features']

        saliency_features = self._extract_saliency_features(enhanced_saliency)
        boundary_features = self._extract_boundary_features(boundary_map)

        fused_features = torch.cat([
            global_features,
            saliency_features,
            boundary_features
        ], dim=1)

        final_logits = self.feature_fusion(fused_features)

        attention_weights = self.attention_mechanism(
            global_features.unsqueeze(1),
            saliency_features.unsqueeze(1),
            boundary_features.unsqueeze(1)
        )

        gradient_enhanced = self.gradient_flow(
            global_features,
            saliency_features,
            boundary_features
        )

        return {
            'final_logits': final_logits,
            'mamba_logits': mamba_outputs['final_logits'],
            'fused_logits': mamba_outputs['fused_logits'],
            'ensemble_logits': mamba_outputs['ensemble_logits'],
            'path_logits': mamba_outputs['path_logits'],
            'saliency_map': enhanced_saliency,
            'boundary_map': boundary_map,
            'complexity': complexity,
            'attention_weights': attention_weights,
            'gradient_enhanced': gradient_enhanced,
            'global_features': global_features
        }

    def _extract_saliency_features(self, saliency_map: torch.Tensor) -> torch.Tensor:
        batch_size = saliency_map.shape[0]

        spatial_features = F.adaptive_avg_pool2d(saliency_map, (4, 4)).view(batch_size, -1)
        statistical_features = torch.cat([
            saliency_map.mean(dim=(2, 3)),
            saliency_map.std(dim=(2, 3)),
            saliency_map.max(dim=2)[0].max(dim=1)[0],
            saliency_map.min(dim=2)[0].min(dim=1)[0]
        ], dim=1)

        return torch.cat([spatial_features, statistical_features], dim=1)

    def _extract_boundary_features(self, boundary_map: torch.Tensor) -> torch.Tensor:
        batch_size = boundary_map.shape[0]

        spatial_features = F.adaptive_avg_pool2d(boundary_map, (4, 4)).view(batch_size, -1)

        boundary_strength = boundary_map.mean(dim=(2, 3))
        boundary_contrast = boundary_map.std(dim=(2, 3))

        sobel_x = F.conv2d(boundary_map,
                           torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                        device=boundary_map.device).float(),
                           padding=1)
        sobel_y = F.conv2d(boundary_map,
                           torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                        device=boundary_map.device).float(),
                           padding=1)

        gradient_magnitude = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).mean(dim=(2, 3))

        statistical_features = torch.cat([
            boundary_strength,
            boundary_contrast,
            gradient_magnitude
        ], dim=1)

        return torch.cat([spatial_features, statistical_features], dim=1)


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_features: torch.Tensor,
                saliency_features: torch.Tensor,
                boundary_features: torch.Tensor) -> torch.Tensor:

        query = self.query_proj(global_features)
        key = self.key_proj(torch.cat([saliency_features, boundary_features], dim=1))
        value = self.value_proj(torch.cat([saliency_features, boundary_features], dim=1))

        attended_features, attention_weights = self.multihead_attn(query, key, value)

        output = self.output_proj(attended_features)

        return output


class GradientFlowModule(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gradient_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        self.gate_mechanism = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_features: torch.Tensor,
                saliency_features: torch.Tensor,
                boundary_features: torch.Tensor) -> torch.Tensor:

        concatenated = torch.cat([global_features, saliency_features, boundary_features], dim=1)

        gradient_flow = self.gradient_net(concatenated)

        gate_weights = self.gate_mechanism(concatenated)

        enhanced_features = global_features + gate_weights * gradient_flow

        return enhanced_features


class SCBMambaWithAuxiliary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.main_model = SCBMamba(config)

        self.auxiliary_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim // 2, config.num_classes)
            ) for _ in range(3)
        ])

        self.complexity_predictor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

        self.feature_regularization = FeatureRegularization(config.hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, start_position: str = 'top_left') -> Dict[str, Any]:
        main_outputs = self.main_model(x, start_position)

        auxiliary_logits = []
        for classifier in self.auxiliary_classifiers:
            aux_logit = classifier(main_outputs['global_features'])
            auxiliary_logits.append(aux_logit)

        complexity_logits = self.complexity_predictor(main_outputs['complexity'].unsqueeze(1))

        regularized_features = self.feature_regularization(main_outputs['global_features'])

        main_outputs['auxiliary_logits'] = torch.stack(auxiliary_logits, dim=1)
        main_outputs['complexity_logits'] = complexity_logits
        main_outputs['regularized_features'] = regularized_features

        return main_outputs


class FeatureRegularization(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.regularization_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attention_weights = nn.Parameter(torch.ones(hidden_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.ones_(self.attention_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        regularized = self.regularization_net(features)

        attention_scores = F.softmax(self.attention_weights, dim=0)
        attended_features = features * attention_scores.unsqueeze(0)

        return regularized + attended_features


class ModelFactory:
    @staticmethod
    def create_scb_mamba(config, model_type: str = 'standard') -> nn.Module:
        if model_type == 'standard':
            return SCBMamba(config)
        elif model_type == 'with_auxiliary':
            return SCBMambaWithAuxiliary(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_optimizer(model: nn.Module, learning_rate: float = 1e-4,
                         weight_decay: float = 1e-5) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer,
                         num_epochs: int,
                         warmup_epochs: int = 5) -> torch.optim.lr_scheduler._LRScheduler:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def create_criterion(auxiliary_weight: float = 0.3,
                         consistency_weight: float = 0.1) -> nn.Module:
        return MultiTaskLoss(
            auxiliary_weight=auxiliary_weight,
            consistency_weight=consistency_weight
        )


class MultiTaskLoss(nn.Module):
    def __init__(self, auxiliary_weight: float = 0.3, consistency_weight: float = 0.1):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight
        self.consistency_weight = consistency_weight

        self.classification_loss = nn.CrossEntropyLoss()
        self.consistency_loss = PathConsistencyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs: Dict[str, Any], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        main_loss = self.classification_loss(outputs['final_logits'], targets)

        auxiliary_loss = 0.0
        if 'auxiliary_logits' in outputs:
            for i in range(outputs['auxiliary_logits'].size(1)):
                aux_logits = outputs['auxiliary_logits'][:, i]
                auxiliary_loss += self.classification_loss(aux_logits, targets)
            auxiliary_loss /= outputs['auxiliary_logits'].size(1)

        consistency_loss = 0.0
        if 'path_logits' in outputs:
            consistency_loss = self.consistency_loss(outputs['path_logits'], outputs['final_logits'])

        complexity_loss = 0.0
        if 'complexity_logits' in outputs:
            complexity_targets = torch.zeros_like(outputs['complexity_logits'])
            for i, comp in enumerate(outputs['complexity']):
                if comp < 50:
                    complexity_targets[i, 0] = 1.0
                elif comp < 150:
                    complexity_targets[i, 1] = 1.0
                else:
                    complexity_targets[i, 2] = 1.0

            complexity_loss = self.kl_loss(
                F.log_softmax(outputs['complexity_logits'], dim=1),
                complexity_targets
            )

        total_loss = (main_loss +
                      self.auxiliary_weight * auxiliary_loss +
                      self.consistency_weight * consistency_loss +
                      0.1 * complexity_loss)

        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'auxiliary_loss': auxiliary_loss,
            'consistency_loss': consistency_loss,
            'complexity_loss': complexity_loss
        }


class ModelCheckpoint:
    def __init__(self, checkpoint_dir: str, model_name: str, patience: int = 15):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.patience = patience
        self.best_accuracy = 0.0
        self.counter = 0
        self.early_stop = False

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler,
                        epoch: int, accuracy: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy
        }

        filename = f"{self.model_name}_epoch_{epoch}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_filename = f"{self.model_name}_best.pth"
            best_filepath = os.path.join(self.checkpoint_dir, best_filename)
            torch.save(checkpoint, best_filepath)

    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer = None,
                        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                        filepath: str = None) -> Dict[str, Any]:
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")

        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)

        return checkpoint

    def check_early_stopping(self, accuracy: float) -> bool:
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop