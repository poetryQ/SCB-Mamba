import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from mamba_core import OverlappingPatchEmbedding, PatchEmbedding, MambaConfig, MambaBlock


class MultiScaleSequenceScanner(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16,
                 embed_dim: int = 512,
                 expansion_factors: List[float] = [1.0, 1.5, 2.0],
                 scanning_strategies: List[str] = ['forward', 'reverse', 'overlapping']):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.expansion_factors = expansion_factors
        self.scanning_strategies = scanning_strategies

        self.patch_embeddings = nn.ModuleList()
        for strategy, expand_factor in zip(scanning_strategies, expansion_factors):
            if strategy == 'overlapping':
                stride = max(1, int(patch_size / expand_factor))
                patch_embed = OverlappingPatchEmbedding(
                    img_size=img_size,
                    patch_size=patch_size,
                    stride=stride,
                    embed_dim=embed_dim
                )
            else:
                patch_embed = PatchEmbedding(
                    img_size=img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim
                )
            self.patch_embeddings.append(patch_embed)

        self.start_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center']

        self.deform_conv = DeformConv2d(3, 3, 3, padding=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_starting_point(self, start_position: str, grid_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = grid_size

        if start_position == 'top_left':
            return (0, 0)
        elif start_position == 'top_right':
            return (0, w - 1)
        elif start_position == 'bottom_left':
            return (h - 1, 0)
        elif start_position == 'bottom_right':
            return (h - 1, w - 1)
        elif start_position == 'center':
            return (h // 2, w // 2)
        else:
            return (0, 0)

    def forward_scanning(self, tokens: torch.Tensor, strategy: str, start_position: str = 'top_left') -> torch.Tensor:
        batch_size, num_tokens, embed_dim = tokens.shape
        grid_h = int(math.sqrt(num_tokens))
        grid_w = grid_h

        tokens_2d = tokens.view(batch_size, grid_h, grid_w, embed_dim)

        start_h, start_w = self.get_starting_point(start_position, (grid_h, grid_w))

        if strategy == 'forward':
            scanned_tokens = self._forward_scan(tokens_2d, start_h, start_w)
        elif strategy == 'reverse':
            scanned_tokens = self._reverse_scan(tokens_2d, start_h, start_w)
        elif strategy == 'overlapping':
            scanned_tokens = self._overlapping_scan(tokens_2d, start_h, start_w)
        else:
            scanned_tokens = self._forward_scan(tokens_2d, start_h, start_w)

        return scanned_tokens

    def _forward_scan(self, tokens_2d: torch.Tensor, start_h: int, start_w: int) -> torch.Tensor:
        batch_size, h, w, dim = tokens_2d.shape
        scanned_tokens = []

        for i in range(h):
            if i % 2 == 0:
                for j in range(w):
                    scanned_tokens.append(tokens_2d[:, i, j, :])
            else:
                for j in range(w - 1, -1, -1):
                    scanned_tokens.append(tokens_2d[:, i, j, :])

        return torch.stack(scanned_tokens, dim=1)

    def _reverse_scan(self, tokens_2d: torch.Tensor, start_h: int, start_w: int) -> torch.Tensor:
        batch_size, h, w, dim = tokens_2d.shape
        scanned_tokens = []

        for i in range(h - 1, -1, -1):
            if (h - 1 - i) % 2 == 0:
                for j in range(w - 1, -1, -1):
                    scanned_tokens.append(tokens_2d[:, i, j, :])
            else:
                for j in range(w):
                    scanned_tokens.append(tokens_2d[:, i, j, :])

        return torch.stack(scanned_tokens, dim=1)

    def _overlapping_scan(self, tokens_2d: torch.Tensor, start_h: int, start_w: int) -> torch.Tensor:
        batch_size, h, w, dim = tokens_2d.shape

        expanded_tokens = F.interpolate(
            tokens_2d.permute(0, 3, 1, 2),
            scale_factor=1.5,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1)

        exp_h, exp_w = expanded_tokens.shape[1:3]

        scanned_tokens = []
        for i in range(exp_h):
            if i % 2 == 0:
                for j in range(exp_w):
                    scanned_tokens.append(expanded_tokens[:, i, j, :])
            else:
                for j in range(exp_w - 1, -1, -1):
                    scanned_tokens.append(expanded_tokens[:, i, j, :])

        return torch.stack(scanned_tokens, dim=1)

    def revert_scanning(self, scanned_tokens: torch.Tensor, original_shape: Tuple[int, int],
                        strategy: str, start_position: str = 'top_left') -> torch.Tensor:
        batch_size, seq_len, embed_dim = scanned_tokens.shape
        grid_h, grid_w = original_shape

        if strategy == 'overlapping':
            grid_h = int(grid_h * 1.5)
            grid_w = int(grid_w * 1.5)

        tokens_2d = torch.zeros(batch_size, grid_h, grid_w, embed_dim, device=scanned_tokens.device)

        start_h, start_w = self.get_starting_point(start_position, (grid_h, grid_w))

        idx = 0
        for i in range(grid_h):
            if (strategy == 'forward' and i % 2 == 0) or (strategy == 'reverse' and (grid_h - 1 - i) % 2 == 0):
                for j in range(grid_w):
                    tokens_2d[:, i, j, :] = scanned_tokens[:, idx, :]
                    idx += 1
            else:
                for j in range(grid_w - 1, -1, -1):
                    tokens_2d[:, i, j, :] = scanned_tokens[:, idx, :]
                    idx += 1

        if strategy == 'overlapping':
            tokens_2d = F.interpolate(
                tokens_2d.permute(0, 3, 1, 2),
                size=original_shape,
                mode='bilinear',
                align_corners=True
            ).permute(0, 2, 3, 1)

        return tokens_2d.view(batch_size, -1, embed_dim)

    def forward(self, images: torch.Tensor, start_position: str = 'top_left') -> List[torch.Tensor]:
        batch_size, channels, height, width = images.shape

        offset = torch.zeros(batch_size, 2 * 3 * 3, height, width, device=images.device)
        guided_images = self.deform_conv(images, offset)

        all_tokens = []
        for i, (patch_embed, strategy) in enumerate(zip(self.patch_embeddings, self.scanning_strategies)):
            tokens = patch_embed(guided_images)
            scanned_tokens = self.forward_scanning(tokens, strategy, start_position)
            all_tokens.append(scanned_tokens)

        return all_tokens


class WeightSharingMambaBackbone(nn.Module):
    def __init__(self,
                 config,
                 num_layers: int = 12,
                 num_paths: int = 3):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.num_paths = num_paths

        mamba_config = MambaConfig(
            d_model=config.hidden_dim,
            d_state=config.state_dim,
            expand=config.expand_factor
        )

        self.shared_mamba_layers = nn.ModuleList([
            MambaBlock(mamba_config) for _ in range(num_layers)
        ])

        self.scanner = MultiScaleSequenceScanner(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.hidden_dim,
            expansion_factors=[1.0, 1.5, 2.0],
            scanning_strategies=['forward', 'reverse', 'overlapping']
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * num_paths, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, num_paths),
            nn.Softmax(dim=-1)
        )

        self.path_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(num_paths)
        ])

        self.dropout = nn.Dropout(config.dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                original_image: torch.Tensor,
                saliency_map: torch.Tensor,
                boundary_map: torch.Tensor,
                start_position: str = 'top_left') -> Tuple[torch.Tensor, torch.Tensor]:

        input_images = torch.cat([original_image, saliency_map, boundary_map], dim=1)

        all_path_tokens = self.scanner(input_images, start_position)

        path_outputs = []
        for i, path_tokens in enumerate(all_path_tokens):
            path_tokens = self.path_norms[i](path_tokens)

            for mamba_layer in self.shared_mamba_layers:
                path_tokens = mamba_layer(path_tokens)

            path_tokens = self.dropout(path_tokens)

            original_shape = (
                self.config.image_size[0] // self.config.patch_size,
                self.config.image_size[1] // self.config.patch_size
            )

            reverted_tokens = self.scanner.revert_scanning(
                path_tokens, original_shape,
                self.scanner.scanning_strategies[i], start_position
            )

            path_outputs.append(reverted_tokens)

        concatenated_features = torch.cat(path_outputs, dim=-1)

        batch_size, num_tokens, _ = concatenated_features.shape
        pooled_features = concatenated_features.mean(dim=1)

        gate_weights = self.fusion_gate(pooled_features)

        fused_features = torch.zeros_like(path_outputs[0])
        for i, path_feat in enumerate(path_outputs):
            fused_features += gate_weights[:, i].view(-1, 1, 1) * path_feat

        global_features = fused_features.mean(dim=1)

        return global_features, fused_features


class AdaptivePathSelection(nn.Module):
    def __init__(self,
                 hidden_dim: int = 512,
                 num_paths: int = 3,
                 num_classes: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_paths = num_paths
        self.num_classes = num_classes

        self.path_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.path_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)
            ) for _ in range(num_paths)
        ])

        self.fusion_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_paths, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.path_weights = nn.Parameter(torch.ones(num_paths) / num_paths)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, path_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = path_features[0].shape[0]

        path_logits = []
        path_attentions = []

        for i, path_feat in enumerate(path_features):
            path_global = path_feat.mean(dim=1)

            attended_feat, attention_weights = self.path_attention(
                path_global.unsqueeze(1),
                path_global.unsqueeze(1),
                path_global.unsqueeze(1)
            )

            logits = self.path_classifiers[i](attended_feat.squeeze(1))
            path_logits.append(logits)
            path_attentions.append(attention_weights)

        concatenated_features = torch.cat([feat.mean(dim=1) for feat in path_features], dim=-1)
        fused_logits = self.fusion_classifier(concatenated_features)

        weighted_path_logits = torch.stack(path_logits, dim=1)
        normalized_weights = F.softmax(self.path_weights, dim=0)
        ensemble_logits = torch.sum(weighted_path_logits * normalized_weights.view(1, -1, 1), dim=1)

        return fused_logits, ensemble_logits, torch.stack(path_logits, dim=1)


class MultiPathMambaWithSelection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = WeightSharingMambaBackbone(config)
        self.path_selector = AdaptivePathSelection(
            hidden_dim=config.hidden_dim,
            num_paths=3,
            num_classes=config.num_classes
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                original_image: torch.Tensor,
                saliency_map: torch.Tensor,
                boundary_map: torch.Tensor,
                start_position: str = 'top_left') -> Dict[str, torch.Tensor]:

        global_features, path_features = self.backbone(
            original_image, saliency_map, boundary_map, start_position
        )

        batch_size, num_tokens, hidden_dim = path_features.shape
        path_features_list = [
            path_features[:, :num_tokens // 3, :],
            path_features[:, num_tokens // 3:2 * num_tokens // 3, :],
            path_features[:, 2 * num_tokens // 3:, :]
        ]

        fused_logits, ensemble_logits, path_logits = self.path_selector(path_features_list)

        final_logits = self.final_classifier(global_features)

        return {
            'final_logits': final_logits,
            'fused_logits': fused_logits,
            'ensemble_logits': ensemble_logits,
            'path_logits': path_logits,
            'global_features': global_features,
            'path_features': path_features
        }


class PathConsistencyLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, path_logits: torch.Tensor, final_logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_paths, num_classes = path_logits.shape

        path_probs = F.softmax(path_logits / self.temperature, dim=-1)
        final_probs = F.softmax(final_logits / self.temperature, dim=-1).unsqueeze(1).expand_as(path_probs)

        consistency_loss = self.kl_loss(
            F.log_softmax(path_logits / self.temperature, dim=-1),
            F.softmax(final_logits.detach() / self.temperature, dim=-1).unsqueeze(1).expand_as(path_probs)
        )

        path_agreement = F.mse_loss(
            path_probs.mean(dim=1),
            final_probs.mean(dim=1)
        )

        return self.alpha * (consistency_loss + path_agreement)


class MultiPathTrainingManager:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 consistency_alpha: float = 0.1):
        self.model = model
        self.consistency_loss = PathConsistencyLoss(alpha=consistency_alpha)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_metrics = []

    def train_epoch(self, dataloader, device: str) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            saliency_maps = torch.randn_like(images) * 0.1 + 0.5
            boundary_maps = torch.randn_like(images) * 0.1 + 0.5

            self.optimizer.zero_grad()

            outputs = self.model(images, saliency_maps, boundary_maps)

            classification_loss = self.criterion(outputs['final_logits'], labels)
            consistency_loss = self.consistency_loss(outputs['path_logits'], outputs['final_logits'])

            loss = classification_loss + consistency_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs['final_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }

    def validate(self, dataloader, device: str) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                saliency_maps = torch.randn_like(images) * 0.1 + 0.5
                boundary_maps = torch.randn_like(images) * 0.1 + 0.5

                outputs = self.model(images, saliency_maps, boundary_maps)

                loss = self.criterion(outputs['final_logits'], labels)

                total_loss += loss.item()

                _, predicted = outputs['final_logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)

        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

        return metrics