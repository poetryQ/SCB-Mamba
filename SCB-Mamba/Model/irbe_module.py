import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import numpy as np
from typing import Tuple, List, Optional, Dict


class FeatureAlignment(nn.Module):
    def __init__(self,
                 in_channels: int,
                 group_channels: int = 32,
                 num_groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.group_channels = group_channels
        self.num_groups = num_groups

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2 * num_groups * 3 * 3, 3, padding=1)
        )

        self.deform_conv = DeformConv2d(
            in_channels, in_channels, 3, padding=1,
            groups=num_groups, bias=False
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat_m: torch.Tensor, feat_n: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feat_m.shape

        if feat_m.size(2) != feat_n.size(2) or feat_m.size(3) != feat_n.size(3):
            feat_n = F.interpolate(feat_n, size=(height, width), mode='bilinear', align_corners=True)

        concat_feat = torch.cat([feat_m, feat_n], dim=1)
        offset = self.offset_conv(concat_feat)

        aligned_feat = self.deform_conv(feat_n, offset)

        return aligned_feat


class GatedFusion(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2, 1),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        if feat1.size(2) != feat2.size(2) or feat1.size(3) != feat2.size(3):
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=True)

        concat_feat = torch.cat([feat1, feat2], dim=1)
        gates = self.gate_conv(concat_feat)

        gate1 = gates[:, 0:1]
        gate2 = gates[:, 1:2]

        fused_feat = gate1 * feat1 + gate2 * feat2

        return fused_feat


class MultiDirectionalEdgeFilter(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_kernels = [
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
        ]

        self.register_buffer('kernels', torch.tensor(sobel_kernels, dtype=torch.float32).unsqueeze(1))

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        edge_maps = []
        for i in range(4):
            kernel = self.kernels[i].repeat(channels, 1, 1, 1)
            edge_map = F.conv2d(x, kernel, padding=1, groups=channels)
            edge_maps.append(edge_map)

        edge_features = torch.cat(edge_maps, dim=1)
        enhanced_edges = self.fusion_conv(edge_features)

        return enhanced_edges


class InjuredRegionBoundaryEnhancement(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 feature_channels: int = 64,
                 gate_channels: int = 32):
        super().__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        self.feature_alignment = FeatureAlignment(
            in_channels=feature_channels,
            group_channels=gate_channels
        )

        self.gated_fusion = GatedFusion(in_channels=feature_channels)

        self.edge_enhancement = MultiDirectionalEdgeFilter()

        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(feature_channels + 1, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self.deform_conv = DeformConv2d(1, 1, 3, padding=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, original_image: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = original_image.shape

        original_features = self.feature_extraction(original_image)

        if saliency_map.size(1) == 1:
            saliency_features = self.feature_extraction(
                saliency_map.repeat(1, 3, 1, 1) if saliency_map.size(1) == 1 else saliency_map
            )
        else:
            saliency_features = self.feature_extraction(saliency_map)

        aligned_saliency = self.feature_alignment(original_features, saliency_features)

        fused_features = self.gated_fusion(original_features, aligned_saliency)

        edge_enhanced = self.edge_enhancement(fused_features)

        boundary_features = torch.cat([fused_features, edge_enhanced], dim=1)
        boundary_map = self.boundary_refinement(boundary_features)

        offset = torch.zeros(batch_size, 2 * 3 * 3, height, width, device=original_image.device)
        refined_boundary = self.deform_conv(boundary_map, offset)

        return refined_boundary


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, boundary_map: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = boundary_map.shape

        boundary_grad_x = torch.abs(boundary_map[:, :, :, 1:] - boundary_map[:, :, :, :-1])
        boundary_grad_y = torch.abs(boundary_map[:, :, 1:, :] - boundary_map[:, :, :-1, :])

        saliency_grad_x = torch.abs(saliency_map[:, :, :, 1:] - saliency_map[:, :, :, :-1])
        saliency_grad_y = torch.abs(saliency_map[:, :, 1:, :] - saliency_map[:, :, :-1, :])

        loss_x = F.mse_loss(boundary_grad_x, saliency_grad_x)
        loss_y = F.mse_loss(boundary_grad_y, saliency_grad_y)

        return self.weight * (loss_x + loss_y)


class BoundaryPrecisionLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predicted_boundary: torch.Tensor, target_boundary: torch.Tensor) -> torch.Tensor:
        intersection = (predicted_boundary * target_boundary).sum()
        predicted_sum = predicted_boundary.sum()

        precision = intersection / (predicted_sum + self.epsilon)

        return 1 - precision


class BoundaryEnhancementOptimizer:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.spatial_loss = SpatialConsistencyLoss()
        self.precision_loss = BoundaryPrecisionLoss()

    def train_step(self,
                   original_images: torch.Tensor,
                   saliency_maps: torch.Tensor,
                   target_boundaries: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        enhanced_boundaries = self.model(original_images, saliency_maps)

        spatial_loss = self.spatial_loss(enhanced_boundaries, saliency_maps)
        precision_loss = self.precision_loss(enhanced_boundaries, target_boundaries)

        total_loss = spatial_loss + precision_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'precision_loss': precision_loss.item()
        }

    def validate(self,
                 original_images: torch.Tensor,
                 saliency_maps: torch.Tensor,
                 target_boundaries: torch.Tensor) -> Dict[str, float]:
        self.model.eval()

        with torch.no_grad():
            enhanced_boundaries = self.model(original_images, saliency_maps)

            spatial_loss = self.spatial_loss(enhanced_boundaries, saliency_maps)
            precision_loss = self.precision_loss(enhanced_boundaries, target_boundaries)

            total_loss = spatial_loss + precision_loss

            boundary_accuracy = self._compute_boundary_accuracy(enhanced_boundaries, target_boundaries)

        return {
            'total_loss': total_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'precision_loss': precision_loss.item(),
            'boundary_accuracy': boundary_accuracy
        }

    def _compute_boundary_accuracy(self,
                                   predicted: torch.Tensor,
                                   target: torch.Tensor,
                                   threshold: float = 0.5) -> float:
        predicted_binary = (predicted > threshold).float()
        target_binary = (target > threshold).float()

        correct = (predicted_binary == target_binary).float().mean()

        return correct.item()