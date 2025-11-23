import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import cv2
from scipy import ndimage
from skimage.segmentation import slic
from skimage.util import img_as_float


class SuperPixelClustering(nn.Module):
    def __init__(self,
                 complexity_thresholds: List[float] = [50, 150],
                 base_clusters: List[int] = [30, 80, 150],
                 cluster_increments: List[int] = [10, 15, 25],
                 grayscale_levels: int = 256):
        super().__init__()
        self.complexity_thresholds = complexity_thresholds
        self.base_clusters = base_clusters
        self.cluster_increments = cluster_increments
        self.grayscale_levels = grayscale_levels

        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def compute_texture_complexity(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = image.shape

        if channels == 3:
            image_gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        else:
            image_gray = image.squeeze(1)

        image_gray = (image_gray * (self.grayscale_levels - 1)).to(torch.int32)

        complexities = []
        for i in range(batch_size):
            img_np = image_gray[i].cpu().numpy().astype(np.uint8)
            complexity = self._compute_glcm_contrast(img_np)
            complexities.append(complexity)

        return torch.tensor(complexities, device=image.device, dtype=torch.float32)

    def _compute_glcm_contrast(self, image: np.ndarray, distance: int = 1, angle: float = 0) -> float:
        rows, cols = image.shape
        glcm = np.zeros((self.grayscale_levels, self.grayscale_levels), dtype=np.float32)

        dx = int(round(distance * np.cos(angle)))
        dy = int(round(distance * np.sin(angle)))

        valid_pixels = 0
        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                pixel1 = image[i, j]
                pixel2 = image[i + dy, j + dx]
                glcm[pixel1, pixel2] += 1
                valid_pixels += 1

        if valid_pixels > 0:
            glcm /= valid_pixels

        contrast = 0.0
        for i in range(self.grayscale_levels):
            for j in range(self.grayscale_levels):
                contrast += glcm[i, j] * (i - j) ** 2

        return contrast

    def get_cluster_numbers(self, complexity: torch.Tensor) -> List[List[int]]:
        batch_size = complexity.shape[0]
        cluster_sets = []

        for i in range(batch_size):
            comp_val = complexity[i].item()

            if comp_val < self.complexity_thresholds[0]:
                base_k = self.base_clusters[0]
                increment = self.cluster_increments[0]
            elif comp_val < self.complexity_thresholds[1]:
                base_k = self.base_clusters[1]
                increment = self.cluster_increments[1]
            else:
                base_k = self.base_clusters[2]
                increment = self.cluster_increments[2]

            cluster_set = [base_k - increment, base_k, base_k + increment]
            cluster_sets.append(cluster_set)

        return cluster_sets

    def apply_slic_clustering(self, image: torch.Tensor, n_segments: int) -> torch.Tensor:
        batch_size, channels, height, width = image.shape

        saliency_maps = []
        for i in range(batch_size):
            if channels == 3:
                img_np = image[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = image[i].squeeze(0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_np = np.stack([img_np] * 3, axis=-1)

            img_float = img_as_float(img_np)

            try:
                segments = slic(img_float, n_segments=n_segments, compactness=10, sigma=1, start_label=1)

                saliency_map = np.zeros((height, width), dtype=np.float32)
                for seg_val in np.unique(segments):
                    mask = (segments == seg_val)
                    region_mean = np.mean(img_np[mask])
                    saliency_map[mask] = region_mean

                saliency_map = saliency_map / 255.0

            except Exception as e:
                saliency_map = np.ones((height, width), dtype=np.float32) * 0.5

            saliency_maps.append(torch.tensor(saliency_map, device=image.device, dtype=torch.float32))

        return torch.stack(saliency_maps).unsqueeze(1)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        complexity = self.compute_texture_complexity(image)
        cluster_sets = self.get_cluster_numbers(complexity)

        saliency_maps = []
        for i, cluster_set in enumerate(cluster_sets):
            cluster_saliencies = []
            for n_segments in cluster_set:
                saliency = self.apply_slic_clustering(image[i:i + 1], n_segments)
                cluster_saliencies.append(saliency)
            saliency_maps.append(torch.cat(cluster_saliencies, dim=1))

        multi_scale_saliency = torch.cat(saliency_maps, dim=0)

        return multi_scale_saliency, complexity, cluster_sets


class GrayScaleDifferenceEnhancement(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: List[int] = [64, 128, 256],
                 complexity_thresholds: List[float] = [50, 150],
                 base_clusters: List[int] = [30, 80, 150],
                 cluster_increments: List[int] = [10, 15, 25]):
        super().__init__()

        self.superpixel_clustering = SuperPixelClustering(
            complexity_thresholds=complexity_thresholds,
            base_clusters=base_clusters,
            cluster_increments=cluster_increments
        )

        self.conv_layers = nn.ModuleList()
        prev_channels = 3
        for hidden_dim in hidden_dims:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(prev_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ))
            prev_channels = hidden_dim

        self.upsample_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i - 1], 3, padding=1),
                nn.BatchNorm2d(hidden_dims[i - 1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ))

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[0] * 3, hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[0], 1, 1),
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

    def align_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        aligned_features = []
        for feat in features:
            if feat.size(2) != features[0].size(2) or feat.size(3) != features[0].size(3):
                feat = F.interpolate(feat, size=features[0].shape[2:], mode='bilinear', align_corners=True)
            aligned_features.append(feat)

        return torch.cat(aligned_features, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape

        multi_scale_saliency, complexity, cluster_sets = self.superpixel_clustering(x)

        multi_scale_features = []
        for i in range(3):
            saliency_map = multi_scale_saliency[:, i:i + 1]

            features = [saliency_map]
            current_feat = saliency_map

            for conv_layer in self.conv_layers:
                current_feat = conv_layer(current_feat)
                features.append(current_feat)

            for j, upsample_layer in enumerate(self.upsample_layers):
                current_feat = upsample_layer(current_feat)
                if j < len(features) - 2:
                    current_feat = current_feat + features[-(j + 3)]

            multi_scale_features.append(current_feat)

        fused_features = self.align_features(multi_scale_features)
        saliency_map = self.fusion_conv(fused_features)

        offset = torch.zeros(batch_size, 2 * 3 * 3, height, width, device=x.device)
        enhanced_saliency = self.deform_conv(saliency_map, offset)

        return enhanced_saliency, multi_scale_saliency, complexity


class AdaptiveSuperPixelGridSearch:
    def __init__(self,
                 complexity_ranges: List[Tuple[float, float]] = [(0, 50), (50, 150), (150, 1000)],
                 k_search_space: List[List[int]] = [[20, 30, 40], [40, 80, 120], [100, 150, 180]],
                 n_search_space: List[List[int]] = [[5, 10, 15], [10, 15, 20], [15, 25, 30]]):
        self.complexity_ranges = complexity_ranges
        self.k_search_space = k_search_space
        self.n_search_space = n_search_space

    def optimize_parameters(self,
                            dataset: torch.utils.data.Dataset,
                            model: nn.Module,
                            num_samples: int = 100) -> Dict[str, Any]:
        best_params = {}
        best_accuracy = 0.0

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=4
        )

        for complexity_range, k_space, n_space in zip(self.complexity_ranges, self.k_search_space, self.n_search_space):
            print(f"Optimizing for complexity range {complexity_range}")

            range_samples = []
            count = 0
            for data in dataloader:
                if count >= num_samples:
                    break

                image = data['image']
                complexity = model.superpixel_clustering.compute_texture_complexity(image)

                if complexity_range[0] <= complexity.item() < complexity_range[1]:
                    range_samples.append(data)
                    count += 1

            if not range_samples:
                continue

            current_best_k = k_space[0]
            current_best_n = n_space[0]
            current_best_acc = 0.0

            for k in k_space:
                for n in n_space:
                    print(f"Testing K={k}, N={n}")

                    accuracy = self._evaluate_parameters(range_samples, model, k, n)

                    if accuracy > current_best_acc:
                        current_best_acc = accuracy
                        current_best_k = k
                        current_best_n = n

            best_params[f'range_{complexity_range[0]}_{complexity_range[1]}'] = {
                'K': current_best_k,
                'N': current_best_n,
                'accuracy': current_best_acc
            }

            if current_best_acc > best_accuracy:
                best_accuracy = current_best_acc

        return best_params

    def _evaluate_parameters(self,
                             samples: List[Dict[str, Any]],
                             model: nn.Module,
                             k: int,
                             n: int) -> float:
        total_correct = 0
        total_samples = 0

        model.eval()
        with torch.no_grad():
            for data in samples:
                image = data['image']
                label = data['label']

                original_k = model.superpixel_clustering.base_clusters
                original_n = model.superpixel_clustering.cluster_increments

                complexity = model.superpixel_clustering.compute_texture_complexity(image)
                comp_val = complexity.item()

                if comp_val < 50:
                    model.superpixel_clustering.base_clusters[0] = k
                    model.superpixel_clustering.cluster_increments[0] = n
                elif comp_val < 150:
                    model.superpixel_clustering.base_clusters[1] = k
                    model.superpixel_clustering.cluster_increments[1] = n
                else:
                    model.superpixel_clustering.base_clusters[2] = k
                    model.superpixel_clustering.cluster_increments[2] = n

                output = model(image)
                pred = output.argmax(dim=1)

                if pred.item() == label.item():
                    total_correct += 1
                total_samples += 1

                model.superpixel_clustering.base_clusters = original_k
                model.superpixel_clustering.cluster_increments = original_n

        return total_correct / total_samples if total_samples > 0 else 0.0