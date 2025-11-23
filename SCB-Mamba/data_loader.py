import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import os
import cv2
from typing import Optional, Callable, Tuple, List, Dict, Any
import json
from pathlib import Path


class OCTDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: Optional[Callable] = None,
                 is_train: bool = True,
                 class_names: List[str] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.target_size = target_size
        self.class_names = class_names or ['normal', 'mild', 'severe']

        self.samples = []
        self.labels = []
        self._load_data()

        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def _load_data(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append(img_path)
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', self.target_size, color='white')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label_idx = self.label_to_idx[label]

        return {
            'image': image,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'label_name': label,
            'image_path': img_path
        }


class OCTDataAugmentation:
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 augmentation: bool = True):

        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.augmentation = augmentation

        if self.augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_train_transform(self):
        return self.train_transform

    def get_val_transform(self):
        return self.val_transform


class OCTDataManager:
    def __init__(self, config):
        self.config = config
        self.augmentation = OCTDataAugmentation(
            target_size=config.resize,
            mean=config.mean,
            std=config.std,
            augmentation=config.augmentation
        )

    def create_datasets(self, data_root: str):
        full_dataset = OCTDataset(
            data_dir=data_root,
            transform=self.augmentation.get_train_transform(),
            is_train=True,
            class_names=self.config.class_names,
            target_size=self.config.resize
        )

        total_size = len(full_dataset)
        train_size = int(self.config.train_ratio * total_size)
        val_size = int(self.config.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_dataset.dataset.transform = self.augmentation.get_val_transform()
        test_dataset.dataset.transform = self.augmentation.get_val_transform()

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self, data_root: str):
        train_dataset, val_dataset, test_dataset = self.create_datasets(data_root)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        return train_loader, val_loader, test_loader

    def get_class_distribution(self, data_root: str):
        dataset = OCTDataset(
            data_dir=data_root,
            transform=None,
            is_train=True,
            class_names=self.config.class_names
        )

        class_counts = {class_name: 0 for class_name in self.config.class_names}
        for label in dataset.labels:
            class_counts[label] += 1

        return class_counts


class DataPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.grayscale_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def compute_glcm_features(self, image: torch.Tensor, distances: List[int] = [1],
                              angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]) -> Dict[str, float]:
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().numpy()
        else:
            image_np = np.array(image)

        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        features = {}

        for distance in distances:
            for angle in angles:
                glcm = self._compute_glcm(image_np, distance, angle)
                if glcm is not None:
                    contrast = self._compute_contrast(glcm)
                    features[f'contrast_d{distance}_a{int(np.degrees(angle))}'] = contrast

        return features

    def _compute_glcm(self, image: np.ndarray, distance: int, angle: float) -> Optional[np.ndarray]:
        try:
            from skimage.feature import greycomatrix
            glcm = greycomatrix(image,
                                distances=[distance],
                                angles=[angle],
                                levels=256,
                                symmetric=True,
                                normed=True)
            return glcm[:, :, 0, 0]
        except ImportError:
            return self._compute_glcm_manual(image, distance, angle)

    def _compute_glcm_manual(self, image: np.ndarray, distance: int, angle: float) -> Optional[np.ndarray]:
        rows, cols = image.shape
        glcm = np.zeros((256, 256), dtype=np.float32)

        dx = int(round(distance * np.cos(angle)))
        dy = int(round(distance * np.sin(angle)))

        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                pixel1 = image[i, j]
                pixel2 = image[i + dy, j + dx]
                glcm[pixel1, pixel2] += 1

        if np.sum(glcm) > 0:
            glcm = glcm / np.sum(glcm)
            return glcm
        return None

    def _compute_contrast(self, glcm: np.ndarray) -> float:
        contrast = 0.0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast

    def compute_texture_complexity(self, image: torch.Tensor) -> float:
        glcm_features = self.compute_glcm_features(image)
        contrast_values = [val for key, val in glcm_features.items() if 'contrast' in key]
        return np.mean(contrast_values) if contrast_values else 0.0


class DataAnalyzer:
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config.resize)

    def analyze_dataset(self, data_root: str):
        dataset = OCTDataset(
            data_dir=data_root,
            transform=None,
            class_names=self.config.class_names
        )

        analysis = {
            'total_samples': len(dataset),
            'class_distribution': {},
            'image_statistics': {},
            'texture_complexity': {}
        }

        class_counts = {}
        image_sizes = []
        texture_complexities = {class_name: [] for class_name in self.config.class_names}

        for i in range(len(dataset)):
            sample = dataset[i]
            label_name = sample['label_name']

            if label_name not in class_counts:
                class_counts[label_name] = 0
            class_counts[label_name] += 1

            image = sample['image']
            if isinstance(image, Image.Image):
                image_size = image.size
                image_tensor = transforms.ToTensor()(image).mean(0).unsqueeze(0)
            else:
                image_size = image.shape[1:]
                image_tensor = image.mean(0).unsqueeze(0) if image.dim() > 2 else image.unsqueeze(0)

            image_sizes.append(image_size)

            complexity = self.preprocessor.compute_texture_complexity(image_tensor)
            texture_complexities[label_name].append(complexity)

        analysis['class_distribution'] = class_counts
        analysis['image_statistics'] = {
            'size_range': f"{(min(image_sizes, key=lambda x: x[0] * x[1]) if image_sizes else (0, 0))} to {(max(image_sizes, key=lambda x: x[0] * x[1]) if image_sizes else (0, 0))}",
            'average_size': f"({np.mean([s[0] for s in image_sizes]):.1f}, {np.mean([s[1] for s in image_sizes]):.1f})" if image_sizes else "(0,0)"
        }

        for class_name in self.config.class_names:
            if texture_complexities[class_name]:
                analysis['texture_complexity'][class_name] = {
                    'mean': np.mean(texture_complexities[class_name]),
                    'std': np.std(texture_complexities[class_name]),
                    'min': np.min(texture_complexities[class_name]),
                    'max': np.max(texture_complexities[class_name])
                }

        return analysis

    def generate_report(self, data_root: str, save_path: Optional[str] = None):
        analysis = self.analyze_dataset(data_root)

        report = []
        report.append("Dataset Analysis Report")
        report.append("=" * 50)
        report.append(f"Total Samples: {analysis['total_samples']}")
        report.append("")

        report.append("Class Distribution:")
        report.append("-" * 20)
        for class_name, count in analysis['class_distribution'].items():
            percentage = (count / analysis['total_samples']) * 100
            report.append(f"{class_name}: {count} ({percentage:.1f}%)")

        report.append("")
        report.append("Image Statistics:")
        report.append("-" * 20)
        for key, value in analysis['image_statistics'].items():
            report.append(f"{key}: {value}")

        report.append("")
        report.append("Texture Complexity:")
        report.append("-" * 20)
        for class_name, stats in analysis['texture_complexity'].items():
            report.append(f"{class_name}:")
            report.append(f"  Mean: {stats['mean']:.4f}")
            report.append(f"  Std: {stats['std']:.4f}")
            report.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)

        return report_str