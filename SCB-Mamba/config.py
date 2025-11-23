import torch
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class ModelConfig:
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 3
    hidden_dim: int = 512
    state_dim: int = 16
    num_layers: int = 12
    expand_factor: int = 2
    conv_bias: bool = True
    bias: bool = False
    dropout: float = 0.1

    superpixel_config: dict = None
    boundary_config: dict = None

    def __post_init__(self):
        if self.superpixel_config is None:
            self.superpixel_config = {
                'complexity_thresholds': [50, 150],
                'base_clusters': [30, 80, 150],
                'cluster_increments': [10, 15, 25],
                'grayscale_levels': 256
            }

        if self.boundary_config is None:
            self.boundary_config = {
                'gate_channels': 32,
                'sobel_kernels': [
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                    [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
                ]
            }


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    patience: int = 15

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True

    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class DataConfig:
    data_root: str = 'data'
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    augmentation: bool = True
    resize: Tuple[int, int] = (224, 224)

    class_names: List[str] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['normal', 'mild', 'severe']