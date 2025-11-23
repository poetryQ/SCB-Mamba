import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import hashlib


class AdvancedLogger:
    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)

        os.makedirs(self.experiment_dir, exist_ok=True)

        self.log_file = os.path.join(self.experiment_dir, 'training.log')
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        self.config_file = os.path.join(self.experiment_dir, 'config.json')

        self.metrics_history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rates': [], 'epoch_times': []
        }

        self._setup_logging()

    def _setup_logging(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)

    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                    learning_rate: float, epoch_time: float):
        metrics_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_loss': train_metrics.get('train_loss', 0),
            'train_accuracy': train_metrics.get('train_accuracy', 0),
            'val_loss': val_metrics.get('val_loss', 0),
            'val_accuracy': val_metrics.get('val_accuracy', 0),
            'learning_rate': learning_rate,
            'epoch_time': epoch_time
        }

        for key, value in metrics_entry.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        self._save_metrics_to_file()

        self.log_info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics.get('train_loss', 0):.4f} | "
            f"Train Acc: {train_metrics.get('train_accuracy', 0):.2f}% | "
            f"Val Loss: {val_metrics.get('val_loss', 0):.4f} | "
            f"Val Acc: {val_metrics.get('val_accuracy', 0):.2f}% | "
            f"LR: {learning_rate:.2e} | "
            f"Time: {epoch_time:.2f}s"
        )

    def _save_metrics_to_file(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def save_config(self, config: Dict):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, cls=CustomJSONEncoder)

    def plot_training_curves(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, 'training_curves.png')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(epochs, self.metrics_history['train_accuracy'], 'b-', label='Train Accuracy', alpha=0.7)
        ax2.plot(epochs, self.metrics_history['val_accuracy'], 'r-', label='Val Accuracy', alpha=0.7)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate
        ax3.plot(epochs, self.metrics_history['learning_rates'], 'g-', alpha=0.7)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Epoch times
        ax4.plot(epochs, self.metrics_history['epoch_times'], 'purple', alpha=0.7)
        ax4.set_title('Epoch Training Time')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path


class ModelAnalyzer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.analysis_results = {}

    def analyze_model_complexity(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        """Analyze model complexity and parameters"""
        total_params = 0
        trainable_params = 0
        layer_breakdown = []

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                num_params = sum(p.numel() for p in module.parameters())
                num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

                if num_params > 0:
                    total_params += num_params
                    trainable_params += num_trainable
                    layer_breakdown.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'parameters': num_params,
                        'trainable_parameters': num_trainable,
                        'percentage': (num_params / total_params) * 100
                    })

        # Estimate FLOPs
        flops = self._estimate_flops(input_shape)

        self.analysis_results['complexity'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'flops': flops,
            'layer_breakdown': layer_breakdown
        }

        return self.analysis_results['complexity']

    def _estimate_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for the model"""
        try:
            from thop import profile
            input_tensor = torch.randn(input_shape)
            flops, params = profile(self.model, inputs=(input_tensor,), verbose=False)
            return flops
        except ImportError:
            # Fallback estimation
            return self._fallback_flops_estimation(input_shape)

    def _fallback_flops_estimation(self, input_shape: Tuple[int, ...]) -> int:
        """Fallback FLOPs estimation when thop is not available"""
        flops = 0
        batch_size, channels, height, width = input_shape

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = (output_h * output_w) * (kernel_h * kernel_w * in_channels) * out_channels
                output_h = (height + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
                output_w = (width + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                flops += output_h * output_w * kernel_ops * module.out_channels

                height, width = output_h, output_w

            elif isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features
                flops += 2 * module.in_features * module.out_features

        return flops

    def analyze_memory_usage(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        """Analyze memory usage of the model"""
        try:
            from torch.utils.benchmark import Timer

            input_tensor = torch.randn(input_shape)

            # Measure memory during forward pass
            def forward_pass():
                with torch.no_grad():
                    _ = self.model(input_tensor)

            timer = Timer(stmt='forward_pass()', globals={'forward_pass': forward_pass})
            result = timer.timeit(10)

            # Estimate memory (this is approximate)
            param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())

            self.analysis_results['memory'] = {
                'parameter_memory_mb': param_memory / (1024 ** 2),
                'forward_time_ms': result.mean * 1000,
                'forward_time_std_ms': result.std * 1000
            }

            return self.analysis_results['memory']

        except Exception as e:
            print(f"Memory analysis failed: {e}")
            return {}

    def generate_model_report(self, save_path: str = None):
        """Generate comprehensive model analysis report"""
        complexity = self.analyze_model_complexity()
        memory = self.analyze_memory_usage()

        report = {
            'model_analysis': {
                'complexity_analysis': complexity,
                'memory_analysis': memory,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, cls=CustomJSONEncoder)

        return report

    def print_model_summary(self):
        """Print model summary to console"""
        complexity = self.analysis_results.get('complexity', {})
        memory = self.analysis_results.get('memory', {})

        print("\n" + "=" * 60)
        print("MODEL ANALYSIS SUMMARY")
        print("=" * 60)

        if complexity:
            print(f"\nParameters:")
            print(f"  Total: {complexity['total_parameters']:,}")
            print(f"  Trainable: {complexity['trainable_parameters']:,}")
            print(f"  Non-trainable: {complexity['non_trainable_parameters']:,}")

            if 'flops' in complexity:
                print(f"  FLOPs: {complexity['flops']:,}")

        if memory:
            print(f"\nMemory Usage:")
            print(f"  Parameter Memory: {memory['parameter_memory_mb']:.2f} MB")
            print(f"  Forward Time: {memory['forward_time_ms']:.2f} ± {memory['forward_time_std_ms']:.2f} ms")

        print(f"\nTop 5 Layers by Parameter Count:")
        layers = complexity.get('layer_breakdown', [])
        layers_sorted = sorted(layers, key=lambda x: x['parameters'], reverse=True)[:5]

        for i, layer in enumerate(layers_sorted, 1):
            print(f"  {i}. {layer['name']} ({layer['type']}): {layer['parameters']:,} params")

        print("=" * 60)


class DataValidator:
    def __init__(self, data_dir: str, expected_classes: List[str]):
        self.data_dir = data_dir
        self.expected_classes = expected_classes

    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset structure and integrity"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Data directory does not exist: {self.data_dir}")
            return validation_results

        # Check class directories
        found_classes = []
        for class_name in self.expected_classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Class directory missing: {class_dir}")
            else:
                found_classes.append(class_name)

        # Count images per class
        class_counts = {}
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        for class_name in found_classes:
            class_dir = os.path.join(self.data_dir, class_name)
            image_files = []

            for file in os.listdir(class_dir):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_files.append(file)

            class_counts[class_name] = len(image_files)

            if len(image_files) == 0:
                validation_results['warnings'].append(f"No images found in class: {class_name}")

        validation_results['statistics']['class_counts'] = class_counts
        validation_results['statistics']['total_images'] = sum(class_counts.values())

        # Validate sample images
        if validation_results['statistics']['total_images'] > 0:
            image_validation = self._validate_sample_images()
            validation_results['image_validation'] = image_validation

            if not image_validation['all_valid']:
                validation_results['warnings'].extend(image_validation['issues'])

        return validation_results

    def _validate_sample_images(self) -> Dict[str, Any]:
        """Validate sample images from each class"""
        results = {
            'all_valid': True,
            'issues': [],
            'image_stats': {}
        }

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        for class_name in self.expected_classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            image_files = [f for f in os.listdir(class_dir)
                           if any(f.lower().endswith(ext) for ext in valid_extensions)]

            if not image_files:
                continue

            # Check first few images
            sample_files = image_files[:min(5, len(image_files))]
            class_stats = {
                'checked_images': len(sample_files),
                'valid_images': 0,
                'sizes': [],
                'formats': []
            }

            for img_file in sample_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        class_stats['valid_images'] += 1
                        class_stats['sizes'].append(img.size)
                        class_stats['formats'].append(img.format)

                        # Check if image can be converted to tensor
                        img_tensor = torch.from_numpy(np.array(img)).float()
                        if img_tensor.numel() == 0:
                            raise ValueError("Empty image tensor")

                except Exception as e:
                    results['issues'].append(f"Invalid image {img_path}: {str(e)}")
                    results['all_valid'] = False

            results['image_stats'][class_name] = class_stats

        return results

    def generate_validation_report(self, save_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        validation_results = self.validate_dataset()

        report = {
            'validation_results': validation_results,
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_path': self.data_dir,
            'expected_classes': self.expected_classes
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report


class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = {}
        self.start_time = time.time()

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self):
        epoch_time = time.time() - self.epoch_start_time
        return epoch_time

    def track_metric(self, metric_name: str, value: float, epoch: int):
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []

        if len(self.metrics_history[metric_name]) <= epoch:
            self.metrics_history[metric_name].extend([None] * (epoch - len(self.metrics_history[metric_name]) + 1))

        self.metrics_history[metric_name][epoch] = value

    def get_metric_trend(self, metric_name: str, window: int = 5) -> Dict[str, Any]:
        if metric_name not in self.metrics_history:
            return {}

        values = self.metrics_history[metric_name]
        valid_values = [v for v in values if v is not None]

        if len(valid_values) < 2:
            return {}

        recent_values = valid_values[-window:]

        # Calculate trend
        if len(recent_values) >= 2:
            trend = recent_values[-1] - recent_values[0]
            trend_percentage = (trend / abs(recent_values[0])) * 100 if recent_values[0] != 0 else 0
        else:
            trend = 0
            trend_percentage = 0

        return {
            'current': recent_values[-1] if recent_values else 0,
            'trend': trend,
            'trend_percentage': trend_percentage,
            'mean': np.mean(recent_values) if recent_values else 0,
            'std': np.std(recent_values) if recent_values else 0,
            'min': min(recent_values) if recent_values else 0,
            'max': max(recent_values) if recent_values else 0
        }

    def should_early_stop(self, metric_name: str, patience: int = 10, min_delta: float = 0.001) -> bool:
        trend = self.get_metric_trend(metric_name, window=patience)

        if not trend:
            return False

        # Check if metric is getting worse
        if trend['trend_percentage'] > min_delta:
            return True

        return False

    def generate_performance_report(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time

        report = {
            'total_training_time': total_time,
            'metrics_summary': {},
            'performance_insights': []
        }

        for metric_name, values in self.metrics_history.items():
            valid_values = [v for v in values if v is not None]

            if valid_values:
                report['metrics_summary'][metric_name] = {
                    'final_value': valid_values[-1],
                    'best_value': min(valid_values) if 'loss' in metric_name.lower() else max(valid_values),
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'num_epochs': len(valid_values)
                }

                # Generate insights
                trend = self.get_metric_trend(metric_name)
                if trend and abs(trend['trend_percentage']) > 1.0:
                    insight = f"{metric_name}: {trend['trend_percentage']:+.2f}% trend over last 5 epochs"
                    report['performance_insights'].append(insight)

        return report


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling various data types"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class ExperimentManager:
    def __init__(self, base_dir: str = 'experiments'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.active_experiments = {}
        self.experiment_history = self._load_experiment_history()

    def _load_experiment_history(self) -> Dict:
        history_file = os.path.join(self.base_dir, 'experiment_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_experiment_history(self):
        history_file = os.path.join(self.base_dir, 'experiment_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.experiment_history, f, indent=2)

    def create_experiment(self, experiment_name: str, config: Dict) -> str:
        """Create a new experiment directory"""
        experiment_id = hashlib.md5(f"{experiment_name}_{datetime.now()}".encode()).hexdigest()[:8]
        full_experiment_name = f"{experiment_name}_{experiment_id}"
        experiment_dir = os.path.join(self.base_dir, full_experiment_name)

        os.makedirs(experiment_dir, exist_ok=True)

        # Save experiment config
        config_file = os.path.join(experiment_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, cls=CustomJSONEncoder)

        # Update experiment history
        self.experiment_history[full_experiment_name] = {
            'created': datetime.now().isoformat(),
            'config': config,
            'status': 'created',
            'experiment_dir': experiment_dir
        }

        self.active_experiments[full_experiment_name] = experiment_dir
        self._save_experiment_history()

        return full_experiment_name, experiment_dir

    def get_experiment_status(self, experiment_name: str) -> Dict:
        """Get experiment status and results"""
        if experiment_name in self.experiment_history:
            return self.experiment_history[experiment_name]
        return {}

    def list_experiments(self, status: str = None) -> List[Dict]:
        """List all experiments, optionally filtered by status"""
        experiments = []

        for name, info in self.experiment_history.items():
            if status is None or info.get('status') == status:
                experiments.append({
                    'name': name,
                    **info
                })

        return sorted(experiments, key=lambda x: x.get('created', ''), reverse=True)

    def update_experiment_status(self, experiment_name: str, status: str, results: Dict = None):
        """Update experiment status and results"""
        if experiment_name in self.experiment_history:
            self.experiment_history[experiment_name]['status'] = status
            self.experiment_history[experiment_name]['updated'] = datetime.now().isoformat()

            if results:
                self.experiment_history[experiment_name]['results'] = results

            self._save_experiment_history()


def setup_random_seeds(seed: int = 42):
    """Setup random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seeds set to: {seed}")


def get_device() -> torch.device:
    """Get available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def save_model(model: nn.Module, path: str, optimizer=None, scheduler=None, epoch=None, metrics=None):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'timestamp': datetime.now().isoformat()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    print(f"Model saved to: {path}")


def load_model(model: nn.Module, path: str, optimizer=None, scheduler=None, device=None):
    """Load model checkpoint"""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    results = {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Model loaded from: {path}")
    print(f"Checkpoint epoch: {results['epoch']}")

    return results


def calculate_class_weights(dataset) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets"""
    class_counts = {}

    for sample in dataset:
        label = sample['label']
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = torch.zeros(num_classes)
    for label, count in class_counts.items():
        weights[label] = total_samples / (num_classes * count)

    return weights


def create_gradcam_visualization(model, image, target_layer, target_class=None):
    """Create Grad-CAM visualization for model interpretability"""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        model.eval()

        # Create Grad-CAM object
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Generate CAM
        grayscale_cam = cam(input_tensor=image.unsqueeze(0), target_category=target_class)
        grayscale_cam = grayscale_cam[0, :]

        # Convert image for visualization
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # Create visualization
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        return visualization

    except ImportError:
        print("Grad-CAM not available. Install with: pip install grad-cam")
        return None


if __name__ == "__main__":
    # Example usage
    print("Utility functions and classes loaded successfully!")

    # Test device detection
    device = get_device()
    print(f"Device: {device}")

    # Test random seeds
    setup_random_seeds(42)