import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SCBMambaTrainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()

        self.metrics_tracker = TrainingMetricsTracker()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.patience_counter = 0

        self._setup_training()

    def _create_optimizer(self):
        param_groups = [
            {'params': self.model.gsde_module.parameters(), 'lr': self.config.learning_rate * 0.1},
            {'params': self.model.irbe_module.parameters(), 'lr': self.config.learning_rate * 0.1},
            {'params': self.model.multi_path_backbone.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.feature_fusion_layers.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.cross_modal_fusion.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.final_classifier.parameters(), 'lr': self.config.learning_rate * 2}
        ]

        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

    def _create_scheduler(self):
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[group['lr'] for group in self.optimizer.param_groups],
            epochs=self.config.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )

    def _create_criterion(self):
        return MultiTaskLossWithRegularization(
            classification_weight=1.0,
            auxiliary_weight=0.3,
            consistency_weight=0.1,
            boundary_weight=0.2,
            complexity_weight=0.05,
            regularization_weight=0.01
        )

    def _setup_training(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        self.train_losses = []
        self.val_metrics = []
        self.learning_rates = []

        print(f"Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Checkpoint Dir: {self.config.checkpoint_dir}")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss_dict = self.criterion(outputs, labels)
            total_loss = loss_dict['total_loss']

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += total_loss.item()

            _, predicted = outputs['final_logits'].max(1)
            total_samples += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100. * correct_predictions / total_samples:.2f}%',
                'LR': f'{current_lr:.2e}'
            })

        epoch_accuracy = 100. * correct_predictions / total_samples
        avg_loss = epoch_loss / len(self.train_loader)

        self.train_losses.append(avg_loss)

        return {
            'train_loss': avg_loss,
            'train_accuracy': epoch_accuracy,
            'learning_rate': current_lr
        }

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)

                loss_dict = self.criterion(outputs, labels)
                val_loss += loss_dict['total_loss'].item()

                probabilities = torch.softmax(outputs['final_logits'], dim=1)
                _, predicted = outputs['final_logits'].max(1)

                total_samples += labels.size(0)
                correct_predictions += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        val_accuracy = 100. * correct_predictions / total_samples
        avg_val_loss = val_loss / len(self.val_loader)

        metrics = {
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities)
        }

        self.val_metrics.append(metrics)

        return metrics

    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_saliency_maps = []
        all_boundary_maps = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)

                loss_dict = self.criterion(outputs, labels)
                test_loss += loss_dict['total_loss'].item()

                probabilities = torch.softmax(outputs['final_logits'], dim=1)
                _, predicted = outputs['final_logits'].max(1)

                total_samples += labels.size(0)
                correct_predictions += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if len(all_saliency_maps) < 100:
                    all_saliency_maps.extend(outputs['saliency_map'].cpu().numpy())
                    all_boundary_maps.extend(outputs['boundary_map'].cpu().numpy())

        test_accuracy = 100. * correct_predictions / total_samples
        avg_test_loss = test_loss / len(self.test_loader)

        test_metrics = {
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities),
            'saliency_maps': np.array(all_saliency_maps),
            'boundary_maps': np.array(all_boundary_maps)
        }

        return test_metrics

    def train(self):
        print("Starting Training...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 50}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.2f}%")

            self.metrics_tracker.update_epoch_metrics(epoch, train_metrics, val_metrics)

            is_best = val_metrics['val_accuracy'] > self.best_accuracy

            if is_best:
                self.best_accuracy = val_metrics['val_accuracy']
                self.patience_counter = 0
                print(f"New best validation accuracy: {self.best_accuracy:.2f}%")
            else:
                self.patience_counter += 1

            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics={
                    'train_accuracy': train_metrics['train_accuracy'],
                    'val_accuracy': val_metrics['val_accuracy'],
                    'best_accuracy': self.best_accuracy
                },
                is_best=is_best
            )

            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        final_metrics = self._finalize_training()
        return final_metrics

    def _finalize_training(self):
        print("\nFinalizing Training...")

        best_checkpoint = self.checkpoint_manager.load_best_checkpoint(self.model)

        test_metrics = self.test()

        final_metrics = {
            'best_validation_accuracy': self.best_accuracy,
            'test_accuracy': test_metrics['test_accuracy'],
            'test_loss': test_metrics['test_loss'],
            'training_time': time.time() - self.metrics_tracker.start_time,
            'total_epochs': self.current_epoch + 1,
            'final_learning_rate': self.learning_rates[-1] if self.learning_rates else 0.0
        }

        self.metrics_tracker.save_final_metrics(final_metrics, test_metrics)
        self.metrics_tracker.generate_training_report()

        print(f"\nFinal Results:")
        print(f"Best Validation Accuracy: {self.best_accuracy:.2f}%")
        print(f"Test Accuracy: {test_metrics['test_accuracy']:.2f}%")
        print(f"Test Loss: {test_metrics['test_loss']:.4f}")

        return final_metrics


class MultiTaskLossWithRegularization(nn.Module):
    def __init__(self, classification_weight=1.0, auxiliary_weight=0.3,
                 consistency_weight=0.1, boundary_weight=0.2,
                 complexity_weight=0.05, regularization_weight=0.01):
        super().__init__()
        self.weights = {
            'classification': classification_weight,
            'auxiliary': auxiliary_weight,
            'consistency': consistency_weight,
            'boundary': boundary_weight,
            'complexity': complexity_weight,
            'regularization': regularization_weight
        }

        self.classification_loss = nn.CrossEntropyLoss()
        self.consistency_loss = nn.MSELoss()
        self.boundary_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        losses = {}

        main_loss = self.classification_loss(outputs['final_logits'], targets)
        losses['classification'] = main_loss

        if 'auxiliary_logits' in outputs:
            aux_loss = 0.0
            for i in range(outputs['auxiliary_logits'].size(1)):
                aux_loss += self.classification_loss(
                    outputs['auxiliary_logits'][:, i], targets
                )
            losses['auxiliary'] = aux_loss / outputs['auxiliary_logits'].size(1)

        if 'path_logits' in outputs:
            consistency_loss = 0.0
            num_paths = outputs['path_logits'].size(1)
            for i in range(num_paths):
                for j in range(i + 1, num_paths):
                    consistency_loss += self.consistency_loss(
                        outputs['path_logits'][:, i],
                        outputs['path_logits'][:, j]
                    )
            losses['consistency'] = consistency_loss / (num_paths * (num_paths - 1) / 2)

        if 'boundary_map' in outputs:
            boundary_target = torch.ones_like(outputs['boundary_map']) * 0.5
            losses['boundary'] = self.boundary_loss(outputs['boundary_map'], boundary_target)

        if 'complexity_logits' in outputs:
            complexity_loss = self.classification_loss(
                outputs['complexity_logits'],
                torch.zeros_like(outputs['complexity_logits'])
            )
            losses['complexity'] = complexity_loss

        total_loss = 0.0
        for loss_name, loss_value in losses.items():
            total_loss += self.weights[loss_name] * loss_value

        losses['total_loss'] = total_loss

        return losses


class TrainingMetricsTracker:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.start_time = time.time()

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []

        os.makedirs(log_dir, exist_ok=True)

    def update_epoch_metrics(self, epoch, train_metrics, val_metrics):
        self.train_losses.append(train_metrics['train_loss'])
        self.train_accuracies.append(train_metrics['train_accuracy'])
        self.val_losses.append(val_metrics['val_loss'])
        self.val_accuracies.append(val_metrics['val_accuracy'])

        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)

        self._save_epoch_metrics(epoch, train_metrics, val_metrics)

    def _save_epoch_metrics(self, epoch, train_metrics, val_metrics):
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['train_loss'],
            'train_accuracy': train_metrics['train_accuracy'],
            'val_loss': val_metrics['val_loss'],
            'val_accuracy': val_metrics['val_accuracy'],
            'learning_rate': train_metrics.get('learning_rate', 0.0),
            'timestamp': datetime.now().isoformat()
        }

        metrics_file = os.path.join(self.log_dir, f'training_metrics.json')

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append(metrics)

        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

    def save_final_metrics(self, final_metrics, test_metrics):
        final_results = {
            'final_metrics': final_metrics,
            'test_metrics': test_metrics,
            'training_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'epoch_times': self.epoch_times
            },
            'completion_time': datetime.now().isoformat()
        }

        results_file = os.path.join(self.log_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

    def generate_training_report(self):
        report = {
            'total_training_time': self.epoch_times[-1] if self.epoch_times else 0,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0,
            'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0,
            'num_epochs': len(self.train_losses)
        }

        report_file = os.path.join(self.log_dir, 'training_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.best_accuracy = 0.0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch:04d}.pth'
        )

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            self.best_accuracy = metrics['val_accuracy']

    def load_checkpoint(self, model, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint

    def load_best_checkpoint(self, model):
        return self.load_checkpoint(model)


def setup_training_environment(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    return device


def create_trainer(config, model, data_manager):
    device = setup_training_environment()

    train_loader, val_loader, test_loader = data_manager.create_dataloaders(config.data_root)

    trainer = SCBMambaTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )

    return trainer


if __name__ == "__main__":
    from config import ModelConfig, TrainingConfig, DataConfig
    from data_loader import OCTDataManager
    from Model.scb_mamba import SCBMamba

    print("SCB-Mamba Training Script")
    print("=" * 50)

    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    data_manager = OCTDataManager(data_config)

    model = SCBMamba(model_config)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = create_trainer(training_config, model, data_manager)

    final_metrics = trainer.train()

    print("\nTraining Completed Successfully!")
    print(f"Best Validation Accuracy: {final_metrics['best_validation_accuracy']:.2f}%")
    print(f"Test Accuracy: {final_metrics['test_accuracy']:.2f}%")