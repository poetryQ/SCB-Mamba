import torch
import numpy as np
import os
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from itertools import cycle
import warnings
from config import ModelConfig, DataConfig
from data_loader import OCTDataManager
from Model.scb_mamba import SCBMamba

warnings.filterwarnings('ignore')


class SCBMambaTester:
    def __init__(self, model, test_loader, device, class_names):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names

        self.metrics_calculator = ComprehensiveMetricsCalculator(
            num_classes=len(class_names),
            class_names=class_names
        )

        self.visualization_tools = ResultVisualizationTools(class_names)

        print(f"Tester initialized with device: {device}")
        print(f"Number of test samples: {len(test_loader.dataset)}")
        print(f"Class names: {class_names}")

    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded from: {checkpoint_path}")

        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")

        return checkpoint

    def test(self, return_detailed=False):
        """Run comprehensive testing"""
        self.model.eval()

        print("Starting comprehensive testing...")
        start_time = time.time()

        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_saliency_maps = []
        all_boundary_maps = []
        all_features = []
        all_image_paths = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                if 'image_path' in batch:
                    all_image_paths.extend(batch['image_path'])

                outputs = self.model(images)

                probabilities = torch.softmax(outputs['final_logits'], dim=1)
                _, predicted = outputs['final_logits'].max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if 'saliency_map' in outputs:
                    all_saliency_maps.extend(outputs['saliency_map'].cpu().numpy())

                if 'boundary_map' in outputs:
                    all_boundary_maps.extend(outputs['boundary_map'].cpu().numpy())

                if 'global_features' in outputs:
                    all_features.extend(outputs['global_features'].cpu().numpy())

        test_time = time.time() - start_time

        # Calculate comprehensive metrics
        test_results = self.metrics_calculator.compute_comprehensive_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
            probabilities=np.array(all_probabilities),
            test_time=test_time
        )

        print(f"\nTesting completed in {test_time:.2f} seconds")
        print(f"Overall Accuracy: {test_results['overall_accuracy']:.4f}")
        print(f"Macro F1-Score: {test_results['macro_f1']:.4f}")

        if return_detailed:
            detailed_results = {
                'predictions': all_predictions,
                'targets': all_targets,
                'probabilities': all_probabilities,
                'saliency_maps': all_saliency_maps,
                'boundary_maps': all_boundary_maps,
                'features': all_features,
                'image_paths': all_image_paths,
                **test_results
            }
            return detailed_results

        return test_results

    def per_class_analysis(self):
        """Perform detailed per-class analysis"""
        detailed_results = self.test(return_detailed=True)

        analysis_results = {}

        # Per-class metrics
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = np.array(detailed_results['targets']) == class_idx
            class_predictions = np.array(detailed_results['predictions'])[class_mask]
            class_targets = np.array(detailed_results['targets'])[class_mask]
            class_probabilities = np.array(detailed_results['probabilities'])[class_mask]

            class_correct = (class_predictions == class_targets).sum()
            class_total = len(class_targets)
            class_accuracy = class_correct / class_total if class_total > 0 else 0

            analysis_results[class_name] = {
                'accuracy': class_accuracy,
                'total_samples': class_total,
                'correct_predictions': class_correct,
                'confidence_mean': class_probabilities[:, class_idx].mean() if class_total > 0 else 0,
                'confidence_std': class_probabilities[:, class_idx].std() if class_total > 0 else 0
            }

        return analysis_results

    def generate_comprehensive_report(self, save_dir='test_results'):
        """Generate comprehensive test report with visualizations"""
        os.makedirs(save_dir, exist_ok=True)

        print("Generating comprehensive test report...")

        detailed_results = self.test(return_detailed=True)
        per_class_results = self.per_class_analysis()

        # Save numerical results
        report_data = {
            'test_results': detailed_results,
            'per_class_analysis': per_class_results,
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

        with open(os.path.join(save_dir, 'test_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2, cls=NumpyEncoder)

        # Generate visualizations
        self.visualization_tools.plot_confusion_matrix(
            detailed_results['targets'],
            detailed_results['predictions'],
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )

        self.visualization_tools.plot_roc_curves(
            detailed_results['targets'],
            detailed_results['probabilities'],
            save_path=os.path.join(save_dir, 'roc_curves.png')
        )

        self.visualization_tools.plot_precision_recall_curves(
            detailed_results['targets'],
            detailed_results['probabilities'],
            save_path=os.path.join(save_dir, 'precision_recall_curves.png')
        )

        # Generate classification report
        classification_rep = classification_report(
            detailed_results['targets'],
            detailed_results['predictions'],
            target_names=self.class_names,
            output_dict=True
        )

        with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
            json.dump(classification_rep, f, indent=2)

        # Print summary
        self._print_test_summary(detailed_results, per_class_results)

        print(f"Comprehensive report saved to: {save_dir}")

        return report_data

    def _print_test_summary(self, test_results, per_class_results):
        """Print formatted test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        print(f"\nOverall Performance:")
        print(f"  Accuracy: {test_results['overall_accuracy']:.4f}")
        print(f"  Macro F1: {test_results['macro_f1']:.4f}")
        print(f"  AUC-ROC: {test_results['auc_roc']:.4f}")
        print(f"  AUC-PR: {test_results['auc_pr']:.4f}")

        print(f"\nPer-Class Performance:")
        for class_name, metrics in per_class_results.items():
            print(f"  {class_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Samples: {metrics['total_samples']}")
            print(f"    Confidence: {metrics['confidence_mean']:.4f} ± {metrics['confidence_std']:.4f}")

        print(f"\nTest Duration: {test_results['test_time']:.2f} seconds")
        print("=" * 60)


class ComprehensiveMetricsCalculator:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names

    def compute_comprehensive_metrics(self, predictions, targets, probabilities, test_time):
        """Compute comprehensive evaluation metrics"""
        metrics = {}

        # Basic classification metrics
        metrics['overall_accuracy'] = np.mean(predictions == targets)
        metrics['test_time'] = test_time

        # Confusion matrix based metrics
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        metrics['confusion_matrix'] = cm

        # Per-class metrics
        class_precisions = []
        class_recalls = []
        class_f1_scores = []
        class_specificities = []

        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1_scores.append(f1)
            class_specificities.append(specificity)

        metrics['class_precisions'] = class_precisions
        metrics['class_recalls'] = class_recalls
        metrics['class_f1_scores'] = class_f1_scores
        metrics['class_specificities'] = class_specificities

        # Aggregate metrics
        metrics['macro_precision'] = np.mean(class_precisions)
        metrics['macro_recall'] = np.mean(class_recalls)
        metrics['macro_f1'] = np.mean(class_f1_scores)
        metrics['macro_specificity'] = np.mean(class_specificities)

        # AUC metrics
        metrics['auc_roc'] = self._compute_multiclass_auc_roc(targets, probabilities)
        metrics['auc_pr'] = self._compute_multiclass_auc_pr(targets, probabilities)

        # Additional metrics
        metrics['balanced_accuracy'] = np.mean(class_recalls)
        metrics['cohen_kappa'] = self._compute_cohen_kappa(cm)
        metrics['matthews_corrcoef'] = self._compute_matthews_corrcoef(cm)

        return metrics

    def _compute_multiclass_auc_roc(self, targets, probabilities):
        """Compute multiclass AUC-ROC"""
        from sklearn.preprocessing import label_binarize

        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            return auc(fpr, tpr)
        else:
            # Binarize the output
            y_test_bin = label_binarize(targets, classes=range(self.num_classes))

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probabilities.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            return roc_auc["micro"]

    def _compute_multiclass_auc_pr(self, targets, probabilities):
        """Compute multiclass AUC-PR"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve, average_precision_score

        if self.num_classes == 2:
            precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
            return auc(recall, precision)
        else:
            y_test_bin = label_binarize(targets, classes=range(self.num_classes))
            return average_precision_score(y_test_bin, probabilities, average='micro')

    def _compute_cohen_kappa(self, cm):
        """Compute Cohen's Kappa"""
        total = np.sum(cm)
        po = np.trace(cm) / total
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total * total)
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    def _compute_matthews_corrcoef(self, cm):
        """Compute Matthews Correlation Coefficient"""
        total = np.sum(cm)
        actual = np.sum(cm, axis=1)
        predicted = np.sum(cm, axis=0)

        tp = np.diag(cm)
        fp = predicted - tp
        fn = actual - tp
        tn = total - (tp + fp + fn)

        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Avoid division by zero
        denominator = np.where(denominator == 0, 1, denominator)

        mcc = numerator / denominator
        return np.mean(mcc)


class ResultVisualizationTools:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)

        plt.style.use('default')
        sns.set_palette("husl")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_roc_curves(self, y_true, y_prob, save_path=None):
        """Plot ROC curves for all classes"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
        else:
            y_test_bin = label_binarize(y_true, classes=range(self.num_classes))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            plt.figure(figsize=(10, 8))

            # Plot each class
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
            for i, color in zip(range(self.num_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')

            # Plot micro-average
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curves')
            plt.legend(loc="lower right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_precision_recall_curves(self, y_true, y_prob, save_path=None):
        """Plot precision-recall curves for all classes"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve, average_precision_score

        if self.num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_precision = average_precision_score(y_true, y_prob[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'Precision-Recall (AP = {avg_precision:.2f})')
        else:
            y_test_bin = label_binarize(y_true, classes=range(self.num_classes))

            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(self.num_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
                average_precision[i] = average_precision_score(y_test_bin[:, i], y_prob[:, i])

            avg_precision_micro = average_precision_score(y_test_bin, y_prob, average="micro")

            plt.figure(figsize=(10, 8))

            colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
            for i, color in zip(range(self.num_classes), colors):
                plt.plot(recall[i], precision[i], color=color, lw=2,
                         label=f'{self.class_names[i]} (AP = {average_precision[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Multi-class Precision-Recall Curves\n(AP: Micro-average = {0:0.2f})'.format(avg_precision_micro))
            plt.legend(loc="lower left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class InferenceEngine:
    def __init__(self, model, device, class_names):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()

    def predict_single(self, image):
        """Predict single image"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs['final_logits'], dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': probabilities.cpu().numpy()[0]
        }

        # Add additional outputs if available
        if 'saliency_map' in outputs:
            result['saliency_map'] = outputs['saliency_map'].cpu().numpy()[0]

        if 'boundary_map' in outputs:
            result['boundary_map'] = outputs['boundary_map'].cpu().numpy()[0]

        return result

    def predict_batch(self, images):
        """Predict batch of images"""
        if isinstance(images, list):
            images = torch.stack(images)

        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs['final_logits'], dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities[torch.arange(probabilities.size(0)),
            torch.argmax(probabilities, dim=1)].cpu().numpy()

        results = []
        for i in range(len(images)):
            result = {
                'predicted_class': predicted_classes[i],
                'predicted_label': self.class_names[predicted_classes[i]],
                'confidence': confidences[i],
                'all_probabilities': probabilities[i].cpu().numpy()
            }
            results.append(result)

        return results


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def setup_testing_environment(seed=42):
    """Setup testing environment"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing device: {device}")

    return device


def create_tester(model, test_loader, class_names, checkpoint_path=None):
    """Create tester instance"""
    device = setup_testing_environment()

    tester = SCBMambaTester(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    if checkpoint_path:
        tester.load_model(checkpoint_path)

    return tester


def create_inference_engine(model, class_names, checkpoint_path=None):
    """Create inference engine"""
    device = setup_testing_environment()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    engine = InferenceEngine(
        model=model,
        device=device,
        class_names=class_names
    )

    return engine


if __name__ == "__main__":


    print("SCB-Mamba Testing Script")
    print("=" * 50)

    # Configuration
    model_config = ModelConfig()
    data_config = DataConfig()

    # Create data manager and load test data
    data_manager = OCTDataManager(data_config)
    _, _, test_loader = data_manager.create_dataloaders(data_config.data_root)

    # Create model
    model = SCBMamba(model_config)

    # Create tester
    tester = create_tester(
        model=model,
        test_loader=test_loader,
        class_names=data_config.class_names,
        checkpoint_path='checkpoints/best_model.pth'  # Update path as needed
    )

    # Run comprehensive testing
    report = tester.generate_comprehensive_report(save_dir='test_results')

    print("\nTesting completed successfully!")