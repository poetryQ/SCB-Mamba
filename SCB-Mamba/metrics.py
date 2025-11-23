import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional


class ClassificationMetrics:
    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        if outputs.dim() > 1 and outputs.size(1) > 1:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        else:
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)

        if self.num_classes == 2:
            return self._compute_binary_metrics(predictions, targets, probabilities)
        else:
            return self._compute_multiclass_metrics(predictions, targets, probabilities)

    def _compute_binary_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                probabilities: np.ndarray) -> Dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        try:
            auc_roc = roc_auc_score(targets, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
        except:
            auc_roc = 0.0

        try:
            auc_pr = average_precision_score(targets, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
        except:
            auc_pr = 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

    def _compute_multiclass_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                    probabilities: np.ndarray) -> Dict[str, float]:
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))

        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0

        precisions = []
        recalls = []
        specificities = []
        f1_scores = []

        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1_scores.append(f1)

        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_specificity = np.mean(specificities)
        macro_f1 = np.mean(f1_scores)

        weighted_precision = np.average(precisions, weights=np.sum(cm, axis=1))
        weighted_recall = np.average(recalls, weights=np.sum(cm, axis=1))
        weighted_f1 = np.average(f1_scores, weights=np.sum(cm, axis=1))

        try:
            if self.num_classes == 2:
                auc_roc = roc_auc_score(targets, probabilities[:, 1])
            else:
                auc_roc = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
        except:
            auc_roc = 0.0

        try:
            if self.num_classes == 2:
                auc_pr = average_precision_score(targets, probabilities[:, 1])
            else:
                auc_pr = average_precision_score(targets, probabilities, average='macro')
        except:
            auc_pr = 0.0

        return {
            'accuracy': accuracy,
            'precision_macro': macro_precision,
            'recall_macro': macro_recall,
            'specificity_macro': macro_specificity,
            'f1_score_macro': macro_f1,
            'precision_weighted': weighted_precision,
            'recall_weighted': weighted_recall,
            'f1_score_weighted': weighted_f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'confusion_matrix': cm,
            'class_precisions': precisions,
            'class_recalls': recalls,
            'class_f1_scores': f1_scores
        }

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        cm = confusion_matrix(self.targets, self.predictions, labels=range(self.num_classes))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_classification_report(self) -> str:
        metrics = self.compute()
        report = []

        report.append("Classification Report:")
        report.append("=" * 50)
        report.append(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        report.append(f"Macro Precision: {metrics['precision_macro']:.4f}")
        report.append(f"Macro Recall: {metrics['recall_macro']:.4f}")
        report.append(f"Macro F1-Score: {metrics['f1_score_macro']:.4f}")
        report.append(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        report.append(f"AUC-PR: {metrics['auc_pr']:.4f}")
        report.append("")

        report.append("Per-class Metrics:")
        report.append("-" * 30)
        for i, class_name in enumerate(self.class_names):
            report.append(f"{class_name}:")
            report.append(f"  Precision: {metrics['class_precisions'][i]:.4f}")
            report.append(f"  Recall: {metrics['class_recalls'][i]:.4f}")
            report.append(f"  F1-Score: {metrics['class_f1_scores'][i]:.4f}")

        return "\n".join(report)


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict: Dict[str, float]):
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_average(self, key: str) -> float:
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return np.mean(self.metrics[key])

    def get_std(self, key: str) -> float:
        if key not in self.metrics or len(self.metrics[key]) < 2:
            return 0.0
        return np.std(self.metrics[key])

    def reset(self):
        self.metrics = {}