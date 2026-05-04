import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class SegmentationMetrics:
    """
    Computes common segmentation metrics for binary masks.
    Expects y_true and y_pred as binary numpy arrays (0 and 1).
    """
    def __init__(self, y_true, y_pred, smooth=1e-6):
        # Flatten the arrays to 1D
        self.y_true = y_true.flatten()
        self.y_pred = y_pred.flatten()
        self.smooth = smooth
        
        # Calculate base components
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def accuracy(self):
        """Pixel Accuracy: (TP + TN) / Total"""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / (total + self.smooth)

    def precision(self):
        """Precision (Positive Predictive Value): TP / (TP + FP)"""
        return self.tp / (self.tp + self.fp + self.smooth)

    def recall(self):
        """Recall (Sensitivity / True Positive Rate): TP / (TP + FN)"""
        return self.tp / (self.tp + self.fn + self.smooth)

    def f1_score(self):
        """F1-Score (Dice Coefficient): 2 * TP / (2 * TP + FP + FN)"""
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r + self.smooth)

    def iou(self):
        """Intersection over Union (Jaccard Index): TP / (TP + FP + FN)"""
        return self.tp / (self.tp + self.fp + self.fn + self.smooth)

    def get_all_metrics(self):
        """Returns a dictionary of all calculated metrics."""
        return {
            "Accuracy": self.accuracy(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1-Score": self.f1_score(),
            "IoU": self.iou()
        }

def print_metrics_table(metrics_dict):
    """Utility to print metrics in a formatted way."""
    print("-" * 30)
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    for name, value in metrics_dict.items():
        print(f"{name:<15} | {value:.4f}")
    print("-" * 30)

def plot_confusion_matrix(tp, tn, fp, fn, save_path="confusion_matrix.png"):
    """
    Plots and saves a confusion matrix heatmap.
    
    Classes: 0 = Background, 1 = Foreground (Person)
    """
    # Matrix format: 
    # [[TN, FP],
    #  [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Create the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Background', 'Foreground'],
                yticklabels=['Background', 'Foreground'])
    
    plt.title('Pixel-wise Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")

def plot_metrics_comparison(metrics_dict, save_path="metrics_summary.png"):
    """
    Plots and saves a bar chart of the various metrics.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create bar chart
    palette = sns.color_palette("viridis", len(names))
    bars = plt.bar(names, values, color=palette)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.title('Background Removal Performance Metrics', fontsize=16, pad=20)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Metrics plot saved to {save_path}")
