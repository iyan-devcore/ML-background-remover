import numpy as np
import cv2
from metrics import SegmentationMetrics, print_metrics_table

def run_demo():
    print("[INFO] Running Evaluation Metrics Demo...")
    
    # Create a synthetic ground truth mask (a 100x100 white square in middle)
    gt = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(gt, (150, 150), (250, 250), 1, -1)
    
    # Create a synthetic prediction (slightly shifted and noisy)
    pred = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(pred, (155, 155), (255, 255), 1, -1)
    
    # Add some random noise
    noise = np.random.choice([0, 1], size=(400, 400), p=[0.99, 0.01]).astype(np.uint8)
    pred = cv2.bitwise_xor(pred, noise)
    
    # Calculate metrics
    m = SegmentationMetrics(gt, pred)
    results = m.get_all_metrics()
    
    print("\n[RESULTS] Synthetic Data Evaluation:")
    print_metrics_table(results)
    
    print("\n[NOTE] These metrics represent:")
    print("- Accuracy: Overall pixel correctness.")
    print("- Precision: How many predicted foreground pixels are correct.")
    print("- Recall: How many actual foreground pixels were captured.")
    print("- F1-Score: Harmonic mean of Precision and Recall (Dice Index).")
    print("- IoU: Overlap between prediction and ground truth.")

if __name__ == "__main__":
    run_demo()
