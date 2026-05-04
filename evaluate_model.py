import os
import sys

# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import numpy as np
import argparse
from metrics import SegmentationMetrics, print_metrics_table, plot_confusion_matrix, plot_metrics_comparison
from background_remover import load_segmentation_model, get_mask

def evaluate(images_dir, masks_dir):
    """
    Evaluates the background removal model on a dataset.
    
    Args:
        images_dir: Path to directory containing source images.
        masks_dir: Path to directory containing ground truth binary masks.
    """
    segmentor = load_segmentation_model()
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    total_metrics = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1-Score": 0,
        "IoU": 0
    }
    
    # Components for global confusion matrix
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    count = 0
    
    print(f"[INFO] Starting evaluation on {len(image_files)} images...")
    
    for img_name in image_files:
        # Construct paths
        img_path = os.path.join(images_dir, img_name)
        # Assuming mask has the same name or prefix
        mask_name = img_name # Modify if masks have different naming convention
        mask_path = os.path.join(masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found for {img_name}, skipping.")
            continue
            
        # Load image and ground truth
        img = cv2.imread(img_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or gt_mask is None:
            continue
            
        # Ensure GT is binary (0 or 1)
        _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)
        
        # Resize GT to match image if necessary
        if gt_mask.shape[:2] != img.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Get prediction from our model
        # Note: background_remover.get_mask returns a blurred float mask
        pred_mask_raw = get_mask(img, segmentor)
        
        # Binarize prediction for metrics (using a standard 0.5 threshold)
        pred_mask = (pred_mask_raw > 0.5).astype(np.uint8)
        
        # Calculate metrics
        metrics_calc = SegmentationMetrics(gt_mask, pred_mask)
        results = metrics_calc.get_all_metrics()
        
        # Add to components for confusion matrix
        total_tp += int(metrics_calc.tp)
        total_tn += int(metrics_calc.tn)
        total_fp += int(metrics_calc.fp)
        total_fn += int(metrics_calc.fn)
        
        for key in total_metrics:
            total_metrics[key] += results[key]
        
        count += 1
        if count % 10 == 0:
            print(f"[PROGRESS] Evaluated {count}/{len(image_files)} images...")

    if count > 0:
        # Average metrics
        avg_metrics = {k: v / count for k, v in total_metrics.items()}
        print("\n" + "="*30)
        print(" FINAL EVALUATION RESULTS")
        print("="*30)
        print(f"Total Images: {count}")
        print_metrics_table(avg_metrics)
        
        # Generate and save graphs
        print("\n[INFO] Generating performance graphs...")
        plot_confusion_matrix(total_tp, total_tn, total_fp, total_fn, "confusion_matrix.png")
        plot_metrics_comparison(avg_metrics, "metrics_summary.png")
        print("[SUCCESS] Evaluation complete. Graphs saved to project directory.")
    else:
        print("[ERROR] No valid image-mask pairs found for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Background Removal Model")
    parser.add_argument("--images", type=str, required=True, help="Path to source images")
    parser.add_argument("--masks", type=str, required=True, help="Path to ground truth masks")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images) or not os.path.exists(args.masks):
        print(f"\n[ERROR] Path not found.")
        print(f" - Images Path: {args.images}")
        print(f" - Masks Path:  {args.masks}")
        print("\nTIP: Please replace 'path/to/images' with your actual data directory.")
    else:
        evaluate(args.images, args.masks)
