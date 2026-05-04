import cv2
import numpy as np
import os

def create_samples(base_dir="eval_samples"):
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"[INFO] Creating sample dataset in {base_dir}...")
    
    # Create 3 sample images and masks
    for i in range(1, 4):
        # 1. Create a background (random noise or solid color)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        # 2. Create a "person" (a simple shape)
        mask = np.zeros((480, 640), dtype=np.uint8)
        
        # Draw a central blob as the "person"
        center = (320, 240)
        axes = (100 + i*20, 150 + i*10)
        cv2.ellipse(img, center, axes, 0, 0, 360, (200, 150, 100), -1) # "Skin" color
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Save files
        img_path = os.path.join(img_dir, f"sample_{i}.jpg")
        mask_path = os.path.join(mask_dir, f"sample_{i}.jpg")
        
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
        
    print(f"[SUCCESS] Samples created!")
    print(f"\nNow you can run:")
    print(f"python evaluate_model.py --images {img_dir} --masks {mask_dir}")

if __name__ == "__main__":
    create_samples()
