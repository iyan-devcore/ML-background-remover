# Real-Time Background Remover — Python & MediaPipe

A high-performance, local Python application for **real-time webcam background removal**. Leveraging Google's **MediaPipe Selfie Segmentation**, this project provides a lag-free experience with professional features like background blurring, custom image replacement, and solid color overlays.

![Project Preview](https://img.shields.io/badge/Tech-OpenCV%20%7C%20MediaPipe%20%7C%20Numpy-blue)
![Python Version](https://img.shields.io/badge/Python-3.11+-green)

---

## 🚀 Features

- **High-Performance Segmentation**: Uses MediaPipe's optimized Selfie Segmentation model for accurate real-time person extraction.
- **Multiple Modes**:
  - **🎨 Solid Color**: Instantly replace your background with any RGB color (useful for virtual green screens).
  - **🖼️ Custom Image**: Upload your own background images or select from pre-loaded stock assets.
  - **🌫️ Professional Blur**: Smoothly blur your existing background for a professional depth-of-field effect.
- **Optimized Engine**:
  - **Threaded Capture**: Webcam reading runs on a separate thread to eliminate input lag.
  - **Vectorized Operations**: Advanced NumPy blending ensures high FPS even on standard hardware.
- **Modern GUI**: A sleek, dark-themed control panel built with **CustomTkinter** for granular adjustments.
- **Screenshot Utility**: Capture and save processed frames with a single click or keypress.

---

## 🛠️ Tech Stack

- **Computer Vision**: [OpenCV](https://opencv.org/) & [MediaPipe](https://mediapipe.dev/)
- **Processing**: [NumPy](https://numpy.org/) (Vectorized image manipulation)
- **UI Framework**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- **Threading**: Python `threading` for asynchronous I/O

---

## 📦 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/background_remover.git
   cd background_remover
   ```

2. **Set up Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: This project specifically uses `numpy<2`, `mediapipe==0.10.14`, and `opencv-python==4.9.0.80` to ensure stability and compatibility on Python 3.11+.

---

## 🎮 Usage

### 1. Premium GUI version (Recommended)
Launch the full interactive interface with sidebar controls:
```bash
python background_remover_gui.py
```
**Controls**:
- **Mode Menu**: Switch between Color, Image, and Blur.
- **Threshold Slider**: Adjust the sensitivity of the segmentation mask.
- **Smoothing Slider**: Change the edge softness/blur of the person cutout.
- **RGB Sliders**: Fine-tune the background color in real-time.
- **Load Custom Image**: Select any file from your PC to use as a backdrop.

### 2. Lightweight CLI Version
Run the core engine with simple keyboard shortcuts:
```bash
python background_remover.py
```
**Keyboard Controls**:
- `b`: Toggle Background Mode (Color → Image → Blur).
- `s`: Save a screenshot to the `screenshots/` directory.
- `q`: Quit the application.

---

## 📂 Project Structure

```text
background_remover/
├── background_remover.py      # Core logic & keyboard-controlled version
├── background_remover_gui.py  # Advanced GUI version (CustomTkinter)
├── requirements.txt           # Pinned dependencies
├── stock_backgrounds/         # Folder for pre-shipped background images
└── screenshots/               # Folder where saved captures are stored
```

---

## ⚙️ Optimization Details

This project is engineered for speed:
- **Mask Smoothing**: Uses Gaussian blurring on the alpha mask to prevent "jagged edges" without significant CPU overhead.
- **BGR Conversion**: Real-time conversion to RGB is handled only for the model processing step to save cycles.
- **Safe Capture**: The `WebcamStream` class handles the camera buffer in the background, preventing frame drops during heavy compositing tasks.

---

## 📊 Evaluation Metrics

The project includes a robust evaluation suite to measure the performance of the background removal model against ground truth data.

### Supported Metrics:
- **Accuracy**: Overall pixel-wise correctness.
- **Precision**: Measures how often the predicted foreground is correct.
- **Recall**: Measures how much of the actual foreground was captured.
- **F1-Score (Dice Coefficient)**: The harmonic mean of Precision and Recall.
- **IoU (Intersection over Union)**: Standard metric for segmentation overlap.

### How to Run Evaluation:

1. **Demo (Synthetic Data)**:
   Verify the metrics logic with a synthetic test:
   ```bash
   python test_metrics_demo.py
   ```

2. **Full Dataset Evaluation**:
   Run the model against your own dataset by providing the actual folder paths:
   ```bash
   # Replace the paths below with your actual data locations
   python evaluate_model.py --images "C:/data/test_images" --masks "C:/data/ground_truth"
   ```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---