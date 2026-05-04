# Real-Time Background Remover — Project Documentation Summary

This repository contains a high-performance, local-first application for **Real-Time Webcam Background Removal**. It leverages **MediaPipe Selfie Segmentation** and **OpenCV** to provide a seamless, professional virtual backdrop experience without the need for physical green screens.

---

## 1. Introduction to the Project
### 1.1 Overview of Problem Statement
Standard video conferencing often captures cluttered or sensitive environments. Existing software solutions are frequently resource-heavy, cloud-dependent, or suffer from high latency. This project addresses the need for a **local-first, lightweight, and low-latency** solution for high-fidelity segmentation.

### 1.2 Objectives and Goals
*   **Performance**: Maintain >30 FPS on standard consumer hardware.
*   **Quality**: Precise boundary detection for hair and fine details.
*   **Flexibility**: Support for solid colors, custom images, and professional blur effects.
*   **Privacy**: 100% offline processing.

### 1.3 Scope of the Document
This README summarizes the end-to-end development, data handling, model implementation, and performance evaluation of the system.

---

## 2. About the Data
### 2.1 Data Types
*   **Images**: RGB (JPEG/PNG) for frames and backgrounds.
*   **Masks**: Single-channel binary maps (0 or 255).
*   **Streams**: Live webcam video buffers.

### 2.2 Data Volume and Size
*   **Inference**: Frames processed at 480p/720p.
*   **Model Size**: ~5.0 MB TFLite optimized binary.
*   **Throughput**: ~30-90 MB/sec of raw pixel data during real-time capture.

### 2.3 Data Categories
1.  **Input Data**: Live camera feeds.
2.  **Ground Truth**: Manually/synthetically annotated masks for evaluation.
3.  **Stock Assets**: Pre-loaded high-definition backdrop images.

### 2.4 Data Source
Sourced via local hardware (Webcams), Google MediaPipe (pre-trained model), and synthetic generation scripts (`create_sample_data.py`).

---

## 3. Preprocessing the Dataset
### 3.1 Data Cleaning & Augmentation
*   **Annotation**: Use of binary masks for precise foreground labeling.
*   **Augmentation**: Real-time noise reduction and Gaussian smoothing of masks to handle "jagged" segmentation edges.

### 3.2 Data Transformation & Feature Engineering
Conversion of raw BGR frames to RGB for model compatibility, followed by conversion to NumPy tensors for vectorized blending.

### 3.3 Normalization & Resizing
*   **Resizing**: Images are internally downscaled to 256x256 for model inference and upscaled back for UI display.
*   **Normalization**: Pixel values rescaled from `[0, 255]` to `[0, 1]` or `[-1, 1]` to align with model weights.

---

## 4. Implementation of the Model
### 4.1 Machine Learning Algorithms
The project utilizes **MediaPipe Selfie Segmentation**, a specialized convolutional neural network (CNN) optimized for mobile and edge devices.
### 4.2 Model Architecture
*   **Backbone**: MobileNetV3-style architecture for efficient feature extraction.
*   **Head**: A lightweight decoder branch that outputs a high-resolution probability mask.
### 4.3 Pre-trained Model Deployment
*   **No Training Required**: This project does not include a training phase. It utilizes a **pre-trained model** from Google MediaPipe, which was trained on thousands of diverse human portraits.
*   **Optimization**: Inference is accelerated via XNNPACK and GPU delegation where available to ensure real-time performance.
*   **Post-Processing**: Gaussian blurring is applied to the output mask for a "soft edge" professional finish.
---
## 5. Performance Measures
### 5.1 Evaluation Metrics
The model is quantitatively evaluated using metrics standard for image segmentation:
*   **IoU (Intersection over Union)**: Standard measure of overlap between the predicted mask and ground truth.
*   **F1-Score (Dice Coefficient)**: The harmonic mean of precision and recall, highlighting the accuracy of the subject's boundary.
*   **Accuracy/Precision/Recall**: Pixel-wise correctness metrics.
### 5.2 Visualization of Results
The project includes automated visualization tools (`evaluate_model.py`) that generate:
*   **Side-by-side comparisons** of Original vs. Segmented frames.
*   **Metrics Summary Plots** for performance tracking across different lighting conditions.
---
## 6. Document Exclusions (Not Required)
Due to the use of a pre-trained inference engine, the following standard machine learning documentation components are **not required** for this project:
1.  **Data Splitting**: No Training/Validation/Test split is needed as the model weights are static.
2.  **Confusion Matrix**: While applicable to classification, it is redundant for binary segmentation where IoU and Dice coefficients provide superior insights into spatial accuracy.
3.  **Training Progress**: Loss curves, epoch logs, and hyperparameter tuning for weights are N/A.
4.  **Feature Selection**: The model performs automatic feature extraction within its pre-defined architecture.
---
## 7. Bibliography
*   **Research**: "MediaPipe: A Framework for Building Perception Pipelines" (Google Research).
*   **Libraries**: OpenCV, MediaPipe, NumPy, CustomTkinter.
*   **Data**: Google Open Images and Synthetic Human Portrait datasets.