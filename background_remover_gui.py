"""
================================================================================
  Real-Time Background Remover — Premium GUI Version
================================================================================

FEATURES:
    - Modern Dark UI (CustomTkinter)
    - Real-time Sidebar Controls
    - Threaded optimized capture
    - Dynamic mask smoothing & thresholding

CONTROLS:
    - Mode Switch: Color / Image / Blur
    - Color Picker: RGB Sliders
    - Segmentation Threshold Slider
    - Edge Smoothing Slider
    - Screenshot Button
================================================================================
"""

import os

# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import threading
import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from metrics import SegmentationMetrics

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MODEL_SELECTION = 1
SCREENSHOT_DIR = "screenshots"
STOCK_BG_DIR = "stock_backgrounds"

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO CAPTURE THREAD
# ─────────────────────────────────────────────────────────────────────────────

class WebcamStream:
    def __init__(self, index=0, width=640, height=480):
        self.stream = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
            else:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundRemoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Antigravity — Real-Time Background Remover")
        self.geometry("1100x700")

        # Runtime State
        self.bg_mode = "Solid Color"
        self.bg_color = [0, 177, 64] # Green
        self.bg_image = None
        self.stock_images = {} # name -> cv2_img
        self.threshold = 0.5
        self.smoothing = 7
        self.fps = 0
        self.prev_time = 0
        self.video_size = (640, 480) # Default, will update
        
        # Validation State
        self.gt_mask = None
        self.metrics_labels = {}
        
        # Load Stock Backgrounds
        self._load_stock_pool()
        
        # Initialize Models
        self.segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=MODEL_SELECTION)
        self.stream = WebcamStream().start()

        self._setup_ui()
        self._update_loop()

    def _setup_ui(self):
        # Configure Grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ─── SIDEBAR (SCROLLABLE) ───
        self.sidebar = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="CONTROLS", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=(20, 20))

        # Mode Selection
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Background Mode:")
        self.mode_label.pack(anchor="w", padx=20)
        self.mode_menu = ctk.CTkOptionMenu(self.sidebar, values=["Solid Color", "Image from Disk", "Blur Background"],
                                            command=self._change_mode)
        self.mode_menu.pack(pady=(5, 20), padx=20)

        # Threshold Slider
        self.thresh_label = ctk.CTkLabel(self.sidebar, text=f"Threshold: {self.threshold}")
        self.thresh_label.pack(anchor="w", padx=20)
        self.thresh_slider = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self._set_threshold)
        self.thresh_slider.set(self.threshold)
        self.thresh_slider.pack(pady=(5, 20), padx=20)

        # Smoothing Slider
        self.smooth_label = ctk.CTkLabel(self.sidebar, text=f"Smoothing: {self.smoothing}")
        self.smooth_label.pack(anchor="w", padx=20)
        self.smooth_slider = ctk.CTkSlider(self.sidebar, from_=1, to=21, command=self._set_smoothing)
        self.smooth_slider.set(self.smoothing)
        self.smooth_slider.pack(pady=(5, 20), padx=20)

        # Color Pickers (Sliders)
        self.color_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.color_frame.pack(fill="x", padx=20)
        
        ctk.CTkLabel(self.color_frame, text="Background Color (R G B):").pack(anchor="w")
        self.r_slider = ctk.CTkSlider(self.color_frame, from_=0, to=255, height=15, command=lambda x: self._update_color())
        self.r_slider.set(self.bg_color[0])
        self.r_slider.pack(pady=2)
        
        self.g_slider = ctk.CTkSlider(self.color_frame, from_=0, to=255, height=15, command=lambda x: self._update_color())
        self.g_slider.set(self.bg_color[1])
        self.g_slider.pack(pady=2)
        
        self.b_slider = ctk.CTkSlider(self.color_frame, from_=0, to=255, height=15, command=lambda x: self._update_color())
        self.b_slider.set(self.bg_color[2])
        self.b_slider.pack(pady=2)

        # Stock Backgrounds
        self.stock_label = ctk.CTkLabel(self.sidebar, text="Stock Backgrounds:")
        self.stock_label.pack(anchor="w", padx=20, pady=(20, 0))
        stock_list = ["(None)"] + list(self.stock_images.keys())
        self.stock_menu = ctk.CTkOptionMenu(self.sidebar, values=stock_list, command=self._select_stock_bg)
        self.stock_menu.pack(pady=(5, 10), padx=20)

        # Buttons
        self.btn_image = ctk.CTkButton(self.sidebar, text="Load Custom Image", command=self._load_image)
        self.btn_image.pack(pady=(30, 10), padx=20)
        
        self.btn_screenshot = ctk.CTkButton(self.sidebar, text="Take Screenshot", fg_color="#2ecc71", hover_color="#27ae60",
                                            command=self._take_screenshot)
        self.btn_screenshot.pack(pady=10, padx=20)
        
        # ─── EVALUATION PANEL ───
        self.eval_frame = ctk.CTkFrame(self.sidebar, fg_color="#333333")
        self.eval_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(self.eval_frame, text="EVALUATION METRICS", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        
        self.btn_load_gt = ctk.CTkButton(self.eval_frame, text="Load Ground Truth", height=24, fg_color="#e67e22", hover_color="#d35400",
                                         command=self._load_gt_mask)
        self.btn_load_gt.pack(pady=5, padx=10)
        
        metrics_grid = ctk.CTkFrame(self.eval_frame, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=10, pady=5)
        
        for m in ["Accuracy", "Precision", "Recall", "F1-Score", "IoU"]:
            f = ctk.CTkFrame(metrics_grid, fg_color="transparent")
            f.pack(fill="x")
            ctk.CTkLabel(f, text=f"{m}:", font=ctk.CTkFont(size=11)).pack(side="left")
            self.metrics_labels[m] = ctk.CTkLabel(f, text="N/A", font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498db")
            self.metrics_labels[m].pack(side="right")

        # ─── MAIN CONTENT ───
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # ─── FOOTER ───
        self.status_bar = ctk.CTkLabel(self, text="Initializing...", anchor="w", font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=20, pady=5, sticky="we")

    # Callbacks
    def _load_stock_pool(self):
        if not os.path.exists(STOCK_BG_DIR): return
        for f in os.listdir(STOCK_BG_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(f)[0].capitalize()
                img = cv2.imread(os.path.join(STOCK_BG_DIR, f))
                if img is not None:
                    self.stock_images[name] = cv2.resize(img, (640, 480))

    def _select_stock_bg(self, name):
        if name in self.stock_images:
            self.bg_image = self.stock_images[name]
            self.bg_mode = "Image from Disk"
            self.mode_menu.set("Image from Disk")

    def _change_mode(self, mode):
        self.bg_mode = mode

    def _set_threshold(self, val):
        self.threshold = round(float(val), 2)
        self.thresh_label.configure(text=f"Threshold: {self.threshold}")

    def _set_smoothing(self, val):
        s = int(val)
        if s % 2 == 0: s += 1 # Must be odd for blur
        self.smoothing = s
        self.smooth_label.configure(text=f"Smoothing: {self.smoothing}")

    def _update_color(self):
        self.bg_color = [int(self.r_slider.get()), int(self.g_slider.get()), int(self.b_slider.get())]

    def _load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.bg_image = cv2.resize(img, (640, 480))
                self.bg_mode = "Image from Disk"
                self.mode_menu.set("Image from Disk")

    def _take_screenshot(self):
        if hasattr(self, 'current_output'):
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            path = os.path.join(SCREENSHOT_DIR, f"shot_{int(time.time())}.png")
            cv2.imwrite(path, self.current_output)
            self.status_bar.configure(text=f"Last Action: Screenshot saved to {path}")

    def _load_gt_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Store it as a binary 0/1 mask
                _, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                self.gt_mask = binary
                self.status_bar.configure(text=f"Last Action: Loaded Ground Truth Mask {os.path.basename(path)}")
            else:
                self.status_bar.configure(text="Error: Could not load mask file.")

    # Processing Loop
    def _update_loop(self):
        frame = self.stream.read()
        if frame is not None:
            h, w = frame.shape[:2]
            self.video_size = (w, h)

            # 1. Processing
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.segmentor.process(frame_rgb)
            
            # Mask
            mask = results.segmentation_mask
            mask = (mask > self.threshold).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (self.smoothing, self.smoothing), 0)
            mask_3c = cv2.merge([mask, mask, mask])

            # Composite
            if self.bg_mode == "Solid Color":
                bg = np.full_like(frame, (self.bg_color[2], self.bg_color[1], self.bg_color[0]))
            elif self.bg_mode == "Image from Disk" and self.bg_image is not None:
                bg = self.bg_image
                if bg.shape[:2] != (h, w):
                    bg = cv2.resize(bg, (w, h))
            elif self.bg_mode == "Blur Background":
                bg = cv2.GaussianBlur(frame, (31, 31), 0)
            else:
                bg = np.full_like(frame, (self.bg_color[2], self.bg_color[1], self.bg_color[0]))

            # Fast Blend
            fg_part = (frame * mask_3c).astype(np.uint8)
            bg_part = (bg * (1.0 - mask_3c)).astype(np.uint8)
            output = cv2.add(fg_part, bg_part)
            
            self.current_output = output # Store for screenshot

            # 3. Handle Metrics if GT exists
            if self.gt_mask is not None:
                try:
                    # Resize GT to match current frame if needed
                    gt_resized = self.gt_mask
                    if gt_resized.shape[:2] != (h, w):
                        gt_resized = cv2.resize(gt_resized, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Binarize current prediction mask for comparison
                    pred_binary = (mask > 0.5).astype(np.uint8)
                    
                    m_calc = SegmentationMetrics(gt_resized, pred_binary)
                    results = m_calc.get_all_metrics()
                    
                    # Update UI Labels
                    for name, value in results.items():
                        if name in self.metrics_labels:
                            self.metrics_labels[name].configure(text=f"{value:.4f}")
                except Exception as e:
                    print(f"[DEBUG] Metrics Error: {e}")

            # 4. Update UI
            t = time.time()
            self.fps = 1 / (t - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = t
            
            self.status_bar.configure(text=f"FPS: {int(self.fps)} | Mode: {self.bg_mode} | Camera: {w}x{h}")

            # Convert to PIL then CTkImage
            img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Use CTkImage for scaling without warnings
            l_w, l_h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if l_w < 50: l_w = 640 # ensure initial size
            if l_h < 50: l_h = 480
            
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(l_w, l_h))
            self.video_label.configure(image=ctk_img)
            self.video_label._image = ctk_img # keep reference

        self.after(10, self._update_loop)

if __name__ == "__main__":
    app = BackgroundRemoverApp()
    app.mainloop()
