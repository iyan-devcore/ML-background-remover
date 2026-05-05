import av
import cv2
import os
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Page config
st.set_page_config(page_title="Background Remover Web App", page_icon="🎥", layout="centered")

st.title("🎥 Real-Time Background Remover")
st.markdown("This web application runs on your PC and uses your phone's camera via the browser!")

# ─────────────────────────────────────────────────────────────────────────────
# UI CONTROLS (Sidebar)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    
    mode = st.radio("Background Mode:", ["Solid Color", "Blur Background", "Stock Image"], index=0)
    
    bg_color_hex = st.color_picker("Background Color (Solid Mode):", "#00b140")
    
    bg_image_array = None
    if mode == "Stock Image":
        stock_bg_dir = "stock_backgrounds"
        if os.path.exists(stock_bg_dir):
            stock_files = [f for f in os.listdir(stock_bg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            stock_names = [os.path.splitext(f)[0].capitalize() for f in stock_files]
            
            selected_stock = st.selectbox("Select Stock Image:", stock_names)
            if selected_stock:
                filename = next(f for f in stock_files if f.lower().startswith(selected_stock.lower()))
                path = os.path.join(stock_bg_dir, filename)
                bg_image_array = cv2.imread(path)
    
    threshold = st.slider("Segmentation Threshold:", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    
    smoothing = st.slider("Edge Smoothing:", min_value=1, max_value=21, value=7, step=2)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

color_rgb = hex_to_rgb(bg_color_hex)

# ─────────────────────────────────────────────────────────────────────────────
# WEBRTC PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundRemoverProcessor(VideoProcessorBase):
    def __init__(self):
        self.threshold = 0.5
        self.smoothing = 7
        self.mode = "Solid Color"
        self.color_rgb = (0, 177, 64)
        self.bg_image_array = None
        import mediapipe as mp
        try:
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        except AttributeError:
            from mediapipe.python.solutions import selfie_segmentation
            self.mp_selfie_segmentation = selfie_segmentation
            
        self.segmentor = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Process with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.segmentor.process(img_rgb)
        
        # Ensure mask exists
        if results.segmentation_mask is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        mask = results.segmentation_mask
        mask = (mask > self.threshold).astype(np.float32)
        
        # Ensure smoothing is odd
        smooth_val = int(self.smoothing)
        if smooth_val % 2 == 0: smooth_val += 1
        
        mask = cv2.GaussianBlur(mask, (smooth_val, smooth_val), 0)
        mask_3c = cv2.merge([mask, mask, mask])
        
        # Composite
        if self.mode == "Solid Color":
            bg = np.full_like(img, (self.color_rgb[2], self.color_rgb[1], self.color_rgb[0]))
        elif self.mode == "Blur Background":
            bg = cv2.GaussianBlur(img, (31, 31), 0)
        elif self.mode == "Stock Image" and self.bg_image_array is not None:
            h, w = img.shape[:2]
            bg = self.bg_image_array
            if bg.shape[:2] != (h, w):
                bg = cv2.resize(bg, (w, h))
        else:
            bg = np.full_like(img, (0, 0, 0))
            
        # Fast Blend
        fg_part = (img * mask_3c).astype(np.uint8)
        bg_part = (bg * (1.0 - mask_3c)).astype(np.uint8)
        output = cv2.add(fg_part, bg_part)
        
        return av.VideoFrame.from_ndarray(output, format="bgr24")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Camera Feed")
st.info("💡 **Note for Phones**: To access this from your phone, make sure your PC and phone are on the same Wi-Fi. Then type `http://<YOUR_PC_IP>:8501` in your mobile browser.")

# Launch WebRTC Streamer
ctx = webrtc_streamer(
    key="bg-remover",
    video_processor_factory=BackgroundRemoverProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Update processor parameters with UI values dynamically
if ctx.video_processor:
    ctx.video_processor.mode = mode
    ctx.video_processor.threshold = threshold
    ctx.video_processor.smoothing = smoothing
    ctx.video_processor.color_rgb = color_rgb
    ctx.video_processor.bg_image_array = bg_image_array if 'bg_image_array' in locals() else None

st.markdown("---")
st.markdown("**Instructions for Local Network Phone Access:**")
st.markdown("1. Find your PC's local IP address (run `ipconfig` in CMD, look for IPv4 Address).")
st.markdown("2. If your browser blocks camera access due to lack of HTTPS, open Chrome on your phone, go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enter `http://<YOUR_PC_IP>:8501`, enable it, and relaunch Chrome.")
