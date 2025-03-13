import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os

# Disable Streamlit file watcher to avoid conflicts with Torch
os.environ["STREAMLIT_WATCHDOG"] = "0"

# Load YOLO model
try:
    model = YOLO("best.pt")
    st.success("YOLO model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Live Nail Segmentation with YOLOv11")

# Define nail color (BGR format)
nail_color = (0, 0, 255)  # Red nails

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        try:
            # Convert frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO segmentation (no asyncio)
            results = model(img, conf=0.3, imgsz=640)

            # Create an empty mask
            colored_mask = np.zeros_like(img, dtype=np.uint8)

            # Process masks safely
            if results and results[0].masks is not None:
                for mask in results[0].masks.data:
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask = (mask > 0.3).astype(np.uint8)
                    colored_mask[mask == 1] = nail_color

            # Blend the mask with the original frame
            output_img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

            return av.VideoFrame.from_ndarray(output_img, format="bgr24")
        except Exception as e:
            st.error(f"Error in processing frame: {e}")
            return frame  # Return original frame on error

webrtc_streamer(
    key="nail-segmentation",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
