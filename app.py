import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import asyncio
import os

os.environ["STREAMLIT_WATCHDOG"] = "0"  # Prevent watchdog issues in Streamlit

# Ensure an asyncio event loop exists
def get_or_create_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

loop = get_or_create_event_loop()

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

            # Run YOLO segmentation asynchronously
            future = loop.run_in_executor(None, lambda: model(img, conf=0.3, imgsz=640))
            results = loop.run_until_complete(future)

            # Create an empty mask
            colored_mask = np.zeros_like(img, dtype=np.uint8)

            # Process masks safely
            if results and results[0].masks is not None:
                for mask in results[0].masks.data:
                    mask = mask.cpu().numpy()  # Convert to numpy array
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize to match input image
                    mask = (mask > 0.3).astype(np.uint8)  # Apply threshold
                    colored_mask[mask == 1] = nail_color  # Apply red color

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
