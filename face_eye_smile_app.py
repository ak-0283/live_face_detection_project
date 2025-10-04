import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Smart Face Detector", page_icon="📷", layout="centered")

st.title("📷 Smart Face, Eye & Smile Detector")
st.markdown("Press **Start** below to begin live detection.")

# ✅ Use built-in Haarcascades (no absolute paths required)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# WebRTC Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.smile_cascade = smile_cascade

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 7)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            if len(eyes) > 0:
                cv2.putText(img, "Eyes Detected", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Detect smiles
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 25)
            if len(smiles) > 0:
                cv2.putText(img, "Smiling", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ✅ Keep WebRTC stable (SENDRECV = send & receive video)
webrtc_streamer(
    key="face-eye-smile-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.info("Press **Stop** to end the webcam feed. Streamlit handles cleanup automatically.")