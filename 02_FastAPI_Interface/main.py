import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from playsound import playsound
import threading
import os

# --- Konstanta untuk Mata ---
DEFAULT_EAR_THRESHOLD = 0.20
DEFAULT_CONSECUTIVE_FRAMES_EYE = 20

# --- Konstanta untuk Mulut (Menguap) ---
DEFAULT_MAR_THRESHOLD = 0.6  # Threshold untuk deteksi mulut terbuka (menguap)
DEFAULT_CONSECUTIVE_FRAMES_YAWN = 25 # Jumlah frame berturut-turut untuk konfirmasi menguap

ALARM_SOUND_PATH = "alarm_cut.mp3"
if not os.path.exists(ALARM_SOUND_PATH):
    st.error(f"File alarm '{ALARM_SOUND_PATH}' tidak ditemukan. Pastikan file tersebut ada di folder yang sama.")
    st.stop()

# --- Inisialisasi MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Landmark indices untuk mata
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Landmark indices untuk mulut ---
MOUTH_VERTICAL_INDICES = [13, 14] 
MOUTH_HORIZONTAL_INDICES = [78, 308]

def calculate_ear(eye_landmarks):
    # Hitung jarak vertikal
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Hitung jarak horizontal
    h1 = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    if h1 == 0:
        return 0
    ear = (v1 + v2) / (2.0 * h1)
    return ear

# --- Fungsi untuk menghitung Mouth Aspect Ratio (MAR) ---
def calculate_mar(mouth_landmarks):
    # Hitung jarak vertikal (antara bibir atas dan bawah)
    v_dist = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    # Hitung jarak horizontal (antara sudut mulut)
    h_dist = dist.euclidean(mouth_landmarks[2], mouth_landmarks[3])
    if h_dist == 0:
        return 0
    mar = v_dist / h_dist
    return mar

class DrowsinessDetector(VideoProcessorBase):
    def __init__(self, ear_threshold, consecutive_frames_eye, mar_threshold, consecutive_frames_yawn):
        self.ear_threshold = ear_threshold
        self.consecutive_frames_eye = consecutive_frames_eye
        self.mar_threshold = mar_threshold
        self.consecutive_frames_yawn = consecutive_frames_yawn

        self.eye_frame_counter = 0
        self.yawn_frame_counter = 0 
        self.drowsy_alert_triggered = False
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        alert_text = ""
        is_drowsy = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                landmarks = np.array([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in face_landmarks.landmark], dtype=np.int32)
                
                # --- Deteksi Mata ---
                left_eye_lm = landmarks[LEFT_EYE_INDICES]
                right_eye_lm = landmarks[RIGHT_EYE_INDICES]
                left_ear = calculate_ear(left_eye_lm)
                right_ear = calculate_ear(right_eye_lm)
                avg_ear = (left_ear + right_ear) / 2.0
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (image.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- Deteksi Menguap ---
                mouth_v_lm = landmarks[MOUTH_VERTICAL_INDICES]
                mouth_h_lm = landmarks[MOUTH_HORIZONTAL_INDICES]
                # Gabungkan landmark untuk fungsi MAR
                mouth_lm = np.concatenate((mouth_v_lm, mouth_h_lm))
                mar = calculate_mar(mouth_lm)
                cv2.putText(image, f"MAR: {mar:.2f}", (image.shape[1] - 150, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # Cek kondisi mata tertutup
                if avg_ear < self.ear_threshold:
                    self.eye_frame_counter += 1
                else:
                    self.eye_frame_counter = 0

                # Cek kondisi menguap
                if mar > self.mar_threshold:
                    self.yawn_frame_counter += 1
                else:
                    self.yawn_frame_counter = 0
                
                # Cek apakah salah satu kondisi kantuk terpenuhi
                if self.eye_frame_counter >= self.consecutive_frames_eye:
                    is_drowsy = True
                    alert_text = "MATA MENGANTUK!"
                
                if self.yawn_frame_counter >= self.consecutive_frames_yawn:
                    is_drowsy = True
                    # Jika mata juga mengantuk, tambahkan teksnya
                    alert_text = "MENGUAP TERDETEKSI!" if alert_text == "" else alert_text + " & MENGUAP"


                if is_drowsy:
                    cv2.putText(image, alert_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not self.drowsy_alert_triggered:
                        self.drowsy_alert_triggered = True
                        try:
                            threading.Thread(target=playsound, args=(ALARM_SOUND_PATH,)).start()
                        except Exception as e:
                            print(f"Error playing sound: {e}")
                else:
                    self.drowsy_alert_triggered = False
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.set_page_config(page_title="Deteksi Kantuk Real-Time", layout="wide")
st.title("😴 Aplikasi Deteksi Kantuk (Mata & Menguap)")
st.write("Aplikasi ini menggunakan webcam untuk mendeteksi tanda kantuk dari mata tertutup (EAR) dan menguap (MAR).")

st.sidebar.title("Pengaturan")
st.sidebar.header("Deteksi Mata")
ear_thresh = st.sidebar.slider("Ambang Batas EAR", 0.10, 0.40, DEFAULT_EAR_THRESHOLD, 0.01)
consecutive_frames_eye = st.sidebar.slider("Frame Mata Tertutup untuk Peringatan", 5, 50, DEFAULT_CONSECUTIVE_FRAMES_EYE, 1)

st.sidebar.header("Deteksi Menguap")
mar_thresh = st.sidebar.slider("Ambang Batas MAR (Menguap)", 0.2, 1.0, DEFAULT_MAR_THRESHOLD, 0.1)
consecutive_frames_yawn = st.sidebar.slider("Frame Mulut Terbuka untuk Peringatan", 5, 50, DEFAULT_CONSECUTIVE_FRAMES_YAWN, 1)


webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=lambda: DrowsinessDetector(
        ear_threshold=ear_thresh,
        consecutive_frames_eye=consecutive_frames_eye,
        mar_threshold=mar_thresh,
        consecutive_frames_yawn=consecutive_frames_yawn
    ),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    st.info("Arahkan wajah Anda ke kamera. Sesuaikan pengaturan di sidebar jika perlu.")
else:
    st.error("Silakan klik 'START' untuk memulai deteksi.")

st.sidebar.info("Aplikasi ini dibuat sebagai demo. Akurasi dapat bervariasi tergantung pada pencahayaan dan posisi wajah.")
