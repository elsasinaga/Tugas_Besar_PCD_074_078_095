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
import time
from collections import deque

# --- KONFIGURASI APLIKASI ---

# --- Pengaturan Suara ---
ALARM_SOUNDS = {
    1: "bip.mp3",
    2: "alarm_medium.mp3",
    3: "alarm.mp3"
}
DEFAULT_ALARM = "alarm.mp3"

if not os.path.exists(DEFAULT_ALARM):
    st.error(f"File alarm utama '{DEFAULT_ALARM}' tidak ditemukan. Aplikasi tidak dapat berjalan.")
    st.stop()

# --- Pengaturan Default untuk Sidebar ---
DEFAULT_EAR_THRESH = 0.20
DEFAULT_EAR_FRAMES = 25
DEFAULT_MAR_THRESH = 0.6
DEFAULT_MAR_FRAMES = 15
DEFAULT_NOD_FRAMES = 10
DEFAULT_NOD_THRESH = 20.0
DEFAULT_BLINK_RATE_THRESH = 10

# --- Inisialisasi MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- INDEKS LANDMARK ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_VERTICAL_INDICES = [13, 14]
MOUTH_HORIZONTAL_INDICES = [78, 308]
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

# --- FUNGSI KALKULASI ---
def calculate_ear(eye_landmarks):
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    h1 = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h1) if h1 != 0 else 0

def calculate_mar(mouth_landmarks):
    v_dist = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    h_dist = dist.euclidean(mouth_landmarks[2], mouth_landmarks[3])
    return v_dist / h_dist if h_dist != 0 else 0

# --- KELAS PROSESOR VIDEO UTAMA ---
class DrowsinessDetector(VideoProcessorBase):
    ## MODIFIKASI: Tambahkan 'lock' dan 'shared_state'
    def _init_(self, settings, lock, shared_state):
        self.settings = settings
        ## BARU: Inisialisasi lock dan state untuk komunikasi dengan UI
        self.lock = lock
        self.shared_state = shared_state
        
        self.drowsiness_score = 0.0
        self.last_sound_play_time = 0
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.blink_counter = 0
        self.blink_status = "OPEN"
        self.blink_rate_start_time = time.time()
        self.blinks_per_minute = 20
        self.head_pitch_history = deque(maxlen=settings["NOD_FRAMES"])
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def _play_sound(self, level):
        current_time = time.time()
        if current_time - self.last_sound_play_time > 5:
            ## MODIFIKASI: Cek apakah alarm di-mute sebelum memainkan suara
            with self.lock:
                if self.shared_state["alarm_muted"]:
                    return # Jangan mainkan suara jika di-mute
            
            sound_path = ALARM_SOUNDS.get(level, DEFAULT_ALARM)
            if not os.path.exists(sound_path):
                sound_path = DEFAULT_ALARM
            try:
                threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
                self.last_sound_play_time = current_time
            except Exception as e:
                print(f"Error playing sound: {e}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        img_h, img_w, _ = image.shape
        self.drowsiness_score = max(0, self.drowsiness_score - 0.05)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks_3d = results.multi_face_landmarks[0].landmark
            landmarks = np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks_3d], dtype=np.int32)
            
            # Analisis Mata
            avg_ear = (calculate_ear(landmarks[LEFT_EYE_INDICES]) + calculate_ear(landmarks[RIGHT_EYE_INDICES])) / 2.0
            if avg_ear < self.settings["EAR_THRESH"] and self.blink_status == "OPEN": self.blink_status = "CLOSED"
            if avg_ear >= self.settings["EAR_THRESH"] and self.blink_status == "CLOSED":
                self.blink_status = "OPEN"; self.blink_counter += 1
            
            elapsed_time = time.time() - self.blink_rate_start_time
            if elapsed_time > 5:
                self.blinks_per_minute = (self.blink_counter / elapsed_time) * 60
                self.blink_counter = 0; self.blink_rate_start_time = time.time()

            if avg_ear < self.settings["EAR_THRESH"]: self.eye_closed_frames += 1
            else: self.eye_closed_frames = 0
                
            # Analisis Mulut
            mar = calculate_mar(np.concatenate((landmarks[MOUTH_VERTICAL_INDICES], landmarks[MOUTH_HORIZONTAL_INDICES])))
            if mar > self.settings["MAR_THRESH"]: self.yawn_frames += 1
            else: self.yawn_frames = 0
                
            # Analisis Kepala
            face_2d = np.array([landmarks[i] for i in HEAD_POSE_LANDMARKS], dtype=np.float64)
            face_3d = np.array([(landmarks_3d[i].x, landmarks_3d[i].y, landmarks_3d[i].z) for i in HEAD_POSE_LANDMARKS]) * [img_w, img_h, 1000]
            cam_matrix = np.array([[img_w, 0, img_w/2], [0, img_w, img_h/2], [0, 0, 1]], dtype=np.float64)
            
            success, rvec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4,1), dtype=np.float64))
            if success:
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])
                pitch = angles[0]
                self.head_pitch_history.append(pitch)
                if len(self.head_pitch_history) == self.settings["NOD_FRAMES"] and pitch - np.mean(self.head_pitch_history) > self.settings["NOD_THRESH"]:
                    self.drowsiness_score = 100
                        
            # Kalkulasi Skor
            if self.blinks_per_minute < self.settings["BLINK_RATE_THRESH"] and elapsed_time > 5: self.drowsiness_score += 0.5 
            if self.yawn_frames > self.settings["MAR_FRAMES"]: self.drowsiness_score += 15; self.yawn_frames = 0
            if self.eye_closed_frames > self.settings["EAR_FRAMES"]: self.drowsiness_score += 40; self.eye_closed_frames = 0
            
            # Tentukan Level & Tampilkan
            current_alert_level = 0
            alert_text = "AMAN"; alert_color = (0, 255, 0)
            
            if self.drowsiness_score > 80:
                current_alert_level, alert_text, alert_color = 3, "!!! BAHAYA - MICROSLEEP !!!", (0, 0, 255)
                overlay = image.copy(); cv2.rectangle(overlay, (0, 0), (img_w, img_h), alert_color, -1)
                image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
            elif self.drowsiness_score > 40:
                current_alert_level, alert_text, alert_color = 2, "PERHATIAN - MENGANTUK", (0, 255, 255)
            elif self.drowsiness_score > 15:
                current_alert_level, alert_text, alert_color = 1, "MULAI LELAH", (255, 165, 0)
            
            ## MODIFIKASI: Komunikasi dengan UI dan mainkan suara
            with self.lock:
                self.shared_state["current_alert_level"] = current_alert_level
                if current_alert_level == 0 and self.shared_state["alarm_muted"]:
                    self.shared_state["alarm_muted"] = False # Auto-unmute
            
            if current_alert_level > 0: self._play_sound(current_alert_level)
            
            cv2.putText(image, f"SKOR KANTUK: {self.drowsiness_score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(image, f"STATUS: {alert_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            cv2.putText(image, f"BPM: {self.blinks_per_minute:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- ANTARMUKA STREAMLIT ---
st.set_page_config(page_title="Deteksi Kantuk V-Max", layout="wide")
st.title("😴 Aplikasi Deteksi Kantuk V-Max")
st.write("Versi paling lengkap dengan analisis multi-parameter, skor kantuk, peringatan bertingkat, dan tombol kontrol alarm.")

# --- Sidebar Pengaturan ---
st.sidebar.title("🔧 Pengaturan Deteksi")
settings = {}
with st.sidebar.expander("Pengaturan Mata & Menguap"):
    settings["EAR_THRESH"] = st.slider("Ambang Batas EAR (Mata)", 0.10, 0.40, DEFAULT_EAR_THRESH, 0.01)
    settings["EAR_FRAMES"] = st.slider("Frame Mata Tertutup", 5, 50, DEFAULT_EAR_FRAMES, 1)
    settings["MAR_THRESH"] = st.slider("Ambang Batas MAR (Menguap)", 0.2, 1.0, DEFAULT_MAR_THRESH, 0.1)
    settings["MAR_FRAMES"] = st.slider("Frame Menguap", 5, 50, DEFAULT_MAR_FRAMES, 1)
with st.sidebar.expander("Pengaturan Lanjutan (Kepala & Kedipan)"):
    settings["NOD_THRESH"] = st.slider("Sensitivitas Anggukan Kepala", 10.0, 30.0, DEFAULT_NOD_THRESH, 0.5)
    settings["NOD_FRAMES"] = st.slider("Frame Cek Anggukan", 5, 20, DEFAULT_NOD_FRAMES, 1)
    settings["BLINK_RATE_THRESH"] = st.slider("Batas Laju Kedipan Rendah (BPM)", 5, 20, DEFAULT_BLINK_RATE_THRESH, 1)

## BARU: Inisialisasi lock dan state bersama untuk kontrol tombol
lock = threading.Lock()
shared_state = {"current_alert_level": 0, "alarm_muted": False}

# --- WEBRTC Streamer ---
## MODIFIKASI: Menggunakan lambda untuk meneruskan state dan lock ke prosesor
webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection-vmax",
    video_processor_factory=lambda: DrowsinessDetector(settings, lock, shared_state),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

## BARU: Logika untuk menampilkan Tombol Hentikan Alarm
if webrtc_ctx.video_processor:
    st.info("Deteksi berjalan. Arahkan wajah Anda ke kamera.")

    with lock:
        alert_level = shared_state["current_alert_level"]
        is_muted = shared_state["alarm_muted"]

    # Buat 2 kolom untuk menempatkan tombol di tengah
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if alert_level > 0 and not is_muted:
            if st.button("🛑 Hentikan Alarm"):
                with lock:
                    shared_state["alarm_muted"] = True
                st.rerun() # Paksa UI untuk refresh dan menampilkan status "muted"
    
    if is_muted:
        st.warning("🔇 Alarm dimatikan sementara. Akan aktif kembali secara otomatis jika kondisi aman terdeteksi.", icon="🤫")
else:
    st.error("Silakan klik 'START' untuk memulai deteksi.")

st.sidebar.markdown("---")
st.sidebar.header("Level Peringatan:")
st.sidebar.markdown("- *Level 1 (Mulai Lelah):* Skor > 15")
st.sidebar.markdown("- *Level 2 (Mengantuk):* Skor > 40")
st.sidebar.markdown("- *Level 3 (Bahaya):* Skor > 80")
