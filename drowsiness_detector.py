import streamlit as st
import cv2
import numpy as np
import threading
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- CONFIG ---
MODEL_PATH = "eye_status_model_final.h5"
EYE_SIZE = (24, 24)
CLOSED_LIMIT = 3.0  # seconds
GAP_FRAMES = 7
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

ALERT_SOUND = "alert.mp3"
USE_PYGAME = True

# --- INIT AUDIO ---
if USE_PYGAME:
    from pygame import mixer
    mixer.init()

    def start_alarm():
        mixer.music.load(ALERT_SOUND)
        mixer.music.play(-1)

    def stop_alarm():
        mixer.music.stop()
else:
    import winsound
    def start_alarm():
        winsound.PlaySound(ALERT_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

    def stop_alarm():
        winsound.PlaySound(None, winsound.SND_PURGE)

# --- INIT MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
model = load_model(MODEL_PATH)

# --- INIT CASCADE ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
eye_cascade_fb = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# --- STREAMLIT UI STATE ---
st.set_page_config(page_title="Drowsiness Detector", layout="centered")
st.title("Drowsiness Detector")

# Gunakan session_state agar loop bisa berhenti ketika checkbox dimatikan
if "run_detect" not in st.session_state:
    st.session_state.run_detect = False

st.checkbox("Start Detection", key="run_detect")

# Placeholder untuk video frame
frame_placeholder = st.empty()

# Penanda apakah alarm sedang berbunyi
alarm_playing = False


def detect_drowsiness():
    """Loop utama deteksi kantuk. Berjalan sepanjang st.session_state.run_detect == True"""

    global alarm_playing

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    eye_closed_start = None
    eyes_last_state = None
    gap_counter = 0
    prev_eyes_boxes = []

    while st.session_state.run_detect:
        # Ambil frame
        ret, frame = cap.read()
        if not ret:
            # Tampilkan placeholder "waiting"
            waiting = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting, "Menunggu webcam...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            frame_placeholder.image(waiting, channels="BGR")
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        now = time.time()
        eyes_closed_this_frame = False
        eyes_detected = False
        new_eyes_boxes = []

        if len(faces):
            (x, y, w, h) = faces[0]  # Ambil wajah pertama
            roi_face = frame[y : y + int(0.55 * h), x : x + w]
            g_face = cv2.cvtColor(roi_face, cv2.COLOR_BGR2GRAY)

            # CLAHE + equalize
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
            g_face = clahe.apply(g_face)
            g_face = cv2.equalizeHist(g_face)

            # Deteksi mata
            eye_min = max(8, int(0.05 * w))
            eyes = eye_cascade.detectMultiScale(g_face, 1.05, 4, minSize=(eye_min, eye_min))
            if len(eyes) == 0:
                eyes = eye_cascade_fb.detectMultiScale(g_face, 1.05, 6, minSize=(eye_min, eye_min))
            if len(eyes) == 0:
                eyes = prev_eyes_boxes  # Gunakan koordinat frame sebelumnya jika ada

            closed_cnt = 0
            for (ex, ey, ew, eh) in eyes[:2]:  # Hanya periksa 2 mata pertama
                eyes_detected = True
                new_eyes_boxes.append((ex, ey, ew, eh))

                # Crop gambar mata
                eye_img = roi_face[ey : ey + eh, ex : ex + ew]

                # ---- PERBAIKAN: validasi eye_img sebelum resize ----
                if eye_img is None or eye_img.size == 0 or eye_img.shape[0] == 0 or eye_img.shape[1] == 0:
                    continue  # Lewati jika kosong / out of bounds

                # Resize ke ukuran model
                eye_res = cv2.resize(eye_img, EYE_SIZE)

                # Normalisasi & prediksi
                eye_norm = eye_res.astype("float32") / 255.0
                eye_arr = np.expand_dims(img_to_array(eye_norm), axis=0)
                prob_open = float(model.predict(eye_arr, verbose=0)[0][0])

                label = "Open" if prob_open > 0.5 else "Closed"
                color = (0, 255, 0) if label == "Open" else (0, 0, 255)
                if label == "Closed":
                    closed_cnt += 1

                # Gambar kotak & label
                gx1, gy1 = x + ex, y + ey
                gx2, gy2 = gx1 + ew, gy1 + eh
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 2)
                cv2.putText(frame, label, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            eyes_closed_this_frame = closed_cnt >= 1
        else:
            cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Simpan koordinat mata untuk frame berikutnya
        prev_eyes_boxes = new_eyes_boxes if eyes_detected else prev_eyes_boxes

        # Logic status mata / gap frame
        if eyes_detected:
            eyes_last_state = "Closed" if eyes_closed_this_frame else "Open"
            gap_counter = 0
        else:
            if len(faces):
                cv2.putText(frame, "Eyes not detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            gap_counter += 1
            if gap_counter <= GAP_FRAMES and eyes_last_state == "Closed":
                eyes_closed_this_frame = True
            elif gap_counter > GAP_FRAMES:
                eyes_last_state, prev_eyes_boxes = None, []

        # Hitung durasi mata tertutup
        if eyes_closed_this_frame:
            eye_closed_start = eye_closed_start or now
        else:
            eye_closed_start = None

        closed_dur = now - eye_closed_start if eye_closed_start else 0
        cv2.putText(
            frame,
            f"Closed: {closed_dur:.1f}s",
            (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )

        # Alarm
        if eye_closed_start and closed_dur >= CLOSED_LIMIT:
            cv2.putText(
                frame,
                "WARNING: DROWSINESS!",
                (140, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )
            if not alarm_playing:
                threading.Thread(target=start_alarm, daemon=True).start()
                alarm_playing = True
        else:
            if alarm_playing:
                stop_alarm()
                alarm_playing = False

        # Tampilkan frame di Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Keluar dari loop, bersihkan
    cap.release()
    stop_alarm()
    frame_placeholder.empty()


# Jalankan loop deteksi saat checkbox aktif
if st.session_state.run_detect:
    detect_drowsiness()
