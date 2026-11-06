"""
Hand Gesture Digit Detector — Streamlit App
Menggunakan YOLOv8 custom model untuk mendeteksi angka (0–5) dari gesture tangan.

Jalankan:
    streamlit run app.py
"""

import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ── Konfigurasi ──────────────────────────────────────────────────────────────
MODEL_PATH     = r"runs/hand_gesture/weights/best.pt"
CONF_THRESHOLD = 0.5
CAMERA_INDEX   = 0

# Label dan emoji per angka
DIGIT_INFO = {
    "0": {"emoji": "🤙", "label": "Zero",  "color": "#6366f1"},
    "1": {"emoji": "☝️",  "label": "One",   "color": "#3b82f6"},
    "2": {"emoji": "✌️",  "label": "Two",   "color": "#10b981"},
    "3": {"emoji": "🤟",  "label": "Three", "color": "#f59e0b"},
    "4": {"emoji": "🖖",  "label": "Four",  "color": "#ef4444"},
    "5": {"emoji": "🖐️", "label": "Five",  "color": "#8b5cf6"},
}

# Warna bounding box per kelas (BGR)
BOX_COLORS = {
    0: (99,  102, 241),
    1: (59,  130, 246),
    2: (16,  185, 129),
    3: (245, 158,  11),
    4: (239,  68,  68),
    5: (139,  92, 246),
}

# ── Halaman Streamlit ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Detector",
    page_icon="🖐️",
    layout="centered",
)

st.title("🖐️ Hand Gesture Digit Detector")
st.caption(f"Model: `{MODEL_PATH}` · Confidence ≥ {CONF_THRESHOLD}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    conf_slider = st.slider(
        "Confidence threshold", 0.10, 0.95, CONF_THRESHOLD, 0.05,
        help="Turunkan jika deteksi terlalu sedikit, naikkan jika terlalu banyak false positive."
    )
    camera_idx = st.number_input("Index kamera", min_value=0, max_value=5, value=CAMERA_INDEX, step=1)
    run_camera  = st.toggle("▶ Aktifkan Kamera", value=False)
    st.divider()
    st.markdown("**Kelas yang dideteksi:**")
    for digit, info in DIGIT_INFO.items():
        st.markdown(f"{info['emoji']} `{digit}` — {info['label']}")

# ── Load model (cache agar tidak reload setiap frame) ─────────────────────────
@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)

model = load_model(MODEL_PATH)

# ── Layout utama ─────────────────────────────────────────────────────────────
camera_placeholder = st.empty()          # frame kamera
st.divider()
status_placeholder  = st.empty()          # teks status besar
card_placeholder    = st.empty()          # kartu info deteksi
metrics_placeholder = st.empty()          # FPS + jumlah deteksi

# ── Fungsi helper ─────────────────────────────────────────────────────────────
def draw_boxes(frame: np.ndarray, results) -> tuple[np.ndarray, list[dict]]:
    """Gambar bounding box + label pada frame, kembalikan annotated + list deteksi."""
    annotated  = frame.copy()
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf   = float(box.conf[0])
            if conf < conf_slider:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label  = model.names.get(cls_id, str(cls_id))
            color  = BOX_COLORS.get(cls_id, (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Background label
            text = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            detections.append({"cls_id": cls_id, "label": label, "conf": conf})

    return annotated, detections


def render_digit_card(detections: list[dict]) -> None:
    """Render kartu info di bawah kamera sesuai angka yang terdeteksi."""
    if not detections:
        card_placeholder.info("👋 Tunjukkan tangan ke kamera…")
        return

    # Ambil deteksi dengan confidence tertinggi
    best = max(detections, key=lambda d: d["conf"])
    digit = best["label"]
    info  = DIGIT_INFO.get(digit, {"emoji": "❓", "label": digit, "color": "#64748b"})

    card_placeholder.markdown(
        f"""
        <div style="
            background: {info['color']}18;
            border: 2px solid {info['color']};
            border-radius: 16px;
            padding: 24px 32px;
            text-align: center;
            margin: 8px 0;
        ">
            <div style="font-size: 72px; line-height: 1;">{info['emoji']}</div>
            <div style="font-size: 48px; font-weight: 800; color: {info['color']}; margin: 8px 0;">
                {digit}
            </div>
            <div style="font-size: 22px; color: #475569; font-weight: 500;">
                {info['label']}
            </div>
            <div style="font-size: 14px; color: #94a3b8; margin-top: 8px;">
                confidence: {best['conf']:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Jika ada lebih dari satu deteksi, tampilkan semua di bawah
    if len(detections) > 1:
        others = detections[:]  # tampilkan semua termasuk best
        cols = status_placeholder.columns(min(len(others), 5))
        for col, d in zip(cols, others):
            di = DIGIT_INFO.get(d["label"], {"emoji": "❓", "label": d["label"], "color": "#64748b"})
            col.markdown(
                f"<div style='text-align:center; font-size:28px'>{di['emoji']}</div>"
                f"<div style='text-align:center; font-weight:700; color:{di['color']}'>{d['label']}</div>"
                f"<div style='text-align:center; font-size:12px; color:#94a3b8'>{d['conf']:.0%}</div>",
                unsafe_allow_html=True,
            )
    else:
        status_placeholder.empty()


# ── Loop kamera ───────────────────────────────────────────────────────────────
if not run_camera:
    camera_placeholder.info("👈 Aktifkan kamera dari sidebar untuk memulai deteksi.")
    card_placeholder.empty()
    metrics_placeholder.empty()
    st.stop()

cap = cv2.VideoCapture(int(camera_idx))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    st.error(f"❌ Tidak dapat membuka kamera index {camera_idx}. Coba ubah index di sidebar.")
    st.stop()

fps_time    = time.time()
frame_count = 0
fps         = 0.0

try:
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)  # mirror horizontal

        # Inferensi
        results    = model.predict(frame, conf=conf_slider, verbose=False)
        annotated, detections = draw_boxes(frame, results)

        # Hitung FPS
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            fps_time    = time.time()
            frame_count = 0

        # Overlay FPS
        cv2.putText(
            annotated, f"FPS {fps:.1f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 230, 0), 2, cv2.LINE_AA,
        )

        # Tampilkan frame (BGR → RGB)
        camera_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        # Render kartu digit
        render_digit_card(detections)

        # Metrics bawah
        det_count = len(detections)
        metrics_placeholder.markdown(
            f"**FPS:** `{fps:.1f}` &nbsp;|&nbsp; "
            f"**Deteksi:** `{det_count}` objek &nbsp;|&nbsp; "
            f"**Threshold:** `{conf_slider:.0%}`"
        )

finally:
    cap.release()
