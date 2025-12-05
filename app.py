# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
from detection import detect_drowsiness

# تنبيه صوتي
def play_alert():
    st.markdown("""
    <script>
    var audio = new Audio('https://www.soundjay.com/buttons/beep-07.mp3');
    audio.play();
    </script>
    """, unsafe_allow_html=True)

# CSS
st.set_page_config(page_title="كشف النعاس الذكي", layout="wide")
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);}
    .title {font-size: 3rem; font-weight: bold; text-align: center; color: #00d4ff; text-shadow: 0 0 10px #00d4ff;}
    .subtitle {font-size: 1.5rem; text-align: center; color: #a0e7ff; margin-bottom: 2rem;}
    .card {background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px);}
    .metric-card {background: rgba(0,212,255,0.2); padding: 1rem; border-radius: 10px; text-align: center;}
    .stButton>button {background: #00d4ff; color: white; font-weight: bold; border-radius: 10px; padding: 0.5rem 2rem;}
    .stButton>button:hover {background: #00b0d4;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">كشف النعاس الذكي</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">تحليل العيون + تنبيه فوري + تاريخ</p>', unsafe_allow_html=True)

# Sidebar
mode = st.sidebar.radio("اختر المصدر", ["كاميرا حية", "رفع فيديو", "رفع صورة"])

# Session State
if 'history' not in st.session_state: st.session_state.history = []
if 'counter' not in st.session_state: st.session_state.counter = 0

col1, col2 = st.columns([3, 1])
with col1: frame_ph = st.empty()
with col2:
    status_ph = st.empty()
    history_ph = st.empty()

# تشغيل
if mode == "كاميرا حية":
    cap = cv2.VideoCapture(0)
    if st.sidebar.button("ابدأ الكاميرا"):
        st.session_state.running = True
    if st.sidebar.button("إيقاف"):
        st.session_state.running = False
        cap.release()

    if st.session_state.get('running'):
        ret, frame = cap.read()
        if ret:
            result, counter, alert, status = detect_drowsiness(frame, [st.session_state.counter])
            st.session_state.counter = counter
            frame_ph.image(result, channels="BGR", use_column_width=True)
            color = "red" if alert else ("orange" if "مغلق" in status else "green")
            status_ph.markdown(f"<h3 style='color:{color}'>الحالة: {status}</h3>", unsafe_allow_html=True)
            if alert and (not st.session_state.history or st.session_state.history[-1][1] != "نائم"):
                play_alert()
            st.session_state.history.append((time.strftime("%H:%M:%S"), status))
            if len(st.session_state.history) > 15: st.session_state.history.pop(0)
            history_ph.dataframe(pd.DataFrame(st.session_state.history, columns=["الوقت", "الحالة"]), use_container_width=True)

elif mode == "رفع فيديو":
    video_file = st.sidebar.file_uploader("ارفع فيديو", type=['mp4', 'avi', 'mov'])
    if video_file:
        tfile = f"temp_{int(time.time())}.mp4"
        with open(tfile, 'wb') as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(tfile)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            result, counter, alert, status = detect_drowsiness(frame, [st.session_state.counter])
            st.session_state.counter = counter
            frame_ph.image(result, channels="BGR")
            if alert: play_alert()
            time.sleep(0.03)
        cap.release()
        os.remove(tfile)

elif mode == "رفع صورة":
    img_file = st.sidebar.file_uploader("ارفع صورة", type=['jpg', 'png'])
    if img_file:
        img = Image.open(img_file)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result, _, alert, status = detect_drowsiness(frame, [0])
        frame_ph.image(result, channels="BGR")
        if alert: play_alert()
