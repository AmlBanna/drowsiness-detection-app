# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import os
from detection import detect_frame

st.set_page_config(page_title="Drowsiness Detection", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);}
    .title {font-size: 3rem; font-weight: bold; text-align: center; color: #00d4ff; text-shadow: 0 0 10px #00d4ff;}
    .stButton>button {background: #00d4ff; color: white; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Drowsiness Detection</h1>', unsafe_allow_html=True)

mode = st.sidebar.radio("Source", ["Live Camera", "Upload Video", "Upload Image"])

# Session state
if 'closed_counter' not in st.session_state: st.session_state.closed_counter = 0
if 'skip_counter' not in st.session_state: st.session_state.skip_counter = 0
if 'last_state' not in st.session_state: st.session_state.last_state = 'open'
if 'fps_start' not in st.session_state: st.session_state.fps_start = time.time()
if 'frame_count' not in st.session_state: st.session_state.frame_count = 0
if 'history' not in st.session_state: st.session_state.history = []

col1, col2 = st.columns([3, 1])
frame_ph = col1.empty()
status_ph = col2.empty()
history_ph = col2.empty()

def play_alert():
    st.markdown("""
    <audio autoplay>
      <source src="https://www.soundjay.com/buttons/beep-07.mp3" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# Live Camera
if mode == "Live Camera":
    cap = cv2.VideoCapture(0)
    start_btn = st.sidebar.button("Start")
    stop_btn = st.sidebar.button("Stop")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
        cap.release()

    if st.session_state.get('running', False):
        ret, frame = cap.read()
        if ret:
            frame, st.session_state.closed_counter, st.session_state.skip_counter, \
            st.session_state.last_state, st.session_state.fps_start, \
            st.session_state.frame_count, alert = detect_frame(
                frame,
                st.session_state.closed_counter,
                st.session_state.skip_counter,
                st.session_state.last_state,
                st.session_state.fps_start,
                st.session_state.frame_count
            )

            frame_ph.image(frame, channels="BGR", use_column_width=True)

            if alert:
                play_alert()
                status_ph.error("DROWSINESS ALERT!")
            else:
                status_ph.success("Eyes Open")

            st.session_state.history.append((time.strftime("%H:%M:%S"), "ALERT" if alert else "OPEN"))
            if len(st.session_state.history) > 10: st.session_state.history.pop(0)
            history_ph.dataframe(pd.DataFrame(st.session_state.history, columns=["Time", "Status"]))

            st.rerun()  # تحديث مستمر
