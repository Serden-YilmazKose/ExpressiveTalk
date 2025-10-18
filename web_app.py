import streamlit as st
import os
from pathlib import Path
from generate_video import generate_video  # MoviePy-based video generator

# --- Configuration ---
UPLOAD_FOLDER = "Uploaded_files"
VIDEO_FOLDER = "Output_video"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

st.title("🎬 ExpressiveTalk")

# --- File Upload Section ---
st.header("Upload Files")
video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
audio_file = st.file_uploader("Upload an Audio", type=["mp3", "wav", "ogg"])

# --- Preview Uploaded Files ---
if video_file:
    st.subheader("Preview Video")
    st.video(video_file)

if audio_file:
    st.subheader("Preview Audio")
    st.audio(audio_file)

# --- Dropdown Section ---
st.header("Select an Emotion")
options = ["Neutral", "Happy", "Sad", "Fear", "Anger", "Surprise", "Disgust"]
selected_option = st.selectbox("Choose an emotion", options)

# --- Emotion Intensity Slider ---
st.subheader("Adjust Emotion Intensity")
intensity_value = st.slider(
    "Select Intensity Level",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Set how intense the selected emotion should be (0 = none, 1 = maximum)."
)
st.write(f"Selected intensity: **{intensity_value:.2f}**")

# --- Process Button ---
if st.button("Process and Play Video"):
    # ✅ Error Checking
    if not video_file and not audio_file:
        st.error("Please upload both a video and an audio file before processing.")
    elif not video_file:
        st.error("Please upload a video file before processing.")
    elif not audio_file:
        st.error("Please upload an audio file before processing.")
    else:
        # ✅ Save uploaded video
        video_path = Path(UPLOAD_FOLDER) / video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.success(f"Video saved to {video_path}")

        # ✅ Save uploaded audio
        audio_path = Path(UPLOAD_FOLDER) / audio_file.name
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"Audio saved to {audio_path}")

        # Show selected emotion and intensity
        st.write(f"Selected emotion: **{selected_option}**")
        st.write(f"Emotion intensity: **{intensity_value:.2f}**")

        # --- Generate video dynamically ---
        output_file_path = Path(VIDEO_FOLDER) / "generated_video.mp4"
        with st.spinner("🎥 Generating video, please wait..."):
            generate_video(str(output_file_path))  # Replace with your actual video generation function

        # --- Play the generated video ---
        if output_file_path.exists():
            st.subheader("🎞️ Playing Generated Video")
            st.video(str(output_file_path))
        else:
            st.error("⚠️ Video generation failed.")
