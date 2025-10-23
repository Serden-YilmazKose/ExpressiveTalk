import os
from pathlib import Path
import subprocess
import sys
import requests

import streamlit as st
import imageio
import imageio_ffmpeg as ffmpeg

# Paths & Constants
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "Uploaded_files"
VIDEO_FOLDER = BASE_DIR / "Output_video"
TEMP_FOLDER = BASE_DIR / "temp"
CHECKPOINT_PATH = BASE_DIR / "lipsync/checkpoints/wav2lip_gan.pth"
INTEGRATION_SCRIPT = BASE_DIR / "integration_withWEB.py"

# ------------------------------
# Ensure folders exist
# ------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ------------------------------
# Ensure FFmpeg is available
# ------------------------------
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg.get_ffmpeg_exe()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎬 ExpressiveTalk")

# --- Mode selection ---
st.header("Mode Selection")
mode = st.radio(
    "Choose Processing Mode",
    ["Emotion", "Emotion and Style"],
    help="Select 'Emotion' to adjust emotional tone only, or 'Emotion and Style' to modify both emotion and visual style."
)
if mode == "Emotion and Style":
    st.info("🖌️ In this mode, both emotion and visual style (color, lighting, artistic filters) are adjusted.")

# --- File upload ---
st.header("Upload Files")
video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "jpeg", "pnj", "jpg"])
audio_file = st.file_uploader("Upload an Audio", type=["mp3", "wav", "ogg"])

if video_file:
    st.subheader("Preview Video")
    st.video(video_file)
if audio_file:
    st.subheader("Preview Audio")
    st.audio(audio_file)

# --- Emotion selection ---
st.header("Select an Emotion")
emotion_options = ["Neutral", "Happy", "Sad", "Fear", "Anger", "Surprise", "Disgust"]
selected_emotion = st.selectbox("Choose an emotion", emotion_options)

emotion_mapping = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Sad": "sad",
    "Fear": "fearful",
    "Anger": "angry",
    "Surprise": "surprised",
    "Disgust": "disgusted"
}

# --- Emotion intensity ---
st.subheader("Adjust Emotion Intensity")
intensity_value = st.slider("Select Intensity Level", 0.0, 1.0, 0.5, 0.01)
st.write(f"Selected intensity: **{intensity_value:.2f}**")

# --- Process button ---
if st.button("Process and Play Video"):
    if not video_file or not audio_file:
        st.error("Please upload both video and audio files before processing.")
    else:
        # Save uploaded files
        video_path = UPLOAD_FOLDER / video_file.name
        audio_path = UPLOAD_FOLDER / audio_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        st.success(f"Saved video to {video_path}")
        st.success(f"Saved audio to {audio_path}")

        output_file_path = VIDEO_FOLDER / "generated_video.mp4"

        with st.spinner("🎥 Generating video, please wait..."):
            try:
                result = subprocess.run([
                    sys.executable,
                    str(INTEGRATION_SCRIPT),
                    "--checkpoint_path", str(CHECKPOINT_PATH),
                    "--face", str(video_path),
                    "--audio", str(audio_path),
                    "--outfile", str(output_file_path),
                    "--emotion", emotion_mapping[selected_emotion],
                    "--emotion_strength", str(intensity_value)
                ], capture_output=True, text=True, check=True)

                st.success("✅ Video generation completed!")
                st.text(result.stdout or "No output captured from script.")

            except subprocess.CalledProcessError as e:
                st.error("❌ Integration script failed:")
                st.text("STDOUT:\n" + (e.stdout or "No stdout") + "\n\nSTDERR:\n" + (e.stderr or "No stderr"))
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {str(e)}")

        # Play and allow download
        if output_file_path.exists():
            st.subheader("🎞️ Generated Video")
            st.video(str(output_file_path))
            with open(output_file_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f.read(),
                    file_name="generated_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("⚠️ Video generation failed.")
