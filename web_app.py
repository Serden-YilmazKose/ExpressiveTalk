import os
from pathlib import Path
import subprocess
import sys

import streamlit as st

# --- Configuration ---
UPLOAD_FOLDER = "Uploaded_files"
VIDEO_FOLDER = "Output_video"

BASE_DIR = Path(__file__).resolve().parent

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs("temp", exist_ok=True)  # Temporary folder

st.title("🎬 ExpressiveTalk")

# --- Mode Toggle Section ---
st.header("Mode Selection")
mode = st.radio(
    "Choose Processing Mode",
    options=["Emotion", "Emotion and Style"],
    help="Select 'Emotion' to adjust emotional tone only, or 'Emotion and Style' to modify both emotion and the visual style of the output video.",
)

if mode == "Emotion and Style":
    st.info(
        "🖌️ **Style Mode Explanation:** In this mode, both the emotional tone and the visual style (such as color, lighting, or artistic filters) are adjusted to better reflect the selected emotion."
    )

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

# Mapping emotions to integration_withWEB
emotion_mapping = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Sad": "sad",
    "Fear": "fearful",
    "Anger": "angry",
    "Surprise": "surprised",
    "Disgust": "disgusted"
}

# --- Emotion Intensity Slider ---
st.subheader("Adjust Emotion Intensity")
intensity_value = st.slider(
    "Select Intensity Level",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Set how intense the selected emotion should be (0 = none, 1 = maximum).",
)
st.write(f"Selected intensity: **{intensity_value:.2f}**")

# --- Process Button ---
if st.button("Process and Play Video"):
    # Error Checking
    if not video_file and not audio_file:
        st.error("Please upload both a video and an audio file before processing.")
    elif not video_file:
        st.error("Please upload a video file before processing.")
    elif not audio_file:
        st.error("Please upload an audio file before processing.")
    else:
        # Save uploaded files
        video_path = Path(UPLOAD_FOLDER) / video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.success(f"Video saved to {video_path}")

        audio_path = Path(UPLOAD_FOLDER) / audio_file.name
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"Audio saved to {audio_path}")

        # Show selections
        st.write(f"Selected emotion: **{selected_option}**")
        st.write(f"Emotion intensity: **{intensity_value:.2f}**")
        st.write(f"Mode selected: **{mode}**")

        # --- Generate video ---
        output_file_path = Path(VIDEO_FOLDER) / "generated_video.mp4"

        with st.spinner("🎥 Generating video, please wait..."):
            try:
                integration_script = BASE_DIR / "integration_withWEB.py"

                # Determine if emotion should be applied
                try:
                    import mediapipe  # check if mediapipe is available
                    apply_emotion = True
                except ImportError:
                    apply_emotion = False
                    st.warning("⚠️ Mediapipe not found. Skipping emotion transfer (lip-sync only).")

                # Pass arguments
                if apply_emotion:
                    emotion_arg = emotion_mapping[selected_option]
                    emotion_strength_arg = str(intensity_value)
                else:
                    emotion_arg = "None"
                    emotion_strength_arg = "0"

                subprocess.run([
                    sys.executable,
                    str(integration_script),
                    "--checkpoint_path", str(BASE_DIR / "checkpoints/wav2lip_gan.pth"),
                    "--face", str(video_path),
                    "--audio", str(audio_path),
                    "--outfile", str(output_file_path),
                    "--emotion", emotion_arg,
                    "--emotion_strength", emotion_strength_arg
                ], check=True)

                st.success("✅ Video generation completed!")

            except subprocess.CalledProcessError as e:
                st.error("❌ Error during video generation.")
                st.text(f"STDOUT:\n{e.stdout}")
                st.text(f"STDERR:\n{e.stderr}")
                st.exception(e)

        # --- Play and Download Video ---
        if output_file_path.exists():
            st.subheader("🎞️ Playing Generated Video")
            st.video(str(output_file_path))

            with open(output_file_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="⬇️ Download Generated Video",
                data=video_bytes,
                file_name="generated_video.mp4",
                mime="video/mp4",
            )
        else:
            st.error("⚠️ Video generation failed.")
