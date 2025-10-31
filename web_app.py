import os
from pathlib import Path
import subprocess
import sys
import requests
import streamlit as st
import imageio
import imageio_ffmpeg as ffmpeg

# ------------------------------
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
video_file = st.file_uploader("Upload a Video or Image", type=["mp4", "mov", "avi", "jpeg", "png", "jpg"])
audio_file = st.file_uploader("Upload an Audio", type=["mp3", "wav", "ogg"])

# --- Style reference upload (only for 'Emotion and Style') ---
style_file = None
if mode == "Emotion and Style":
    st.subheader("🎨 Upload a Style Reference Video or Image")
    style_file = st.file_uploader("Upload Style Reference", type=["mp4", "mov", "avi", "jpeg", "png", "jpg"])

# --- Preview uploaded files ---
if video_file:
    st.subheader("Preview Input (Video/Image)")
    if video_file.type.startswith("video"):
        st.video(video_file)
    else:
        st.image(video_file)

if style_file:
    st.subheader("Preview Style Reference")
    if style_file.type.startswith("video"):
        st.video(style_file)
    else:
        st.image(style_file)

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

# --- Process button ---
if st.button("Process and Play Video"):
    if not video_file or not audio_file:
        st.error("Please upload both a video/image and an audio file before processing.")
    elif mode == "Emotion and Style" and not style_file:
        st.error("Please upload a style reference when using Emotion and Style mode.")
    else:
        # Save uploaded files
        video_path = UPLOAD_FOLDER / video_file.name
        audio_path = UPLOAD_FOLDER / audio_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        st.success(f"Saved video/image to {video_path}")
        st.success(f"Saved audio to {audio_path}")

        style_path = None
        if style_file:
            style_path = UPLOAD_FOLDER / style_file.name
            with open(style_path, "wb") as f:
                f.write(style_file.getbuffer())
            st.success(f"Saved style reference to {style_path}")

        output_file_path = VIDEO_FOLDER / "generated_video.mp4"

        # Run subprocess
        with st.spinner("🎥 Generating video, please wait..."):
            try:
                command = [
                    sys.executable,
                    str(INTEGRATION_SCRIPT),
                    "--checkpoint_path", str(CHECKPOINT_PATH),
                    "--face", str(video_path),
                    "--audio", str(audio_path),
                    "--outfile", str(output_file_path),
                    "--emotion", emotion_mapping[selected_emotion],
                    "--emotion_strength", str(intensity_value)
                ]

                # Add style reference argument only if available
                if style_path:
                    command.extend(["--style_ref", str(style_path)])

                result = subprocess.run(
                    command, capture_output=True, text=True, check=True
                )

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
