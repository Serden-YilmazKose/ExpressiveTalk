import os
from pathlib import Path
import subprocess
import sys
import gdown

import streamlit as st
import imageio_ffmpeg as ffmpeg

# ------------------------------
# Paths & Constants
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "Uploaded_files"
VIDEO_FOLDER = BASE_DIR / "Output_video"
TEMP_FOLDER = BASE_DIR / "temp"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "wav2lip_gan.pth"
INTEGRATION_SCRIPT = BASE_DIR / "integration_withWEB.py"

# ID correct du fichier Wav2Lip GAN sur Google Drive
GOOGLE_DRIVE_FILE_ID = "1fQtBSYEyuai9MjbOOrIMDJbLyDj92lLJ"

# ------------------------------
# Ensure folders exist
# ------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ------------------------------
# Ensure FFmpeg is available
# ------------------------------
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg.get_ffmpeg_exe()

# ------------------------------
# Download checkpoint if missing
# ------------------------------
@st.cache_resource
def download_checkpoint():
    """Télécharge le modèle Wav2Lip GAN si nécessaire"""
    if not CHECKPOINT_PATH.exists():
        with st.spinner("📥 Téléchargement du modèle Wav2Lip GAN (~320MB)..."):
            try:
                # Utiliser gdown avec l'URL complète
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, str(CHECKPOINT_PATH), quiet=False)
                
                # Vérifier que le fichier est valide
                if CHECKPOINT_PATH.exists():
                    file_size = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
                    if file_size < 100:  # Le modèle devrait faire ~320MB
                        CHECKPOINT_PATH.unlink()
                        st.error("❌ Le fichier téléchargé est trop petit. Téléchargement échoué.")
                        return False
                    st.success(f"✅ Modèle téléchargé avec succès ({file_size:.1f} MB)")
                    return True
                else:
                    st.error("❌ Échec du téléchargement")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du téléchargement: {str(e)}")
                st.info("""
                **Téléchargement manuel requis:**
                1. Téléchargez le modèle depuis: https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k
                2. Cherchez le fichier `wav2lip_gan.pth` (320 MB)
                3. Placez-le dans le dossier `checkpoints/`
                """)
                return False
    return True

# Télécharger le modèle au démarrage
model_ready = download_checkpoint()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎬 ExpressiveTalk")
st.markdown("*Synchronisation labiale avec transfert d'émotion*")

# Vérifier l'état du système
with st.sidebar:
    st.header("📊 État du système")
    st.text(f"🐍 Python: {sys.version.split()[0]}")
    
    if model_ready and CHECKPOINT_PATH.exists():
        file_size = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        st.success(f"✅ Modèle chargé ({file_size:.1f} MB)")
    else:
        st.error("❌ Modèle manquant")
        if st.button("🔄 Réessayer le téléchargement"):
            st.cache_resource.clear()
            st.rerun()

# --- Mode selection ---
st.header("⚙️ Configuration")
mode = st.radio(
    "Mode de traitement",
    ["Lip-sync uniquement", "Lip-sync + Émotion"],
    help="Choisissez d'ajouter ou non le transfert d'émotion"
)

# --- File upload ---
st.header("📤 Charger les fichiers")

col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader(
        "Vidéo du visage",
        type=["mp4", "mov", "avi"],
        help="Vidéo contenant le visage à animer"
    )
    if video_file:
        st.video(video_file)

with col2:
    audio_file = st.file_uploader(
        "Fichier audio",
        type=["mp3", "wav", "ogg"],
        help="Audio pour la synchronisation labiale"
    )
    if audio_file:
        st.audio(audio_file)

# --- Emotion settings ---
emotion_to_use = None
intensity_value = 0.0

if mode == "Lip-sync + Émotion":
    st.header("😊 Paramètres d'émotion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        emotion_options = ["Neutral", "Happy", "Sad", "Fear", "Anger", "Surprise", "Disgust"]
        selected_emotion = st.selectbox("Émotion", emotion_options)
    
    with col2:
        intensity_value = st.slider("Intensité", 0.0, 1.0, 0.6, 0.05)
    
    emotion_mapping = {
        "Neutral": "neutral",
        "Happy": "happy",
        "Sad": "sad",
        "Fear": "fearful",
        "Anger": "angry",
        "Surprise": "surprised",
        "Disgust": "disgusted"
    }
    
    if intensity_value > 0:
        emotion_to_use = emotion_mapping[selected_emotion]
    
    st.caption(f"✨ Émotion: **{selected_emotion}** ({intensity_value:.0%})")

# --- Process button ---
st.header("🎬 Génération")

# Vérifier que tout est prêt
can_process = True
error_messages = []

if not model_ready or not CHECKPOINT_PATH.exists():
    error_messages.append("🤖 Le modèle Wav2Lip n'est pas disponible")
    can_process = False

if not video_file:
    error_messages.append("📹 Veuillez charger une vidéo")
    can_process = False

if not audio_file:
    error_messages.append("🎵 Veuillez charger un fichier audio")
    can_process = False

if error_messages:
    for msg in error_messages:
        st.warning(msg)

# Bouton de traitement
if st.button("🚀 Générer la vidéo", disabled=not can_process, type="primary"):
    try:
        # Sauvegarder les fichiers uploadés
        video_path = UPLOAD_FOLDER / video_file.name
        audio_path = UPLOAD_FOLDER / audio_file.name
        
        with st.status("💾 Sauvegarde des fichiers...", expanded=True) as status:
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.write(f"✓ Vidéo: `{video_path.name}`")
            
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.write(f"✓ Audio: `{audio_path.name}`")
            
            status.update(label="✅ Fichiers sauvegardés", state="complete")
        
        # Générer la vidéo
        output_file_path = VIDEO_FOLDER / f"result_{video_file.name}"
        
        # Construire la commande
        cmd = [
            sys.executable,
            str(INTEGRATION_SCRIPT),
            "--checkpoint_path", str(CHECKPOINT_PATH),
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_file_path),
        ]
        
        # Ajouter les paramètres d'émotion si nécessaire
        if emotion_to_use:
            cmd.extend([
                "--emotion", emotion_to_use,
                "--emotion_strength", str(intensity_value)
            ])
        
        with st.spinner("🎥 Génération en cours... Cela peut prendre plusieurs minutes."):
            # Afficher les logs en temps réel
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Container pour les logs
            log_container = st.expander("📋 Logs de traitement", expanded=False)
            with log_container:
                stdout_area = st.empty()
                stderr_area = st.empty()
                
                stdout_lines = []
                stderr_lines = []
                
                # Lire les sorties
                while True:
                    # Lire stdout
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        stdout_lines.append(stdout_line.strip())
                        stdout_area.code("\n".join(stdout_lines[-20:]))  # Dernières 20 lignes
                    
                    # Lire stderr
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        stderr_lines.append(stderr_line.strip())
                        stderr_area.code("\n".join(stderr_lines[-20:]))  # Dernières 20 lignes
                    
                    # Vérifier si le processus est terminé
                    if process.poll() is not None:
                        break
                
                # Lire les dernières lignes
                remaining_stdout = process.stdout.read()
                if remaining_stdout:
                    stdout_lines.extend(remaining_stdout.strip().split('\n'))
                
                remaining_stderr = process.stderr.read()
                if remaining_stderr:
                    stderr_lines.extend(remaining_stderr.strip().split('\n'))
            
            # Vérifier le code de retour
            if process.returncode != 0:
                st.error(f"❌ Le processus a échoué avec le code: {process.returncode}")
                with st.expander("🔍 Détails de l'erreur", expanded=True):
                    st.code("\n".join(stderr_lines))
            else:
                st.success("✅ Vidéo générée avec succès!")
        
        # Afficher et télécharger la vidéo
        if output_file_path.exists():
            st.header("🎞️ Résultat")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.video(str(output_file_path))
            
            with col2:
                file_size = output_file_path.stat().st_size / (1024 * 1024)
                st.metric("Taille", f"{file_size:.1f} MB")
                
                with open(output_file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Télécharger",
                        data=f.read(),
                        file_name=output_file_path.name,
                        mime="video/mp4",
                        use_container_width=True
                    )
        else:
            st.error("⚠️ Le fichier de sortie n'a pas été créé")
            
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        with st.expander("🔍 Détails de l'erreur"):
            st.exception(e)

# --- Footer ---
st.markdown("---")
st.caption("ExpressiveTalk - Powered by Wav2Lip & MediaPipe")