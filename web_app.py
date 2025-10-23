import os
import sys
from pathlib import Path
import streamlit as st

# Vérifier si mediapipe est disponible
try:
    import mediapipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Vérifier si PyTorch est disponible
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("⚠️ PyTorch n'est pas installé. L'application ne peut pas fonctionner.")
    st.stop()

from integration_withWEB import main_video_gen

# --- Configuration ---
UPLOAD_FOLDER = "Uploaded_files"
VIDEO_FOLDER = "Output_video"
CHECKPOINT_PATH = "checkpoints/wav2lip_gan.pth"

# Créer les dossiers nécessaires
for folder in [UPLOAD_FOLDER, VIDEO_FOLDER, "temp", "checkpoints"]:
    os.makedirs(folder, exist_ok=True)

# --- Header ---
st.title("🎬 ExpressiveTalk")
st.markdown("*Synchronisation labiale avec transfert d'émotion*")

# --- Vérification des modèles ---
with st.sidebar:
    st.header("📊 État du système")
    
    # Python version
    st.text(f"🐍 Python: {sys.version.split()[0]}")
    
    # MediaPipe
    if MEDIAPIPE_AVAILABLE:
        st.success("✅ MediaPipe disponible")
    else:
        st.warning("⚠️ MediaPipe indisponible")
        st.caption("Le transfert d'émotion sera désactivé")
    
    # PyTorch
    if TORCH_AVAILABLE:
        st.success("✅ PyTorch disponible")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"Device: {device}")
    
    # Modèle Wav2Lip
    if os.path.exists(CHECKPOINT_PATH):
        st.success("✅ Modèle Wav2Lip chargé")
        file_size = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
        st.caption(f"Taille: {file_size:.1f} MB")
    else:
        st.error("❌ Modèle Wav2Lip manquant")
        st.markdown("""
        **Instructions de téléchargement:**
        1. Téléchargez le modèle depuis [GitHub](https://github.com/Rudrabha/Wav2Lip#getting-the-weights)
        2. Placez `wav2lip_gan.pth` dans le dossier `checkpoints/`
        3. Ou exécutez: `python download_models.py`
        """)

# Afficher un avertissement global si mediapipe n'est pas disponible
if not MEDIAPIPE_AVAILABLE:
    st.info("""
    ℹ️ **Mode de fonctionnement limité**
    
    Le transfert d'émotion nécessite MediaPipe qui n'est pas disponible dans cet environnement.
    Seule la synchronisation labiale (lip-sync) fonctionnera.
    
    **Pour activer toutes les fonctionnalités:**
    1. Créez un fichier `.python-version` avec `3.12`
    2. Poussez sur GitHub et redémarrez l'application
    """)

# --- Mode Selection ---
st.header("⚙️ Configuration")

if MEDIAPIPE_AVAILABLE:
    mode = st.radio(
        "Mode de traitement",
        options=["Lip-sync uniquement", "Lip-sync + Émotion"],
        help="Choisissez d'ajouter ou non le transfert d'émotion"
    )
    enable_emotion = mode == "Lip-sync + Émotion"
else:
    st.info("🎤 Mode: Lip-sync uniquement")
    enable_emotion = False

# --- File Upload ---
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

# --- Emotion Settings ---
emotion_to_use = None
intensity_value = 0.0

if enable_emotion:
    st.header("😊 Paramètres d'émotion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        emotion_options = ["Neutral", "Happy", "Sad", "Fear", "Anger", "Surprise", "Disgust"]
        selected_emotion = st.selectbox(
            "Émotion",
            emotion_options,
            help="Choisissez l'émotion à appliquer"
        )
    
    with col2:
        intensity_value = st.slider(
            "Intensité",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Force de l'émotion (0 = aucune, 1 = maximum)"
        )
    
    # Mapper les émotions
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
    
    # Afficher un aperçu
    st.caption(f"✨ Émotion sélectionnée: **{selected_emotion}** ({intensity_value:.0%})")

# --- Process Button ---
st.header("🎬 Génération")

# Vérifier que tout est prêt
can_process = True
error_messages = []

if not video_file:
    error_messages.append("📹 Veuillez charger une vidéo")
    can_process = False

if not audio_file:
    error_messages.append("🎵 Veuillez charger un fichier audio")
    can_process = False

if not os.path.exists(CHECKPOINT_PATH):
    error_messages.append("🤖 Le modèle Wav2Lip est manquant")
    can_process = False

if error_messages:
    for msg in error_messages:
        st.warning(msg)

# Bouton de traitement
if st.button("🚀 Générer la vidéo", disabled=not can_process, type="primary"):
    try:
        # Sauvegarder les fichiers uploadés
        video_path = Path(UPLOAD_FOLDER) / video_file.name
        audio_path = Path(UPLOAD_FOLDER) / audio_file.name
        
        with st.status("💾 Sauvegarde des fichiers...", expanded=True) as status:
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.write(f"✓ Vidéo sauvegardée: `{video_path.name}`")
            
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.write(f"✓ Audio sauvegardé: `{audio_path.name}`")
            
            status.update(label="✅ Fichiers sauvegardés", state="complete")
        
        # Afficher les paramètres
        with st.expander("📋 Paramètres de génération", expanded=False):
            st.json({
                "Vidéo": str(video_path),
                "Audio": str(audio_path),
                "Émotion": emotion_to_use or "Aucune",
                "Intensité": f"{intensity_value:.2f}",
                "Device": "CUDA" if torch.cuda.is_available() else "CPU"
            })
        
        # Générer la vidéo
        output_file_path = Path(VIDEO_FOLDER) / f"result_{video_file.name}"
        
        with st.spinner("🎥 Génération de la vidéo en cours... Cela peut prendre plusieurs minutes."):
            progress_bar = st.progress(0, text="Initialisation...")
            
            # Appeler la fonction de génération
            main_video_gen(
                checkpoint_path=CHECKPOINT_PATH,
                face=str(video_path),
                audio=str(audio_path),
                outfile=str(output_file_path),
                emotion=emotion_to_use,
                emotion_strength=intensity_value,
                emotion_fps=None
            )
            
            progress_bar.progress(100, text="Terminé!")
        
        st.success("✅ Vidéo générée avec succès!")
        
        # Afficher et télécharger la vidéo
        if output_file_path.exists():
            st.header("🎞️ Résultat")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.video(str(output_file_path))
            
            with col2:
                file_size = os.path.getsize(output_file_path) / (1024 * 1024)
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
        st.error(f"❌ Erreur lors de la génération: {str(e)}")
        with st.expander("🔍 Détails de l'erreur"):
            st.exception(e)

# --- Footer ---
st.markdown("---")
st.caption("ExpressiveTalk - Synchronisation labiale avec transfert d'émotion | Powered by Wav2Lip & MediaPipe")