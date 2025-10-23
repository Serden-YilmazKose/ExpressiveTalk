"""
Script pour télécharger les modèles nécessaires
"""
import os
import urllib.request
import gdown
from pathlib import Path

def download_file(url, output_path):
    """Télécharge un fichier depuis une URL"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Downloaded {output_path}")

def download_wav2lip_models():
    """Télécharge les modèles Wav2Lip"""
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Wav2Lip GAN model (meilleure qualité)
    wav2lip_gan_url = "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?download=1"
    wav2lip_gan_path = checkpoints_dir / "wav2lip_gan.pth"
    
    if not wav2lip_gan_path.exists():
        print("Downloading Wav2Lip GAN model...")
        try:
            gdown.download(
                "https://drive.google.com/uc?id=1fQtBSYEyuai9MjbOOrIMDJbLyDj92lLJ",
                str(wav2lip_gan_path),
                quiet=False
            )
            print("✓ Wav2Lip GAN model downloaded")
        except Exception as e:
            print(f"✗ Error downloading Wav2Lip GAN: {e}")
            print("Please download manually from:")
            print("https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
    else:
        print(f"✓ Wav2Lip model already exists at {wav2lip_gan_path}")
    
    # Face detection model (s3fd)
    face_det_dir = Path("lipsync/face_detection/detection/sfd")
    face_det_dir.mkdir(parents=True, exist_ok=True)
    face_det_path = face_det_dir / "s3fd.pth"
    
    if not face_det_path.exists():
        print("Downloading face detection model...")
        try:
            gdown.download(
                "https://drive.google.com/uc?id=1NW6EORK8yrEYSIy-wFzFfzlXb5F8g0P7",
                str(face_det_path),
                quiet=False
            )
            print("✓ Face detection model downloaded")
        except Exception as e:
            print(f"✗ Error downloading face detection: {e}")
    else:
        print(f"✓ Face detection model already exists")

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading required models for ExpressiveTalk")
    print("=" * 60)
    download_wav2lip_models()
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)