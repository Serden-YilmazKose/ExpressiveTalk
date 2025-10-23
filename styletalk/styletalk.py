import os, subprocess
from tools import extract_audio, extract_3dmm, extract_phonemes
from pathlib import Path

def generate_styletalk_video(style_video_path, ref_video_path, output_path="./output", audio_path=None):
    if not audio_path:
      audio_path = extract_audio(ref_video_path)

    style_video_path = os.path.abspath(style_video_path)
    ref_video_path =  os.path.abspath(ref_video_path)
    audio_path = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)
    
    phoneme_json = extract_phonemes(ref_video_path, audio_path)
    style_clip_png, style_clip_mat = extract_3dmm(style_video_path, output_path)
    ref_png, ref_mat = extract_3dmm(ref_video_path, output_path)

    phoneme_json = os.path.abspath(phoneme_json)
    style_clip_png = os.path.abspath(style_clip_png)
    style_clip_mat = os.path.abspath(style_clip_mat)
    ref_png = os.path.abspath(ref_png)
    ref_mat = os.path.abspath(ref_mat)

    cmd = [
        "python", "inference_for_demo.py",
        "--audio_path", phoneme_json,
        "--style_clip_path", style_clip_mat,
        "--pose_path", style_clip_mat,
        "--src_img_path", ref_png,
        "--wav_path", audio_path,
        "--output_path", output_path
    ]
    repo_root = Path("styletalk").resolve()

    subprocess.run(cmd, cwd=repo_root, check=True, text=True)

    cap = cv2.VideoCapture(output_path)
    frames = []
    ok, f = cap.read()
    while ok:
        frames.append(f)
        ok, f = cap.read()
    cap.release()
    return frames

if __name__ == "__main__":
    generate_styletalk_video(
      style_video_path="./Obama_clip3.mp4", 
      ref_video_path="./KristiNoem.mp4", 
      audio_path="KristiNoem.wav"
    )