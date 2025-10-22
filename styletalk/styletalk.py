import os
from tools import extract_audio, extract_3dmm, extract_phonemes

def generate_styletalk_video(style_video_path, pose_path, reference_image_path, output_path, audio_path=None):
    if not audio_path:
        audio_path = extract_audio(style_video_path)
    phoneme_json = extract_phonemes(audio_path)
    style_clip_mat = extract_3dmm(style_video_path, output_path)
    pose_mat = extract_3dmm(pose_path, output_path)

    cmd = f"""
    python ./styletalk/inference_for_demo.py \
    --audio_path {phoneme_json} \
    --style_clip_path {style_clip_mat} \
    --pose_path {pose_mat} \
    --src_img_path {reference_image_path} \
    --wav_path {audio_path} \
    --output_path {output_path}
    """

    os.system(cmd)
    cap = cv2.VideoCapture(out_path)
    frames = []
    ok, f = cap.read()
    while ok
        frames.append(f)
        ok, f = cap.read()
    cap.release()
    return frames
    #return output_path

if __name__ == "__main__":
    generate_styletalk_video(style_video_path="./ref_img", pose_path="./", reference_image_path="./styletalk/samples/source_video/image/andrew_clip_1.png", output_path="./", audio_path="Obama_clip1.wav")
