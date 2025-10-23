import re
import json
import ffmpeg
import whisper
import pronouncing
from g2p_en import G2p
import os, subprocess
from bournemouth_aligner import PhonemeTimestampAligner
from pathlib import Path

PHINDEX_PATH = "styletalk/phindex.json"
IPA_TO_ARPABET_PATH = "./ipa_to_arpabet.json"
WHISPER_MODEL = "base"

def extract_audio(video_path, output_wav="audio.wav"):
    """
    Uses ffmpeg to extract a .wav file from an input video
    """
    (
        ffmpeg
        .input(video_path)
        .output(output_wav, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_wav


def transcribe_with_whisper(audio_path):
    """
    Transcribes audio using the OpenAI whisper model
    """
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path)
    return result["text"]


def tokenize_words(text):
    """
    Normalizes the words and returns a list of them in lowercase
    """
    _word_re = re.compile(r"[A-Za-z']+")
    return _word_re.findall(text.lower())

def word_to_arpabet(word):
    """
    Try CMUdict via pronouncing; fallback to g2p_en
    Returns a list of ARPAbet phonemes
    """
    _g2p = G2p()
    # Try CMUdict via pronouncing
    phones = pronouncing.phones_for_word(word)
    if phones:
        # choose first candidate
        phone_str = phones[0]
        tokens = phone_str.split()
    else:
        # fallback
        g2p_out = _g2p(word) 
        tokens = [t.upper() for t in g2p_out if re.fullmatch(r"[A-Za-z0-9]+", t)]
    
    # strip any numeric stress markers (e.g. AH0 -> AH)
    stripped = [re.sub(r'\d', '', t).upper() for t in tokens]
    return stripped

def load_phindex(phindex_path):
    """
    Map a sequence of ARPAbet tokens -> numeric indices via phindex.json (available in the styletalk repo)
    """
    with open(phindex_path, "r") as f:
        phindex = json.load(f)
    phindex_up = {k.upper(): int(v) for k, v in phindex.items()}
    return phindex_up

def map_arpabet_sequence_to_ids(arpabet_seq, phindex):
    """
    Maps the arpabet phoneme sequence to IDs from the phindex dictionary
    """
    ids = []
    for p in arpabet_seq:
        if p in phindex:
            ids.append(phindex[p])
        else:
            ids.append(phindex.get('SIL', 0))
    return ids

def arpabet_to_id(arpabet, phindex):
    """
    Map a single ARPAbet token to its numeric index via phindex.json
    """
    return phindex.get(arpabet, phindex.get('SIL', 0))

def text_to_arpabet_tokens(text):
    """
    Use above helper functions to convert text to arpabet tokens
    """
    words = tokenize_words(text)
    arpabet_list = []
    for w in words:
        arp = word_to_arpabet(w)
        arpabet_list.extend(arp)
    print(arpabet_list)
    return arpabet_list

def extract_phoneme_timestamps(audio_path, output_json="output_timestamps.json"):
    """
    Based on an example from the Bournemouth Aligner repo: https://github.com/tabahi/bournemouth-forced-aligner
    """
    extractor = PhonemeTimestampAligner(
        preset="en-us",
        duration_max=30,
        device='cpu'
    )

    text = transcribe_with_whisper(audio_path)

    # Load and process
    audio_wav = extractor.load_audio(audio_path)

    timestamps = extractor.process_sentence(
        text,
        audio_wav,
        ts_out_path=None,
        extract_embeddings=False,
        vspt_path=None,
        do_groups=True,
        debug=True,
    )

    with open(output_json, "w") as f:
        json.dump(timestamps, f)
    
    return output_json

def align_phonemes_to_frames(timestamps_json, fps, phindex_json=PHINDEX_PATH, ipa_to_arpabet_json=IPA_TO_ARPABET_PATH, output_json="aligned_frames.json"):
    """
    Returns a list of phoneme ids with the same length as the number of video frames,
    where each entry corresponds to the phoneme id active at that frame.
    """
    ts = json.load(open(timestamps_json, "r"))
    phindex = {k.upper(): int(v) for k, v in json.load(open(phindex_json, "r")).items()}
    sil_id = phindex.get("SIL", None)

    items = []
    for segment in ts["segments"]:
        for ph in segment["phoneme_ts"]:
            items.append(ph)

    print(items)

    frame_duration_ms = 1000.0 / fps
    aligned_ids = []
    current_time_ms = 0.0

    for item in items:
        ipa_char = item["phoneme_label"]
        start_ms = item["start_ms"]
        end_ms = item["end_ms"]

        arpabet_char = ipa_to_arpabet(ipa_char, ipa_to_arpabet_json)
        phoneme_id = phindex.get(arpabet_char, sil_id)

        gap = start_ms - current_time_ms
        if gap > 0:
            num_sil_frames = max(0, int(round(gap * fps / 1000.0)))
            aligned_ids.extend([sil_id] * num_sil_frames)
            current_time_ms += num_sil_frames * frame_duration_ms

        duration = end_ms - start_ms
        num_frames = max(0, int(round(duration * fps / 1000.0)))
        if num_frames == 0:
            num_frames = 1
        aligned_ids.extend([phoneme_id] * num_frames)
        current_time_ms += num_frames * frame_duration_ms

    with open(output_json, "w") as f:
        json.dump(aligned_ids, f)
    return output_json

def ipa_to_arpabet(ipa_char, ipa_to_arpabet_json=IPA_TO_ARPABET_PATH):
    """
    Converts an IPA character to its corresponding ARPAbet representation
    using a provided mapping JSON file.
    """
    with open(ipa_to_arpabet_json, "r") as f:
        ipa_to_arpabet_map = json.load(f)
    
    arpabet_char = ipa_to_arpabet_map.get(ipa_char, None)
    if arpabet_char is None:
        # Insert SIL for unknown characters
        arpabet_char = "SIL"

    return arpabet_char

def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of a video using ffprobe
    """
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    if not video_streams:
        raise ValueError("No video stream found")
    fps_str = video_streams[0]['r_frame_rate']
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom
    return fps

def extract_phonemes(video_path, audio_path, output_json="output_seq.json", phindex_path=PHINDEX_PATH, ipa_to_arpabet_path=IPA_TO_ARPABET_PATH):
    """
    Extract phonemes from a video 
    """
    timestamps_json = extract_phoneme_timestamps(audio_path, output_json="temp_timestamps.json")
    fps = get_video_fps(video_path)
    output_json = align_phonemes_to_frames(timestamps_json, fps, phindex_json=phindex_path, ipa_to_arpabet_json=ipa_to_arpabet_path, output_json=output_json)
    return output_json

def extract_3dmm(video_file, output_dir):
    print(output_dir)
    repo_root = Path("Deep3DFaceRecon_pytorch").resolve()
    cmd = [
        "python",
        "extract_kp_videos.py",
        "--input_video", video_file,
        "--output_dir", output_dir,
        "--device_id", "0"
    ]
    print("running extract_kp_videos")
    subprocess.run(cmd, cwd=repo_root, capture_output=True)

    video_filename = os.path.splitext(os.path.basename(video_file))[0]
    cmd = [
        "python",
        "face_recon_videos.py",
        "--input_video", video_file,
        "--keypoint_file", f"{output_dir}/{video_filename}.txt",
        "--output_dir", f"{output_dir}/final",
        "--inference_batch_size", "100",
        "--name", "epoch_20",
        "--epoch", "20",
        "--model", "facerecon",
        "--use_opengl", "false"
    ]
    print("running face_recon_videos")
    subprocess.run(cmd, cwd=repo_root, capture_output=True)

    out_png_path = os.path.abspath(os.path.join(output_dir, "final", f"{video_filename}.png"))
    out_mat_path = os.path.abspath(os.path.join(output_dir, "final", f"{video_filename}.mat"))

    return out_png_path, out_mat_path