import re
import json
import ffmpeg
import whisper
import pronouncing
from g2p_en import G2p
import os, subprocess

# Configuration
PHINDEX_PATH = "styletalk/phindex.json"  # path to phindex.json
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

def extract_phonemes(audio_path, output_json="output_seq.json", phindex_path=PHINDEX_PATH):
    """
    Extract phonemes from a video 
    """
    text = transcribe_with_whisper(audio_path)
    arpabet_tokens = text_to_arpabet_tokens(text)
    phindex = load_phindex(phindex_path)
    ids = map_arpabet_sequence_to_ids(arpabet_tokens, phindex)
    # Save
    with open(output_json, "w") as f:
        json.dump(ids, f)
    print(f"Saved phoneme id sequence to {output_json}, length={len(ids)}")
    return output_json


def extract_3dmm(video_path, output_dir):
    cmd = f"""
    python Deep3DFaceRecon_pytorch/extract_kp_videos.py \
    --input_dir {video_path} \
    --output_dir {video_path}/keypoint \
    --device_ids 0,1,2,3 \
    --workers 12
    """
    os.system(cmd)

    cmd = f"""
    python Deep3DFaceRecon_pytorch/face_recon_videos.py \
    --input_dir {video_path} \
    --keypoint_dir {video_path}/keypoint \
    --output_dir {output_dir} \
    --inference_batch_size 100 \
    --name=epoch_20 \
    --epoch=20 \
    --model facerecon \
    --use_opengl false
    """
    os.system(cmd)

    return output_dir