"""
The design follows these principles:

1. **Modular functions pulled in from other scripts** – The piecewise affine
   warp for images is copied from ``wrap.py`` verbatim.  The emotion
   transformation logic is provided by importing the ``EmotionTransfer``
   class from ``emotion.py`` if available.  If the required packages (such
   as ``mediapipe``) are not installed then emotion transfer will be
   skipped and the user warned at runtime.

2. **Wav2Lip inference** – The original ``inference.py`` has been condensed
   into a function called ``run_wav2lip``.  It handles reading frames from
   a video, computing mel chunks from the audio, loading a Wav2Lip model
   checkpoint, running inference in batches and assembling the output video.
   The code is largely unchanged except for being wrapped in a function and
   accepting parameters explicitly rather than via a global ``argparse``.

3. **Command line interface** – At the bottom of this file there is a
   command line parser which exposes the common parameters from both
   ``emotion.py`` and ``inference.py``.  You can optionally perform
   emotion transfer on your input face video before running Wav2Lip.  If
   you do not specify an emotion then emotion transfer will be skipped.

Usage examples::

    # Lip sync a video without any emotion transfer
    python integration_script.py \
        --checkpoint_path path/to/wav2lip.pth \
        --face path/to/input_face.mp4 \
        --audio path/to/input_audio.wav \
        --outfile path/to/output.mp4

    # Apply a happy emotion at 75% strength and then lip sync
    python integration_script.py \
        --checkpoint_path path/to/wav2lip.pth \
        --face path/to/input_face.mp4 \
        --audio path/to/input_audio.wav \
        --outfile path/to/output.mp4 \
        --emotion happy --emotion_strength 0.75

Dependencies:
    * Python packages: ``numpy``, ``scipy``, ``cv2`` (OpenCV), ``librosa``,
      ``tqdm`` and ``torch`` for Wav2Lip inference.  For emotion transfer
      you additionally need ``mediapipe``.  These packages are not bundled
      with this script, so please install them in your environment.
"""
import os, sys
# Force include the lipsync folder in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lipsync")))
import argparse
import os
import subprocess
import sys
from typing import List, Tuple, Optional, Iterable

import cv2
import numpy as np
if not hasattr(np, 'complex'):
    np.complex = complex
from tqdm import tqdm

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is required for audio processing. Please install it via pip before using this script."
    )

# -----------------------------------------------------------------------------
# Audio helpers (Wav2Lip compatible)
# -----------------------------------------------------------------------------

def load_wav(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file and resample it to the target sampling rate.

    Args:
        path: Path to the input wav/ogg/mp3/etc.
        sample_rate: Desired sample rate. Wav2Lip uses 16 kHz audio.

    Returns:
        A 1‑D NumPy array containing the audio signal normalized to
        ``[-1, 1]``.
    """
    wav, sr = librosa.load(path, sr=sample_rate)
    # Normalize to [-1, 1]
    if wav.dtype != np.float32 and wav.dtype != np.float64:
        wav = wav.astype(np.float32)
    wav = wav / np.max(np.abs(wav) + 1e-8)
    return wav


def _preemphasis(wav: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """Apply a pre‑emphasis filter to the waveform.

    Pre‑emphasis can help models focus on the higher frequencies and is
    commonly used in speech processing.  The filter is defined by the
    difference equation ``y[t] = x[t] - coefficient * x[t-1]``.

    Args:
        wav: Input waveform.
        coefficient: Pre‑emphasis coefficient, typically around 0.97.

    Returns:
        Filtered waveform of the same length as ``wav``.
    """
    return np.append(wav[0], wav[1:] - coefficient * wav[:-1])


def melspectrogram(
    wav: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 800,
    hop_size: int = 200,
    win_size: int = 800,
    num_mels: int = 80,
    fmin: float = 55.0,
    fmax: float = 7600.0,
    preemphasis_coef: float = 0.97,
    ref_level_db: float = 20.0,
    min_level_db: float = -100.0,
) -> np.ndarray:
    """Compute a mel spectrogram from a waveform using parameters similar to Wav2Lip.

    The implementation follows the logic of the original Wav2Lip ``audio.py``.
    It applies a pre‑emphasis filter, computes a short‑time Fourier transform
    (STFT), converts the linear spectrogram to the mel scale and finally
    converts amplitudes to decibels.  Values are then normalized to a range
    roughly between ``[0, 1]``.

    Args:
        wav: Input audio waveform (1‑D array).
        sample_rate: Sampling rate of ``wav``.
        n_fft: Number of FFT bins.
        hop_size: Hop length between STFT frames.
        win_size: Window size for STFT.
        num_mels: Number of mel bands.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        preemphasis_coef: Coefficient for pre‑emphasis filter.
        ref_level_db: Reference level for amplitude to dB conversion.
        min_level_db: Floor for decibel values to avoid log of zero.

    Returns:
        A 2‑D NumPy array of shape ``(num_mels, T)`` where ``T`` is the number
        of time frames.
    """
    # Apply pre‑emphasis filter
    wav = _preemphasis(wav, preemphasis_coef)

    # Compute STFT
    stft_matrix = librosa.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window='hann',
        center=True,
    )
    magnitude = np.abs(stft_matrix)

    # Build mel filter bank and project
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_spectrogram = np.dot(mel_basis, magnitude**2)

    # Convert amplitude to decibels
    mel_spectrogram_db = 10.0 * np.log10(np.maximum(1e-10, mel_spectrogram))

    # Normalize
    mel_spectrogram_db = mel_spectrogram_db - ref_level_db
    mel_spectrogram_norm = np.clip((mel_spectrogram_db - min_level_db) / (-min_level_db), 0.0, 1.0)
    return mel_spectrogram_norm.astype(np.float32)


def get_mel_chunks(
    audio_path: str,
    fps: float,
    mel_step_size: int = 16,
    **mel_kwargs
) -> Tuple[List[np.ndarray], float]:
    """Split a waveform into mel spectrogram chunks aligned to video frames.

    Wav2Lip expects mel spectrograms segmented so that each chunk corresponds
    roughly to the duration of a video frame.  This helper loads the audio,
    computes a full mel spectrogram and then splits it into overlapping
    segments of length ``mel_step_size`` with a stride determined by the
    frame rate.

    Args:
        audio_path: Path to the audio file.
        fps: Frames per second of the target video.
        mel_step_size: Number of STFT frames per mel chunk (Wav2Lip uses 16).
        **mel_kwargs: Optional overrides for ``melspectrogram`` arguments.

    Returns:
        A tuple containing a list of mel chunks (each of shape ``(num_mels, mel_step_size)``)
        and the total length of the underlying mel spectrogram in frames.
    """
    wav = load_wav(audio_path, sample_rate=mel_kwargs.get('sample_rate', 16000))
    mel = melspectrogram(wav, **mel_kwargs)
    mel_idx_multiplier = 80.0 / fps
    mel_chunks: List[np.ndarray] = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    return mel_chunks, mel.shape[1]


# -----------------------------------------------------------------------------
# Image warping (from wrap.py)
# -----------------------------------------------------------------------------

def piecewise_affine_warp(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warp an image by mapping triangles from ``src_pts`` to ``dst_pts``.

    This function is a direct copy of the implementation in ``wrap.py``.  It
    subdivides the set of source points via Delaunay triangulation, computes
    local affine transforms for each triangle and applies them to the image.
    Optionally a ``preserve_mask`` can be provided to retain regions such as
    lips when morphing faces.

    Args:
        img: Input image as a NumPy array (BGR or RGB).
        src_pts: Source landmark coordinates, shape ``(N, 2)``.
        dst_pts: Destination landmark coordinates, shape ``(N, 2)``.
        preserve_mask: Optional binary mask of the same height/width as ``img``.
            Pixels with value > 0 will be blended from the original image
            instead of the warped result.

    Returns:
        A warped copy of ``img`` of the same shape.
    """
    h, w = img.shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for (x, y) in src_pts:
        subdiv.insert((float(x), float(y)))
    triangles = subdiv.getTriangleList().astype(np.float32).reshape(-1, 3, 2)

    out = np.zeros_like(img)
    accum_mask = np.zeros((h, w), np.uint8)

    for tri in triangles:
        src_tri = tri
        # find the indices of the vertices in the original point list
        idx = [np.argmin(np.linalg.norm(src_pts - v, axis=1)) for v in tri]
        dst_tri = dst_pts[idx]

        r1 = cv2.boundingRect(np.float32([src_tri]))
        r2 = cv2.boundingRect(np.float32([dst_tri]))

        src_offset = np.float32([[p[0] - r1[0], p[1] - r1[1]] for p in src_tri])
        dst_offset = np.float32([[p[0] - r2[0], p[1] - r2[1]] for p in dst_tri])

        M = cv2.getAffineTransform(src_offset, dst_offset)
        patch = img[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
        warped = cv2.warpAffine(
            patch,
            M,
            (r2[2], r2[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_offset), 255)

        roi = out[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
        roi_mask = accum_mask[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

        roi[mask > 0] = warped[mask > 0]
        roi_mask[mask > 0] = 255

    # Blend with original to preserve certain regions (e.g. lips)
    if preserve_mask is not None:
        preserve = cv2.bitwise_and(img, img, mask=preserve_mask)
        inv = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(preserve_mask))
        out = cv2.add(inv, preserve)

    return out


# -----------------------------------------------------------------------------
#  Emotion transfer
# -----------------------------------------------------------------------------

try:
    from emotion import EmotionTransfer
except Exception:
    EmotionTransfer = None  # type: ignore


def apply_emotion_to_video(
    input_path: str,
    output_path: str,
    emotion: Optional[str] = None,
    strength: float = 1.0,
    fps: Optional[int] = None,
) -> None:
    """Apply an emotion to every frame of a video using ``EmotionTransfer``.

    If ``emotion`` is ``None`` or the emotion module could not be imported,
    this function simply copies ``input_path`` to ``output_path``.  When an
    emotion is specified and ``EmotionTransfer`` is available, it will
    iterate through the frames of the video, morph the facial landmarks
    accordingly and save a new video.

    Args:
        input_path: Path to the source video.
        output_path: Where to save the processed video.
        emotion: One of ``['neutral','happy','sad','angry','surprised','fearful','disgusted']``.
        strength: Scalar multiplier for the emotion intensity (0 disables the effect).
        fps: Optionally override the frames per second of the output video.

    Raises:
        RuntimeError: If ``emotion`` is provided but the EmotionTransfer
        implementation is unavailable due to missing dependencies.
    """
    # No emotion requested; simply copy input to output
    if not emotion or strength <= 0.0:
        # Avoid unnecessary processing by hard linking or copying
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            subprocess.check_call(['ffmpeg', '-y', '-i', input_path, '-c', 'copy', output_path])
        return

    if EmotionTransfer is None:
        raise RuntimeError(
            "EmotionTransfer could not be imported. Make sure 'mediapipe' is installed if you want emotion effects."
        )

    et = EmotionTransfer()
    et.process_video(input_path, output_path, emotion, strength=strength, fps=fps)


# -----------------------------------------------------------------------------
# Wav2Lip inference (adapted from inference.py)
# -----------------------------------------------------------------------------

def run_wav2lip(
    checkpoint_path: str,
    face_path: str,
    audio_path: str,
    outfile: str,
    static: bool = False,
    fps: float = 25.0,
    pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 128,
    resize_factor: int = 1,
    crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
    box: Tuple[int, int, int, int] = (-1, -1, -1, -1),
    rotate: bool = False,
    nosmooth: bool = False,
    mel_kwargs: Optional[dict] = None,
) -> None:
    """Perform lip sync using a pre‑trained Wav2Lip model.

    This function encapsulates the bulk of ``inference.py``.  It loads the
    input video (or image), extracts face regions, aligns the mel
    spectrogram with the video frames, passes them through a Wav2Lip model
    and writes the result to disk.  It does not perform any emotion
    transfer; if you want to morph the face prior to lip syncing then call
    :func:`apply_emotion_to_video` first.

    Args:
        checkpoint_path: Path to the Wav2Lip model weights (.pth file).
        face_path: Path to the input video or image containing the face.
        audio_path: Path to the audio file (.wav recommended).
        outfile: Path where the final video with synchronized lips will be saved.
        static: If True and ``face_path`` points to an image, the same image
            will be used for all frames (useful for static avatars).
        fps: Frames per second of the input video (ignored if the video has
            embedded FPS information).
        pads: Padding around detected face boxes (top, bottom, left, right).
        face_det_batch_size: Batch size used when detecting faces.
        wav2lip_batch_size: Batch size for the Wav2Lip model.
        resize_factor: Downsampling factor for the input frames.
        crop: Crop rectangle (top, bottom, left, right).  Use -1 to auto
            infer boundaries.
        box: Use a fixed face bounding box instead of detection.
        rotate: Rotate video 90 degrees clockwise.
        nosmooth: Disable smoothing of face detections across time.
        mel_kwargs: Extra arguments for the ``melspectrogram`` computation.

    Note:
        This function depends on the ``face_detection`` and ``models``
        modules from the original Wav2Lip repository as well as PyTorch.
        If these are not available you will see an ImportError.
    """
    # Avoid circular import when these modules are missing
    try:
        import torch
        import face_detection
        from models import Wav2Lip
    except ImportError as e:
        raise ImportError(
            f"Required module for Wav2Lip inference missing: {e}.\n"
            "Please install PyTorch, face_detection and the Wav2Lip models package to use this function."
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for inference.')

    if not os.path.isfile(face_path):
        raise ValueError('--face argument must be a valid path to video/image file')

    static_input = static
    if os.path.isfile(face_path) and face_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        static_input = True

    # -------------------------------------------------------------------------
    # Helper functions scoped inside run_wav2lip to capture arguments
    # -------------------------------------------------------------------------
    def get_smoothened_boxes(boxes: List[np.ndarray], T: int) -> List[np.ndarray]:
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T :]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(images: List[np.ndarray]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, flip_input=False, device=device
        )
        batch_size = face_det_batch_size
        while True:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch = np.array(images[i : i + batch_size])
                    predictions.extend(detector.get_detections_for_batch(batch))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU. Try increasing --resize_factor')
                batch_size //= 2
                print(f'Recovering from OOM error; new batch size: {batch_size}')
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth:
            boxes = get_smoothened_boxes(boxes.tolist(), T=5)

        # ensure all coordinates are integers
        boxes = np.round(np.array(boxes)).astype(int)

        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]
        del detector
        return results

    def datagen(frames: List[np.ndarray], mels: List[np.ndarray]):
        img_batch: List[np.ndarray] = []
        mel_batch: List[np.ndarray] = []
        frame_batch: List[np.ndarray] = []
        coords_batch: List[Tuple[int, int, int, int]] = []

        if box[0] == -1:
            if not static_input:
                face_det_results = face_detect(frames)  # BGR to RGB inside the model
            else:
                face_det_results = face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = box
            face_det_results = [
                [f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames
            ]

        for i, mel in enumerate(mels):
            idx = 0 if static_input else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx]
            face = cv2.resize(face, (96, 96))
            img_batch.append(face)
            mel_batch.append(mel)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            if len(img_batch) >= wav2lip_batch_size:
                img_batch_np = np.asarray(img_batch)
                mel_batch_np = np.asarray(mel_batch)
                img_masked = img_batch_np.copy()
                img_masked[:, img_masked.shape[1] // 2 :, :] = 0
                img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
                mel_batch_np = mel_batch_np[:, :, :, None]
                yield img_batch_np, mel_batch_np, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        # flush remaining
        if len(img_batch) > 0:
            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)
            img_masked = img_batch_np.copy()
            img_masked[:, img_masked.shape[1] // 2 :, :] = 0
            img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
            mel_batch_np = mel_batch_np[:, :, :, None]
            yield img_batch_np, mel_batch_np, frame_batch, coords_batch

    # -------------------------------------------------------------------------
    # Read input frames
    # -------------------------------------------------------------------------
    full_frames: List[np.ndarray] = []
    if face_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        frame = cv2.imread(face_path)
        if frame is None:
            raise IOError(f'Could not load image {face_path}')
        full_frames = [frame]
    else:
        video_stream = cv2.VideoCapture(face_path)
        if not video_stream.isOpened():
            raise IOError(f'Cannot open face video: {face_path}')
        fps_in = video_stream.get(cv2.CAP_PROP_FPS)
        if fps_in > 0 and not static_input:
            fps = fps_in
        print('Reading video frames...')
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(
                    frame,
                    (
                        frame.shape[1] // resize_factor,
                        frame.shape[0] // resize_factor,
                    ),
                )
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    print(f'Number of frames available for inference: {len(full_frames)}')

    # -------------------------------------------------------------------------
    # Load and process audio
    # -------------------------------------------------------------------------
    if not audio_path.lower().endswith('.wav'):
        print('Extracting raw audio to a temporary .wav file...')
        temp_audio = 'temp/temp.wav'
        os.makedirs('temp', exist_ok=True)
        command = f'ffmpeg -y -i "{audio_path}" -strict -2 "{temp_audio}"'
        subprocess.call(command, shell=True)
        audio_path = temp_audio

    mel_kwargs = mel_kwargs or {}
    mel_chunks, mel_length = get_mel_chunks(audio_path, fps=fps, **mel_kwargs)
    print(f'Length of mel chunks: {len(mel_chunks)}')

    # Align frames and mel chunks
    full_frames = full_frames[: len(mel_chunks)]

    # Load model
    def load_model(path_: str):
        model = Wav2Lip()
        print(f'Loading checkpoint from: {path_}')
        if device == 'cuda':
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location=lambda storage, loc: storage)
        s = checkpoint['state_dict']
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        return model.to(device).eval()

    model = None
    writer = None
    for i, (img_batch, mel_batch, frames_batch, coords_batch) in enumerate(
        tqdm(
            datagen(full_frames.copy(), mel_chunks),
            total=int(np.ceil(float(len(mel_chunks)) / wav2lip_batch_size)),
        )
    ):
        if i == 0:
            model = load_model(checkpoint_path)
            print('Model loaded successfully')
            frame_h, frame_w = full_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                'temp/result.avi',
                cv2.VideoWriter_fourcc(*'DIVX'),
                fps,
                (frame_w, frame_h),
            )
        img_batch_t = torch.FloatTensor(img_batch.transpose(0, 3, 1, 2)).to(device)
        mel_batch_t = torch.FloatTensor(mel_batch.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            pred = model(mel_batch_t, img_batch_t)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for p, f, c in zip(pred, frames_batch, coords_batch):
            y1, y2, x1, x2 = c
            p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p_resized
            writer.write(f)
    if writer:
        writer.release()
    # Combine generated video with audio
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    command = f'ffmpeg -y -i "{audio_path}" -i temp/result.avi -strict -2 -q:v 1 "{outfile}"'
    subprocess.call(command, shell=(os.name != 'nt'))


def main():
    parser = argparse.ArgumentParser(
        description='Run emotion transfer and Wav2Lip inference from a single script.'
    )
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Wav2Lip checkpoint (.pth)')
    parser.add_argument('--face', type=str, required=True, help='Path to video/image containing the face')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file (.wav recommended)')
    parser.add_argument('--outfile', type=str, default='results/integration_result.mp4', help='Output video file')
    # Emotion options
    parser.add_argument(
        '--emotion',
        type=str,
        default=None,
        choices=['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted'],
        help='Apply this emotion to the input video before lip syncing',
    )
    parser.add_argument(
        '--emotion_strength',
        type=float,
        default=1.0,
        help='Strength of emotion (0 disables emotion transfer)'
    )
    parser.add_argument('--emotion_fps', type=int, default=None, help='Override FPS when saving emotion‑processed video')
    # Wav2Lip options (subset)
    parser.add_argument('--static', action='store_true', help='Use only first frame for inference when input is an image')
    parser.add_argument('--fps', type=float, default=25.0, help='Override FPS if input is an image')
    parser.add_argument('--nosmooth', action='store_true', help='Disable smoothing of face detections')
    parser.add_argument('--resize_factor', type=int, default=1, help='Downsample input frames by this factor')
    parser.add_argument('--crop', nargs=4, type=int, default=(0, -1, 0, -1), help='Crop (top, bottom, left, right) from frames')
    parser.add_argument('--box', nargs=4, type=int, default=(-1, -1, -1, -1), help='Fixed face bounding box (top, bottom, left, right)')
    args = parser.parse_args()

    temp_emotion_video = args.face
    # Perform emotion transfer if requested
    if args.emotion and args.emotion_strength > 0.0:
        temp_emotion_video = 'temp/emotion_processed.mp4'
        os.makedirs('temp', exist_ok=True)
        print(
            f'Applying emotion "{args.emotion}" at strength {args.emotion_strength} to {args.face}...'
        )
        apply_emotion_to_video(
            input_path=args.face,
            output_path=temp_emotion_video,
            emotion=args.emotion,
            strength=args.emotion_strength,
            fps=args.emotion_fps,
        )
        print('Emotion transfer complete.')

    # Run Wav2Lip inference on the (possibly emotion‑processed) video
    run_wav2lip(
        checkpoint_path=args.checkpoint_path,
        face_path=temp_emotion_video,
        audio_path=args.audio,
        outfile=args.outfile,
        static=args.static,
        fps=args.fps,
        resize_factor=args.resize_factor,
        crop=tuple(args.crop),
        box=tuple(args.box),
        nosmooth=args.nosmooth,
    )
    print(f'Lip‑synchronised video saved to {args.outfile}')


if __name__ == '__main__':
    main()