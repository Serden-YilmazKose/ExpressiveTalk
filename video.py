import cv2
import numpy as np
from landmarks import face_landmarks_rgb
from emotion import emotion_offsets, lips_mask_from_landmarks
from warp import piecewise_affine_warp

def apply_emotion(frame_bgr, emotion, intensity):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pts = face_landmarks_rgb(rgb)
    if pts is None:
        return frame_bgr
    delta = emotion_offsets(pts, emotion, intensity)
    dst_pts = pts + delta

    lips_preserve = lips_mask_from_landmarks(frame_bgr.shape, pts)
    out = piecewise_affine_warp(frame_bgr, pts, dst_pts, preserve_mask=lips_preserve)
    return out
