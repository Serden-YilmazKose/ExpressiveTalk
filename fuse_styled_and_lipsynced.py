import cv2
from pipeline.video import apply_emotion
from pipeline.Landmarks import face_landmarks_rgb
from pipeline.Emotion import lips_mask_from_landmarks

def fuse_frames(styled_frames, lipsynced_frames, emotion="neutral", intensity=0.6):
    """
    Align by min length, copy the mouth region from lipsynced_frames onto styled_frames,
    then apply emotion deformation (which preserves lips).
    """
    T = min(len(styled_frames), len(lipsynced_frames))
    out = []
    for i in range(T):
        base = styled_frames[i].copy()
        lip = lipsynced_frames[i]

        rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        pts = face_landmarks_rgb(rgb)
        if pts is None:
            out.append(base); continue

        mask = lips_mask_from_landmarks(base.shape, pts)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        bg = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(lip,  lip,  mask=mask)
        blended = cv2.add(bg, fg)

        emo = apply_emotion(blended, emotion, intensity)
        out.append(emo)
    return out
