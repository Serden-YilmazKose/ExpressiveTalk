# emotion_transfer.py
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
from typing import Optional, Iterable, List
import matplotlib.pyplot as plt
import argparse
import os
import sys

class EmotionTransfer:
    def __init__(self, detection_confidence: float = 0.9):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence
        )
        self.emotion_profiles = {
            'happy': self._happy_profile,
            'sad': self._sad_profile,
            'angry': self._angry_profile,
            'surprised': self._surprised_profile,
            'fearful': self._fearful_profile,
            'disgusted': self._disgusted_profile,
            'neutral': self._neutral_profile
        }
        self.face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        self.brow_l = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70]
        self.brow_r = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 300]
        self.eye_l_u = [159, 158, 157, 173, 133, 246, 130]
        self.eye_l_d = [145, 144, 153, 154, 155, 143]
        self.eye_r_u = [386, 387, 388, 466, 263, 249, 359]
        self.eye_r_d = [374, 380, 381, 382, 373, 362]
        self.nose = [1, 2, 98, 327, 195, 5, 4, 94, 331, 64, 294, 168]
        self.mouth_o = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
        self.mouth_i = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]
        self.jaw = [172, 136, 150, 176, 149, 148, 152, 377, 400, 379, 378, 183]
        self.cheek_l = [205, 147, 123, 50]
        self.cheek_r = [425, 352, 450, 351]
        self.control_idx = sorted(set(self.face_oval + self.brow_l + self.brow_r + self.eye_l_u + self.eye_l_d + self.eye_r_u + self.eye_r_d + self.nose + self.mouth_o + self.mouth_i + self.jaw + self.cheek_l + self.cheek_r))
        self.last_landmarks = None

    def __del__(self):
        # make sure mediapipe resources are cleaned
        try:
            self.face_mesh.close()
        except Exception:
            pass

    # ---------- public API ----------
    def transfer_emotion(self, image: np.ndarray, target_emotion: str, strength: float = 1.0) -> np.ndarray:
        if target_emotion not in self.emotion_profiles or strength <= 0:
            return image.copy()
        lm = self._get_landmarks(image)
        if lm is None:
            if self.last_landmarks is not None:
                print("No face detected. Using last known landmarks.")
                lm = self.last_landmarks
            else:
                print("No face detected. Skipping frame.")
                return image.copy()
        else:
            self.last_landmarks = lm.copy()
        disp = self.emotion_profiles[target_emotion](lm, strength)
        dst = lm + disp * strength
        return self._warp_piecewise(image, lm[self.control_idx], dst[self.control_idx])

    def process_video(self, input_path: str, output_path: str, target_emotion: str,
                      strength: float = 1.0, fps: Optional[int] = None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        in_fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps if fps else in_fps, (w, h))
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]
            processed_frame = self.transfer_emotion(frame, target_emotion, strength)
            out.write(processed_frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_path}")

    def process_image(self, input_path: str, output_path: str, target_emotion: str, strength: float = 1.0):
        img = cv2.imread(input_path)
        if img is None:
            raise IOError(f"Cannot open image: {input_path}")
        out = self.transfer_emotion(img, target_emotion, strength)
        cv2.imwrite(output_path, out)
        print(f"Saved: {output_path}")

    def visualize_all_emotions(self, image: np.ndarray, strength: float = 0.85):
        emotions = ['neutral', 'happy', 'sad', 'fearful', 'angry', 'disgusted', 'surprised']
        fig, axes = plt.subplots(1, len(emotions), figsize=(22, 4.5))
        for i, emotion in enumerate(emotions):
            result = self.transfer_emotion(image, emotion, strength)
            axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[i].set_title(emotion.capitalize())
            axes[i].axis('off')
        plt.tight_layout()
        return fig

    # ---------- internals ----------
    def _get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        if w > 1280:
            scale = 1280 / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
            h, w = image.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        landmarks = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)
        return landmarks

    def _s(self, lm: np.ndarray, g: float) -> float:
        return float(np.linalg.norm(lm[33] - lm[263])) * g

    def _neutral_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        return np.zeros_like(lm)

    def _happy_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        d = np.zeros_like(lm); k = self._s(lm, 0.016) * s
        d[61, 1] -= 4.2 * k; d[291, 1] -= 4.2 * k
        d[61, 0] -= 0.5 * k; d[291, 0] += 0.5 * k
        for i in [62, 185, 40, 37, 0, 267, 269, 292]: d[i, 1] -= 2.0 * k
        for i in [146, 91, 181, 84, 17, 314, 405, 321, 375]: d[i, 1] -= 2.6 * k
        for i in [116, 123, 345, 352]: d[i, 1] -= 1.6 * k
        for i in self.eye_l_d + self.eye_r_d: d[i, 1] -= 0.8 * k
        for i in self.cheek_l + self.cheek_r: d[i, 1] -= 1.4 * k
        return d

    def _sad_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        d = np.zeros_like(lm); k = self._s(lm, 0.015) * s
        for i in [70, 63, 105, 336, 296, 334]: d[i, 1] -= 2.6 * k
        d[70, 0] += 0.8 * k; d[300, 0] -= 0.8 * k
        for i in [46, 53, 276, 283]: d[i, 1] += 1.1 * k
        d[61, 1] += 3.2 * k; d[291, 1] += 3.2 * k
        for i in [62, 185, 40, 37, 0, 267, 269, 292]: d[i, 1] += 1.6 * k
        for i in [146, 91, 181, 84, 17, 314, 405, 321]: d[i, 1] += 1.2 * k
        return d

    def _angry_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        d = np.zeros_like(lm); k = self._s(lm, 0.015) * s
        for i in self.brow_l: d[i, 1] += 2.6 * k; d[i, 0] += 1.4 * k
        for i in self.brow_r: d[i, 1] += 2.4 * k; d[i, 0] -= 1.4 * k
        for i in self.eye_l_u + self.eye_r_u: d[i, 1] += 1.4 * k
        for i in self.eye_l_d + self.eye_r_d: d[i, 1] -= 1.2 * k
        for i in [61, 185, 40, 0, 267, 269, 291]: d[i, 1] += 0.8 * k
        for i in [146, 91, 84, 17, 314, 321]: d[i, 1] -= 1.2 * k
        d[64, 0] -= 0.9 * k; d[294, 0] += 0.9 * k
        return d

    def _surprised_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        d = np.zeros_like(lm); k = self._s(lm, 0.017) * s
        for i in self.brow_l + self.brow_r: d[i, 1] -= 4.2 * k
        for i in self.eye_l_u + self.eye_r_u: d[i, 1] -= 2.8 * k
        for i in self.eye_l_d + self.eye_r_d: d[i, 1] += 2.0 * k
        for i in [61, 185, 40, 37, 0, 267, 269, 291]: d[i, 1] -= 1.2 * k
        for i in [146, 91, 181, 84, 17, 314, 405, 321, 375]: d[i, 1] += 3.8 * k
        for i in self.jaw: d[i, 1] += 2.0 * k
        return d

    def _fearful_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        d = np.zeros_like(lm); k = self._s(lm, 0.016) * s
        for i in [70, 63, 105, 336, 296, 334]: d[i, 1] -= 3.6 * k
        d[70, 0] += 1.5 * k; d[300, 0] -= 1.5 * k
        for i in [46, 53, 276, 283]: d[i, 1] -= 2.0 * k
        for i in self.eye_l_u + self.eye_r_u: d[i, 1] -= 2.4 * k
        for i in self.eye_l_d + self.eye_r_d: d[i, 1] += 1.6 * k
        for i in [146, 91, 181, 84, 17, 314, 405, 321]: d[i, 1] += 1.4 * k
        d[61, 0] += 1.2 * k; d[291, 0] -= 1.2 * k
        return d

    def _disgusted_profile(self, lm: np.ndarray, s: float) -> np.ndarray:
        d = np.zeros_like(lm); k = self._s(lm, 0.015) * s
        for i in [61, 62, 185, 40, 267, 269, 291, 292]: d[i, 1] -= 2.8 * k
        d[61, 1] -= 1.0 * k
        for i in [1, 2, 98, 327]: d[i, 1] -= 1.4 * k
        d[64, 0] -= 1.2 * k; d[294, 0] += 1.2 * k
        for i in self.eye_l_d + self.eye_r_d: d[i, 1] -= 1.6 * k
        for i in self.cheek_l + self.cheek_r: d[i, 1] += 1.0 * k
        for i in [70, 63, 105, 300, 293, 334]: d[i, 1] += 1.3 * k
        d[70, 0] += 0.8 * k; d[300, 0] -= 0.8 * k
        return d

    def _warp_piecewise(self, img: np.ndarray, src_ctl: np.ndarray, dst_ctl: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        tri = Delaunay(src_ctl)
        out = img.copy()
        for tri_idx in tri.simplices:
            t1 = src_ctl[tri_idx].astype(np.float32)
            t2 = dst_ctl[tri_idx].astype(np.float32)
            r1 = cv2.boundingRect(t1)
            r2 = cv2.boundingRect(t2)
            t1r = t1 - np.array([r1[0], r1[1]], np.float32)
            t2r = t2 - np.array([r2[0], r2[1]], np.float32)
            M = cv2.getAffineTransform(t1r, t2r)
            patch = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
            warped = cv2.warpAffine(patch, M, (r2[2], r2[3]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
            mask = np.zeros((r2[3], r2[2]), np.uint8)
            cv2.fillConvexPoly(mask, np.int32(t2r), 255)
            roi = out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
            m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 2.0)[..., None]
            roi[:] = roi * (1 - m) + warped * m
            out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = roi

        hull_dst = dst_ctl[np.unique(Delaunay(dst_ctl).convex_hull.flatten())].astype(np.int32)
        face_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(hull_dst), 255)
        face_mask = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (61, 61), 18)[..., None]
        return (out * face_mask + img * (1 - face_mask)).astype(np.uint8)

# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Emotion transfer on image/video using MediaPipe landmarks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # image
    p_img = sub.add_parser("image", help="Process a single image")
    p_img.add_argument("--in", dest="inp", required=True, help="Input image path")
    p_img.add_argument("--out", dest="out", required=True, help="Output image path")
    p_img.add_argument("--emotion", required=True, choices=["neutral","happy","sad","angry","surprised","fearful","disgusted"])
    p_img.add_argument("--strength", type=float, default=1.0)

    # video
    p_vid = sub.add_parser("video", help="Process a video")
    p_vid.add_argument("--in", dest="inp", required=True, help="Input video path")
    p_vid.add_argument("--out", dest="out", required=True, help="Output video path (.mp4)")
    p_vid.add_argument("--emotion", required=True, choices=["neutral","happy","sad","angry","surprised","fearful","disgusted"])
    p_vid.add_argument("--strength", type=float, default=1.0)
    p_vid.add_argument("--fps", type=int, default=None)

    return p

def main():
    args = build_arg_parser().parse_args()
    et = EmotionTransfer()

    if args.cmd == "image":
        et.process_image(args.inp, args.out, args.emotion, args.strength)
    elif args.cmd == "video":
        et.process_video(args.inp, args.out, args.emotion, args.strength, fps=args.fps)
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
