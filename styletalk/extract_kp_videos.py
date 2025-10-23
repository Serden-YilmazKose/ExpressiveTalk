import os
import cv2
import time
import argparse
import face_alignment
import numpy as np
from PIL import Image

class KeypointExtractor():
    def __init__(self):
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D)   

    def extract_keypoint(self, images, name=None):
        if isinstance(images, list):
            keypoints = []
            for image in images:
                current_kp = self.extract_keypoint(image)
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    keypoints = self.detector.get_landmarks_from_image(np.array(images))[0]
                    break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break    
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)                    
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def run(filename, output_dir, device_id=None):
    if device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']

    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(filename)))
    target_dir = os.path.join(output_dir, parent_dir) if parent_dir else output_dir
    os.makedirs(target_dir, exist_ok=True)
    output_name = os.path.join(target_dir, os.path.basename(filename))
    kp_extractor.extract_keypoint(images, name=output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_video', type=str, required=True, help='path to the input mp4 file')
    parser.add_argument('--output_dir', type=str, required=True, help='directory for the extracted keypoints')
    parser.add_argument('--device_id', type=str, default='0', help='CUDA device to use; use "cpu" to force CPU execution')

    opt = parser.parse_args()

    input_video = os.path.abspath(opt.input_video)
    output_dir = os.path.abspath(opt.output_dir)

    if not os.path.isfile(input_video):
        raise FileNotFoundError(f'Input video not found: {input_video}')
    if not input_video.lower().endswith('.mp4'):
        raise ValueError('Input file must be an .mp4 video')

    os.makedirs(output_dir, exist_ok=True)

    device_id = None if opt.device_id.lower() == 'cpu' else opt.device_id
    run(input_video, output_dir, device_id)
