"""
This script is originally from https://github.com/RenYurui/PIRender/tree/main/scripts,
but has been heavily modified by Jesper Nyman to fit our needs.
"""

import os
import cv2
import numpy as np
from PIL import Image
from scipy.io import savemat

import torch

from models import create_model
from options.inference_options import InferenceOptions
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import tensor2im, save_image

class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = filenames
        self.txt_filenames = txt_filenames
        self.lm3d_std = load_lm3d(bfm_folder) 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        txt_filename = self.txt_filenames[index]
        frames = self.read_video(filename)
        lm = np.loadtxt(txt_filename).astype(np.float32)
        try:
            lm = lm.reshape([len(frames), -1, 2])
        except ValueError as exc:
            raise ValueError(f'Keypoint data in {txt_filename} does not match frame count {len(frames)}') from exc
        out_images, out_trans_params = list(), list()
        for i in range(len(frames)):
            out_img, _, out_trans_param \
                = self.image_transform(frames[i], lm[i])
            out_images.append(out_img[None])
            out_trans_params.append(out_trans_param[None])
        return {
            'imgs': torch.cat(out_images, 0),
            'trans_param':torch.cat(out_trans_params, 0),
            'filename': filename
        }
        
    def read_video(self, filename):
        frames = list()
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

    def image_transform(self, images, lm):
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params        

def load_sample(video_path, keypoint_path, bfm_folder):
    dataset = VideoPathDataset([video_path], [keypoint_path], bfm_folder)
    return dataset[0]


def main(opt, model):
    input_video = os.path.abspath(opt.input_video)
    keypoint_path = os.path.abspath(opt.keypoint_file)
    output_dir = os.path.abspath(opt.output_dir)

    if not os.path.isfile(input_video):
        raise FileNotFoundError(f'Input video not found: {input_video}')
    if not input_video.lower().endswith('.mp4'):
        raise ValueError('Input file must be an .mp4 video')
    if not os.path.isfile(keypoint_path):
        raise FileNotFoundError(f'Keypoint file not found: {keypoint_path}')

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_video))[0]
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(input_video)))
    target_dir = os.path.join(output_dir, parent_dir) if parent_dir else output_dir
    os.makedirs(target_dir, exist_ok=True)

    sample = load_sample(input_video, keypoint_path, opt.bfm_folder)
    frames = sample['imgs']
    trans_params = sample['trans_param'].cpu().numpy()
    total_frames = frames.shape[0]
    batch_size = opt.inference_batch_size
    pred_coeffs = []
    thumbnail_saved = False

    if batch_size <= 0:
        raise ValueError('inference_batch_size must be a positive integer')
    if total_frames == 0:
        raise ValueError(f'No frames found in video: {input_video}')

    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        data_input = {
            'imgs': frames[start:end],
        }
        model.set_input(data_input)
        model.test()
        pred_coeff = {key: model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'],
            pred_coeff['exp'],
            pred_coeff['tex'],
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans'],
        ], 1)
        pred_coeffs.append(pred_coeff)

        if not thumbnail_saved and model.input_img is not None:
            original_tensor = model.input_img[0].detach()
            image_numpy = tensor2im(original_tensor)
            save_image(image_numpy, os.path.join(target_dir, f'{base_name}.png'))
            thumbnail_saved = True

    pred_coeffs = np.concatenate(pred_coeffs, 0)
    output_name = f'{base_name}.mat'
    savemat(
        os.path.join(target_dir, output_name),
        {'coeff': pred_coeffs, 'transform_params': trans_params}
    )

if __name__ == '__main__':
    opt = InferenceOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    gpu_ids = [gid.strip() for gid in opt.gpu_ids.split(',') if gid.strip()]
    use_cpu = len(gpu_ids) == 0 or all(gid == '-1' for gid in gpu_ids)
    device = torch.device('cpu') if use_cpu else torch.device('cuda:0')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model.device = device
    model.parallelize()
    model.eval()

    main(opt, model)


