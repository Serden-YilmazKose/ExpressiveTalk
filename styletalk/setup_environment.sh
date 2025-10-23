#!/bin/bash
set -e

echo "Installing system dependencies..."
apt-get update
apt-get install -y build-essential cmake libgl1-mesa-glx libglib2.0-0 ffmpeg espeak

pip install --upgrade pip
pip install -r requirements.txt

echo "Cloning Deep3DFaceRecon_pytorch"
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
echo "Cloning styletalk"
git clone https://github.com/FuxiVirtualHuman/styletalk.git

mv "extract_kp_videos.py" "Deep3DFaceRecon_pytorch"
mv "face_recon_videos.py" "Deep3DFaceRecon_pytorch"
mv "inference_options.py" "Deep3DFaceRecon_pytorch/options"

echo "Installing nvdiffrast"
git clone https://github.com/NVlabs/nvdiffrast temp_1
cd temp_1
pip install .
cd ..
rm -rf temp_1

echo "Installing ArcFace Torch (InsightFace)..."
git clone https://github.com/deepinsight/insightface.git temp_2
cp -r temp_2/recognition/arcface_torch Deep3DFaceRecon_pytorch/models/
rm -rf temp_2

echo "Downloading model assets..."

# Create dirs if needed
mkdir -p Deep3DFaceRecon_pytorch/checkpoints
mkdir -p Deep3DFaceRecon_pytorch/checkpoints/epoch_20
mkdir -p Deep3DFaceRecon_pytorch/BFM
mkdir -p Deep3DFaceRecon_pytorch/BFM/temp
mkdir -p styletalk/checkpoints

pip install gdown

echo "Downloading styletalk checkpoints..."
gdown 1z54FymEiyPQ0mPGrVePt8GMtDe-E2RmN \
    -O styletalk/checkpoints/styletalk_checkpoint.pth
gdown 1wFAtFQjybKI3hwRWvtcBDl4tpZzlDkja \
  -O styletalk/checkpoints/renderer_checkpoint.pt

echo "Downloading Deep3DFaceRecon checkpoint..."
gdown 1BlDBB4dLLrlN3cJhVL4nmrd_g6Jx6uP0 \
    -O Deep3DFaceRecon_pytorch/checkpoints/epoch_20/epoch_20.pth

echo "Downloading Basel Face Model (BFM09)..."
gdown 1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6 \
    -O Deep3DFaceRecon_pytorch/BFM/Exp_Pca.bin
wget --user "usr" --password "pwd" https://faces.dmi.unibas.ch/bfm/content/basel_face_model/downloads/restricted/BaselFaceModel.tgz \
    -O "Deep3DFaceRecon_pytorch/BFM/temp/BaselFaceModel.tgz"
tar -xvzf Deep3DFaceRecon_pytorch/BFM/temp/BaselFaceModel.tgz -C Deep3DFaceRecon_pytorch/BFM/temp
cp "Deep3DFaceRecon_pytorch/BFM/temp/PublicMM1/01_MorphableModel.mat" "Deep3DFaceRecon_pytorch/BFM"

echo "All necessary assets downloaded!"
