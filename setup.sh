#!/bin/bash

echo "Installing deps..."

pip install -r requirements.txt
apt -y install -qq aria2 ffmpeg git

echo "Cloning ComfyUI..."

git clone https://github.com/Isi-dev/ComfyUI

cd ComfyUI/custom_nodes
git clone https://github.com/Isi-dev/ComfyUI_GGUF.git

cd ComfyUI_GGUF
pip install -r requirements.txt
cd ../../

echo "Creating model dirs..."

mkdir -p ComfyUI/models/unet
mkdir -p ComfyUI/models/vae
mkdir -p ComfyUI/models/text_encoders

echo "Downloading MINIMAL models..."

# UNET (Q4 small)
aria2c -x 16 -s 16 -k 1M \
https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_0.gguf \
-d ComfyUI/models/unet -o wan2.1-i2v-14b-480p-Q4_0.gguf

# TEXT ENCODER
aria2c -x 16 -s 16 -k 1M \
https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
-d ComfyUI/models/text_encoders -o umt5_xxl_fp8_e4m3fn_scaled.safetensors

# VAE
aria2c -x 16 -s 16 -k 1M \
https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors \
-d ComfyUI/models/vae -o wan_2.1_vae.safetensors

echo "✅ Setup done"
