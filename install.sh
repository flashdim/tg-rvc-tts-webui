#!/bin/bash
python3.10 -m venv venv
. ./venv/bin/activate

# Install PyTorch manually if you want to use NVIDIA GPU (Windows)
# See https://pytorch.org/get-started/locally/ for more details
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

pip install -r ./requirements.txt

# Download models in root directory
curl -L -O "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
curl -L -O "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

