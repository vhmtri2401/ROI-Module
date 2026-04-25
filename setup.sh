#!/bin/bash
# ==============================================================
# Setup script cho ROI_module trên máy mới
# ==============================================================
set -e

echo "============================================"
echo " ROI_module Setup"
echo "============================================"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create conda env
echo "[1/5] Creating conda environment 'roi'..."
conda create -n roi python=3.8 -y 2>/dev/null || echo "Environment 'roi' already exists"

# Activate
eval "$(conda shell.bash hook)"
conda activate roi

# Install PyTorch
echo "[2/5] Installing PyTorch + CUDA..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install TensorRT
echo "[3/5] Installing TensorRT..."
pip install tensorrt==8.5.3.1

# Install torch2trt from bundled source
echo "[4/5] Installing torch2trt..."
cd "$SCRIPT_DIR/torch2trt_src"
pip install -e .

# Install YOLOX from bundled source
echo "[5/5] Installing YOLOX + other deps..."
cd "$SCRIPT_DIR/YOLOX"
pip install -e .
cd "$SCRIPT_DIR"
pip install -r requirements.txt

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Tiếp theo:"
echo "  conda activate roi"
echo ""
echo "  # Convert checkpoint (BẮT BUỘC trên mỗi GPU mới):"
echo "  python convert_torch_to_trt.py \\"
echo "      --torch-ckpt weights/yolox_nano_416_roi_torch.pth \\"
echo "      --output weights/yolox_nano_416_roi_trt.pth \\"
echo "      --verify"
echo ""
echo "  # Crop ROI:"
echo "  python crop_roi.py \\"
echo "      --input-dir /path/to/images \\"
echo "      --output-dir /path/to/output \\"
echo "      --weight weights/yolox_nano_416_roi_trt.pth"
