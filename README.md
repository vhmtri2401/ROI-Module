# ROI_module — YOLOX ROI Converter & Cropper

Module **self-contained** để convert checkpoint YOLOX PyTorch → TensorRT và crop ROI trên ảnh mammography. Có thể copy toàn bộ folder sang máy khác chạy.

> ⚠️ **Checkpoint TRT phụ thuộc GPU cụ thể.** Khi sang máy/GPU mới, **phải convert lại** từ checkpoint PyTorch gốc.

## Cấu trúc

```
ROI_module/
├── convert_torch_to_trt.py   # Convert PyTorch → TensorRT
├── crop_roi.py                # Crop ROI (DICOM/PNG → PNG)
├── setup.sh                   # Setup tự động trên máy mới
├── requirements.txt           # Python dependencies
├── README.md                  # File này
├── weights/
│   └── yolox_nano_416_roi_torch.pth   # Checkpoint PyTorch (portable)
├── YOLOX/                     # YOLOX source code (bundled)
└── torch2trt_src/             # torch2trt source code (bundled)
```

## Setup trên máy mới

### Cách 1: Tự động (recommended)
```bash
cd ROI_module
chmod +x setup.sh
bash setup.sh
```

### Cách 2: Thủ công
```bash
conda create -n roi python=3.8 -y
conda activate roi
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install tensorrt==8.5.3.1
cd torch2trt_src && pip install -e . && cd ..
cd YOLOX && pip install -e . && cd ..
pip install -r requirements.txt
```

## Sử dụng

### Bước 1: Convert checkpoint (BẮT BUỘC trên mỗi GPU mới)
```bash
conda activate roi
python convert_torch_to_trt.py \
    --torch-ckpt weights/yolox_nano_416_roi_torch.pth \
    --output weights/yolox_nano_416_roi_trt.pth \
    --verify
```

### Bước 2: Crop ROI
```bash
# Thư mục DICOM
python crop_roi.py \
    --input-dir /path/to/dicoms \
    --output-dir /path/to/output \
    --weight weights/yolox_nano_416_roi_trt.pth \
    --save-bbox-viz

# File đơn lẻ
python crop_roi.py \
    --input-file /path/to/image.dcm \
    --output-dir /path/to/output \
    --weight weights/yolox_nano_416_roi_trt.pth

# Với resize + flip
python crop_roi.py \
    --input-dir /path/to/dicoms \
    --output-dir /path/to/output \
    --weight weights/yolox_nano_416_roi_trt.pth \
    --target-size 1536 1024 \
    --flip-left \
    --save-bbox-viz

# Dùng PyTorch backend (không cần convert TRT)
python crop_roi.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --weight weights/yolox_nano_416_roi_torch.pth \
    --backend pytorch
```

## Yêu cầu
- NVIDIA GPU + CUDA >= 11.7
- TensorRT >= 8.5
- Python 3.8
