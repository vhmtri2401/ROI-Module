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

### Tham số đầy đủ

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--input-dir` | — | Thư mục chứa DICOM/PNG (tìm đệ quy) |
| `--input-file` | — | File đơn lẻ (dùng thay `--input-dir`) |
| `--output-dir` | `./output_roi` | Thư mục lưu kết quả |
| `--weight` | **(bắt buộc)** | Đường dẫn file weight YOLOX (.pth) |
| `--backend` | `auto` | `auto` / `trt` / `pytorch` |
| `--input-size` | `416 416` | Kích thước input cho YOLOX |
| `--conf-thres` | `0.5` | Ngưỡng confidence |
| `--nms-thres` | `0.9` | Ngưỡng NMS |
| `--target-size` | — | Resize output (H W), vd: `1536 1024` |
| `--flip-left` | `false` | Lật ảnh vú trái (laterality=L) |
| `--save-bbox-viz` | `false` | Lưu ảnh bounding box visualization |

## Xử lý DICOM PhotometricInterpretation

Khi đọc file DICOM, chương trình **tự động phát hiện** `PhotometricInterpretation` của từng ảnh:

| Giá trị | Ý nghĩa | Xử lý |
|---|---|---|
| **MONOCHROME1** | Pixel thấp = mô đặc (sáng), pixel cao = nền (tối) | **Tự động invert** (`max - pixel`) trước khi convert |
| **MONOCHROME2** | Pixel cao = mô đặc (sáng) — chuẩn hiển thị | Giữ nguyên, không invert |

### Output báo cáo

Sau khi xử lý xong, chương trình in **Photometric Interpretation Report**:

```
============================================================
SUMMARY: 21/21 success, avg=4.3ms/image
Output: /path/to/output

  Photometric Interpretation Report:
  ────────────────────────────────────────
  MONOCHROME1 (inverted)  : 5
  MONOCHROME2 (normal)    : 16
  ────────────────────────────────────────
  ⚠ 5 image(s) were MONOCHROME1 → pixels inverted before conversion
    - IM-0001-0001.dcm
    - IM-0001-0003.dcm
    - ...
============================================================
```

### Thông tin trong `results.json`

Mỗi file trong `results.json` chứa 2 trường mới:

```json
{
  "file": "IM-0001-0001.dcm",
  "method": "YOLOX",
  "bbox": [1174, 238, 2414, 3162],
  "photometric_interpretation": "MONOCHROME1",
  "is_monochrome1_inverted": true,
  "..."
}
```

- `photometric_interpretation`: Giá trị gốc từ DICOM header (`MONOCHROME1`, `MONOCHROME2`, hoặc `N/A (non-DICOM)` cho file PNG/JPG)
- `is_monochrome1_inverted`: `true` nếu ảnh đã bị invert trước khi convert sang PNG

> ⚠️ **Lưu ý:** Nếu ảnh gốc là **MONOCHROME1**, file PNG đầu ra đã được invert để hiển thị đúng (mô đặc = sáng). Không cần invert lại khi sử dụng.

## Yêu cầu
- NVIDIA GPU + CUDA >= 11.7
- TensorRT >= 8.5
- Python 3.8
