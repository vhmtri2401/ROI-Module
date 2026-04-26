#!/usr/bin/env python3
"""
Crop ROI trên ảnh mammography (DICOM hoặc PNG) sử dụng YOLOX.

Hỗ trợ:
  - TensorRT backend (nhanh, cần convert trước)
  - PyTorch backend (chậm hơn, không cần convert)
  - Otsu fallback (nếu YOLOX fail)

Usage:
    # Crop thư mục DICOM với TRT
    python crop_roi.py \
        --input-dir /path/to/dicoms \
        --output-dir /path/to/output \
        --weight weights/yolox_nano_416_roi_trt.pth \
        --backend trt

    # Crop với PyTorch backend
    python crop_roi.py \
        --input-dir /path/to/images \
        --output-dir /path/to/output \
        --weight weights/yolox_nano_416_roi_torch.pth \
        --backend pytorch

    # Crop 1 file
    python crop_roi.py \
        --input-file /path/to/image.dcm \
        --output-dir /path/to/output \
        --weight weights/yolox_nano_416_roi_trt.pth
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOX_DIR = os.path.join(SCRIPT_DIR, 'YOLOX')
if os.path.isdir(YOLOX_DIR) and YOLOX_DIR not in sys.path:
    sys.path.insert(0, YOLOX_DIR)

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    return torch.meshgrid(*tensors)


# ============================================================================
# Otsu fallback
# ============================================================================
def extract_roi_otsu(img, gkernel=(5, 5)):
    """Fallback ROI detection using Otsu thresholding."""
    ori_h, ori_w = img.shape[:2]
    upper = np.percentile(img, 95)
    img[img > upper] = np.min(img)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    img_bin = cv2.dilate(img_bin, element)
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None, None
    areas = np.array([cv2.contourArea(c) for c in cnts])
    idx = np.argmax(areas)
    area_pct = areas[idx] / (ori_h * ori_w)
    x0, y0, w, h = cv2.boundingRect(cnts[idx])
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    return [x0, y0, x1, y1], area_pct, None


# ============================================================================
# ROI Extractor
# ============================================================================
class RoiExtractor:
    """YOLOX-based ROI extractor (TensorRT or PyTorch backend)."""

    HW = [(52, 52), (26, 26), (13, 13)]
    STRIDES = [8, 16, 32]

    def __init__(self, weight_path, input_size=(416, 416),
                 num_classes=1, conf_thres=0.5, nms_thres=0.9,
                 area_pct_thres=0.04, backend='auto'):
        self.input_h, self.input_w = input_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.area_pct_thres = area_pct_thres
        self.hw = self.HW
        self.strides = self.STRIDES
        self.backend = backend
        self.model = self._load(weight_path)

    def _load(self, path):
        if self.backend in ('auto', 'trt'):
            try:
                from torch2trt import TRTModule
                m = TRTModule()
                m.load_state_dict(torch.load(path, map_location='cuda',
                                             weights_only=False))
                self.backend = 'trt'
                logger.info(f"Loaded TRT model: {path}")
                return m
            except Exception as e:
                if self.backend == 'trt':
                    raise RuntimeError(f"TRT load failed: {e}")
                logger.info(f"TRT unavailable ({e}), trying PyTorch...")

        # PyTorch fallback
        return self._load_pytorch(path)

    def _load_pytorch(self, path):
        import torch.nn as nn
        try:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        except ImportError:
            raise RuntimeError(
                "Cannot import yolox. Ensure YOLOX/ directory exists "
                "or install: cd YOLOX && pip install -e ."
            )

        depth, width = 0.33, 0.25
        in_ch = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_ch,
                             act='silu', depthwise=True)
        head = YOLOXHead(self.num_classes, width, in_channels=in_ch,
                         act='silu', depthwise=True)
        model = YOLOX(backbone, head)

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model = model.cuda().eval()

        # Probe hw/strides
        with torch.no_grad():
            dummy = torch.ones(1, 3, self.input_h, self.input_w).cuda()
            model(dummy)
            self.hw = model.head.hw
            self.strides = model.head.strides

        self.backend = 'pytorch'
        logger.info(f"Loaded PyTorch model: {path}")
        return model

    def decode_outputs(self, outputs):
        dtype = outputs.type()
        grids, strides_list = [], []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides_list.append(torch.full((*grid.shape[:2], 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides_cat = torch.cat(strides_list, dim=1).type(dtype)
        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides_cat,
            torch.exp(outputs[..., 2:4]) * strides_cat,
            outputs[..., 4:],
        ], dim=-1)
        return outputs

    def post_process(self, pred):
        box_corner = pred.new(pred.shape)
        box_corner[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        box_corner[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        box_corner[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        box_corner[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
        pred[:, :, :4] = box_corner[:, :, :4]

        output = [None]
        image_pred = pred[0]
        if not image_pred.size(0):
            return output
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + self.num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.conf_thres)
        dets = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        dets = dets[conf_mask]
        if not dets.size(0):
            return output
        nms_idx = torchvision.ops.batched_nms(
            dets[:, :4], dets[:, 4] * dets[:, 5], dets[:, 6], self.nms_thres)
        output[0] = dets[nms_idx]
        return output

    def preprocess_single(self, img):
        ori_h, ori_w = img.size(0), img.size(1)
        ratio = min(self.input_h / ori_h, self.input_w / ori_w)
        resized = F.interpolate(img.view(1, 1, ori_h, ori_w), mode="bilinear",
                                scale_factor=ratio, recompute_scale_factor=True)[0, 0]
        padded = torch.full((self.input_h, self.input_w), 114,
                            dtype=resized.dtype, device='cuda')
        padded[:resized.size(0), :resized.size(1)] = resized
        padded = padded.unsqueeze(-1).expand(-1, -1, 3).permute(2, 0, 1).float()
        return padded, resized, ratio, ori_h, ori_w

    @torch.no_grad()
    def detect_single(self, img):
        """
        Detect ROI in single grayscale image tensor (GPU, float, ~[0,255]).
        Returns: (xyxy, area_pct, confidence) or (None, 0.0, None)
        """
        padded, resized, ratio, ori_h, ori_w = self.preprocess_single(img)
        output = self.model(padded.unsqueeze(0))
        if self.backend == 'pytorch' and isinstance(output, (tuple, list)):
            output = output[0]
        output = self.decode_outputs(output)
        output = self.post_process(output)[0]

        if output is not None:
            output[:, :4] /= ratio
            output[:, 4] *= output[:, 5]
            best = output[output[:, 4].argmax()]
            try:
                if torch.isinf(best).any() or torch.isnan(best).any():
                    raise ValueError("inf/nan")
                x0 = min(max(int(best[0]), 0), ori_w)
                y0 = min(max(int(best[1]), 0), ori_h)
                x1 = min(max(int(best[2]), 0), ori_w)
                y1 = min(max(int(best[3]), 0), ori_h)
                area_pct = (x1 - x0) * (y1 - y0) / (ori_h * ori_w)
                if area_pct >= self.area_pct_thres:
                    return [x0, y0, x1, y1], area_pct, float(best[4])
            except Exception:
                pass

        # Otsu fallback
        xyxy, area_pct, _ = extract_roi_otsu(
            resized.to(torch.uint8).cpu().numpy())
        if xyxy is not None and area_pct is not None:
            if area_pct >= self.area_pct_thres:
                x0, y0, x1, y1 = xyxy
                x0 = min(max(int(x0 / ratio), 0), ori_w)
                y0 = min(max(int(y0 / ratio), 0), ori_h)
                x1 = min(max(int(x1 / ratio), 0), ori_w)
                y1 = min(max(int(y1 / ratio), 0), ori_h)
                return [x0, y0, x1, y1], area_pct, None

        return None, 0.0, None


# ============================================================================
# DICOM utilities
# ============================================================================
def read_dicom(path):
    """Read DICOM → (float32 array, metadata dict).
    
    Detects PhotometricInterpretation (MONOCHROME1 / MONOCHROME2)
    and logs the result. MONOCHROME1 images are inverted so that
    pixel intensity matches the standard (bright = dense tissue).
    """
    import pydicom
    ds = pydicom.dcmread(path)
    px = ds.pixel_array.astype(np.float32)

    # Detect PhotometricInterpretation
    photometric = str(getattr(ds, 'PhotometricInterpretation', 'UNKNOWN'))
    is_monochrome1 = (photometric == 'MONOCHROME1')

    if is_monochrome1:
        logger.warning(f"  [MONO1] {os.path.basename(path)} → "
                       f"PhotometricInterpretation=MONOCHROME1, inverting pixels")
        px = np.max(px) - px
    else:
        logger.info(f"  [MONO2] {os.path.basename(path)} → "
                    f"PhotometricInterpretation={photometric}")

    # Windowing
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)
    if wc is not None and ww is not None:
        if hasattr(wc, '__len__'):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
        lo, hi = wc - ww / 2, wc + ww / 2
        windowed = np.clip(px, lo, hi)
        windowed = ((windowed - lo) / (hi - lo + 1e-8) * 255).astype(np.float32)
    else:
        lo, hi = np.percentile(px, 1), np.percentile(px, 99)
        windowed = np.clip(px, lo, hi)
        windowed = ((windowed - lo) / (hi - lo + 1e-8) * 255).astype(np.float32)

    lat = str(getattr(ds, 'ImageLaterality',
                       getattr(ds, 'Laterality', 'U')))
    meta = {
        'laterality': lat,
        'view': str(getattr(ds, 'ViewPosition', 'Unknown')),
        'patient_id': str(getattr(ds, 'PatientID', 'Unknown')),
        'rows': int(ds.Rows),
        'columns': int(ds.Columns),
        'photometric_interpretation': photometric,
        'is_monochrome1_inverted': is_monochrome1,
    }
    return px, windowed, meta


def read_image(path):
    """Read PNG/DICOM → (raw_float32, windowed_float32, meta)."""
    ext = Path(path).suffix.lower()
    if ext == '.dcm':
        return read_dicom(path)
    # PNG / JPG (no DICOM header → photometric info not available)
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    raw = img.astype(np.float32)
    if img.dtype == np.uint8:
        windowed = raw.copy()
    else:
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
        windowed = np.clip(raw, lo, hi)
        windowed = ((windowed - lo) / (hi - lo + 1e-8) * 255).astype(np.float32)
    return raw, windowed, {
        'laterality': 'U',
        'view': 'Unknown',
        'patient_id': 'Unknown',
        'photometric_interpretation': 'N/A (non-DICOM)',
        'is_monochrome1_inverted': False,
    }


# ============================================================================
# Main crop logic
# ============================================================================
def crop_single(extractor, img_path, out_dir, target_size=None,
                flip_left=False, save_viz=False):
    """Crop ROI from a single image. Returns result dict."""
    raw, windowed, meta = read_image(img_path)
    ori_h, ori_w = raw.shape[:2]

    # Normalize for YOLOX
    lo = np.percentile(raw, 0.5)
    hi = np.percentile(raw, 99.5)
    normed = np.clip(raw, lo, hi)
    normed = ((normed - lo) / (hi - lo + 1e-8) * 255).astype(np.float32)
    img_tensor = torch.from_numpy(normed).cuda()

    t0 = time.time()
    xyxy, area_pct, conf = extractor.detect_single(img_tensor)
    dt = (time.time() - t0) * 1000

    method = 'YOLOX'
    if xyxy is None:
        xyxy = [0, 0, ori_w, ori_h]
        area_pct = 1.0
        conf = 0.0
        method = 'full_frame'

    # Crop on windowed 8-bit
    img_8bit = np.clip(windowed, 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = xyxy
    roi = img_8bit[y0:y1, x0:x1]
    if roi.size == 0:
        roi = img_8bit
        method = 'full_frame'

    # Resize
    if target_size is not None:
        roi = cv2.resize(roi, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LANCZOS4)

    # Flip laterality
    lat = meta.get('laterality', 'U')
    if flip_left and lat == 'L':
        roi = cv2.flip(roi, 1)

    # Save
    stem = Path(img_path).stem
    out_path = os.path.join(out_dir, f"{stem}_cropped.png")
    cv2.imwrite(out_path, roi)

    # Optional bbox visualization
    if save_viz:
        viz = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 0), 3)
        label = f"{method}"
        if conf:
            label += f" {conf:.3f}"
        cv2.putText(viz, label, (x0, max(y0 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        viz_path = os.path.join(out_dir, f"{stem}_bbox.png")
        # Resize viz for reasonable file size
        max_dim = 1024
        h, w = viz.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            viz = cv2.resize(viz, (int(w * s), int(h * s)))
        cv2.imwrite(viz_path, viz)

    return {
        'file': os.path.basename(img_path),
        'method': method,
        'bbox': xyxy,
        'area_pct': round((area_pct or 0) * 100, 1),
        'conf': round(conf, 4) if conf else 0.0,
        'time_ms': round(dt, 1),
        'output': out_path,
        'laterality': lat,
        'size': f"{ori_h}x{ori_w}",
        'photometric_interpretation': meta.get('photometric_interpretation', 'N/A'),
        'is_monochrome1_inverted': meta.get('is_monochrome1_inverted', False),
    }


def find_images(input_dir=None, input_file=None):
    """Find all DICOM/PNG files."""
    if input_file:
        return [input_file]
    patterns = ['**/*.dcm', '**/*.png', '**/*.jpg', '**/*.jpeg']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p), recursive=True))
    return sorted(set(files))


def main():
    p = argparse.ArgumentParser(description="YOLOX ROI Crop")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input-dir', help='Directory with DICOM/PNG files')
    g.add_argument('--input-file', help='Single image file')
    p.add_argument('--output-dir', default='./output_roi')
    p.add_argument('--weight', required=True, help='YOLOX weight path (.pth)')
    p.add_argument('--backend', default='auto', choices=['auto', 'trt', 'pytorch'])
    p.add_argument('--input-size', nargs=2, type=int, default=[416, 416])
    p.add_argument('--conf-thres', type=float, default=0.5)
    p.add_argument('--nms-thres', type=float, default=0.9)
    p.add_argument('--target-size', nargs=2, type=int, default=None,
                   metavar=('H', 'W'), help='Resize output (H W)')
    p.add_argument('--flip-left', action='store_true',
                   help='Flip left breast images')
    p.add_argument('--save-bbox-viz', action='store_true',
                   help='Save bounding box visualization')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find files
    files = find_images(args.input_dir, args.input_file)
    print(f"Found {len(files)} images")
    if not files:
        return

    # Load model
    print(f"Loading model: {args.weight} (backend={args.backend})")
    t0 = time.time()
    extractor = RoiExtractor(
        weight_path=args.weight,
        input_size=tuple(args.input_size),
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        backend=args.backend,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s (backend: {extractor.backend})")

    # Process
    results = []
    total_time = 0
    for i, f in enumerate(files):
        logger.info(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
        try:
            r = crop_single(extractor, f, args.output_dir,
                            target_size=tuple(args.target_size) if args.target_size else None,
                            flip_left=args.flip_left,
                            save_viz=args.save_bbox_viz)
            results.append(r)
            total_time += r['time_ms']
            logger.info(f"  → {r['method']} bbox={r['bbox']} "
                        f"area={r['area_pct']}% conf={r['conf']} "
                        f"t={r['time_ms']}ms")
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results.append({'file': os.path.basename(f), 'error': str(e)})

    # Summary
    n = len(results)
    ok = sum(1 for r in results if 'error' not in r)
    avg = total_time / ok if ok else 0

    # Photometric interpretation report
    mono1_count = sum(1 for r in results
                      if r.get('is_monochrome1_inverted', False))
    mono2_count = sum(1 for r in results
                      if 'error' not in r
                      and r.get('photometric_interpretation') == 'MONOCHROME2')
    other_count = ok - mono1_count - mono2_count

    print(f"\n{'='*60}")
    print(f"SUMMARY: {ok}/{n} success, avg={avg:.1f}ms/image")
    print(f"Output: {args.output_dir}")
    print(f"")
    print(f"  Photometric Interpretation Report:")
    print(f"  {'─'*40}")
    print(f"  MONOCHROME1 (inverted)  : {mono1_count}")
    print(f"  MONOCHROME2 (normal)    : {mono2_count}")
    if other_count > 0:
        print(f"  Other / non-DICOM       : {other_count}")
    print(f"  {'─'*40}")
    if mono1_count > 0:
        print(f"  ⚠ {mono1_count} image(s) were MONOCHROME1 → pixels inverted before conversion")
        # List individual MONOCHROME1 files
        for r in results:
            if r.get('is_monochrome1_inverted', False):
                print(f"    - {r['file']}")
    print(f"{'='*60}")

    # Save JSON
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results: {json_path}")


if __name__ == '__main__':
    main()
