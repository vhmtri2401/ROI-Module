#!/usr/bin/env python3
"""
Convert YOLOX PyTorch checkpoint → TensorRT checkpoint.

TRT checkpoint phụ thuộc GPU cụ thể → phải convert lại trên mỗi máy/GPU mới.

Usage:
    conda activate roi
    python convert_torch_to_trt.py \
        --torch-ckpt weights/yolox_nano_416_roi_torch.pth \
        --output weights/yolox_nano_416_roi_trt.pth \
        --verify
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add YOLOX to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOX_DIR = os.path.join(SCRIPT_DIR, 'YOLOX')
if os.path.isdir(YOLOX_DIR) and YOLOX_DIR not in sys.path:
    sys.path.insert(0, YOLOX_DIR)


def build_yolox_nano(num_classes=1):
    """Build YOLOX-Nano model (depth=0.33, width=0.25, depthwise=True)."""
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    depth, width = 0.33, 0.25
    in_channels = [256, 512, 1024]

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    backbone = YOLOPAFPN(depth, width, in_channels=in_channels,
                         act='silu', depthwise=True)
    head = YOLOXHead(num_classes, width, in_channels=in_channels,
                     act='silu', depthwise=True)
    model = YOLOX(backbone, head)
    model.apply(init_yolo)
    return model


def convert(args):
    import tensorrt as trt
    from torch2trt import torch2trt

    print("=" * 60)
    print(" YOLOX PyTorch → TensorRT Converter")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA không khả dụng!")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"TensorRT: {trt.__version__}")
    print()

    # 1) Build model
    print("[1/4] Khởi tạo YOLOX-Nano model...")
    model = build_yolox_nano(num_classes=args.num_classes)

    # 2) Load weights
    print(f"[2/4] Load checkpoint: {args.torch_ckpt}")
    ckpt = torch.load(args.torch_ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    model.head.decode_in_inference = False

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")

    # 3) Convert
    H, W = args.input_size
    fp16 = not args.no_fp16
    print(f"[3/4] Converting to TRT (input={H}x{W}, fp16={fp16})...")
    x = torch.ones(1, 3, H, W).cuda()
    t0 = time.time()
    model_trt = torch2trt(model, [x], fp16_mode=fp16,
                          log_level=trt.Logger.INFO,
                          max_workspace_size=(1 << args.workspace),
                          max_batch_size=1)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    # 4) Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    torch.save(model_trt.state_dict(), args.output)
    sz = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[4/4] Saved: {args.output} ({sz:.1f} MB)")

    # Save engine
    engine_path = args.output.replace('.pth', '.engine')
    try:
        with open(engine_path, 'wb') as f:
            f.write(model_trt.engine.serialize())
        print(f"  Engine: {engine_path}")
    except Exception as e:
        print(f"  Warning: engine save failed: {e}")

    # 5) Verify
    if args.verify:
        print()
        print("=" * 60)
        print(" Verification: PyTorch vs TRT")
        print("=" * 60)
        for i in range(3):
            inp = torch.rand(1, 3, H, W).cuda() * 255
            with torch.no_grad():
                out_pt = model(inp)
                out_trt = model_trt(inp)
            diff = (out_pt - out_trt).abs()
            print(f"  Test {i+1}: max_diff={diff.max():.6f}, mean={diff.mean():.6f}")

        # Speed test
        for _ in range(5):  # warmup
            with torch.no_grad():
                model_trt(x)
        torch.cuda.synchronize()
        t0 = time.time()
        N = 50
        for _ in range(N):
            with torch.no_grad():
                model_trt(x)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / N * 1000
        print(f"  TRT speed: {ms:.1f} ms/frame")

    print()
    print("HOÀN TẤT!")
    print(f"  Checkpoint TRT: {args.output}")
    print(f"  GPU: {gpu}")
    print(f"  Sử dụng: python crop_roi.py --weight {args.output} ...")


def main():
    p = argparse.ArgumentParser(description="Convert YOLOX PyTorch → TensorRT")
    p.add_argument('--torch-ckpt', required=True, help='PyTorch checkpoint path')
    p.add_argument('--output', default='weights/yolox_nano_416_roi_trt.pth')
    p.add_argument('--input-size', nargs=2, type=int, default=[416, 416])
    p.add_argument('--num-classes', type=int, default=1)
    p.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    p.add_argument('--workspace', type=int, default=32)
    p.add_argument('--verify', action='store_true', help='Verify after convert')
    args = p.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
