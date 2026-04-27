#!/usr/bin/env python3
"""
Test script: Verify MONOCHROME1/MONOCHROME2 detection on DICOM files.
Only requires pydicom + numpy (no GPU/YOLOX needed).

Usage:
    python test_monochrome.py --input-dir <path_to_dicoms>
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

def test_dicom_monochrome(dicom_dir):
    """Read all DICOM files and report PhotometricInterpretation."""
    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom not installed. Run: pip install pydicom")
        sys.exit(1)

    # Find all DICOM files (.dcm and .dicom)
    patterns = ['**/*.dcm', '**/*.dicom']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(dicom_dir, p), recursive=True))
    files = sorted(set(files))

    if not files:
        print(f"No DICOM files found in: {dicom_dir}")
        return

    print(f"Found {len(files)} DICOM files in: {dicom_dir}")
    print(f"{'='*80}")

    results = []
    mono1_files = []
    mono2_files = []
    other_files = []
    errors = []

    for i, fpath in enumerate(files):
        fname = os.path.relpath(fpath, dicom_dir)
        try:
            ds = pydicom.dcmread(fpath)
            photometric = str(getattr(ds, 'PhotometricInterpretation', 'UNKNOWN'))
            rows = int(getattr(ds, 'Rows', 0))
            cols = int(getattr(ds, 'Columns', 0))
            bits = int(getattr(ds, 'BitsStored', 0))
            lat = str(getattr(ds, 'ImageLaterality',
                               getattr(ds, 'Laterality', 'N/A')))
            view = str(getattr(ds, 'ViewPosition', 'N/A'))
            patient_id = str(getattr(ds, 'PatientID', 'N/A'))

            # Read pixel data to verify
            px = ds.pixel_array.astype(np.float32)
            px_min, px_max = float(np.min(px)), float(np.max(px))
            px_mean = float(np.mean(px))

            is_mono1 = (photometric == 'MONOCHROME1')

            # Classify
            if is_mono1:
                mono1_files.append(fname)
                tag = "[MONO1]"
            elif photometric == 'MONOCHROME2':
                mono2_files.append(fname)
                tag = "[MONO2]"
            else:
                other_files.append(fname)
                tag = "[OTHER]"

            result = {
                'file': fname,
                'photometric': photometric,
                'is_mono1': is_mono1,
                'size': f"{rows}x{cols}",
                'bits': bits,
                'laterality': lat,
                'view': view,
                'patient_id': patient_id,
                'pixel_range': f"[{px_min:.0f}, {px_max:.0f}]",
                'pixel_mean': f"{px_mean:.1f}",
            }
            results.append(result)

            print(f"  [{i+1:2d}/{len(files)}] {tag} {fname}")
            print(f"           Size={rows}x{cols}  Bits={bits}  "
                  f"Lat={lat}  View={view}")
            print(f"           Pixels: range={result['pixel_range']}  "
                  f"mean={result['pixel_mean']}")

            # Simulate inversion for MONOCHROME1
            if is_mono1:
                px_inv = np.max(px) - px
                inv_min, inv_max = float(np.min(px_inv)), float(np.max(px_inv))
                inv_mean = float(np.mean(px_inv))
                print(f"           After invert: range=[{inv_min:.0f}, {inv_max:.0f}]  "
                      f"mean={inv_mean:.1f}")

        except Exception as e:
            errors.append((fname, str(e)))
            print(f"  [{i+1:2d}/{len(files)}] [ERROR] {fname}")
            print(f"           {e}")

    # Summary report
    print(f"\n{'='*80}")
    print(f"PHOTOMETRIC INTERPRETATION REPORT")
    print(f"{'='*80}")
    print(f"  Total DICOM files     : {len(files)}")
    print(f"  Successfully read     : {len(results)}")
    print(f"  Errors                : {len(errors)}")
    print(f"")
    print(f"  MONOCHROME1 (inverted): {len(mono1_files)}")
    print(f"  MONOCHROME2 (normal)  : {len(mono2_files)}")
    if other_files:
        print(f"  Other                 : {len(other_files)}")
    print(f"  {'-'*40}")

    if mono1_files:
        print(f"\n  Files with MONOCHROME1 (will be inverted during conversion):")
        for f in mono1_files:
            print(f"    - {f}")

    if mono2_files:
        print(f"\n  Files with MONOCHROME2 (no inversion needed):")
        for f in mono2_files:
            print(f"    - {f}")

    if errors:
        print(f"\n  Files with errors:")
        for f, e in errors:
            print(f"    - {f}: {e}")

    # Group by patient
    patients = {}
    for r in results:
        pid = r['patient_id']
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(r)

    print(f"\n{'='*80}")
    print(f"PER-PATIENT SUMMARY")
    print(f"{'='*80}")
    for pid, imgs in sorted(patients.items()):
        mono1_n = sum(1 for r in imgs if r['is_mono1'])
        mono2_n = len(imgs) - mono1_n
        print(f"\n  Patient: {pid}  ({len(imgs)} images)")
        print(f"    MONO1={mono1_n}  MONO2={mono2_n}")
        for r in imgs:
            tag = "MONO1" if r['is_mono1'] else "MONO2"
            print(f"    [{tag}] {r['file']}  "
                  f"Lat={r['laterality']}  View={r['view']}  "
                  f"Size={r['size']}  Bits={r['bits']}")

    print(f"\n{'='*80}")
    print(f"DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Test DICOM Monochrome Detection")
    p.add_argument('--input-dir', required=True,
                   help='Directory with DICOM files')
    args = p.parse_args()
    test_dicom_monochrome(args.input_dir)
