#!/usr/bin/env python3
"""
dicom_metadata_extractor.py

Usage:
    python dicom_metadata_extractor.py /path/to/dicom_or_folder /path/to/output.json

If the input is a single .dcm file, it extracts metadata for that series.
If the input is a folder, it reads all .dcm files in that folder (one series expected)
and builds a 3D volume to compute simple measurements: voxel spacing, slice count,
volume shape, HU statistics (mean, std, percentiles), and a basic alveolar-ridge proxy:
 - center-sagittal mean HU per slice
 - approximate vertical extent of bone in centerline (threshold-based)

Outputs a JSON metadata file (human-friendly).
"""
import os, sys, json, numpy as np
from pathlib import Path
import pydicom
from pydicom import dcmread
from collections import defaultdict

def is_dicom_file(p: Path):
    try:
        dcmread(str(p), stop_before_pixels=True)
        return True
    except Exception:
        return False

def load_series_from_folder(folder):
    files = [Path(folder)/f for f in sorted(os.listdir(folder))]
    dicoms = []
    for f in files:
        if f.is_file() and f.suffix.lower() in (".dcm", ""):
            try:
                ds = dcmread(str(f))
                dicoms.append(ds)
            except Exception:
                continue
    if not dicoms:
        raise RuntimeError("No DICOMs found in folder.")
    # sort by InstanceNumber or by SliceLocation
    dicoms = sorted(dicoms, key=lambda x: getattr(x, "InstanceNumber", getattr(x, "SliceLocation", 0)))
    return dicoms

def build_volume_and_meta(datasets):
    # get pixel spacing and slice thickness (try consistent values)
    first = datasets[0]
    px = None
    try:
        px = [float(x) for x in first.PixelSpacing]  # [row, col]
    except Exception:
        px = None
    slice_thickness = None
    try:
        slice_thickness = float(first.get("SliceThickness", 0.0))
    except Exception:
        slice_thickness = None

    # stack pixel arrays (apply rescale)
    arrs = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1.0))
        interp = float(ds.get("RescaleIntercept", 0.0))
        arr = arr * slope + interp
        arrs.append(arr)
    volume = np.stack(arrs, axis=0)  # shape (Z, H, W)

    # basic geometry
    Z, H, W = volume.shape
    spacing_z = slice_thickness if slice_thickness else (float(datasets[1].ImagePositionPatient[2]) - float(datasets[0].ImagePositionPatient[2]) if len(datasets) > 1 and hasattr(datasets[0],'ImagePositionPatient') else 1.0)
    spacing_xy = tuple(px) if px else (1.0, 1.0)

    # basic HU stats
    vol_flat = volume.ravel()
    stats = {
        "mean": float(np.mean(vol_flat)),
        "std": float(np.std(vol_flat)),
        "min": float(np.min(vol_flat)),
        "max": float(np.max(vol_flat)),
        "p25": float(np.percentile(vol_flat, 25)),
        "p50": float(np.percentile(vol_flat, 50)),
        "p75": float(np.percentile(vol_flat, 75)),
    }

    # center sagittal analysis (approx)
    cx = W // 2
    cy = H // 2
    centerline_mean_per_slice = [float(np.mean(vol[z, :, cx-10:cx+10])) for z in range(Z)]
    # approximate bone vertical extent where centerline mean > threshold (threshold use 200 HU as proxy)
    thresh = stats["p50"] * 0.6 if stats["p50"] > 0 else 200
    bone_slices = [i for i, val in enumerate(centerline_mean_per_slice) if val > thresh]
    if bone_slices:
        bone_extent_mm = len(bone_slices) * spacing_z
        bone_extent_slice_range = (min(bone_slices), max(bone_slices))
    else:
        bone_extent_mm = 0.0
        bone_extent_slice_range = (0,0)

    meta = {
        "shape": {"z": int(Z), "y": int(H), "x": int(W)},
        "spacing_mm": {"z": float(spacing_z), "y": float(spacing_xy[0]), "x": float(spacing_xy[1])},
        "hu_stats": stats,
        "centerline_mean_per_slice_sample": centerline_mean_per_slice[:12],
        "bone_extent_mm_centerline": float(bone_extent_mm),
        "bone_extent_slice_range": bone_extent_slice_range,
        "slice_count": int(Z)
    }
    # add series-level DICOM tags (from first)
    tags = {}
    for k in ("PatientID","PatientName","PatientAge","PatientSex","StudyDate","Manufacturer","ManufacturerModelName","SeriesDescription"):
        if k in first:
            tags[k]=str(first[k])
    meta["dicom_tags"] = tags
    return meta

def main(input_path, out_json):
    p = Path(input_path)
    if p.is_dir():
        datasets = load_series_from_folder(p)
    elif p.is_file():
        # if single .dcm that belongs to a series, check if it's just one slice
        if is_dicom_file(p):
            ds = dcmread(str(p))
            # if file is single-frame 3D (rare), treat as series length 1
            datasets = [ds]
        else:
            raise RuntimeError("File is not a DICOM.")
    else:
        raise RuntimeError("Input path not found.")

    meta = build_volume_and_meta(datasets)
    # write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {out_json}")
    return meta

if __name__=="__main__":
    if len(sys.argv) < 3:
        print("Usage: python dicom_metadata_extractor.py /path/to/dicom_or_folder /path/to/output.json")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    meta = main(inp, out)
    print(json.dumps(meta, indent=2))
