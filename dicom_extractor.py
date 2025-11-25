#!/usr/bin/env python3
"""
dicom_extractor.py
Usage:
    python dicom_extractor.py <input_path> <out_json>

- input_path: single .dcm file OR path to a .zip (DICOM series) OR a folder containing .dcm files
- out_json: path to write the output JSON metadata summary

Produces a compact JSON with:
{
  "slice_count": int,
  "spacing_mm": {"z":..., "y":..., "x":...} or null,
  "hu_stats": {"mean":..., "std":..., "min":..., "max":...} or null,
  "bone_extent_mm_centerline": optional heuristic number or null,
  "dicom_tags": {PatientID, PatientName, PatientAge, PatientSex, StudyDate, Manufacturer, ManufacturerModelName, Modality}
}
"""

import sys
import os
import json
import tempfile
import zipfile
from pathlib import Path
from collections import defaultdict

try:
    import pydicom
    import numpy as np
except Exception as e:
    print("Missing dependencies. Install: pip install pydicom numpy", file=sys.stderr)
    raise

def gather_dicom_files(input_path: str):
    p = Path(input_path)
    files = []
    if p.is_file():
        if p.suffix.lower() == ".zip":
            # extract zip to temp dir
            tmp = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(str(p), "r") as z:
                z.extractall(tmp.name)
            base = Path(tmp.name)
            for f in base.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".dcm",""):
                    files.append(f)
            return files, tmp  # caller should keep tmp alive
        else:
            # single file
            return [p], None
    elif p.is_dir():
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".dcm",""):
                files.append(f)
        return sorted(files), None
    else:
        return [], None

def read_pixel_arrays(dcms, max_slices=512):
    """
    Read PixelData from DICOMs where present. Returns stacked numpy array (z,y,x) if possible.
    Limits memory by only reading up to max_slices.
    """
    arrays = []
    count = 0
    for f in dcms:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array.astype(np.float32)
                arrays.append(arr)
            else:
                # skip if no pixels
                continue
            count += 1
            if count >= max_slices:
                break
        except Exception:
            continue
    if not arrays:
        return None
    # try to stack, if shapes match
    shapes = {a.shape for a in arrays}
    if len(shapes) == 1:
        stacked = np.stack(arrays, axis=0)  # z,y,x
        return stacked
    else:
        # variable shapes: return None (we avoid resampling)
        return None

def safe_get(tagmap, keys):
    for k in keys:
        if k in tagmap:
            return tagmap[k]
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python dicom_extractor.py <input_path> <out_json>", file=sys.stderr)
        sys.exit(2)
    inp = sys.argv[1]
    out_json = sys.argv[2]

    files, tmp = gather_dicom_files(inp)
    if not files:
        payload = {"error": "no DICOM files found"}
        with open(out_json, "w") as fo:
            json.dump(payload, fo, indent=2)
        print("Wrote", out_json)
        return

    # read first valid DICOM for tags and geometry
    tags = {}
    spacing = None
    slice_count = 0
    manufacturer = manufacturer_model = modality = None

    valid_ds = None
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            valid_ds = ds
            break
        except Exception:
            continue

    if valid_ds:
        def get_tag(name):
            return getattr(valid_ds, name, None)
        tags_map = {
            "PatientID": get_tag("PatientID"),
            "PatientName": str(get_tag("PatientName")) if get_tag("PatientName") else None,
            "PatientAge": get_tag("PatientAge"),
            "PatientSex": get_tag("PatientSex"),
            "StudyDate": get_tag("StudyDate"),
            "Manufacturer": get_tag("Manufacturer"),
            "ManufacturerModelName": get_tag("ManufacturerModelName"),
            "Modality": get_tag("Modality")
        }
        tags = {k:v for k,v in tags_map.items() if v is not None}

        # spacing: try PixelSpacing and SliceThickness or SpacingBetweenSlices
        px = getattr(valid_ds, "PixelSpacing", None)
        st = getattr(valid_ds, "SliceThickness", None)
        sb = getattr(valid_ds, "SpacingBetweenSlices", None)

        try:
            if px is not None:
                # PixelSpacing is [row, col]
                spacing = {"z": float(st) if st else (float(sb) if sb else None),
                           "y": float(px[0]),
                           "x": float(px[1])}
            else:
                spacing = None
        except Exception:
            spacing = None

    # attempt reading pixel arrays (up to memory limit)
    arr = read_pixel_arrays(files, max_slices=512)
    hu_stats = None
    if arr is not None:
        # attempt to convert to HU if RescaleSlope/Intercept present on first slice
        hu = arr
        try:
            ds0 = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
            slope = float(getattr(ds0, "RescaleSlope", 1.0))
            intercept = float(getattr(ds0, "RescaleIntercept", 0.0))
            hu = hu * slope + intercept
        except Exception:
            pass
        hu_stats = {"mean": float(np.nanmean(hu)), "std": float(np.nanstd(hu)),
                    "min": float(np.nanmin(hu)), "max": float(np.nanmax(hu))}
        slice_count = int(hu.shape[0])
    else:
        slice_count = len(files)

    # simple heuristic for "bone_extent_mm_centerline" - try to estimate thickness of dense voxels along middle Z slice
    bone_extent = None
    if arr is not None:
        try:
            zmid = arr.shape[0] // 2
            mid = arr[zmid]
            # simple threshold at 300 HU as bone proxy if HU available, else use raw intensity percentile
            if hu_stats:
                thresh = 300.0
                mask = (hu[zmid] >= thresh)
                # estimate extent in mm using pixel spacing if available
                if spacing and spacing.get("x"):
                    # compute largest contiguous vertical extent in center column
                    cx = mid.shape[1] // 2
                    col = mask[:, cx]
                    # find longest run of True
                    runs = []
                    run = 0
                    for v in col:
                        if v:
                            run += 1
                        else:
                            if run:
                                runs.append(run)
                            run = 0
                    if run:
                        runs.append(run)
                    maxrun = max(runs) if runs else 0
                    bone_extent = float(maxrun * spacing.get("x"))
                else:
                    bone_extent = float(int(np.sum(mask))/10.0)  # fallback heuristic
        except Exception:
            bone_extent = None

    out = {
        "slice_count": slice_count,
        "spacing_mm": spacing,
        "hu_stats": hu_stats,
        "bone_extent_mm_centerline": bone_extent,
        "dicom_tags": tags
    }

    with open(out_json, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2)

    print("Wrote metadata to", out_json)
    # keep any temp dir alive until process ends
    if tmp is not None:
        # tmp is TemporaryDirectory object; it will be cleaned at process exit
        pass

if __name__ == "__main__":
    main()
