# dicom_reader.py
"""
Robust DICOM reader for CBCT:
- Accepts path to .zip (extracted), folder of .dcm files, or single .dcm (including multi-frame)
- Tries pydicom + pylibjpeg decoders first
- Falls back to dcm2niix (if available) for hardest vendor encodings
- Returns dict: {
    'volume': np.ndarray (z,y,x) or None,
    'affine': np.eye(4) or nifti affine,
    'metadata': {...},
    'hu_stats': {'mean':..., 'min':..., 'max':...} or {},
    'preview': {'axial': path, 'coronal': path, 'sagittal': path} or None,
    'warnings': [...]
  }
"""
import os
import zipfile
import tempfile
import subprocess
import shutil
import json
from pathlib import Path
import numpy as np
import pydicom
from pydicom.uid import UID
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _extract_zip_to_tmp(zip_path):
    tmp = tempfile.mkdtemp(prefix="cbct_unzip_")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)
    return tmp

def _collect_dcm_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(".dcm") or fn.lower().endswith(".dicom"):
                files.append(os.path.join(root, fn))
    # sometimes files have no .dcm extension — also consider all files with DICOM magic
    if not files:
        for root, _, filenames in os.walk(folder):
            for fn in filenames:
                p = os.path.join(root, fn)
                try:
                    with open(p, "rb") as fh:
                        head = fh.read(132)
                        if b"DICM" in head:
                            files.append(p)
                except Exception:
                    pass
    files.sort()
    return files

def _try_pydicom_multiframe(path):
    """
    Try reading a single DICOM file that contains multiple frames (NumberOfFrames)
    """
    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception:
        return None, None
    # check for pixel data and multiframe
    if not hasattr(ds, "PixelData"):
        return None, None
    nframes = getattr(ds, "NumberOfFrames", None)
    if nframes is None:
        # some vendors use PerFrameFunctionalGroups; still may be multi-frame
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence') and isinstance(getattr(ds, 'PerFrameFunctionalGroupsSequence'), (list, tuple)):
            # try treating as multiframe
            nframes = len(ds.PerFrameFunctionalGroupsSequence)
    if not nframes:
        return None, None
    try:
        arr = ds.pixel_array  # triggers pylibjpeg if installed
    except Exception:
        return None, None
    # arr shape may be (frames, rows, cols) or (rows, cols, frames, ...)
    if arr.ndim == 3:
        vol = arr.astype(np.float32)
    elif arr.ndim == 4:
        # e.g., (frames, rows, cols, channels) -> take first channel or convert grayscale
        vol = arr[...,0].astype(np.float32)
    else:
        return None, None
    # basic metadata
    affine = np.eye(4)
    pixel_spacing = None
    slice_thickness = None
    try:
        if hasattr(ds, "PixelSpacing"):
            pixel_spacing = float(ds.PixelSpacing[0])
    except Exception:
        pass
    try:
        if hasattr(ds, "SliceThickness"):
            slice_thickness = float(ds.SliceThickness)
    except Exception:
        pass
    metadata = {"pixel_spacing": pixel_spacing, "slice_thickness": slice_thickness, "dcm_multiframe": True, "sop_class": getattr(ds,'SOPClassUID', None)}
    return vol, (affine, metadata)

def _try_pydicom_stack(dcm_paths):
    """
    Read series using pydicom; return 3D numpy array if possible.
    """
    slices = []
    positions = []
    spacings = []
    warnings = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, force=True)
        except Exception as e:
            warnings.append(f"pydicom read failed for {p}: {e}")
            return None, None
        try:
            arr = ds.pixel_array
        except Exception as e:
            warnings.append(f"pixel_array decoding failed for {p}: {e}")
            return None, None
        # ensure 2D grayscale for each slice
        if arr.ndim == 3 and arr.shape[2] in (3,4):
            # color image - convert to gray
            arr = np.mean(arr, axis=2)
        slices.append(arr.astype(np.float32))
        # z position
        zpos = None
        if hasattr(ds, "ImagePositionPatient"):
            try:
                zpos = float(ds.ImagePositionPatient[2])
            except Exception:
                zpos = None
        elif hasattr(ds, "SliceLocation"):
            try:
                zpos = float(ds.SliceLocation)
            except Exception:
                zpos = None
        positions.append(zpos)
        # spacing
        ps = None
        st = None
        if hasattr(ds, "PixelSpacing"):
            try:
                ps = float(ds.PixelSpacing[0])
            except Exception:
                ps = None
        if hasattr(ds, "SliceThickness"):
            try:
                st = float(ds.SliceThickness)
            except Exception:
                st = None
        spacings.append((ps, st))
    # attempt ordering
    try:
        if any([p is not None for p in positions]):
            order = np.argsort([p if p is not None else i for i,p in enumerate(positions)])
            slices = [slices[i] for i in order]
    except Exception:
        pass
    vol = np.stack(slices, axis=0).astype(np.float32)
    affine = np.eye(4)
    pixel_spacing = None
    slice_thickness = None
    ps_vals = [s[0] for s in spacings if s[0]]
    st_vals = [s[1] for s in spacings if s[1]]
    if ps_vals:
        pixel_spacing = float(np.mean(ps_vals))
    if st_vals:
        slice_thickness = float(np.mean(st_vals))
    metadata = {"pixel_spacing": pixel_spacing, "slice_thickness": slice_thickness, "n_slices": vol.shape[0]}
    metadata["warnings"] = warnings
    return vol, (affine, metadata)

def _dcm2niix_conversion(folder):
    """
    Run dcm2niix on the folder and return the first NIfTI path produced.
    Requires dcm2niix in PATH.
    """
    outdir = tempfile.mkdtemp(prefix="cbct_dcm2niix_out_")
    cmd = ["dcm2niix", "-z", "n", "-o", outdir, folder]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except Exception as e:
        return None
    nifti_files = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    nifti_files.sort()
    if not nifti_files:
        return None
    return nifti_files[0]

def _make_preview_images(vol, out_folder, prefix="preview"):
    """
    Generate axial/coronal/sagittal PNG previews (mid slices).
    Returns dict of paths.
    """
    _ensure_dir(out_folder)
    previews = {}
    if vol is None:
        return None
    try:
        z,y,x = vol.shape
        axial = vol[z//2,:,:]
        coronal = vol[:,y//2,:]
        sagittal = vol[:,:,x//2]
        # simple autoscale and save helper
        def save_img(arr, fname):
            arr = np.nan_to_num(arr)
            # scale to 0-255 for PNG
            lo, hi = np.percentile(arr, [2,98])
            if hi - lo <= 0:
                lo, hi = arr.min(), arr.max()
                if hi - lo == 0:
                    hi = lo + 1.0
            arr_clip = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            img = (arr_clip * 255).astype(np.uint8)
            im = Image.fromarray(img)
            im = im.convert("L")
            path = os.path.join(out_folder, fname)
            im.save(path)
            return path
        previews['axial'] = save_img(axial, f"{prefix}_axial.png")
        previews['coronal'] = save_img(coronal, f"{prefix}_coronal.png")
        previews['sagittal'] = save_img(sagittal, f"{prefix}_sagittal.png")
    except Exception as e:
        # if preview creation fails, return None
        return None
    return previews

def load_cbct_from_path(path):
    """
    Main entry. path can be:
    - /path/to/zip.zip (extracted)
    - /path/to/folder_with_dcms/
    - /path/to/single.dcm (multi-frame or single)
    """
    cleanup_tmp = None
    warnings = []
    try:
        if str(path).lower().endswith(".zip"):
            tmp_dir = _extract_zip_to_tmp(path)
            folder = tmp_dir
            cleanup_tmp = tmp_dir
        else:
            # if it's a file that is a single dcm, we may still want to treat folder as containing that file's parent
            if os.path.isfile(path) and (path.lower().endswith(".dcm") or path.lower().endswith(".dicom")):
                # We'll attempt to handle single-file multi-frame; otherwise treat parent dir as folder.
                folder = os.path.dirname(path)
                single_file = path
            else:
                folder = path
                single_file = None

        # If single_file exists and is multiframe, try that first
        if 'single_file' in locals() and single_file:
            vol, meta_tuple = _try_pydicom_multiframe(single_file)
            if vol is not None:
                affine, md = meta_tuple
                metadata = {"source_file": single_file}
                metadata.update(md)
                # previews
                previews_dir = os.path.join(os.getcwd(), "uploaded_cases", "previews")
                _ensure_dir(previews_dir)
                previews = _make_preview_images(vol, previews_dir, prefix=Path(single_file).stem)
                hu_stats = {}
                try:
                    hu_stats = {"mean": float(np.nanmean(vol)), "min": float(np.nanmin(vol)), "max": float(np.nanmax(vol))}
                except Exception:
                    hu_stats = {}
                if cleanup_tmp:
                    shutil.rmtree(cleanup_tmp, ignore_errors=True)
                return {"volume": vol, "affine": affine, "metadata": metadata, "hu_stats": hu_stats, "preview": previews, "warnings": warnings}

        # collect DICOM files in folder
        dcm_files = _collect_dcm_files(folder)
        if not dcm_files:
            # try dcm2niix on folder
            nifti_path = _dcm2niix_conversion(folder)
            if nifti_path:
                img = nib.load(nifti_path)
                vol = img.get_fdata().astype(np.float32)
                affine = img.affine
                metadata = {"source_nifti": nifti_path}
                previews_dir = os.path.join(os.getcwd(), "uploaded_cases", "previews")
                _ensure_dir(previews_dir)
                previews = _make_preview_images(vol, previews_dir, prefix=Path(nifti_path).stem)
                hu_stats = {}
                try:
                    hu_stats = {"mean": float(np.nanmean(vol)), "min": float(np.nanmin(vol)), "max": float(np.nanmax(vol))}
                except Exception:
                    hu_stats = {}
                if cleanup_tmp:
                    shutil.rmtree(cleanup_tmp, ignore_errors=True)
                return {"volume": vol, "affine": affine, "metadata": metadata, "hu_stats": hu_stats, "preview": previews, "warnings": warnings}
            else:
                if cleanup_tmp:
                    shutil.rmtree(cleanup_tmp, ignore_errors=True)
                raise RuntimeError("No DICOM files found and dcm2niix conversion failed.")

        # try pydicom decoding stack
        vol_meta = _try_pydicom_stack(dcm_files)
        if vol_meta[0] is not None:
            vol, meta_tuple = vol_meta
            affine, md = meta_tuple
            metadata = {"n_slices": vol.shape[0], **md}
            # preview generation
            previews_dir = os.path.join(os.getcwd(), "uploaded_cases", "previews")
            _ensure_dir(previews_dir)
            previews = _make_preview_images(vol, previews_dir, prefix=Path(dcm_files[0]).stem)
            hu_stats = {}
            try:
                hu_stats = {"mean": float(np.nanmean(vol)), "min": float(np.nanmin(vol)), "max": float(np.nanmax(vol))}
            except Exception:
                hu_stats = {}
            if cleanup_tmp:
                shutil.rmtree(cleanup_tmp, ignore_errors=True)
            return {"volume": vol, "affine": affine, "metadata": metadata, "hu_stats": hu_stats, "preview": previews, "warnings": warnings + metadata.get("warnings", [])}

        # fallback: use dcm2niix
        nifti_path = _dcm2niix_conversion(folder)
        if nifti_path:
            img = nib.load(nifti_path)
            vol = img.get_fdata().astype(np.float32)
            affine = img.affine
            metadata = {"source_nifti": nifti_path}
            previews_dir = os.path.join(os.getcwd(), "uploaded_cases", "previews")
            _ensure_dir(previews_dir)
            previews = _make_preview_images(vol, previews_dir, prefix=Path(nifti_path).stem)
            hu_stats = {}
            try:
                hu_stats = {"mean": float(np.nanmean(vol)), "min": float(np.nanmin(vol)), "max": float(np.nanmax(vol))}
            except Exception:
                hu_stats = {}
            if cleanup_tmp:
                shutil.rmtree(cleanup_tmp, ignore_errors=True)
            return {"volume": vol, "affine": affine, "metadata": metadata, "hu_stats": hu_stats, "preview": previews, "warnings": warnings}

        # final fail
        if cleanup_tmp:
            shutil.rmtree(cleanup_tmp, ignore_errors=True)
        raise RuntimeError("Failed to decode DICOM series (all methods).")
    except Exception as e:
        if cleanup_tmp:
            shutil.rmtree(cleanup_tmp, ignore_errors=True)
        raise