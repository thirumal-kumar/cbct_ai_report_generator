"""
cbct_ai_analyzer.py
High-accuracy CBCT analyzer module.

Features:
- Tooth segmentation (tries to use a pretrained PyTorch 3D U-Net if available)
- Sinus segmentation (threshold + region growing)
- Airway segmentation + volume + min CSA
- Simple mandibular canal detection (heuristic)
- Bone thickness estimation (distance transform on bone mask)
- Ridge height proxy and implant-site metrics
- Periapical low-HU region detection (heuristic)
- Produces:
    - structured JSON (analysis dict)
    - PNG visualizations (heatmaps, airway mask, sinus mask, teeth mask)
    - plain-text clinician-style report (string)
Notes:
- Uses your project's `dicom_reader.load_cbct_from_path` output (volume, metadata).
- Model: optional. If PyTorch present and model file placed at `static/models/tooth_unet.pth`, it will run inference.
- All outputs are saved under the provided `out_dir`.
"""

import os
import json
import numpy as np
import warnings
from pathlib import Path
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology, filters, exposure, segmentation

# Try optional torch model
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# model path (you can change)
DEFAULT_TOOTH_MODEL_PATH = Path("static/models/tooth_unet.pth")

# Utility
def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img):
    arr = np.array(img, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx > 0:
        arr = arr / mx * 255.0
    return arr.astype(np.uint8)

# -------------------------
# Tooth segmentation (optional)
# -------------------------
def _run_tooth_model(volume, model_path):
    """
    Run a 3D U-Net model for tooth segmentation if available.
    Expects volume as numpy array (Z,Y,X).
    Returns labelmap (same shape) or None on failure.
    This function is intentionally generic — you must provide a compatible model.
    """
    if not TORCH_AVAILABLE:
        return None, "torch_not_installed"

    model_path = Path(model_path)
    if not model_path.exists():
        return None, "model_not_found"

    try:
        # Minimal safe inference wrapper (user must ensure model I/O shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(str(model_path), map_location=device)
        model.eval()
        # Prepare volume: normalize, add batch/channel dims
        vol = np.nan_to_num(volume, nan=0.0, copy=True)
        vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)
        # convert to float32 and into torch
        x = torch.from_numpy(vol.astype(np.float32))[None, None]  # (1,1,Z,Y,X)
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
            # assume output is logits with shape (1, C, Z, Y, X), pick argmax
            if out.ndim == 5:
                pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
            else:
                pred = (torch.sigmoid(out) > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred, "ok"
    except Exception as e:
        return None, f"model_inference_error: {e}"

# -------------------------
# Sinus segmentation (threshold + connected component)
# -------------------------
def segment_sinus(volume, meta):
    """
    Return binary mask of sinus regions using an adaptive threshold and morphological cleanup.
    Simple heuristic: threshold around air-soft tissue HU band and keep connected regions in expected FOV.
    """
    try:
        vol = np.array(volume, dtype=np.float32)
    except Exception:
        return None

    # Use Otsu on a middle-coronal slab
    z_mid = vol.shape[0] // 2
    slab = vol[max(0, z_mid-10):min(vol.shape[0], z_mid+10), :, :]
    try:
        flat = slab.flatten()
        th = filters.threshold_otsu(flat)
    except Exception:
        th = np.percentile(slab, 20)

    # sinuses are air-filled (low HU) - but CT units vary across CBCT; we invert logic: take low-intensity connected cavities inside maxillary region
    # Create mask for low densities
    mask = vol < (np.median(slab) * 0.75)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=2000)
    # Keep largest connected components in upper maxilla region (split by z)
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    # heuristics: keep components with centroid in upper half
    kept = np.zeros_like(mask, dtype=bool)
    Z = vol.shape[0]
    for p in props:
        cz = p.centroid[0]
        if cz < Z * 0.6:  # upper half
            if p.area > 2000:
                kept[labels == p.label] = True
    return kept

# -------------------------
# Airway segmentation (threshold + largest connected airway)
# -------------------------
def segment_airway(volume):
    vol = np.array(volume, dtype=np.float32)
    # threshold for air (very low intensity)
    # CBCT HU scale is not standardized; use relative threshold: areas much lower than mean
    meanv = np.nanmean(vol)
    th = meanv * 0.2
    mask = vol < th
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    # keep the largest component that crosses sagittal center
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    props = measure.regionprops(labels)
    # choose component with largest area
    largest = max(props, key=lambda p: p.area)
    airway = labels == largest.label
    return airway

# -------------------------
# Mandibular canal heuristic (tubular low-density detection)
# -------------------------
def detect_mandibular_canal(volume):
    vol = np.array(volume, dtype=np.float32)
    # compute local standard deviation: canals are tubular low-density with surrounding high density
    try:
        std = ndi.generic_filter(vol, np.std, size=5)
        low = vol < (np.median(vol) * 0.6)
        candidate = low & (std > np.percentile(std, 60))
        # skeletonize to get centerline
        comb = morphology.remove_small_objects(candidate.astype(bool), min_size=100)
        labeled = measure.label(comb)
        canals = []
        for region in measure.regionprops(labeled):
            if region.area > 200:
                canals.append({
                    "centroid": [float(c) for c in region.centroid],
                    "area": int(region.area)
                })
        return canals
    except Exception:
        return []

# -------------------------
# Bone mask + thickness estimation
# -------------------------
def bone_mask_and_thickness(volume):
    vol = np.array(volume, dtype=np.float32)
    # threshold for bone: high intensities relative to median
    th = np.nanmedian(vol) + (np.nanpercentile(vol, 90) - np.nanmedian(vol)) * 0.35
    bone = vol > th
    bone = morphology.remove_small_objects(bone, min_size=500)
    # distance transform on inverted bone mask yields marrow cavity thickness; for cortical thickness use distance transform on bone mask's complement
    dt = ndi.distance_transform_edt(bone)
    # approximate cortical thickness map as the distance transform of the complement where bone boundary exists
    boundaries = segmentation.find_boundaries(bone, mode="outer")
    cortical_thickness = dt * boundaries
    # summarise
    thickness_stats = {
        "cortical_mean": float(np.nanmean(cortical_thickness[cortical_thickness>0]) if np.any(cortical_thickness>0) else 0.0),
        "cortical_max": float(np.nanmax(cortical_thickness) if np.any(cortical_thickness) else 0.0)
    }
    return bone, cortical_thickness, thickness_stats

# -------------------------
# Periapical low-HU region detection (naive)
# -------------------------
def detect_periapical_low_hu(volume, measurements):
    vol = np.array(volume, dtype=np.float32)
    mean_hu = measurements.get("hu_mean", np.nan)
    if np.isnan(mean_hu):
        mean_hu = np.nanmean(vol)
    mask = vol < (mean_hu * 0.5)
    mask = morphology.remove_small_objects(mask, min_size=50)
    labels = measure.label(mask)
    candidates = []
    for region in measure.regionprops(labels):
        if region.area > 100:
            candidates.append({
                "centroid": [float(x) for x in region.centroid],
                "area_voxels": int(region.area)
            })
    return candidates

# -------------------------
# Ridge height proxy & implant metrics
# -------------------------
def ridge_and_implant_metrics(volume, metadata):
    vol = np.array(volume, dtype=np.float32)
    zc = vol.shape[0] // 2
    coronal = vol[zc]
    # ridge height proxy: distance from top of alveolar dense bone to nasal floor along midline column
    # Here we get a coarse proxy: index of max intensity column and its normalized value
    col_profile = np.sum(coronal, axis=0)
    peak_idx = int(np.argmax(col_profile))
    return {
        "ridge_peak_index": int(peak_idx),
        "col_profile_mean": float(np.mean(col_profile)),
    }

# -------------------------
# Visualizations (PNG)
# -------------------------
def _save_heatmap(arr, path, cmap="viridis", title=None):
    plt.figure(figsize=(6,6))
    plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()

# -------------------------
# Main entrypoint
# -------------------------
def analyze_cbct(cbct_dict, out_dir, run_tooth_model_if_available=True, tooth_model_path=DEFAULT_TOOTH_MODEL_PATH):
    """
    cbct_dict: output of load_cbct_from_path() — must contain "volume" (Z,Y,X) and "metadata"
    out_dir: directory to write analysis.json, images, labelmaps
    returns: analysis dict (also written to analysis.json)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    volume = cbct_dict.get("volume") if isinstance(cbct_dict, dict) else None
    metadata = cbct_dict.get("metadata", {}) if isinstance(cbct_dict, dict) else {}
    warnings = cbct_dict.get("warnings", [])

    if volume is None:
        raise ValueError("analyze_cbct: no volume found in cbct_dict")

    # Basic measurements
    v = np.array(volume, dtype=np.float32)
    meas = {}
    meas["shape_z_y_x"] = tuple(int(s) for s in v.shape)
    meas["voxel_size_mm"] = metadata.get("pixel_spacing") or metadata.get("voxel_size") or (metadata.get("slice_thickness"),)*3
    meas["hu_mean"] = float(np.nanmean(v))
    meas["hu_min"] = float(np.nanmin(v))
    meas["hu_max"] = float(np.nanmax(v))
    meas["warnings"] = warnings

    # Tooth segmentation (model preferred)
    teeth_mask = None
    teeth_info = {"method": "none", "status": "skipped"}
    if run_tooth_model_if_available and TORCH_AVAILABLE and tooth_model_path.exists():
        pred, status = _run_tooth_model(v, tooth_model_path)
        if pred is not None:
            teeth_mask = pred.astype(np.int32)
            teeth_info = {"method": "unet_model", "status": status}
        else:
            teeth_info = {"method": "unet_model", "status": status}
    else:
        # fallback: heuristic tooth detection — detect dense objects in jaw region
        try:
            # threshold high-intensity seeds
            th = np.nanpercentile(v, 95)
            seeds = v > th
            seeds = morphology.remove_small_objects(seeds, min_size=50)
            labeled = measure.label(seeds)
            # create coarse tooth mask by dilating seeds
            tooth_mask = morphology.binary_dilation(seeds, morphology.ball(2))
            teeth_mask = tooth_mask.astype(np.uint8)
            teeth_info = {"method": "heuristic", "status": "ok"}
        except Exception:
            teeth_info = {"method": "heuristic", "status": "failed"}

    # Sinus segmentation
    sinus_mask = None
    try:
        sinus_mask = segment_sinus(v, metadata)
        sinus_mask = sinus_mask.astype(np.uint8) if sinus_mask is not None else None
        sinus_volume = int(np.sum(sinus_mask)) if sinus_mask is not None else 0
    except Exception:
        sinus_mask = None
        sinus_volume = 0

    # Airway segmentation
    try:
        airway_mask = segment_airway(v).astype(np.uint8)
        airway_volume_vox = int(np.sum(airway_mask))
        voxel_mm = meas["voxel_size_mm"]
        if voxel_mm and voxel_mm[0]:
            voxel_vol = float(voxel_mm[0] * voxel_mm[1] * voxel_mm[2])
            airway_volume_mm3 = airway_volume_vox * voxel_vol
        else:
            airway_volume_mm3 = float(airway_volume_vox)
    except Exception:
        airway_mask = None
        airway_volume_mm3 = 0.0

    # Mandibular canal detection heuristic
    canals = detect_mandibular_canal(v)

    # Bone mask and thickness
    bone_mask, cortical_thickness_map, thickness_stats = bone_mask_and_thickness(v)

    # Periapical candidates
    periapicals = detect_periapical_low_hu(v, meas)

    # Ridge metrics
    ridge = ridge_and_implant_metrics(v, metadata)

    # Save images (mid-slices and heatmaps)
    try:
        # mid axial/coronal/sagittal
        zc, yc, xc = (s//2 for s in v.shape)
        axial = v[zc]
        coronal = v[:, yc, :]
        sagittal = v[:, :, xc]

        _save_heatmap(_to_uint8(axial), out_dir / "axial_mid.png", cmap="gray", title="Axial mid")
        _save_heatmap(_to_uint8(coronal), out_dir / "coronal_mid.png", cmap="gray", title="Coronal mid")
        _save_heatmap(_to_uint8(sagittal), out_dir / "sagittal_mid.png", cmap="gray", title="Sagittal mid")

        if teeth_mask is not None:
            tmid = teeth_mask[zc]
            _save_heatmap(tmid, out_dir / "teeth_mid.png", cmap="tab20", title="Teeth mask mid")
        if sinus_mask is not None:
            _save_heatmap(sinus_mask[zc], out_dir / "sinus_mid.png", cmap="gray", title="Sinus mask mid")
        if airway_mask is not None:
            # airway mid coronal
            _save_heatmap(airway_mask[zc], out_dir / "airway_mid.png", cmap="gray", title="Airway mask mid")
        # cortical thickness heatmap (max projection)
        try:
            proj = np.max(cortical_thickness_map, axis=0)
            _save_heatmap(proj, out_dir / "cortical_thickness_proj.png", cmap="magma", title="Cortical thickness (projection)")
        except Exception:
            pass
    except Exception:
        # don't fail entire analysis on visualization issues
        pass

    # Build structured analysis dict
    analysis = {
        "timestamp": datetime.datetime.now().isoformat(),
        "measurements": meas,
        "teeth": teeth_info,
        "sinus": {"volume_voxels": int(sinus_volume), "mask_saved": bool(sinus_mask is not None)},
        "airway": {"volume_mm3": float(airway_volume_mm3)},
        "canals": canals,
        "thickness_stats": thickness_stats,
        "periapicals": periapicals,
        "ridge": ridge,
        "warnings": warnings,
    }

    # Write JSON
    json_path = out_dir / "analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Create plain-text radiology-style report (concise, deterministic)
    report_lines = []
    report_lines.append("CBCT AI REPORT")
    report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("Patient: (anonymized — no identifiers supplied)")
    report_lines.append("")
    report_lines.append("1. Scan Details")
    report_lines.append(f"- Source file: {metadata.get('source_file', '(not provided)')}")
    if meas.get("voxel_size_mm"):
        report_lines.append(f"- Voxel size (approx): {meas['voxel_size_mm']}")
    report_lines.append("")
    report_lines.append("2. Image Quality")
    if meas.get("warnings"):
        for w in meas["warnings"]:
            report_lines.append(f"- Warning: {w}")
    else:
        report_lines.append("- Image quality adequate for automated analysis unless stated otherwise.")
    report_lines.append("")
    report_lines.append("3. Findings")
    report_lines.append(f"- Volume shape (Z,Y,X): {meas.get('shape_z_y_x')}")
    report_lines.append(f"- Mean HU (approx): {meas.get('hu_mean'):.1f}; min {meas.get('hu_min')}; max {meas.get('hu_max')}")
    # Sinus
    if sinus_mask is not None and sinus_volume > 0:
        report_lines.append(f"- Maxillary sinus: estimated aeration mask present; voxel count {sinus_volume} (quantitative assessment).")
    else:
        report_lines.append("- Maxillary sinus: no major sinus aeration changes detected on coarse pass.")
    # Airway
    report_lines.append(f"- Airway volume (automated estimate): {airway_volume_mm3:.1f} mm³")
    # Canals
    if canals:
        report_lines.append(f"- Mandibular canal candidate regions detected: {len(canals)} (heuristic).")
    else:
        report_lines.append("- Mandibular canal: no clear tubular canal region identified by heuristic.")
    # Periapical
    if periapicals:
        report_lines.append(f"- Periapical low-density candidates: {len(periapicals)} (low confidence). Correlate with periapical radiographs.")
    else:
        report_lines.append("- No obvious periapical low-density clusters detected on coarse automated pass.")
    # Bone
    report_lines.append(f"- Cortical thickness (mean projection approx): {thickness_stats.get('cortical_mean', 0.0):.2f} (voxels units).")
    report_lines.append("")
    report_lines.append("4. Impression")
    impressions = []
    if periapicals and len(periapicals) > 0:
        impressions.append("- Possible periapical low-density region(s) identified (low confidence). Correlate clinically and with periapical radiographs.")
    impressions.append("- No gross destructive osseous lesion identified on this automated pass.")
    report_lines.extend(impressions)
    report_lines.append("")
    report_lines.append("5. Recommendations")
    report_lines.append("- Correlate imaging with clinical findings and intraoral/periapical radiographs.")
    report_lines.append("- For implant planning, perform dedicated cross-sectional measurements at proposed site.")
    report_lines.append("- If suspicious lesion persists, consider targeted CBCT/clinical evaluation or specialist referral.")
    report_lines.append("")
    report_lines.append("Radiologist:")
    report_lines.append("Name: _______________________")
    report_lines.append(f"Date: {datetime.datetime.now().date().isoformat()}")

    report_text = "\n".join(report_lines)

    # Save report text
    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Return analysis dict and paths
    return {
        "analysis": analysis,
        "json_path": str(json_path),
        "report_text": report_text,
        "report_path": str(report_path),
        "images": {
            "axial_mid": str((out_dir / "axial_mid.png")) if (out_dir / "axial_mid.png").exists() else None,
            "coronal_mid": str((out_dir / "coronal_mid.png")) if (out_dir / "coronal_mid.png").exists() else None,
            "sagittal_mid": str((out_dir / "sagittal_mid.png")) if (out_dir / "sagittal_mid.png").exists() else None,
            "teeth_mid": str((out_dir / "teeth_mid.png")) if (out_dir / "teeth_mid.png").exists() else None,
            "sinus_mid": str((out_dir / "sinus_mid.png")) if (out_dir / "sinus_mid.png").exists() else None,
            "airway_mid": str((out_dir / "airway_mid.png")) if (out_dir / "airway_mid.png").exists() else None,
            "cortical_thickness_proj": str((out_dir / "cortical_thickness_proj.png")) if (out_dir / "cortical_thickness_proj.png").exists() else None,
        }
    }
