# cbct_measurements.py
"""
CBCT measurement utilities (conservative & explainable)

Design goals:
- Provide useful, reproducible, conservative measurements without overclaiming.
- If required spatial metadata is missing, mark results as 'approximate' and include warnings.
- Fast, numpy-only implementations (no heavy segmentation).
- Returns: (measurements_dict, warnings_list) and a human-readable summary string.

Primary functions:
- get_voxel_size_mm(cbct_result) -> (dz, dy, dx, approx_flag, warnings)
- basic_volume_stats(cbct_result) -> dict
- estimate_bone_density(cbct_result, roi_center=None, roi_size_mm=5) -> list of ROI results
- estimate_ridge_height_candidates(cbct_result, arch='both', sample_count=9) -> list of candidate site measures
- detect_periapical_candidates(cbct_result, min_vol_mm3=8) -> list of candidate lesion descriptors (very conservative)
- build_measurements_summary(cbct_result, retrieval_snippets=[]) -> text summary (used by prompts)
- helper: mm_from_voxels(n_voxels, voxel_size_mm)

SAMPLE_TEMPLATE_PATH is provided as a reference to one template file uploaded earlier in this conversation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import math
import logging
from pathlib import Path

logger = logging.getLogger("cbct_measurements")
logger.setLevel(logging.INFO)

# A sample template path from conversation history (replace with a local path if needed)
SAMPLE_TEMPLATE_PATH = "/mnt/data/Mr. Renin Martin report- CBCT Full Skull (11.09.2024).docx"


# -------------------------
# Helpers
# -------------------------
def _safe_get_volume(cbct_result: Dict[str, Any]) -> Optional[np.ndarray]:
    vol = cbct_result.get("volume", None)
    if vol is None:
        return None
    if not isinstance(vol, np.ndarray):
        try:
            vol = np.asarray(vol, dtype=np.float32)
        except Exception:
            return None
    return vol


def mm_from_voxels(n_voxels: float, voxel_mm: float) -> float:
    return float(n_voxels) * float(voxel_mm)


# -------------------------
# Voxel size extraction
# -------------------------
def get_voxel_size_mm(cbct_result: Dict[str, Any]) -> Tuple[Tuple[float, float, float], bool, List[str]]:
    """
    Return voxel size (dz, dy, dx) in mm, approx_flag (True if approximate fallback used), warnings
    - Prefer explicit metadata keys 'pixel_spacing' and 'slice_thickness' from loader
    - If missing or inconsistent, fallback to default_voxel_mm (0.4 mm) but mark approx_flag
    """
    warnings = []
    md = cbct_result.get("metadata", {}) or {}
    # attempt to read pixel spacing and slice thickness
    slice_thickness = md.get("slice_thickness", None)  # assumed mm
    pixel_spacing = md.get("pixel_spacing", None)  # may be None
    # Some metadata packs pixel spacing as tuple/list
    if isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 1:
        try:
            pixel_spacing = float(pixel_spacing[0])
        except Exception:
            pixel_spacing = None

    approx = False
    # If none present, fallback
    default_voxel_mm = 0.4  # conservative fallback (typical CBCT)
    if slice_thickness is None and pixel_spacing is None:
        warnings.append(f"No pixel spacing or slice_thickness in metadata; using fallback {default_voxel_mm} mm (approx).")
        dz = dy = dx = default_voxel_mm
        approx = True
    else:
        # Use slice_thickness as dz; pixel_spacing as dy/dx if available
        try:
            dz = float(slice_thickness) if slice_thickness is not None else default_voxel_mm
        except Exception:
            dz = default_voxel_mm
            warnings.append(f"Invalid slice_thickness; using fallback {default_voxel_mm} mm.")
            approx = True
        try:
            if pixel_spacing is not None:
                dy = dx = float(pixel_spacing)
            else:
                dy = dx = dz  # assume isotropic if pixel spacing missing
                if slice_thickness is not None:
                    warnings.append("Pixel spacing missing; assuming isotropic voxels equal to slice_thickness.")
                    approx = True
        except Exception:
            dy = dx = default_voxel_mm
            warnings.append(f"Invalid pixel spacing; using fallback {default_voxel_mm} mm.")
            approx = True

    return (dz, dy, dx), approx, warnings


# -------------------------
# Basic volume stats
# -------------------------
def basic_volume_stats(cbct_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return dictionary of basic stats:
    - shape (Z,Y,X)
    - hu_stats: mean/min/max
    - percentiles (5th,50th,95th)
    - approximate flag for missing spacing
    """
    vol = _safe_get_volume(cbct_result)
    if vol is None:
        return {"error": "volume_missing"}

    hu = cbct_result.get("hu_stats", {}) or {}
    # compute percentiles if not provided
    try:
        arr = vol.astype(np.float32)
        p5 = float(np.percentile(arr, 5))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        hu_stats = {
            "mean": float(np.nanmean(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "p5": p5,
            "p50": p50,
            "p95": p95
        }
    except Exception as e:
        logger.exception("basic_volume_stats failed: %s", e)
        hu_stats = hu or {}

    (dz, dy, dx), approx, vs_warnings = get_voxel_size_mm(cbct_result)
    meta = {
        "shape": vol.shape,
        "voxel_size_mm": {"dz": dz, "dy": dy, "dx": dx},
        "voxel_size_approx": bool(approx),
        "hu_stats": hu_stats,
        "warnings": vs_warnings
    }
    return meta


# -------------------------
# Bone density estimate (ROI sampling)
# -------------------------
def estimate_bone_density(cbct_result: Dict[str, Any], roi_centers_vox: Optional[List[Tuple[int,int,int]]] = None, roi_size_mm: float = 5.0) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Estimate mean HU in small cubic ROIs.

    Parameters:
    - roi_centers_vox: list of (z,y,x) voxel centers. If None, function will sample a set of canonical candidate sites across the arch:
        - midline center
        - left/right quadrants at mid Z
    - roi_size_mm: cube size in mm (default 5 mm)
    Returns:
    - list of dicts for each ROI: {center_vox, center_mm, size_mm, mean_hu, min_hu, max_hu, voxels_sampled}
    - warnings list
    """
    warnings = []
    vol = _safe_get_volume(cbct_result)
    if vol is None:
        return [], ["No volume available"]

    (dz, dy, dx), approx, vs_warnings = get_voxel_size_mm(cbct_result)
    warnings.extend(vs_warnings)
    # compute roi in voxels (use ceil)
    vz = max(1, int(math.ceil(roi_size_mm / dz)))
    vy = max(1, int(math.ceil(roi_size_mm / dy)))
    vx = max(1, int(math.ceil(roi_size_mm / dx)))

    shape = vol.shape
    centers = []
    if roi_centers_vox:
        centers = roi_centers_vox
    else:
        # default sampling: center, left mid, right mid at mid Z
        zc = shape[0] // 2
        yc = shape[1] // 2
        xc = shape[2] // 2
        centers = [
            (zc, yc, xc),
            (zc, int(yc*0.6), int(xc*0.4)),  # left-ish
            (zc, int(yc*0.6), int(xc*1.6) if int(xc*1.6) < shape[2] else int(xc*1.4)),  # right-ish (bounded)
        ]

    results = []
    for c in centers:
        cz, cy, cx = c
        # bound indices
        z0 = max(0, cz - vz//2); z1 = min(shape[0], cz + vz//2 + 1)
        y0 = max(0, cy - vy//2); y1 = min(shape[1], cy + vy//2 + 1)
        x0 = max(0, cx - vx//2); x1 = min(shape[2], cx + vx//2 + 1)
        sub = vol[z0:z1, y0:y1, x0:x1]
        if sub.size == 0:
            warnings.append(f"ROI at {c} produced empty sample; skipped.")
            continue
        mean_hu = float(np.nanmean(sub))
        min_hu = float(np.nanmin(sub))
        max_hu = float(np.nanmax(sub))
        vox_count = int(sub.size)
        center_mm = (cz*dz, cy*dy, cx*dx)
        results.append({
            "center_vox": (int(cz), int(cy), int(cx)),
            "center_mm": tuple([float(round(v,3)) for v in center_mm]),
            "size_mm": float(roi_size_mm),
            "mean_hu": mean_hu,
            "min_hu": min_hu,
            "max_hu": max_hu,
            "voxels_sampled": vox_count,
            "voxel_size_mm": {"dz": dz, "dy": dy, "dx": dx},
            "voxel_size_approx": bool(approx)
        })
    return results, warnings


# -------------------------
# Ridge height candidate estimation (coarse)
# -------------------------
def estimate_ridge_height_candidates(cbct_result: Dict[str, Any], arch: str = "both", sample_count: int = 9) -> Tuple[List[Dict[str,Any]], List[str]]:
    """
    Coarse estimation of alveolar ridge height candidates across the arch.
    Method (conservative & explainable):
    - Use central axial slab (±5 slices) to estimate bone surface (threshold > bone_HU_threshold)
    - For each sample column across X (sample_count evenly spaced), compute topmost bone voxel (from superior) and bottommost bone voxel (inferior) -> approximate height
    - Return list of candidate positions with measured height in mm and a confidence score (0..1)
    Notes:
    - This is an approximate heuristic: results are only valid when bone surface is continuous and well-resolved.
    - If metadata is missing or bone not found, function returns empty list and warnings.
    """
    warnings = []
    vol = _safe_get_volume(cbct_result)
    if vol is None:
        return [], ["No volume available"]

    (dz, dy, dx), approx, vs_warnings = get_voxel_size_mm(cbct_result)
    warnings.extend(vs_warnings)

    z, y, x = vol.shape

    # choose axial slab: middle 11 slices (if available)
    slab_half = min(5, z//4)
    mid = z//2
    slab = vol[max(0, mid - slab_half): min(z, mid + slab_half + 1), :, :]

    # threshold for bone — conservative threshold (CBCT vendor dependent)
    bone_thresh = 300  # heuristic
    # compute bone mask for slab (2D projection: max across slab)
    proj = np.max(slab, axis=0)
    bone_mask = proj >= bone_thresh
    if bone_mask.sum() < 50:
        warnings.append("Bone mask sparse in central slab — ridge height measurement unreliable.")
        # return empty to avoid false results
        return [], warnings

    # sample columns along X axis (across left-right) at mid Y
    sample_x = np.linspace(10, x-10, num=sample_count, dtype=int)
    results = []
    for sx in sample_x:
        # take vertical column through Y dimension at sx (project across small width)
        column = proj[:, max(0, sx-2):min(x, sx+3)]
        # compute crest location: topmost row index where bone present
        rows = np.where(np.max(column, axis=1) >= bone_thresh)[0]
        if len(rows) == 0:
            # no bone in this column
            continue
        crest_row = int(rows[0])  # topmost bone voxel row index (0 = superior-most)
        # find inferior-most bone in a vertical window below crest (to measure thickness)
        # compute column along Z dimension at this x using central slab aggregated in Z
        # We approximate height by searching along superior-inferior axis in the whole volume for continuous bone
        # Map crest_row (in projection) back to volume Y index
        cy = crest_row
        # Now find superior-most bone voxel in full volume at this x, y column
        ys = max(0, cy-5); ye = min(y, cy+6)
        # find index along Z of first slice where intensity at (z, cy, sx) >= bone_thresh
        top_z_indices = [zi for zi in range(z) if vol[zi, cy, sx] >= bone_thresh]
        if not top_z_indices:
            continue
        top_z = top_z_indices[0]
        # find bottom-most bone index along Z in a far inferior region (coarse)
        bottom_z_indices = [zi for zi in range(z-1, -1, -1) if vol[zi, cy, sx] >= bone_thresh]
        if not bottom_z_indices:
            continue
        bottom_z = bottom_z_indices[0]
        height_vox = max(0, bottom_z - top_z + 1)
        height_mm = mm_from_voxels(height_vox, dz)
        # compute local confidence: proportion of bone voxels in a small neighborhood
        local_mask = vol[:, max(0,cy-2):min(y,cy+3), max(0,sx-2):min(x,sx+3)]
        bone_frac = float(np.sum(local_mask >= bone_thresh)) / float(local_mask.size)
        confidence = min(1.0, bone_frac * 50.0)  # scaled
        results.append({
            "sample_x": int(sx),
            "crest_row_y": int(cy),
            "top_z_slice": int(top_z),
            "bottom_z_slice": int(bottom_z),
            "height_voxels": int(height_vox),
            "height_mm": float(round(height_mm,3)),
            "confidence": float(round(confidence, 2)),
            "voxel_size_mm": {"dz": dz, "dy": dy, "dx": dx},
            "voxel_size_approx": bool(approx)
        })

    if not results:
        warnings.append("Ridge height detection found no reliable columns; results unavailable.")
    return results, warnings


# -------------------------
# Periapical candidate detection (conservative)
# -------------------------
def detect_periapical_candidates(cbct_result: Dict[str,Any], min_vol_mm3: float = 8.0) -> Tuple[List[Dict[str,Any]], List[str]]:
    """
    Conservative detection of low-density pockets that may represent periapical lesions.
    Steps:
    - Threshold at low HU (<= 50) to find radiolucent regions
    - Small morphological filtering using voxel-count thresholds
    - Compute bounding boxes in voxels -> convert to mm
    - Only return candidates whose estimated volume >= min_vol_mm3
    Note: This is a heuristic; false positives/negatives may occur. Results include confidence (low/medium).
    """
    warnings = []
    vol = _safe_get_volume(cbct_result)
    if vol is None:
        return [], ["No volume available"]

    (dz, dy, dx), approx, vs_warnings = get_voxel_size_mm(cbct_result)
    warnings.extend(vs_warnings)

    # threshold
    lucent_thresh = 50  # heuristic
    try:
        mask = vol <= lucent_thresh
        # label connected components in 3D (simple flood-fill) - implement iterative scan for small volumes
        visited = np.zeros_like(mask, dtype=bool)
        candidates = []
        zmax, ymax, xmax = vol.shape
        # scan through mask and flood fill small components (limit for speed)
        for zi in range(zmax):
            for yi in range(0, ymax, max(1, ymax//50)):  # coarse sampling stride for speed
                for xi in range(0, xmax, max(1, xmax//50)):
                    if not mask[zi, yi, xi] or visited[zi, yi, xi]:
                        continue
                    # flood-fill BFS limited to a max voxel count
                    stack = [(zi, yi, xi)]
                    comp = []
                    visited[zi, yi, xi] = True
                    while stack and len(comp) < 200000:  # cap for safety
                        cz, cy, cx = stack.pop()
                        comp.append((cz, cy, cx))
                        # neighbors 6-connectivity
                        for dzn, dyn, dxn in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                            nz, ny, nx = cz+dzn, cy+dyn, cx+dxn
                            if 0 <= nz < zmax and 0 <= ny < ymax and 0 <= nx < xmax:
                                if mask[nz, ny, nx] and not visited[nz, ny, nx]:
                                    visited[nz, ny, nx] = True
                                    stack.append((nz, ny, nx))
                    if not comp:
                        continue
                    vox_count = len(comp)
                    vol_mm3 = vox_count * dz * dy * dx
                    if vol_mm3 < min_vol_mm3:
                        continue
                    # bounding box
                    zs = [c[0] for c in comp]; ys = [c[1] for c in comp]; xs = [c[2] for c in comp]
                    bbox = {
                        "z0": int(min(zs)), "z1": int(max(zs)),
                        "y0": int(min(ys)), "y1": int(max(ys)),
                        "x0": int(min(xs)), "x1": int(max(xs))
                    }
                    bbox_mm = {
                        "z_mm": float(round((bbox["z1"]-bbox["z0"]+1)*dz,3)),
                        "y_mm": float(round((bbox["y1"]-bbox["y0"]+1)*dy,3)),
                        "x_mm": float(round((bbox["x1"]-bbox["x0"]+1)*dx,3)),
                    }
                    mean_hu = float(np.mean([vol[c] for c in comp]))
                    # conservative confidence: larger volumes and lower mean HU increase confidence
                    conf = min(1.0, max(0.1, (min_vol_mm3/vol_mm3) * 0.5 + (50 - mean_hu)/200.0))
                    candidates.append({
                        "voxels": vox_count,
                        "volume_mm3": float(round(vol_mm3,3)),
                        "bbox_vox": bbox,
                        "bbox_mm": bbox_mm,
                        "mean_hu": float(round(mean_hu,2)),
                        "confidence": float(round(max(0.05, min(1.0, conf)), 2)),
                        "voxel_size_mm": {"dz": dz, "dy": dy, "dx": dx},
                        "voxel_size_approx": bool(approx)
                    })
        # sort by volume desc
        candidates = sorted(candidates, key=lambda x: x["volume_mm3"], reverse=True)
        if not candidates:
            warnings.append("No periapical-size lucent regions detected above threshold/size.")
        return candidates, warnings
    except Exception as e:
        logger.exception("detect_periapical_candidates failed: %s", e)
        return [], [f"Exception during detection: {e}"]


# -------------------------
# Build the summary text for prompts
# -------------------------
def build_measurements_summary(cbct_result: Dict[str, Any], retrieval_snippets: Optional[List[str]] = None) -> str:
    """
    Construct a concise measurement summary string listing:
    - volume shape & voxel size (noting approximation if any)
    - HU stats
    - bone density sample results
    - ridge height candidate list (top 3 with confidence)
    - periapical candidates count & top sizes
    This is intentionally conservative and phrased for LLM consumption.
    """
    lines = []
    warnings = []
    vol = _safe_get_volume(cbct_result)
    if vol is None:
        return "No volumetric data available."

    # basic stats
    stats = basic_volume_stats(cbct_result)
    shape = stats.get("shape")
    voxel_info = stats.get("voxel_size_mm", {})
    approx_flag = stats.get("voxel_size_approx", False)
    hu = stats.get("hu_stats", {})
    lines.append(f"- Volume shape (Z,Y,X): {shape}; voxel size (dz,dy,dx) mm: ({voxel_info.get('dz')},{voxel_info.get('dy')},{voxel_info.get('dx')}) {'(approx)' if approx_flag else ''}")
    if hu:
        lines.append(f"- HU mean: {hu.get('mean'):.1f}; min: {hu.get('min'):.1f}; max: {hu.get('max'):.1f}")

    # bone density sampling (3 ROIs)
    rois, r_warnings = estimate_bone_density(cbct_result, roi_centers_vox=None, roi_size_mm=5.0)
    warnings.extend(r_warnings)
    if rois:
        for i, r in enumerate(rois):
            lines.append(f"- ROI {i+1} at {r['center_mm']}: mean HU {r['mean_hu']:.1f} (n={r['voxels_sampled']})")

    # ridge height candidates
    ridge_results, ridge_warnings = estimate_ridge_height_candidates(cbct_result, arch="both", sample_count=7)
    warnings.extend(ridge_warnings)
    if ridge_results:
        # sort by confidence descending
        sorted_ridge = sorted(ridge_results, key=lambda x: x["confidence"], reverse=True)
        top3 = sorted_ridge[:3]
        for i, rr in enumerate(top3):
            lines.append(f"- Ridge candidate {i+1}: height {rr['height_mm']:.2f} mm (confidence {rr['confidence']:.2f}) at sample_x {rr['sample_x']}")

    # periapical
    peris, per_warnings = detect_periapical_candidates(cbct_result, min_vol_mm3=8.0)
    warnings.extend(per_warnings)
    if peris:
        # top 2
        for i, p in enumerate(peris[:2]):
            lines.append(f"- Periapical candidate {i+1}: vol {p['volume_mm3']:.1f} mm³; mean HU {p['mean_hu']:.1f}; confidence {p['confidence']:.2f}")

    # append retrieval snippets hint (if present)
    if retrieval_snippets:
        lines.append(f"- Evidence snippets available: {len(retrieval_snippets)}")

    # include warnings in final string
    if warnings:
        lines.append("- WARNINGS:")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


# -------------------------
# Small CLI for quick test
# -------------------------
if __name__ == "__main__":
    import argparse
    from dicom_reader import load_cbct_from_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcm", type=str, default="uploaded_cases/temp_extracted.dcm")
    args = parser.parse_args()
    print("Loading:", args.dcm)
    cbct = load_cbct_from_path(args.dcm)
    print("Basic stats:")
    print(basic_volume_stats(cbct))
    print("\nROI densities:")
    rois, warns = estimate_bone_density(cbct)
    print(rois)
    print("Warnings:", warns)
    print("\nRidge candidates (coarse):")
    r, rw = estimate_ridge_height_candidates(cbct)
    print(r)
    print("Ridge warnings:", rw)
    print("\nPeriapical candidates (conservative):")
    p, pw = detect_periapical_candidates(cbct)
    print(p[:3])
    print("Periapical warnings:", pw)
