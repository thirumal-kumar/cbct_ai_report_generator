# condition_detector.py
"""
Condition detector for CBCT volumes.

- Input: cbct_result (the dict returned by dicom_reader.load_cbct_from_path)
- Uses:
    - cbct_measurements.build_measurements_summary (summary text)
    - report_templates.list_conditions() + snippets
- Output: list of suggestions: [{ "condition": str, "score": float, "reason": str }, ...]
- Lightweight, deterministic heuristics with explainable reasons.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import re
import math
import logging

# Local imports (existing modules in your project)
try:
    from report_templates import list_conditions, get_template_snippet
except Exception:
    # defensive fallback for testability
    def list_conditions(): return []
    def get_template_snippet(c): return None

try:
    import cbct_measurements as measurements
except Exception:
    # minimal stub if not present
    def build_measurements_summary(cbct, snippets):
        return ""
    measurements = type("MStub", (), {"build_measurements_summary": staticmethod(lambda a,b: "")})

logger = logging.getLogger("condition_detector")
logger.setLevel(logging.INFO)


# --------------------------
# Utility scoring helpers
# --------------------------
def _score_range(value: float, good_min: float, good_max: float, margin: float = 0.2) -> float:
    """
    Return a score [0,1] for how well 'value' fits within [good_min, good_max].
    The margin expands the desirable window by fraction (default 20%).
    """
    if good_min is None or good_max is None:
        return 0.0
    low = good_min - abs(good_min) * margin
    high = good_max + abs(good_max) * margin
    if value < low or value > high:
        # outside range; smooth decline
        dist = min(abs(value - low), abs(value - high))
        width = max(1.0, abs(high - low))
        return max(0.0, 1.0 - (dist / (3.0 * width)))
    return 1.0


def _normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vmax = max(d.values())
    if vmax <= 0:
        return {k: 0.0 for k in d}
    return {k: float(v / vmax) for k, v in d.items()}


# --------------------------
# Heuristic detectors
# --------------------------
def _detect_implant_by_metal_voxels(vol: np.ndarray, threshold: float = 2500.0) -> Tuple[bool, float]:
    """
    Implants/screws (metal) often register at very high voxel values in CBCT (vendor dependent).
    We detect fraction of voxels above a high threshold. Returns (present, fraction_score).
    """
    if vol is None:
        return False, 0.0
    try:
        total = vol.size
        cnt = float(np.sum(vol >= threshold))
        frac = cnt / float(total)
        # small fractions (e.g. 1e-5) can indicate metal; scale score to [0,1]
        score = min(1.0, math.log1p(frac * 1e6) / 6.0) if frac > 0 else 0.0
        return (frac > 1e-6), score
    except Exception as e:
        logger.exception("_detect_implant_by_metal_voxels failed: %s", e)
        return False, 0.0


def _detect_full_vs_sectional(vol_shape: Tuple[int, ...]) -> Tuple[str, float]:
    """
    Heuristic:
    - Full skull / full arch scans tend to have more slices (Z dimension) and larger XY.
    - Sectional scans have small Z (<200) and smaller FOV.
    Returns ('full'|'sectional'|'unknown', confidence)
    """
    if not vol_shape:
        return "unknown", 0.0
    z, y, x = vol_shape
    # thresholds tuned for typical CBCT scans
    if z >= 300 or max(x, y) >= 512:
        return "full", min(1.0, (z / 400.0))
    if z < 200 and max(x, y) <= 512:
        return "sectional", 0.8
    return "unknown", 0.3


def _detect_sinus_involvement(vol: np.ndarray, metadata: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Very simple heuristic: mean intensity in upper slices (approx superior region) higher --> possible sinus mucosal thickening.
    This is approximate and conservative.
    """
    if vol is None:
        return False, 0.0
    try:
        z, y, x = vol.shape
        # examine top 25% slices (assuming axial slices z from inferior->superior; vendor dependent)
        top = max(1, int(z * 0.25))
        region = vol[-top:, :, :]
        mean = float(np.nanmean(region))
        # mucosal thickening often increases soft-tissue density; use heuristic thresholds
        if mean > 200 and mean < 1500:
            # score relative to common CBCT soft-tissue region
            score = min(1.0, (mean - 200) / 800.0)
            return True, score
        return False, 0.0
    except Exception as e:
        logger.exception("_detect_sinus_involvement failed: %s", e)
        return False, 0.0


def _detect_periapical_candidates(vol: np.ndarray, metadata: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Rough method: find small low-density blobs near tooth-bearing regions.
    This is a rough heuristic and returns low confidence detection.
    """
    if vol is None:
        return False, 0.0
    try:
        # low density patch detection: fraction of voxels under a low HU threshold in central 50% of XY
        z, y, x = vol.shape
        cz = vol[:, y//4: 3*y//4, x//4: 3*x//4]
        low_frac = float(np.sum(cz <= 40)) / float(cz.size)
        # If there are localized pockets, low_frac may be small but non-zero
        score = min(1.0, math.log1p(low_frac * 1e3) / 3.0) if low_frac > 0 else 0.0
        return (low_frac > 1e-4), score
    except Exception as e:
        logger.exception("_detect_periapical_candidates failed: %s", e)
        return False, 0.0


def _detect_fracture_candidates(vol: np.ndarray) -> Tuple[bool, float]:
    """
    Heuristic: fractures often produce linear high-contrast discontinuities in cortical bone.
    Use simple gradient-based measure to detect high-gradient voxels concentrated in thin shell.
    Conservative and low-confidence.
    """
    if vol is None:
        return False, 0.0
    try:
        # compute central slice gradient magnitude
        z, y, x = vol.shape
        slice_idx = z // 2
        sl = vol[slice_idx, :, :].astype(np.float32)
        gy, gx = np.gradient(sl)
        grad = np.sqrt(gy**2 + gx**2)
        high_grad_frac = float(np.sum(grad > (np.mean(grad) + 3*np.std(grad)))) / float(grad.size)
        score = min(1.0, math.log1p(high_grad_frac * 1e4) / 4.0) if high_grad_frac > 0 else 0.0
        return (high_grad_frac > 1e-4), score
    except Exception as e:
        logger.exception("_detect_fracture_candidates failed: %s", e)
        return False, 0.0


def _detect_periodontitis(vol: np.ndarray, metadata: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Heuristic: generalized bone loss -> reduced cortical thickness across alveolar region.
    Very approximate: examine mean HU across central rows and compare to higher expected cortical HU.
    """
    if vol is None:
        return False, 0.0
    try:
        z, y, x = vol.shape
        # sample mid-slices and mid-column ROI
        sample = vol[z//2 - 5: z//2 + 5, y//3: 2*y//3, x//4:3*x//4]
        mean_hu = float(np.nanmean(sample))
        # bone mean approximate thresholds depend on scanner; use heuristic thresholds
        if mean_hu < 300:  # lower mean suggests reduced dense bone in region
            score = min(1.0, (300 - mean_hu) / 300.0)
            return True, score
        return False, 0.0
    except Exception as e:
        logger.exception("_detect_periodontitis failed: %s", e)
        return False, 0.0


# --------------------------
# Template-keyword matching
# --------------------------
def _keyword_score_against_snippet(snippet: str, keywords: List[str]) -> float:
    if not snippet:
        return 0.0
    s = snippet.lower()
    score = 0.0
    for k in keywords:
        if k.lower() in s:
            score += 1.0
    # normalize by number of keywords
    return float(score) / max(1.0, float(len(keywords)))


def _score_templates_for_condition(condition: str, cbct_summary_text: str) -> float:
    """
    Score how well the condition's templates match the cbct_summary_text (which includes measurements & metadata).
    Uses the template snippet text as additional context.
    """
    snippet = get_template_snippet(condition) or ""
    # keywords extracted from the condition name and snippet
    key_candidates = [condition]
    # common keywords to bump certain conditions
    common_map = {
        "Implant": ["implant", "ridge", "sinus", "implant planning", "bone height"],
        "Periapical": ["periapical", "apical", "radiolucency"],
        "Periodontitis": ["periodontitis", "bone loss", "furcation", "attachment loss"],
        "Fractured": ["fracture", "fractured", "crack"],
        "TMJ": ["tmj", "condyle", "joint"],
        "Cyst": ["cyst", "radiolucent", "expansile"],
        "Sinus": ["sinus", "maxillary sinus", "mucosal"],
        "Impacted": ["impacted", "impaction", "third molar", "impacted tooth"],
        "Full Skull": ["full skull", "panfacial", "full face"],
    }
    for k, kws in common_map.items():
        if k.lower() in condition.lower():
            key_candidates.extend(kws)
    # also add words from the cbct_summary_text (split unique tokens)
    tokens = re.findall(r"[A-Za-z0-9]{3,}", cbct_summary_text.lower())
    # take top unique tokens (avoid huge lists)
    tokens = list(dict.fromkeys(tokens))[:40]
    key_candidates.extend(tokens)
    # compute keyword overlap with template snippet
    ks = _keyword_score_against_snippet(snippet, key_candidates)
    return ks


# --------------------------
# Main API: detect_conditions
# --------------------------
def detect_conditions(cbct_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Input: cbct_result returned by dicom_reader.load_cbct_from_path()
    Output: ranked suggestions: [{"condition":str, "score":0..1, "reason":str}, ...]
    """
    suggestions: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}

    volume = cbct_result.get("volume")
    metadata = cbct_result.get("metadata", {})
    hu_stats = cbct_result.get("hu_stats", {})

    vol_shape = None
    if volume is not None:
        vol_shape = volume.shape

    # 1) structural heuristic: full vs sectional
    fs_type, fs_conf = _detect_full_vs_sectional(vol_shape)
    if fs_type == "full":
        suggestions["Full Skull"] = fs_conf
        reasons.setdefault("Full Skull", []).append(f"Large volume (shape={vol_shape}) suggests a full/panfacial scan.")
    elif fs_type == "sectional":
        # find matching sectional keys
        for key in list_conditions():
            if "sectional" in key.lower() or "sectional" in key.lower():
                suggestions[key] = max(suggestions.get(key, 0.0), fs_conf * 0.9)
                reasons.setdefault(key, []).append(f"Volume small (shape={vol_shape}) suggests sectional scan.")

    # 2) metal/implant detection
    impl_present, impl_score = _detect_implant_by_metal_voxels(volume)
    if impl_present:
        # propose likely implant-related templates
        for cond in list_conditions():
            if "implant" in cond.lower() or "screw" in cond.lower() or "surgical" in cond.lower():
                suggestions[cond] = max(suggestions.get(cond, 0.0), impl_score * 1.2)
                reasons.setdefault(cond, []).append(f"High-value voxels fraction indicates metal/implant (score={impl_score:.2f}).")

    # 3) sinus involvement heuristic
    sinus_present, sinus_score = _detect_sinus_involvement(volume, metadata)
    if sinus_present:
        for cond in list_conditions():
            if "sinus" in cond.lower() or "maxillary" in cond.lower():
                suggestions[cond] = max(suggestions.get(cond, 0.0), sinus_score * 1.1)
                reasons.setdefault(cond, []).append(f"Upper-slice mean intensity suggests sinus mucosal thickening (score={sinus_score:.2f}).")

    # 4) periapical candidates
    pa_present, pa_score = _detect_periapical_candidates(volume, metadata)
    if pa_present:
        for cond in list_conditions():
            if "periapical" in cond.lower() or "apical" in cond.lower():
                suggestions[cond] = max(suggestions.get(cond, 0.0), pa_score * 1.0)
                reasons.setdefault(cond, []).append(f"Low-density focal pockets detected (periapical candidate score={pa_score:.2f}).")

    # 5) fracture
    fr_present, fr_score = _detect_fracture_candidates(volume)
    if fr_present:
        for cond in list_conditions():
            if "fracture" in cond.lower() or "fractured" in cond.lower():
                suggestions[cond] = max(suggestions.get(cond, 0.0), fr_score * 1.2)
                reasons.setdefault(cond, []).append(f"High-gradient lines in bone section suggest possible fracture (score={fr_score:.2f}).")

    # 6) periodontitis
    per_present, per_score = _detect_periodontitis(volume, metadata)
    if per_present:
        for cond in list_conditions():
            if "periodontitis" in cond.lower() or "periodontal" in cond.lower():
                suggestions[cond] = max(suggestions.get(cond, 0.0), per_score * 1.1)
                reasons.setdefault(cond, []).append(f"Lower bone density in alveolar ROI suggests periodontal bone loss (score={per_score:.2f}).")

    # 7) template snippet matching using measurements summary
    cbct_summary_text = measurements.build_measurements_summary(cbct_result, [])
    for cond in list_conditions():
        ks = _score_templates_for_condition(cond, cbct_summary_text)
        if ks > 0:
            suggestions[cond] = max(suggestions.get(cond, 0.0), ks * 0.9)
            reasons.setdefault(cond, []).append(f"Template snippet overlap score={ks:.2f} against measurements summary.")

    # 8) normalize and build final list
    norm = _normalize_scores(suggestions)
    results = []
    for cond, raw in sorted(norm.items(), key=lambda kv: kv[1], reverse=True):
        if raw <= 0.02:
            continue  # drop near-zero
        results.append({
            "condition": cond,
            "score": float(raw),
            "reason": "; ".join(reasons.get(cond, [])) or "Heuristic match"
        })
    return results


# --------------------------
# Utility: suggest conditions for a file path using dicom_reader
# --------------------------
def suggest_for_filepath(path: str) -> List[Dict[str, Any]]:
    """
    Helper that loads the CBCT via dicom_reader and runs detect_conditions.
    """
    try:
        from dicom_reader import load_cbct_from_path
    except Exception as e:
        raise RuntimeError(f"dicom_reader import failed: {e}")
    cbct = load_cbct_from_path(path)
    return detect_conditions(cbct)


# --------------------------
# CLI / quick test
# --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick test for CBCT condition detector")
    parser.add_argument("--dcm", type=str, default="uploaded_cases/temp_extracted.dcm", help="Path to DICOM (file or folder)")
    args = parser.parse_args()

    print("Loading:", args.dcm)
    try:
        suggestions = suggest_for_filepath(args.dcm)
        if not suggestions:
            print("No confident suggestions. You can still choose manually.")
        else:
            print("Top suggestions (condition, score, reason):")
            for s in suggestions[:10]:
                print(f" - {s['condition']}: {s['score']:.2f} — {s['reason']}")
    except Exception as e:
        print("Detection failed:", e)


# --------------------------
# Example: sample template file path (from upload history)
# --------------------------
# Some downstream tools expect a 'url' or path to a sample template file.
# If you need an example path to pass into a tool call, here is one from the conversation history:
SAMPLE_TEMPLATE_PATH = "/mnt/data/Mr. Renin Martin report- CBCT Full Skull (11.09.2024).docx"
# Note: replace SAMPLE_TEMPLATE_PATH with a local path inside your templates folder when using locally.
