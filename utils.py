# utils.py
import re
from pathlib import Path
from typing import Optional

def clean_answer(text: str) -> str:
    """Remove internal doc IDs, paths, chunk markers, noise."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"doc-\d+", "", text)
    text = re.sub(r"rag_db/[\w\-/\.]+", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def short_summary_from_metadata(meta: dict) -> str:
    if not meta:
        return ""
    parts = []
    if meta.get("slice_count") is not None:
        parts.append(f"slices: {meta.get('slice_count')}")
    if meta.get("spacing_mm"):
        sp = meta.get("spacing_mm")
        parts.append(f"spacing(mm): z={sp.get('z')}, y={sp.get('y')}, x={sp.get('x')}")
    if meta.get("hu_mean") is not None:
        parts.append(f"mean HU: {meta.get('hu_mean'):.1f}")
    if meta.get("bone_extent_mm_centerline") is not None:
        parts.append(f"bone extent (mm): {meta.get('bone_extent_mm_centerline'):.1f}")
    return "; ".join(parts)

def ensure_str(x: Optional[str]) -> str:
    return x if isinstance(x, str) else ""
