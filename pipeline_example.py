# pipeline_example.py
from dicom_reader import load_cbct_from_path
from cbct_measurements import compute_hu_stats, slice_montage, estimate_bone_thickness_along_line
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# local test file (from conversation)
LOCAL_ZIP = "CareStream3D-CBCT-sample.zip"   # <-- your uploaded file path

def run_pipeline(path):
    print("Loading CBCT from:", path)
    out = load_cbct_from_path(path)
    vol = out["volume"]
    meta = out.get("metadata", {})
    print("Volume shape:", vol.shape)
    stats = compute_hu_stats(vol)
    print("HU stats:", stats)
    # create montage and save
    montage = slice_montage(vol)
    plt.imsave("cbct_montage_preview.png", montage, cmap='gray')
    # crude thickness example: pick two voxels roughly center canal line
    zc = vol.shape[0]//2
    yc = vol.shape[1]//2
    xc = vol.shape[2]//2
    p1 = [zc, yc-50, xc]
    p2 = [zc, yc+50, xc]
    thickness = estimate_bone_thickness_along_line(vol, p1, p2, num_samples=200)
    print("Example thickness (vox):", thickness)
    summary = {"shape": vol.shape, "metadata": meta, "stats": stats, "thickness_example": thickness}
    with open("cbct_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved preview -> cbct_montage_preview.png and summary -> cbct_summary.json")

if __name__ == "__main__":
    run_pipeline(LOCAL_ZIP)
