import os
import zipfile
import tempfile
import numpy as np
from PIL import Image
import gdcm


def extract_first_dicom(zip_path):
    """Extracts the first DICOM from the ZIP into a temp folder."""
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(".dcm"):
                out_path = os.path.join(tmp, os.path.basename(name))
                z.extract(name, tmp)
                return os.path.join(tmp, name)
    return None


def decode_with_gdcm(dcm_path):
    """Decodes JPEG12/16 DICOM slices using GDCM."""
    reader = gdcm.ImageReader()
    reader.SetFileName(dcm_path)

    if not reader.Read():
        raise ValueError("GDCM failed to read DICOM")

    img = reader.GetImage()
    pixel_bytes = img.GetBuffer()

    if pixel_bytes is None:
        raise ValueError("Pixel buffer is None (bad JPEG stream).")

    # Extract dimensions
    width = img.GetDimension(0)
    height = img.GetDimension(1)
    samples_per_pixel = img.GetNumberOfDimensions()

    # 12-bit can be stored as 16-bit signed or unsigned
    pixel_format = img.GetPixelFormat()
    bits = pixel_format.GetBitsStored()

    # Convert buffer → numpy array
    if bits <= 8:
        arr = np.frombuffer(pixel_bytes, dtype=np.uint8)
    else:
        arr = np.frombuffer(pixel_bytes, dtype=np.uint16)

    # Reshape to 2D slice
    arr = arr.reshape(height, width)

    return arr


def normalize_to_png(arr):
    """Normalizes DICOM pixel data to 8-bit PNG."""
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() != 0:
        arr /= arr.max()
    arr *= 255
    return arr.astype(np.uint8)


def create_preview(zip_path, out_file="preview.png"):
    """
    Extracts ONE slice from CBCT zip and saves preview PNG.
    Returns path or None.
    """

    try:
        dcm_path = extract_first_dicom(zip_path)
        if not dcm_path:
            print("No DICOM found inside ZIP.")
            return None

        arr = decode_with_gdcm(dcm_path)
        arr = normalize_to_png(arr)

        # Save
        img = Image.fromarray(arr)
        img.save(out_file)

        return out_file

    except Exception as e:
        print("DICOM preview error:", e)
        return None
