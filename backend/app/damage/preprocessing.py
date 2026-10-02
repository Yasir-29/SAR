import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any
from app.config import settings

def preprocess_and_compare(
    before_img_path: str,
    after_img_path: str,
    request_id: str
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """
    Step 6 & Step 8: Preprocessing & Change Image Generation.
    1. Loads BEFORE and AFTER images.
    2. Resizes both images to consistent resolution (256x256).
    3. Normalizes image values consistently.
    4. Calculates absolute difference: ABS(AFTER - BEFORE).
    5. Saves comparison image to data/comparison/{request_id}_difference.png.
    """
    before_img = Image.open(before_img_path).convert("RGB").resize((256, 256))
    after_img = Image.open(after_img_path).convert("RGB").resize((256, 256))

    before_arr = np.array(before_img, dtype=np.float32) / 255.0
    after_arr = np.array(after_img, dtype=np.float32) / 255.0

    # Step 8: Absolute Difference Image calculation
    diff_arr = np.abs(after_arr - before_arr)
    
    # Generate enhanced heat map representation (amplifying change signal for visualization)
    diff_intensity = np.mean(diff_arr, axis=2)
    diff_heatmap = np.zeros((256, 256, 3), dtype=np.uint8)

    # Apply colormap (Jet / Infernal-like contrast)
    diff_heatmap[:, :, 0] = np.clip(diff_intensity * 255.0 * 3.5, 0, 255).astype(np.uint8)  # Red channel
    diff_heatmap[:, :, 1] = np.clip(diff_intensity * 255.0 * 1.2, 0, 255).astype(np.uint8)  # Green channel
    diff_heatmap[:, :, 2] = np.clip((1.0 - diff_intensity) * 100.0, 0, 255).astype(np.uint8) # Blue channel

    diff_path = settings.COMPARISON_DIR / f"{request_id}_difference.png"
    diff_pil = Image.fromarray(diff_heatmap)
    diff_pil.save(diff_path)

    # Cloud and quality checks
    valid_pixel_ratio = float(np.mean(before_arr > 0.01))
    quality_meta = {
        "valid_pixels_percent": round(valid_pixel_ratio * 100.0, 2),
        "mean_change_magnitude": round(float(np.mean(diff_arr)), 4),
        "max_change_magnitude": round(float(np.max(diff_arr)), 4),
        "dimensions": [256, 256, 3]
    }

    return str(diff_path), diff_arr, quality_meta
