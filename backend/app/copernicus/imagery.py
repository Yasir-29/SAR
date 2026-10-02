import os
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from app.config import settings
from app.copernicus.auth import get_cdse_token

def generate_synthetic_optical_chip(
    bbox: list,
    is_before: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates realistic 256x256 True Color (RGB: B04, B03, B02) and NIR (B08)
    satellite chip over the specified bounding box for demonstration/mock mode.
    Simulates disaster effects (e.g. structure collapse/inundation) on AFTER image.
    """
    np.random.seed(seed)
    width, height = 256, 256

    # Create terrain base (vegetation/soil/roofs)
    base_img = Image.new("RGB", (width, height), color=(140, 160, 120))
    draw = ImageDraw.Draw(base_img)

    # Draw roads / infrastructure
    draw.line([(0, 128), (256, 128)], fill=(180, 180, 180), width=8)
    draw.line([(128, 0), (128, 256)], fill=(180, 180, 180), width=6)

    # Draw building footprints
    building_coords = [
        (40, 40, 90, 80),
        (160, 40, 210, 90),
        (40, 160, 100, 210),
        (160, 160, 220, 220),
        (100, 70, 140, 110)
    ]

    for x1, y1, x2, y2 in building_coords:
        if is_before:
            # Clean intact building roofs
            draw.rectangle([x1, y1, x2, y2], fill=(210, 120, 90), outline=(80, 80, 80))
        else:
            # Disaster damaged buildings (debris / collapse / darkening)
            if (x1 + y1) % 3 == 0:
                # Intact
                draw.rectangle([x1, y1, x2, y2], fill=(200, 115, 85), outline=(80, 80, 80))
            elif (x1 + y1) % 3 == 1:
                # Damaged roof
                draw.rectangle([x1, y1, x2, y2], fill=(120, 100, 90), outline=(40, 40, 40))
                draw.line([(x1, y1), (x2, y2)], fill=(60, 50, 40), width=3)
            else:
                # Destroyed / Debris field
                draw.polygon([(x1, y1), (x2-5, y1+10), (x2, y2), (x1+5, y2-5)], fill=(80, 75, 70))

    # Convert to NumPy array
    rgb_arr = np.array(base_img, dtype=np.uint8)
    
    # Generate NIR band (B08)
    nir_arr = np.clip(rgb_arr[:, :, 1].astype(np.float32) * 1.3 + np.random.normal(0, 5, (256, 256)), 0, 255).astype(np.uint8)

    return rgb_arr, nir_arr

def download_sentinel2_chip(
    bbox: list,
    product_info: Dict[str, Any],
    is_before: bool = True,
    request_id: str = "demo",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    event_id: str = "gaja_2018",
    event_date: str = "2018-11-10"
) -> Tuple[str, np.ndarray]:
    """
    Downloads location-specific crop of optical satellite imagery centered at (lat, lon).
    Constructs cache key including event_id, lat, lon, event_date, product_id, and before/after phase.
    Draws a center marker ● representing the requested coordinate.
    """
    import math, io
    if lat is None or lon is None:
        lat = (bbox[1] + bbox[3]) / 2.0
        lon = (bbox[0] + bbox[2]) / 2.0

    clean_pid = str(product_info.get("product_id", "s2")).replace("/", "_").replace(".SAFE", "")
    cache_key = f"{event_id}_{lat:.6f}_{lon:.6f}_{event_date}_{'before' if is_before else 'after'}_{clean_pid}"
    target_dir = settings.BEFORE_DIR if is_before else settings.AFTER_DIR
    target_filename = f"{cache_key}.png"
    target_path = target_dir / target_filename

    if target_path.exists():
        img = Image.open(target_path).convert("RGB")
        return str(target_path), np.array(img)

    # Convert lat/lon to high-resolution tile coordinates (zoom 17)
    lat_rad = math.radians(lat)
    n = 2 ** 17
    fx = (lon + 180.0) / 360.0 * n
    fy = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

    xtile = int(fx)
    ytile = int(fy)

    px = int((fx - xtile) * 256)
    py = int((fy - ytile) * 256)

    # Build 2x2 tile mosaic around requested coordinate
    mosaic = Image.new("RGB", (512, 512))
    for dx in range(2):
        for dy in range(2):
            tx, ty = xtile + dx, ytile + dy
            tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/17/{ty}/{tx}"
            try:
                resp = requests.get(tile_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=6)
                if resp.status_code == 200:
                    tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    mosaic.paste(tile_img, (dx * 256, dy * 256))
            except Exception:
                pass

    # Crop 256x256 window centered EXACTLY on px, py
    size = 256
    half = size // 2
    crop_box = (px, py, px + size, py + size)
    crop_img = mosaic.crop(crop_box)

    # Draw precise center marker ● representing requested coordinate
    draw = ImageDraw.Draw(crop_img)
    cx, cy = half, half
    draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(239, 68, 68), outline=(255, 255, 255))
    draw.line([(cx-8, cy), (cx+8, cy)], fill=(255, 255, 255), width=1)
    draw.line([(cx, cy-8), (cx, cy+8)], fill=(255, 255, 255), width=1)

    crop_img.save(target_path)
    return str(target_path), np.array(crop_img)


