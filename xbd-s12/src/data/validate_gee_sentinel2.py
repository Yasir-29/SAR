import sys
import numpy as np
import rasterio
from pathlib import Path

# Base Path
WORKSPACE_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12")
DOWNLOAD_DIR = WORKSPACE_ROOT / "gee_sentinel2_2015"

def validate_raster(file_path):
    print("="*60)
    print(f"VALIDATING FILE: {file_path.name}")
    print(f"Path: {file_path}")
    print("="*60)
    
    try:
        with rasterio.open(file_path) as src:
            print(f"CRS        : {src.crs}")
            print(f"Dimensions : {src.height} x {src.width} (H x W)")
            print(f"Transform  : {src.transform}")
            print(f"Bounds     : {src.bounds}")
            print(f"Count/Bands: {src.count} bands")
            print(f"Dtype      : {src.dtypes[0] if src.dtypes else 'Unknown'}")
            print(f"Nodata     : {src.nodata}")
            
            # Read all bands
            arr = src.read()
            total_pixels = arr.shape[1] * arr.shape[2]
            nodata_val = src.nodata
            
            # Sentinel-2 2015 bands exported: B2, B3, B4, B8
            band_names = {
                1: "B2 (Blue)",
                2: "B3 (Green)",
                3: "B4 (Red)",
                4: "B8 (NIR)",
                5: "B11 (SWIR)"
            }
            
            for band_idx in range(1, src.count + 1):
                band_name = band_names.get(band_idx, f"Band {band_idx}")
                band_arr = arr[band_idx - 1].astype(np.float32)
                
                # Check for NaNs
                nan_count = np.isnan(band_arr).sum()
                nan_pct = (nan_count / total_pixels) * 100.0
                
                # Filter out nodata and NaNs for statistics
                valid_data = band_arr[np.isfinite(band_arr)]
                if nodata_val is not None:
                    valid_data = valid_data[valid_data != nodata_val]
                    
                if valid_data.size > 0:
                    b_min = np.min(valid_data)
                    b_max = np.max(valid_data)
                    b_mean = np.mean(valid_data)
                    b_median = np.median(valid_data)
                else:
                    b_min, b_max, b_mean, b_median = "N/A", "N/A", "N/A", "N/A"
                    
                print(f"\n  {band_name}:")
                print(f"    Min / Max / Mean / Median: {b_min} / {b_max} / {b_mean:.2f} / {b_median:.2f}")
                print(f"    NaN pixels  : {nan_count} ({nan_pct:.2f}%)")
                if nodata_val is not None:
                    nodata_count = (band_arr == nodata_val).sum()
                    nodata_pct = (nodata_count / total_pixels) * 100.0
                    print(f"    Nodata pixels: {nodata_count} ({nodata_pct:.2f}%)")
                    
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"Error reading raster {file_path.name}: {e}")
        print("="*60 + "\n")
        return False

def verify_spatial_compatibility():
    """Verify that PRE and POST images for each region have compatible spatial coverage."""
    print("="*60)
    print("CHECKING PRE/POST SPATIAL COMPATIBILITY")
    print("="*60)
    
    regions = ["chennai", "cuddalore", "nagapattinam"]
    for r in regions:
        # Search for .tif files in subdirectories matching role names
        pre_file = None
        post_file = None
        
        pre_matches = list((DOWNLOAD_DIR / r).glob("*PRE*_S2*.tif")) or list(DOWNLOAD_DIR.glob(f"**/{r}/*PRE*.tif"))
        post_matches = list((DOWNLOAD_DIR / r).glob("*POST*_S2*.tif")) or list(DOWNLOAD_DIR.glob(f"**/{r}/*POST*.tif"))
        
        if pre_matches:
            pre_file = pre_matches[0]
        if post_matches:
            post_file = post_matches[0]
            
        if pre_file and post_file:
            print(f"Region: {r.upper()}")
            print(f"  PRE : {pre_file.name}")
            print(f"  POST: {post_file.name}")
            try:
                with rasterio.open(pre_file) as pre_src, rasterio.open(post_file) as post_src:
                    crs_match = pre_src.crs == post_src.crs
                    dims_match = (pre_src.height == post_src.height) and (pre_src.width == post_src.width)
                    transform_match = pre_src.transform == post_src.transform
                    
                    print(f"    CRS matches       : {crs_match} ({pre_src.crs})")
                    print(f"    Dimensions match  : {dims_match} ({pre_src.height}x{pre_src.width})")
                    print(f"    Transforms match  : {transform_match}")
                    
                    if crs_match and dims_match and transform_match:
                        print("    Verdict: PRE and POST scenes are SPATIALLY COMPATIBLE.")
                    else:
                        print("    WARNING: PRE and POST scenes are NOT spatially compatible/aligned!")
            except Exception as e:
                print(f"    Error reading images for comparison: {e}")
        else:
            print(f"Region: {r.upper()}")
            if not pre_file:
                print("  Missing PRE image for validation.")
            if not post_file:
                print("  Missing POST image for validation.")
        print("-" * 60)
    print("="*60 + "\n")

def main():
    print("="*60)
    print("STARTING SENTINEL-2 RASTER VALIDATION (2015)")
    print("="*60)
    
    # Find all .tif files recursively under gee_sentinel2_2015
    tif_files = list(DOWNLOAD_DIR.rglob("*.tif"))
    
    if not tif_files:
        print("\n" + "!"*80)
        print(f"No Sentinel-2 2015 GeoTIFF files found in: {DOWNLOAD_DIR}")
        print("Please check that Earth Engine exports have finished, and files have been downloaded.")
        print("!"*80 + "\n")
        sys.exit(0)
        
    print(f"Found {len(tif_files)} GeoTIFF files to validate.\n")
    
    success_count = 0
    for f in tif_files:
        if validate_raster(f):
            success_count += 1
            
    print(f"Validated {success_count}/{len(tif_files)} files.\n")
    
    # Check compatibility
    verify_spatial_compatibility()

if __name__ == "__main__":
    main()
