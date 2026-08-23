import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Input directories
INPUT_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12/data/south_coastal")
OUTPUT_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12/data/processed")

REGIONS = ["chennai", "cuddalore", "nagapattinam"]

def to_db(data):
    """Convert Sentinel-1 linear backscatter to dB, masking zeros/NaNs with np.nan."""
    invalid = (data <= 0) | np.isnan(data)
    safe_data = np.maximum(data, 1e-10)
    db = 10.0 * np.log10(safe_data)
    db[invalid] = np.nan
    return db

def process_region(region_name):
    print("\n" + "=" * 60)
    print(f"PROCESSING REGION: {region_name.upper()}")
    print("=" * 60)
    
    region_in_dir = INPUT_ROOT / region_name
    region_out_dir = OUTPUT_ROOT / region_name
    region_out_dir.mkdir(parents=True, exist_ok=True)
    
    pre_path = region_in_dir / "PRE_VV_VH.tif"
    post_path = region_in_dir / "POST_VV_VH.tif"
    
    if not pre_path.exists() or not post_path.exists():
        raise FileNotFoundError(f"Missing input TIFFs in {region_in_dir}")
        
    # 1. Load and Verify Alignment
    print("1. Inspecting and verifying alignments...")
    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        pre_crs = pre.crs
        post_crs = post.crs
        pre_res = pre.res
        post_res = post.res
        pre_transform = pre.transform
        post_transform = post.transform
        pre_width, pre_height = pre.width, pre.height
        post_width, post_height = post.width, post.height
        pre_bounds = pre.bounds
        post_bounds = post.bounds
        
        print(f"  PRE: CRS={pre_crs}, Res={pre_res}, Shape={pre_width}x{pre_height}")
        print(f"  POST: CRS={post_crs}, Res={post_res}, Shape={post_width}x{post_height}")
        
        # Checking match
        crs_match = pre_crs == post_crs
        res_match = pre_res == post_res
        transform_match = pre_transform == post_transform
        shape_match = (pre_width == post_width) and (pre_height == post_height)
        
        # Bounds floating comparison
        bounds_match = True
        for k in ["left", "bottom", "right", "top"]:
            if abs(getattr(pre_bounds, k) - getattr(post_bounds, k)) > 1e-5:
                bounds_match = False
                break
                
        print(f"  Alignment verification:")
        print(f"    CRS matches: {crs_match}")
        print(f"    Resolution matches: {res_match}")
        print(f"    Transform matches: {transform_match}")
        print(f"    Dimensions match: {shape_match}")
        print(f"    Bounds match: {bounds_match}")
        
        if not (crs_match and res_match and transform_match and shape_match and bounds_match):
            raise ValueError(f"Spatial alignment failure for {region_name}! PRE and POST grids do not match.")
            
        print("  Spatial alignment: verified successfully ✅")
        
        # Read arrays
        pre_vv = pre.read(1).astype(np.float32)
        pre_vh = pre.read(2).astype(np.float32)
        post_vv = post.read(1).astype(np.float32)
        post_vh = post.read(2).astype(np.float32)
        
        profile = pre.profile.copy()
        
    # 2. Convert to dB safely (handling 0.0 values)
    print("2. Converting linear intensity to dB scale...")
    pre_vv_db = to_db(pre_vv)
    pre_vh_db = to_db(pre_vh)
    post_vv_db = to_db(post_vv)
    post_vh_db = to_db(post_vh)
    
    # 3. Calculate Change Features
    print("3. Calculating change features...")
    vv_change = post_vv_db - pre_vv_db
    vh_change = post_vh_db - pre_vh_db
    
    # 4. Save processed outputs separately
    print("4. Saving output TIFFs...")
    profile.update(count=1, dtype="float32", compress="deflate", nodata=np.nan)
    
    outputs = {
        "PRE_VV_dB.tif": pre_vv_db,
        "PRE_VH_dB.tif": pre_vh_db,
        "POST_VV_dB.tif": post_vv_db,
        "POST_VH_dB.tif": post_vh_db,
        "CHANGE_VV.tif": vv_change,
        "CHANGE_VH.tif": vh_change
    }
    
    created_files = []
    for fname, data in outputs.items():
        out_file = region_out_dir / fname
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(data, 1)
        created_files.append(out_file)
        print(f"  Wrote: {out_file.name} (Min={np.nanmin(data):.4f}, Max={np.nanmax(data):.4f})")
        
    # 5. Generate visualizations
    print("5. Generating plots and dashboard...")
    plot_configs = [
        {"data": pre_vv_db, "title": "PRE VV (dB)", "cmap": "viridis", "vmin": -25, "vmax": 0, "out": "PRE_VV.png"},
        {"data": post_vv_db, "title": "POST VV (dB)", "cmap": "viridis", "vmin": -25, "vmax": 0, "out": "POST_VV.png"},
        {"data": vv_change, "title": "VV Change (dB)", "cmap": "RdBu", "vmin": -5, "vmax": 5, "out": "CHANGE_VV.png"},
        {"data": pre_vh_db, "title": "PRE VH (dB)", "cmap": "viridis", "vmin": -30, "vmax": -5, "out": "PRE_VH.png"},
        {"data": post_vh_db, "title": "POST VH (dB)", "cmap": "viridis", "vmin": -30, "vmax": -5, "out": "POST_VH.png"},
        {"data": vh_change, "title": "VH Change (dB)", "cmap": "RdBu", "vmin": -5, "vmax": 5, "out": "CHANGE_VH.png"}
    ]
    
    # Save individual plots
    for p in plot_configs:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(p["data"], cmap=p["cmap"], vmin=p["vmin"], vmax=p["vmax"])
        ax.set_title(f"{region_name.capitalize()} {p['title']}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out_img = region_out_dir / p["out"]
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()
        
    # Save combined dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, p in enumerate(plot_configs):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        im = ax.imshow(p["data"], cmap=p["cmap"], vmin=p["vmin"], vmax=p["vmax"])
        ax.set_title(p["title"])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fig.suptitle(f"Sentinel-1 SAR Processing Dashboard: {region_name.capitalize()}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    dashboard_out = region_out_dir / "DASHBOARD.png"
    plt.savefig(dashboard_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Wrote dashboard: {dashboard_out.name}")
    
    # Compute statistics for reporting
    stats = {}
    for name, change_arr in [("VV", vv_change), ("VH", vh_change)]:
        valid_mask = ~np.isnan(change_arr)
        valid_count = np.sum(valid_mask)
        total_count = change_arr.size
        valid_pct = (valid_count / total_count) * 100.0
        masked_pct = 100.0 - valid_pct
        
        if valid_count > 0:
            valid_vals = change_arr[valid_mask]
            min_val = float(np.min(valid_vals))
            max_val = float(np.max(valid_vals))
            mean_val = float(np.mean(valid_vals))
            median_val = float(np.median(valid_vals))
            std_val = float(np.std(valid_vals))
        else:
            min_val = max_val = mean_val = median_val = std_val = 0.0
            
        stats[name] = {
            "valid_pct": valid_pct,
            "masked_pct": masked_pct,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": median_val,
            "std": std_val
        }
        
        print(f"  {name} Change Stats:")
        print(f"    Valid Pixels: {valid_pct:.2f}% | Masked: {masked_pct:.2f}%")
        print(f"    Min: {min_val:.4f} dB | Max: {max_val:.4f} dB | Mean: {mean_val:.4f} dB")
        print(f"    Median: {median_val:.4f} dB | StdDev: {std_val:.4f} dB")
        
    # Return stats for reporting
    return {
        "region": region_name,
        "files_created": [f.name for f in created_files] + [p["out"] for p in plot_configs] + ["DASHBOARD.png"],
        "dimensions": f"{pre_width} x {pre_height}",
        "crs": str(pre_crs),
        "resolution": str(pre_res),
        "stats": stats
    }

def main():
    print("Starting Sentinel-1 SAR-Only processing pipeline...")
    reports = []
    for r in REGIONS:
        try:
            report = process_region(r)
            reports.append(report)
        except Exception as e:
            print(f"Error processing {r}: {e}")
            
    # Save the pipeline stats report
    import json
    report_json = OUTPUT_ROOT / "pipeline_report.json"
    with open(report_json, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nPipeline execution finished. Summary saved to {report_json}")

if __name__ == "__main__":
    main()
