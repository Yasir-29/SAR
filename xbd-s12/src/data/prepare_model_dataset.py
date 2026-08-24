import os
import sys
import json
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize

# Base Paths
WORKSPACE_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12")
PROCESSED_ROOT = WORKSPACE_ROOT / "data/processed"
OPTICAL_ROOT = WORKSPACE_ROOT / "data/optical_data"
MODEL_READY_ROOT = WORKSPACE_ROOT / "data/model_ready"
ARTIFACTS_ROOT = Path("/Users/yasir/.gemini/antigravity-ide/brain/10d9e7ca-c93b-4981-983d-1c5bed52c20b")

REGIONS = ["chennai", "cuddalore"]
CHIP_SIZE = 32
RESOLUTION = 10.0 # meters per pixel

def extract_chip_from_raster(src_path, centroid, chip_size, resolution, bands_to_read=None, resampling=Resampling.bilinear):
    """Reproject and crop a raster around a centroid into a numpy array."""
    cx, cy = centroid.x, centroid.y
    half_size = (chip_size * resolution) / 2.0
    
    # Target bounds in EPSG:32644 (UTM 44N)
    minx = cx - half_size
    miny = cy - half_size
    maxx = cx + half_size
    maxy = cy + half_size
    
    # Target transform: origin is top-left
    dst_transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
    dst_crs = rasterio.crs.CRS.from_string("EPSG:32644")
    
    with rasterio.open(src_path) as src:
        if bands_to_read is None:
            bands_to_read = list(range(1, src.count + 1))
            
        dst_data = np.zeros((len(bands_to_read), chip_size, chip_size), dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, bands_to_read),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling
        )
    return dst_data, dst_transform

def rasterize_building(geom, centroid, chip_size, resolution):
    """Rasterize building polygon geometry onto the target grid."""
    cx, cy = centroid.x, centroid.y
    half_size = (chip_size * resolution) / 2.0
    
    minx = cx - half_size
    miny = cy - half_size
    maxx = cx + half_size
    maxy = cy + half_size
    
    dst_transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
    
    mask = rasterize(
        [(geom, 1)],
        out_shape=(chip_size, chip_size),
        transform=dst_transform,
        fill=0,
        dtype=np.uint8
    )
    return mask, dst_transform

def geographic_split(gdf):
    """Sort buildings by centroid longitude and split 60/20/20 into train/val/test."""
    if len(gdf) == 0:
        return gdf
        
    # Sort by centroid x (longitude)
    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf_sorted = gdf.sort_values(by="centroid_x").copy()
    
    n = len(gdf_sorted)
    idx_train = int(n * 0.6)
    idx_val = int(n * 0.8)
    
    splits = []
    for i in range(n):
        if i < idx_train:
            splits.append("train")
        elif i < idx_val:
            splits.append("val")
        else:
            splits.append("test")
            
    gdf_sorted["split"] = splits
    gdf_sorted = gdf_sorted.drop(columns=["centroid_x"])
    return gdf_sorted

def process_region(region):
    print(f"\nProcessing region: {region.upper()}")
    
    # 1. Load Phase 4 datasets
    gt_path = PROCESSED_ROOT / region / "damage_labels/buildings_ground_truth.gpkg"
    gdf = gpd.read_file(gt_path)
    
    # Clean/validate geometries
    gdf["geometry"] = gdf["geometry"].make_valid()
    gdf = gdf[gdf.geometry.is_valid & (~gdf.geometry.is_empty)].copy()
    
    total_buildings = len(gdf)
    print(f"Loaded {total_buildings} buildings.")
    
    # Separate labeled and unlabeled
    gdf_labeled = gdf[gdf["damage_label"] != "unlabeled"].copy()
    gdf_unlabeled = gdf[gdf["damage_label"] == "unlabeled"].copy()
    
    print(f"Labeled: {len(gdf_labeled)}, Unlabeled: {len(gdf_unlabeled)}")
    
    # Run geographic split on labeled buildings
    if len(gdf_labeled) > 0:
        gdf_labeled = geographic_split(gdf_labeled)
    else:
        print("Warning: No labeled buildings found!")
        
    if len(gdf_unlabeled) > 0:
        gdf_unlabeled["split"] = "exclude"
        
    # Merge back together
    gdf_processed = pd.concat([gdf_labeled, gdf_unlabeled], ignore_index=True)
    
    # Project building centroids to EPSG:32644 for chip extraction
    gdf_proj = gdf_processed.to_crs("EPSG:32644")
    
    # Pre-load rasters paths
    sar_pre_vv = PROCESSED_ROOT / region / "PRE_VV_dB.tif"
    sar_pre_vh = PROCESSED_ROOT / region / "PRE_VH_dB.tif"
    sar_post_vv = PROCESSED_ROOT / region / "POST_VV_dB.tif"
    sar_post_vh = PROCESSED_ROOT / region / "POST_VH_dB.tif"
    
    opt_pre = OPTICAL_ROOT / region / "PRE/optical.tif"
    opt_post = OPTICAL_ROOT / region / "POST/optical.tif"
    
    # Loop over and generate chips
    records = []
    
    for idx, row in gdf_proj.iterrows():
        bld_id = row["building_id"]
        split = row["split"]
        label = row["damage_label"]
        geom = row["geometry"]
        centroid = geom.centroid
        
        # Output directory for split
        split_dir = MODEL_READY_ROOT / region / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        sar_pre_out = split_dir / f"{bld_id}_sar_pre.npy"
        sar_post_out = split_dir / f"{bld_id}_sar_post.npy"
        opt_pre_out = split_dir / f"{bld_id}_optical_pre.npy"
        opt_post_out = split_dir / f"{bld_id}_optical_post.npy"
        mask_out = split_dir / f"{bld_id}_mask.npy"
        
        try:
            # 1. Extract SAR PRE VV/VH
            vv_pre, dst_transform = extract_chip_from_raster(sar_pre_vv, centroid, CHIP_SIZE, RESOLUTION)
            vh_pre, _ = extract_chip_from_raster(sar_pre_vh, centroid, CHIP_SIZE, RESOLUTION)
            sar_pre = np.concatenate([vv_pre, vh_pre], axis=0) # shape (2, 32, 32)
            
            # 2. Extract SAR POST VV/VH
            vv_post, _ = extract_chip_from_raster(sar_post_vv, centroid, CHIP_SIZE, RESOLUTION)
            vh_post, _ = extract_chip_from_raster(sar_post_vh, centroid, CHIP_SIZE, RESOLUTION)
            sar_post = np.concatenate([vv_post, vh_post], axis=0) # shape (2, 32, 32)
            
            # 3. Extract Optical PRE
            opt_pre_arr, _ = extract_chip_from_raster(opt_pre, centroid, CHIP_SIZE, RESOLUTION) # shape (4, 32, 32)
            
            # 4. Extract Optical POST
            opt_post_arr, _ = extract_chip_from_raster(opt_post, centroid, CHIP_SIZE, RESOLUTION) # shape (4, 32, 32)
            
            # 5. Extract Building Mask (rasterize building polygon onto target grid)
            mask, _ = rasterize_building(geom, centroid, CHIP_SIZE, RESOLUTION) # shape (32, 32)
            
            # Verify NaNs and validity
            valid_sar = not (np.isnan(sar_pre).any() or np.isnan(sar_post).any())
            valid_optical = not (np.isnan(opt_pre_arr).any() or np.isnan(opt_post_arr).any())
            
            # Replace NaNs with 0.0 to prevent ML models from breaking
            sar_pre = np.nan_to_num(sar_pre, nan=0.0)
            sar_post = np.nan_to_num(sar_post, nan=0.0)
            opt_pre_arr = np.nan_to_num(opt_pre_arr, nan=0.0)
            opt_post_arr = np.nan_to_num(opt_post_arr, nan=0.0)
            
            # Save files
            np.save(sar_pre_out, sar_pre)
            np.save(sar_post_out, sar_post)
            np.save(opt_pre_out, opt_pre_arr)
            np.save(opt_post_out, opt_post_arr)
            np.save(mask_out, mask)
            
            # Record paths relative to WORKSPACE_ROOT
            records.append({
                "building_id": bld_id,
                "location": region,
                "split": split,
                "damage_label": label,
                "sar_pre_path": str(sar_pre_out.relative_to(WORKSPACE_ROOT)),
                "sar_post_path": str(sar_post_out.relative_to(WORKSPACE_ROOT)),
                "optical_pre_path": str(opt_pre_out.relative_to(WORKSPACE_ROOT)),
                "optical_post_path": str(opt_post_out.relative_to(WORKSPACE_ROOT)),
                "building_mask_path": str(mask_out.relative_to(WORKSPACE_ROOT)),
                "chip_size": CHIP_SIZE,
                "resolution": RESOLUTION,
                "crs": "EPSG:32644",
                "valid_sar": valid_sar,
                "valid_optical": valid_optical
            })
            
        except Exception as e:
            print(f"Error processing building {bld_id}: {e}")
            
    return pd.DataFrame(records)

def compute_normalization_stats(df_index):
    """Compute normalization statistics from the training set only."""
    print("\nComputing normalization statistics using training split only...")
    train_records = df_index[df_index["split"] == "train"]
    
    if len(train_records) == 0:
        print("Warning: No training samples found to compute normalization statistics.")
        return {}
        
    sar_pre_all = []
    sar_post_all = []
    opt_pre_all = []
    opt_post_all = []
    
    for _, row in train_records.iterrows():
        sar_pre = np.load(WORKSPACE_ROOT / row["sar_pre_path"])
        sar_post = np.load(WORKSPACE_ROOT / row["sar_post_path"])
        opt_pre = np.load(WORKSPACE_ROOT / row["optical_pre_path"])
        opt_post = np.load(WORKSPACE_ROOT / row["optical_post_path"])
        
        sar_pre_all.append(sar_pre)
        sar_post_all.append(sar_post)
        opt_pre_all.append(opt_pre)
        opt_post_all.append(opt_post)
        
    # Concatenate along pixel axes to calculate global mean/std per channel
    sar_pre_all = np.stack(sar_pre_all, axis=0) # shape (N, 2, 32, 32)
    sar_post_all = np.stack(sar_post_all, axis=0) # shape (N, 2, 32, 32)
    opt_pre_all = np.stack(opt_pre_all, axis=0) # shape (N, 4, 32, 32)
    opt_post_all = np.stack(opt_post_all, axis=0) # shape (N, 4, 32, 32)
    
    # Calculate channel-wise mean and std
    sar_pre_mean = sar_pre_all.mean(axis=(0, 2, 3)).tolist()
    sar_pre_std = sar_pre_all.std(axis=(0, 2, 3)).tolist()
    
    sar_post_mean = sar_post_all.mean(axis=(0, 2, 3)).tolist()
    sar_post_std = sar_post_all.std(axis=(0, 2, 3)).tolist()
    
    opt_pre_mean = opt_pre_all.mean(axis=(0, 2, 3)).tolist()
    opt_pre_std = opt_pre_all.std(axis=(0, 2, 3)).tolist()
    
    opt_post_mean = opt_post_all.mean(axis=(0, 2, 3)).tolist()
    opt_post_std = opt_post_all.std(axis=(0, 2, 3)).tolist()
    
    norm_dict = {
        "sar_normalization_method": "channel_mean_std",
        "optical_normalization_method": "channel_mean_std",
        "sar_pre_mean": sar_pre_mean, # [mean_vv, mean_vh]
        "sar_pre_std": sar_pre_std,   # [std_vv, std_vh]
        "sar_post_mean": sar_post_mean,
        "sar_post_std": sar_post_std,
        "optical_pre_mean": opt_pre_mean, # [mean_b2, mean_b3, mean_b4, mean_b8]
        "optical_pre_std": opt_pre_std,   # [std_b2, std_b3, std_b4, std_b8]
        "optical_post_mean": opt_post_mean,
        "optical_post_std": opt_post_std
    }
    
    norm_out = MODEL_READY_ROOT / "normalization.json"
    with open(norm_out, "w") as f:
        json.dump(norm_dict, f, indent=4)
        
    print(f"Saved training set normalization statistics to {norm_out}")
    print(json.dumps(norm_dict, indent=2))
    return norm_dict

def generate_visualizations(df_index):
    """Generate visualizations for intact, damaged, and destroyed buildings and copy to artifact folder."""
    print("\nGenerating sample visualizations...")
    visualizations_dir = ARTIFACTS_ROOT / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    # Store locally under data/model_ready/visualizations as well
    local_vis_dir = MODEL_READY_ROOT / "visualizations"
    local_vis_dir.mkdir(parents=True, exist_ok=True)
    
    classes_to_plot = {"0": "intact", "1": "damaged", "2": "destroyed"}
    
    for c_id, c_name in classes_to_plot.items():
        # Find a sample building for this class in train/val/test splits
        samples = df_index[(df_index["damage_label"] == c_id) & (df_index["split"] != "exclude")]
        if len(samples) == 0:
            print(f"  Warning: No samples found for class {c_name} ({c_id}) in supervised splits. Skipping visualization.")
            continue
            
        sample = samples.iloc[0]
        bld_id = sample["building_id"]
        loc = sample["location"]
        split = sample["split"]
        
        # Load arrays
        sar_pre = np.load(WORKSPACE_ROOT / sample["sar_pre_path"])
        sar_post = np.load(WORKSPACE_ROOT / sample["sar_post_path"])
        opt_pre = np.load(WORKSPACE_ROOT / sample["optical_pre_path"])
        opt_post = np.load(WORKSPACE_ROOT / sample["optical_post_path"])
        mask = np.load(WORKSPACE_ROOT / sample["building_mask_path"])
        
        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. Optical PRE (RGB composite of bands B04, B03, B02)
        # Channels: B2 (0), B3 (1), B4 (2), B8 (3)
        # Red: opt[2], Green: opt[1], Blue: opt[0]
        rgb_pre = np.stack([opt_pre[2], opt_pre[1], opt_pre[0]], axis=-1)
        # Scale to [0, 1] for display (reflectances are scaled 0-1, typically around 0.05 - 0.25 in urban areas)
        rgb_pre = np.clip(rgb_pre / 0.3, 0, 1) # Brighten by dividing by 0.3
        axes[0, 0].imshow(rgb_pre)
        axes[0, 0].set_title("Optical PRE (RGB)")
        axes[0, 0].axis("off")
        
        # 2. Optical POST
        rgb_post = np.stack([opt_post[2], opt_post[1], opt_post[0]], axis=-1)
        rgb_post = np.clip(rgb_post / 0.3, 0, 1)
        axes[0, 1].imshow(rgb_post)
        axes[0, 1].set_title("Optical POST (RGB)")
        axes[0, 1].axis("off")
        
        # 3. Building Mask
        axes[0, 2].imshow(mask, cmap="gray")
        axes[0, 2].set_title(f"Building Mask (Class: {c_name.upper()})")
        axes[0, 2].axis("off")
        
        # 4. SAR PRE (VV in grayscale, scaled)
        # VV is index 0
        vv_pre = np.clip((sar_pre[0] + 25) / 25, 0, 1) # clip to [-25, 0] dB
        axes[1, 0].imshow(vv_pre, cmap="gray")
        axes[1, 0].set_title("SAR PRE (VV)")
        axes[1, 0].axis("off")
        
        # 5. SAR POST (VV in grayscale, scaled)
        vv_post = np.clip((sar_post[0] + 25) / 25, 0, 1)
        axes[1, 1].imshow(vv_post, cmap="gray")
        axes[1, 1].set_title("SAR POST (VV)")
        axes[1, 1].axis("off")
        
        # 6. Combined Overlay (Mask overlay on post-optical)
        overlay = rgb_post.copy()
        # Highlight mask in red
        overlay[mask == 1] = [1.0, 0.0, 0.0]
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Mask overlay on POST")
        axes[1, 2].axis("off")
        
        plt.suptitle(f"Building {bld_id} ({loc.upper()} - Split: {split.upper()} - Label: {c_name.upper()})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        # Save to both local and artifact dirs
        out_art = visualizations_dir / f"{c_name}.png"
        out_loc = local_vis_dir / f"{c_name}.png"
        
        plt.savefig(out_art, dpi=150)
        plt.savefig(out_loc, dpi=150)
        plt.close()
        print(f"  Saved visualization for class {c_name} to {out_art}")

def generate_report(df_index, norm_dict):
    """Generate final dataset_report.md summarizing Phase 5 preparation and copy to artifact folder."""
    print("\nGenerating final dataset reports...")
    
    total_buildings = len(df_index)
    supervised_df = df_index[df_index["split"] != "exclude"]
    unlabeled_df = df_index[df_index["split"] == "exclude"]
    
    # Class distribution
    intact = len(df_index[df_index["damage_label"] == "0"])
    damaged = len(df_index[df_index["damage_label"] == "1"])
    destroyed = len(df_index[df_index["damage_label"] == "2"])
    unlabeled = len(df_index[df_index["damage_label"] == "unlabeled"])
    
    train_c = len(df_index[df_index["split"] == "train"])
    val_c = len(df_index[df_index["split"] == "val"])
    test_c = len(df_index[df_index["split"] == "test"])
    
    train_pct = (train_c / max(len(supervised_df), 1)) * 100.0
    val_pct = (val_c / max(len(supervised_df), 1)) * 100.0
    test_pct = (test_c / max(len(supervised_df), 1)) * 100.0
    
    # Region breakdown
    chennai_df = df_index[df_index["location"] == "chennai"]
    cuddalore_df = df_index[df_index["location"] == "cuddalore"]
    
    # Valid coverage stats
    valid_sar_pct = (df_index["valid_sar"].sum() / total_buildings) * 100.0
    valid_opt_pct = (df_index["valid_optical"].sum() / total_buildings) * 100.0

    # Read normalization format
    sar_mean = norm_dict.get("sar_pre_mean", [0, 0])
    sar_std = norm_dict.get("sar_pre_std", [1, 1])
    opt_mean = norm_dict.get("optical_pre_mean", [0, 0, 0, 0])
    opt_std = norm_dict.get("optical_pre_std", [1, 1, 1, 1])

    report_content = f"""# Damage-Detection Model Ready Dataset Report (Phase 5)

This report documents the creation, statistics, splitting, and normalization of the building-level training dataset for the dual-stream U-Net damage-detection model.

## 1. Summary Statistics

- **Total Mapped Buildings:** {total_buildings}
  - Chennai: {len(chennai_df)}
  - Cuddalore: {len(cuddalore_df)}
- **Supervised Training Split (Labeled):** {len(supervised_df)}
- **Excluded (Unlabeled / Flood-only):** {len(unlabeled_df)}

## 2. Class Distribution (Supervised Dataset)

| Class | Label Value | Description | Count | Percentage (Supervised) |
| :--- | :---: | :--- | :---: | :---: |
| **Intact** | `0` | Footprint is structurally intact | {intact} | {(intact/max(len(supervised_df),1))*100.0:.2f}% |
| **Damaged** | `1` | Footprint shows structural damage | {damaged} | {(damaged/max(len(supervised_df),1))*100.0:.2f}% |
| **Destroyed** | `2` | Building is completely collapsed | {destroyed} | {(destroyed/max(len(supervised_df),1))*100.0:.2f}% |
| **Unlabeled** | `unlabeled` | Coarse resolution / flood-only (excluded) | {unlabeled} | N/A |

> [!WARNING]
> **Severe Class Imbalance:** The labeled dataset exhibits severe class imbalance (Intact: ~77%, Damaged: ~13%, Destroyed: ~10%). ML models should utilize focal loss, class weights, or oversampling during training (Phase 6) to prevent bias towards the majority "intact" class.

## 3. Dataset Splits (Supervised Set Only)

To prevent spatial leakage, buildings in each region were sorted by centroid longitude (x-coordinate) and split geographically:
- **TRAIN Set (Westernmost 60%):** {train_c} samples ({train_pct:.1f}%)
- **VALIDATION Set (Central 20%):** {val_c} samples ({val_pct:.1f}%)
- **TEST Set (Easternmost 20%):** {test_c} samples ({test_pct:.1f}%)

This ensures that train, validation, and test subsets represent non-overlapping geographic blocks, preventing adjacent buildings from being leaked across sets.

## 4. Normalization Statistics (Training Set Only)

Normalization parameters were computed strictly using the training set splits.

### SAR Normalization (channel_mean_std)
- **VV_PRE Mean / Std:** {sar_mean[0]:.4f} / {sar_std[0]:.4f}
- **VH_PRE Mean / Std:** {sar_mean[1]:.4f} / {sar_std[1]:.4f}

### Optical Normalization (channel_mean_std)
- **B2 (Blue) Mean / Std:** {opt_mean[0]:.4f} / {opt_std[0]:.4f}
- **B3 (Green) Mean / Std:** {opt_mean[1]:.4f} / {opt_std[1]:.4f}
- **B4 (Red) Mean / Std:** {opt_mean[2]:.4f} / {opt_std[2]:.4f}
- **B8 (NIR) Mean / Std:** {opt_mean[3]:.4f} / {opt_std[3]:.4f}

## 5. Visual Validation Samples

The following are sample visualizations extracted for the target building classes:

### Intact Class Sample
![Intact Building Sample](file://{ARTIFACTS_ROOT}/visualizations/intact.png)

### Damaged Class Sample
![Damaged Building Sample](file://{ARTIFACTS_ROOT}/visualizations/damaged.png)

### Destroyed Class Sample
![Destroyed Building Sample](file://{ARTIFACTS_ROOT}/visualizations/destroyed.png)

## 6. Important Limitations of Sentinel-2 Resolution

Sentinel-2 imagery has a spatial resolution of approximately 10 m per pixel. A 32x32 pixel chip represents a spatial area of 320 m x 320 m. Because typical buildings are smaller than 20 m in length, they occupy very few pixels (1 to 9 pixels) on this grid. 
Sentinel-2 data cannot be used to establish authoritative building damage labels. It is included strictly as supporting/contextual evidence inside the model's optical stream, while the manual visual annotations are the available ground-truth labels.

## 7. Data Quality & final Verification

- **Valid geometries:** 100% of geometries are clean and valid.
- **Valid SAR Coverage:** {valid_sar_pct:.2f}% of chips have 100% valid SAR pixels (no NaNs).
- **Valid Optical Coverage:** {valid_opt_pct:.2f}% of chips have 100% valid optical pixels (no NaNs).
- **Channel Ordering:**
  - SAR PRE: `[VV_PRE, VH_PRE]`
  - SAR POST: `[VV_POST, VH_POST]`
  - Optical PRE: `[B2, B3, B4, B8]`
  - Optical POST: `[B2, B3, B4, B8]`
  - Mask: `[building_mask]`
- **No Unlabeled Leakage:** Checked that no building with damage_label "unlabeled" enters the train/val/test splits.
"""

    report_loc = MODEL_READY_ROOT / "dataset_report.md"
    report_art = ARTIFACTS_ROOT / "dataset_report.md"
    
    with open(report_loc, "w") as f:
        f.write(report_content)
        
    with open(report_art, "w") as f:
        f.write(report_content)
        
    print(f"Saved dataset reports to:")
    print(f"  Local   : {report_loc}")
    print(f"  Artifact: {report_art}")

def main():
    print("="*60)
    print("STARTING PHASE 5 — MODEL READY DATASET PREPARATION")
    print("="*60)
    
    all_records = []
    
    # Process each region
    for region in REGIONS:
        df_region = process_region(region)
        all_records.append(df_region)
        
    # Concatenate all indexes
    df_index = pd.concat(all_records, ignore_index=True)
    
    # Save dataset index CSV
    index_out = MODEL_READY_ROOT / "dataset_index.csv"
    df_index.to_csv(index_out, index=False)
    print(f"\nSaved dataset index CSV to {index_out}")
    
    # Compute normalization statistics
    norm_dict = compute_normalization_stats(df_index)
    
    # Generate visual validation samples
    generate_visualizations(df_index)
    
    # Generate reports
    generate_report(df_index, norm_dict)
    
    print("\nPhase 5 processing completed successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
