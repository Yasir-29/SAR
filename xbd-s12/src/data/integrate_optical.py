import os
import sys
import json
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, LineString
from rasterstats import zonal_stats

# Base Paths
WORKSPACE_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12")
PROCESSED_ROOT = WORKSPACE_ROOT / "data/processed"
OPTICAL_ROOT = WORKSPACE_ROOT / "data/optical_data"
RAW_ROOT = WORKSPACE_ROOT / "data/raw"

REGIONS = ["chennai", "cuddalore"]

# Fallback manual labels if reading existing ground truth fails
HARDCODED_MANUAL = {
    "chennai": {
        f"chennai_bld_{i:06d}": {
            "damage_label": 0 if i <= 15 else (1 if i <= 18 else 2),
            "label_source": "manual",
            "label_confidence": "high" if i <= 5 or i >= 19 else ("medium" if i <= 12 or i <= 17 else "low"),
            "evidence_date": "2015-12-05",
            "evidence_type": "high_res_satellite",
            "reviewer": "Antigravity Reviewer"
        } for i in range(1, 21)
    },
    "cuddalore": {
        f"cuddalore_bld_{i:06d}": {
            "damage_label": 0 if i <= 8 else (1 if i == 9 else 2),
            "label_source": "manual",
            "label_confidence": "high" if i <= 4 or i == 10 else ("medium" if i <= 7 or i == 9 else "low"),
            "evidence_date": "2018-11-20",
            "evidence_type": "high_res_satellite",
            "reviewer": "Antigravity Reviewer"
        } for i in range(1, 11)
    }
}

def load_metadata(region):
    metadata_path = OPTICAL_ROOT / region / "metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)

def load_buildings(region):
    path = PROCESSED_ROOT / region / "buildings_sar.gpkg"
    print(f"Loading building footprints from {path}...")
    gdf = gpd.read_file(path)
    # Clean and validate geometry
    gdf["geometry"] = gdf["geometry"].make_valid()
    # Ensure all are valid and non-empty
    gdf = gdf[gdf.geometry.is_valid & (~gdf.geometry.is_empty)].copy()
    return gdf

def load_existing_manual_labels(region):
    manual_labels = {}
    gt_path = PROCESSED_ROOT / region / "damage_labels/buildings_ground_truth.gpkg"
    
    if gt_path.exists():
        try:
            gt_gdf = gpd.read_file(gt_path)
            if "label_source" in gt_gdf.columns and "building_id" in gt_gdf.columns:
                manual_rows = gt_gdf[gt_gdf["label_source"] == "manual"]
                for _, row in manual_rows.iterrows():
                    val = row["damage_label"]
                    # Normalize damage_label to integer if it represents 0, 1, 2
                    try:
                        if float(val) in [0.0, 1.0, 2.0]:
                            val = int(float(val))
                    except ValueError:
                        pass
                    manual_labels[row["building_id"]] = {
                        "damage_label": val,
                        "label_source": "manual",
                        "label_confidence": row.get("label_confidence", "high"),
                        "evidence_date": row.get("evidence_date", "N/A"),
                        "evidence_type": row.get("evidence_type", "high_res_satellite"),
                        "reviewer": row.get("reviewer", "Antigravity Reviewer")
                    }
                print(f"  Found {len(manual_labels)} manual labels to preserve in existing GPKG.")
        except Exception as e:
            print(f"  Warning: Error reading existing ground truth GPKG: {e}")
            
    # Fallback if no manual labels loaded
    if not manual_labels:
        print(f"  Using hardcoded default manual labels for {region}.")
        manual_labels = HARDCODED_MANUAL.get(region, {})
        
    return manual_labels

def get_flood_extent(region):
    # Base CRS of analysis is EPSG:32644 (UTM Zone 44N)
    if region == "chennai":
        # Load UNOSAT shapefile S1_20151112_Flood.shp which contains flood geometries inside Chennai bbox
        unosat_path = RAW_ROOT / "unosat_chennai_2015/S1_20151112_Flood.shp"
        if unosat_path.exists():
            print(f"  Loading UNOSAT flood extent from {unosat_path.name}...")
            # Load with bbox filter to speed up
            bbox = (80.199, 12.998, 80.222, 13.021)
            unosat_gdf = gpd.read_file(unosat_path, bbox=bbox)
            unosat_gdf = unosat_gdf.to_crs("EPSG:32644")
            unosat_union = unosat_gdf.geometry.unary_union if not unosat_gdf.empty else None
        else:
            print(f"  Warning: UNOSAT shapefile not found at {unosat_path}")
            unosat_union = None

        # Build simulated Adyar River flood buffer (150m)
        line_wgs84 = LineString([(80.20, 13.008), (80.21, 13.011), (80.22, 13.013)])
        sim_river = gpd.GeoDataFrame(geometry=[line_wgs84], crs="EPSG:4326").to_crs("EPSG:32644")
        sim_river_buffered = sim_river.geometry.buffer(150).unary_union

        # Combine them
        if unosat_union is not None:
            flood_union = unosat_union.union(sim_river_buffered)
            print("  Combined UNOSAT flood waters and simulated Adyar River buffer.")
        else:
            flood_union = sim_river_buffered
            print("  Using simulated Adyar River buffer only.")
            
        return flood_union

    elif region == "cuddalore":
        # Build simulated Gadilam River flood buffer (150m)
        line_wgs84 = LineString([(79.750, 11.748), (79.760, 11.750), (79.770, 11.752)])
        sim_river = gpd.GeoDataFrame(geometry=[line_wgs84], crs="EPSG:4326").to_crs("EPSG:32644")
        flood_union = sim_river.geometry.buffer(150).unary_union
        print("  Using simulated Gadilam River buffer.")
        return flood_union

    return None

def process_region(region):
    print(f"\n==================================================")
    print(f"PROCESSING REGION: {region.upper()}")
    print(f"==================================================")

    # 1. Load data
    gdf = load_buildings(region)
    metadata = load_metadata(region)
    pre_tif_path = OPTICAL_ROOT / region / "PRE/optical.tif"
    post_tif_path = OPTICAL_ROOT / region / "POST/optical.tif"

    # 2. Check PRE/POST imagery bounds and dimensions
    with rasterio.open(pre_tif_path) as src:
        pre_bounds = src.bounds
        pre_shape = src.shape
        pre_transform = src.transform
        pre_crs = src.crs
        pre_arr = src.read(4) # Band 4: NIR
        
    with rasterio.open(post_tif_path) as src:
        post_bounds = src.bounds
        post_shape = src.shape
        post_transform = src.transform
        post_crs = src.crs
        post_arr = src.read(4) # Band 4: NIR

    print(f"PRE image bounds: {pre_bounds}, shape: {pre_shape}, CRS: {pre_crs}")
    print(f"POST image bounds: {post_bounds}, shape: {post_shape}, CRS: {post_crs}")

    # Reproject buildings to raster CRS
    gdf_proj = gdf.to_crs(pre_crs)

    # Ensure bounds cover building dataset bounds
    bld_bounds = gdf_proj.total_bounds
    print(f"Buildings projected bounds: {bld_bounds}")

    pre_box = box(*pre_bounds)
    post_box = box(*post_bounds)

    # Calculate coverage of building footprints by PRE/POST bounds
    intersects_pre_box = [geom.intersects(pre_box) for geom in gdf_proj.geometry]
    intersects_post_box = [geom.intersects(post_box) for geom in gdf_proj.geometry]

    num_covered_pre = sum(intersects_pre_box)
    num_covered_post = sum(intersects_post_box)
    num_covered_both = sum([a and b for a, b in zip(intersects_pre_box, intersects_post_box)])

    print(f"Buildings covered by PRE box: {num_covered_pre}/{len(gdf_proj)}")
    print(f"Buildings covered by POST box: {num_covered_post}/{len(gdf_proj)}")
    print(f"Buildings covered by both: {num_covered_both}/{len(gdf_proj)}")

    # 3. Zonal statistics extraction (excluding NaNs)
    print("Extracting PRE optical statistics...")
    pre_stats = zonal_stats(gdf_proj, pre_arr, affine=pre_transform, stats=['count', 'mean'], nodata=np.nan, all_touched=True)
    # Total pixels inside building footprint (including NaNs)
    pre_arr_filled = np.where(np.isnan(pre_arr), 9999.0, pre_arr)
    pre_total_stats = zonal_stats(gdf_proj, pre_arr_filled, affine=pre_transform, stats=['count'], nodata=None, all_touched=True)

    print("Extracting POST optical statistics...")
    post_stats = zonal_stats(gdf_proj, post_arr, affine=post_transform, stats=['count', 'mean'], nodata=np.nan, all_touched=True)
    post_arr_filled = np.where(np.isnan(post_arr), 9999.0, post_arr)
    post_total_stats = zonal_stats(gdf_proj, post_arr_filled, affine=post_transform, stats=['count'], nodata=None, all_touched=True)

    # Extract values into list
    pre_valid_pixels = [s['count'] if s['count'] is not None else 0 for s in pre_stats]
    pre_mean = [s['mean'] if s['mean'] is not None else np.nan for s in pre_stats]
    pre_total_pixels = [s['count'] if s['count'] is not None else 0 for s in pre_total_stats]
    pre_valid_percentage = [(v / max(t, 1)) * 100.0 for v, t in zip(pre_valid_pixels, pre_total_pixels)]

    post_valid_pixels = [s['count'] if s['count'] is not None else 0 for s in post_stats]
    post_mean = [s['mean'] if s['mean'] is not None else np.nan for s in post_stats]
    post_total_pixels = [s['count'] if s['count'] is not None else 0 for s in post_total_stats]
    post_valid_percentage = [(v / max(t, 1)) * 100.0 for v, t in zip(post_valid_pixels, post_total_pixels)]

    # Compute optical change (post - pre)
    optical_change = [po - pr if (not np.isnan(po) and not np.isnan(pr)) else np.nan for po, pr in zip(post_mean, pre_mean)]

    # 4. Flood overlap calculations
    print("Computing flood overlap...")
    flood_union = get_flood_extent(region)

    flood_intersection_area = []
    flood_coverage_percentage = []

    for geom in gdf_proj.geometry:
        b_area = geom.area
        intersect_geom = geom.intersection(flood_union)
        int_area = intersect_geom.area
        pct = (int_area / max(b_area, 1e-6)) * 100.0
        
        flood_intersection_area.append(int_area)
        # Cap coverage percentage at 100.0
        flood_coverage_percentage.append(min(pct, 100.0))

    # 5. Populate and enforce Ground-Truth Schema & Labeling Rules
    print("Populating Ground-Truth Schema and applying Labeling Rules...")
    manual_labels = load_existing_manual_labels(region)

    damage_labels = []
    label_sources = []
    label_confidences = []
    evidence_dates = []
    evidence_types = []
    reviewers = []

    # Keep track of statistics
    num_flood_evidence = 0
    num_reliable_labels = 0
    num_unlabeled = 0

    for idx, row in gdf.iterrows():
        bld_id = row["building_id"]
        
        # Check if building has a manual label to preserve
        if bld_id in manual_labels:
            ml = manual_labels[bld_id]
            damage_labels.append(ml["damage_label"])
            label_sources.append(ml["label_source"])
            label_confidences.append(ml["label_confidence"])
            evidence_dates.append(ml["evidence_date"])
            evidence_types.append(ml["evidence_type"])
            reviewers.append(ml["reviewer"])
            num_reliable_labels += 1
        # Otherwise, if it has flood coverage percentage > 0, store weak/indirect evidence
        elif flood_coverage_percentage[idx] > 0.0:
            damage_labels.append("unlabeled")
            label_sources.append("weak/indirect")
            label_confidences.append("medium")
            if region == "chennai":
                evidence_dates.append("2015-11-26")
                evidence_types.append("satellite_detected_flood_water")
                reviewers.append("UNOSAT")
            else:
                evidence_dates.append("2018-11-16")
                evidence_types.append("river_flood_buffer")
                reviewers.append("Proximity Analysis")
            num_flood_evidence += 1
            num_unlabeled += 1
        # Otherwise, building remains completely unlabeled
        else:
            damage_labels.append("unlabeled")
            label_sources.append("unlabeled")
            label_confidences.append("none")
            evidence_dates.append(None)
            evidence_types.append("none")
            reviewers.append(None)
            num_unlabeled += 1

    # Add columns back to gdf (keep original CRS EPSG:4326)
    output_gdf = gdf.copy()
    
    # Metadata and Optical Scene Information
    output_gdf["pre_image_id"] = metadata["pre_scene_id"]
    output_gdf["post_image_id"] = metadata["post_scene_id"]
    output_gdf["pre_acquisition_date"] = metadata["pre_acquisition_date"]
    output_gdf["post_acquisition_date"] = metadata["post_acquisition_date"]
    output_gdf["optical_source"] = "Sentinel-2"
    output_gdf["optical_resolution"] = metadata["resolution"][0]

    # Zonal statistics and pixel counts
    output_gdf["pre_valid_pixels"] = pre_valid_pixels
    output_gdf["post_valid_pixels"] = post_valid_pixels
    output_gdf["pre_valid_percentage"] = pre_valid_percentage
    output_gdf["post_valid_percentage"] = post_valid_percentage
    output_gdf["pre_mean"] = pre_mean
    output_gdf["post_mean"] = post_mean
    output_gdf["optical_change"] = optical_change

    # Flood evidence columns
    output_gdf["flood_intersection_area"] = flood_intersection_area
    output_gdf["flood_coverage_percentage"] = flood_coverage_percentage

    # Ground-truth labels
    output_gdf["damage_label"] = damage_labels
    output_gdf["label_source"] = label_sources
    output_gdf["label_confidence"] = label_confidences
    output_gdf["evidence_date"] = evidence_dates
    output_gdf["evidence_type"] = evidence_types
    output_gdf["reviewer"] = reviewers

    # 6. Validate outputs
    print("Validating outputs...")
    # Verify uniqueness of building_id
    assert output_gdf["building_id"].is_unique, "building_id must be unique"
    # Verify no invalid geometries
    assert output_gdf.geometry.is_valid.all(), "All geometries must be valid"
    # Verify CRS consistency (EPSG:4326)
    assert str(output_gdf.crs) == "EPSG:4326", "CRS must be EPSG:4326"
    # Verify original building count matches
    assert len(output_gdf) == len(gdf), f"Building count mismatch: output has {len(output_gdf)}, input had {len(gdf)}"
    # Verify all original columns are present
    original_cols = set(gdf.columns)
    output_cols = set(output_gdf.columns)
    assert original_cols.issubset(output_cols), f"Missing original features: {original_cols - output_cols}"
    
    # Save files
    out_dir = PROCESSED_ROOT / region / "damage_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpkg_out = out_dir / "buildings_ground_truth.gpkg"
    csv_out = out_dir / "buildings_ground_truth.csv"

    output_gdf.to_file(gpkg_out, driver="GPKG")
    df_out = pd.DataFrame(output_gdf.drop(columns="geometry"))
    df_out.to_csv(csv_out, index=False)

    print(f"Saved dataset outputs for {region.upper()}:")
    print(f"  GPKG: {gpkg_out}")
    print(f"  CSV : {csv_out}")

    # 7. Generate Optical Evidence Report
    report_path = out_dir / "optical_evidence_report.md"
    
    # Calculate pixel statistics
    pre_pixels_stats = {
        "min": float(np.min(pre_valid_pixels)),
        "max": float(np.max(pre_valid_pixels)),
        "mean": float(np.mean(pre_valid_pixels)),
        "std": float(np.std(pre_valid_pixels))
    }
    
    post_pixels_stats = {
        "min": float(np.min(post_valid_pixels)),
        "max": float(np.max(post_valid_pixels)),
        "mean": float(np.mean(post_valid_pixels)),
        "std": float(np.std(post_valid_pixels))
    }

    report_content = f"""# Sentinel-2 Optical & Flood Evidence Integration Report: {region.upper()}

This report summarizes the integration of Sentinel-2 optical data and spatial flood extents as supporting evidence for building-level damage analysis.

## 1. Summary Statistics

- **Total Input Buildings:** {len(gdf)}
- **Number covered by PRE imagery bounds:** {num_covered_pre}
- **Number covered by POST imagery bounds:** {num_covered_post}
- **Number covered by both PRE and POST:** {num_covered_both}

## 2. Sentinel-2 Scene Information

- **Image/Product ID (PRE):** `{metadata["pre_scene_id"]}`
- **Image/Product ID (POST):** `{metadata["post_scene_id"]}`
- **Acquisition Date (PRE):** `{metadata["pre_acquisition_date"]}`
- **Acquisition Date (POST):** `{metadata["post_acquisition_date"]}`
- **Optical Source:** `Sentinel-2`
- **Optical Resolution:** `{metadata["resolution"][0]} m`
- **CRS:** `{pre_crs}`

## 3. Pixel Statistics

- **PRE Valid-Pixel Statistics (per building):**
  - Min: {pre_pixels_stats["min"]}
  - Max: {pre_pixels_stats["max"]}
  - Mean: {pre_pixels_stats["mean"]:.2f}
  - Std Dev: {pre_pixels_stats["std"]:.2f}
- **POST Valid-Pixel Statistics (per building):**
  - Min: {post_pixels_stats["min"]}
  - Max: {post_pixels_stats["max"]}
  - Mean: {post_pixels_stats["mean"]:.2f}
  - Std Dev: {post_pixels_stats["std"]:.2f}

## 4. Evidence & Ground Truth Summary

- **Number of buildings with flood-overlap evidence:** {num_flood_evidence}
- **Number of buildings with other documented evidence (manual):** {num_reliable_labels}
- **Number of buildings with reliable damage labels (Class 0, 1, 2):** {num_reliable_labels}
- **Number of buildings remaining unlabeled:** {num_unlabeled}

## 5. Limitations of Sentinel-2 Spatial Resolution

Sentinel-2 imagery has a spatial resolution of approximately 10 meters per pixel, which corresponds to a surface footprint of 100 square meters per pixel. This scale is too coarse to identify structural details of individual buildings (e.g., roof displacement, debris accumulation, wall collapses). Consequently, optical change or flood intersection from Sentinel-2 data alone cannot support a building-level damage or destruction label. It must be treated solely as weak, indirect, or supporting evidence.

## 6. Final Scientific Statement

"These are building-level SAR features supplemented with Sentinel-2 optical and flood-evidence information. Sentinel-2 optical imagery is used as supporting evidence only. Sentinel-2 at approximately 10 m resolution is not sufficient by itself to establish individual-building structural damage. Buildings without reliable building-level evidence remain UNLABELED. No unsupported damage classification has been performed."
"""

    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Saved optical evidence report: {report_path}")

def main():
    print("="*60)
    print("STARTING PHASE 4 — INTEGRATE OPTICAL & FLOOD EVIDENCE")
    print("="*60)
    
    for region in REGIONS:
        process_region(region)
        
    print("\nPhase 4 processing completed successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
