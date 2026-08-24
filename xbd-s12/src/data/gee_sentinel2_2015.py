import os
import sys
import json
import csv
import argparse
from pathlib import Path
import datetime as dt

# Base Paths
WORKSPACE_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12")
PROCESSED_ROOT = WORKSPACE_ROOT / "data/processed"
REPORT_DIR = PROCESSED_ROOT / "sentinel2"

# AOI Coordinates [min_lon, min_lat, max_lon, max_lat]
AOIS = {
    "chennai": [80.0, 12.85, 80.35, 13.20],
    "cuddalore": [79.65, 11.65, 79.90, 11.90],
    "nagapattinam": [79.65, 10.65, 79.90, 10.90]
}

# Date ranges for Chennai, Cuddalore, Nagapattinam (PRE: Nov 2015, POST: Dec 2015)
DATE_RANGES = {
    "PRE": ("2015-10-15", "2015-11-30"),
    "POST": ("2015-12-01", "2016-01-15")
}

def initialize_earth_engine():
    """Initialize Google Earth Engine API. Fail clearly if not authenticated."""
    try:
        import ee
    except ImportError:
        print("Error: The 'earthengine-api' package is not installed. Please run 'uv pip install earthengine-api'.")
        sys.exit(1)
        
    print("Initializing Google Earth Engine...")
    try:
        ee.Initialize(project="shining-axon-432413-v3")
        project = None
        try:
            project = ee._state.get_state().cloud_api_user_project
        except Exception:
            pass
        if not project:
            project = "default/environment-configured"
            
        print(f"Successfully authenticated with Earth Engine.")
        print(f"Earth Engine project being used: {project}")
        return ee
    except Exception as e:
        print("\n" + "="*80)
        print("EARTH ENGINE INITIALIZATION FAILED!")
        print("="*80)
        print("Please verify that Earth Engine is authenticated on your system.")
        print("You can authenticate by running: earthengine authenticate")
        print(f"Details: {e}")
        print("="*80 + "\n")
        sys.exit(1)

def query_candidates(ee, region, role, bbox, start_date, end_date):
    """Query S2_HARMONIZED and join with S2_CLOUD_PROBABILITY to calculate valid pixel percentages."""
    geom = ee.Geometry.BBox(*bbox)
    
    # Load collections
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(start_date, end_date)
    s2_cloud = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(geom).filterDate(start_date, end_date)
    
    # Check if empty
    s2_size = int(s2.size().getInfo())
    if s2_size == 0:
        return []
        
    # Join on system:index
    join = ee.Join.inner()
    filter_index = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = join.apply(s2, s2_cloud, filter_index)
    
    def merge_bands(feature):
        primary = ee.Image(feature.get("primary"))
        secondary = ee.Image(feature.get("secondary"))
        return primary.addBands(secondary.select(["probability"]))
        
    combined = ee.ImageCollection(joined.map(merge_bands))
    
    def add_cloud_stats(image):
        prob = image.select("probability")
        qa = image.select("QA60")
        
        # Cloud mask: probability > 50 or QA60 cloud bit 10 or 11 active
        prob_cloud = prob.gt(50)
        qa_cloud = qa.bitwiseAnd(1024).neq(0).Or(qa.bitwiseAnd(2048).neq(0))
        is_cloud = prob_cloud.Or(qa_cloud).rename("is_cloud")
        is_valid = is_cloud.Not().rename("is_valid")
        
        mean_dict = ee.Image.cat([is_cloud, is_valid]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=20,
            maxPixels=1e8
        )
        
        # Fallback values if mean_dict is null or empty
        cloud_val = ee.Number(ee.Algorithms.If(mean_dict.get("is_cloud"), mean_dict.get("is_cloud"), 1.0))
        valid_val = ee.Number(ee.Algorithms.If(mean_dict.get("is_valid"), mean_dict.get("is_valid"), 0.0))
        
        return image.set({
            "aoi_cloud_pct": cloud_val.multiply(100.0),
            "aoi_valid_pct": valid_val.multiply(100.0),
            "aoi_masked_pct": cloud_val.multiply(100.0),
            "acquisition_date": ee.Date(image.get("system:time_start")).format("YYYY-MM-DD")
        })
        
    processed_col = combined.map(add_cloud_stats)
    
    # Retrieve candidates list
    candidates_info = processed_col.toList(100).getInfo()
    
    candidates = []
    for info in candidates_info:
        props = info["properties"]
        candidates.append({
            "image_id": info["id"],
            "product_bundle_id": props.get("PRODUCT_BUNDLE_ID", info["id"].split("/")[-1]),
            "date": props.get("acquisition_date"),
            "cloud_pct_metadata": props.get("CLOUDY_PIXEL_PERCENTAGE"),
            "aoi_cloud_pct": props.get("aoi_cloud_pct"),
            "aoi_valid_pct": props.get("aoi_valid_pct"),
            "aoi_masked_pct": props.get("aoi_masked_pct"),
            "image": ee.Image(info["id"])
        })
        
    return candidates

def rank_candidates(candidates, role):
    """Sort candidates prioritizing: 1. Coverage, 2. Date proximity to event, 3. Cloudiness."""
    # PRE: Event occurred around Dec 1. We want PRE images closest to Nov 30 (latest in range).
    # POST: Event occurred around Dec 1. We want POST images closest to Dec 1 (earliest in range).
    target_date = dt.date(2015, 11, 30) if role == "PRE" else dt.date(2015, 12, 1)
    
    def sorting_key(c):
        c_date = dt.datetime.strptime(c["date"], "%Y-%m-%d").date()
        date_diff = abs((c_date - target_date).days)
        # Group valid percentage in 20% bins (bin 0 is 80-100%, bin 1 is 60-80% etc.)
        valid_bin = -int(c["aoi_valid_pct"] // 20)
        return (valid_bin, date_diff, c["aoi_cloud_pct"])
        
    return sorted(candidates, key=sorting_key)

def perform_building_test(ee, pre_img, post_img):
    """Extract a small chip around a real building to test visual suitability."""
    import geopandas as gpd
    from shapely.geometry import box
    import requests
    import zipfile
    import io
    
    gpkg_path = PROCESSED_ROOT / "chennai/damage_labels/buildings_ground_truth.gpkg"
    if not gpkg_path.exists():
        print(f"  Building test skipped: Ground-truth GPKG not found at {gpkg_path}")
        return
        
    try:
        print("\n" + "-"*60)
        print("STEP 11: RUNNING BUILDING-LEVEL TEST (CHENNAI)")
        print("-"*60)
        
        gdf = gpd.read_file(gpkg_path)
        if gdf.empty:
            print("  Building test skipped: Ground-truth GPKG is empty.")
            return
            
        # Select first building
        first_bld = gdf.iloc[0]
        bld_id = first_bld["building_id"]
        geom_wgs = first_bld.geometry
        
        # Buffer building by 150m (WGS84 rough equivalent is 0.0015 degrees)
        # Buffer in projected UTM 44N for metric accuracy
        gdf_proj = gpd.GeoDataFrame(geometry=[geom_wgs], crs="EPSG:4326").to_crs("EPSG:32644")
        buffered_proj = gdf_proj.geometry.buffer(150)
        buffered_wgs = buffered_proj.to_crs("EPSG:4326").iloc[0]
        
        # Get bounding box of buffer
        minx, miny, maxx, maxy = buffered_wgs.bounds
        ee_box = ee.Geometry.BBox(minx, miny, maxx, maxy)
        
        print(f"  Selected building: {bld_id}")
        print(f"  AOI Bounding Box: [{minx:.5f}, {miny:.5f}, {maxx:.5f}, {maxy:.5f}]")
        
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Fetch pre and post chips
        for role, img in [("PRE", pre_img), ("POST", post_img)]:
            # Select RGB bands
            rgb = img.select(["B4", "B3", "B2"])
            
            # Export via getDownloadURL
            url = rgb.getDownloadURL({
                "scale": 10,
                "crs": "EPSG:4326",
                "region": ee_box,
                "format": "GEO_TIFF"
            })
            
            # Download and extract GeoTIFF
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            
            z = zipfile.ZipFile(io.BytesIO(r.content))
            tif_files = [f for f in z.namelist() if f.endswith(".tif")]
            
            tif_path = REPORT_DIR / f"CHN_TEST_BUILDING_{role}.tif"
            png_path = REPORT_DIR / f"CHN_TEST_BUILDING_{role}.png"
            
            if tif_files:
                # Save GeoTIFF
                with open(tif_path, "wb") as f_out:
                    f_out.write(z.read(tif_files[0]))
                print(f"  Saved {role} building chip GeoTIFF: {tif_path}")
                
                # Generate RGB PNG visualization
                import rasterio
                import numpy as np
                import matplotlib.pyplot as plt
                
                with rasterio.open(tif_path) as src:
                    # Read R, G, B
                    red = src.read(1)
                    green = src.read(2)
                    blue = src.read(3)
                    
                    rgb_stack = np.dstack((red, green, blue)).astype(np.float32)
                    # Sentinel-2 L1C ranges roughly 0 - 10000. Scale to 0 - 3000 for visibility
                    rgb_stack = np.clip(rgb_stack / 3000.0, 0.0, 1.0)
                    
                    # Save RGB plot
                    plt.imsave(png_path, rgb_stack)
                print(f"  Saved {role} building chip visualization: {png_path}")
                
        # Suitability verdict
        print("\n  Sentinel-2 (10m resolution) Building Suitability Verdict:")
        print("  - At 10m scale, a typical building (~15m footprint) is represented by only 1 to 4 pixels.")
        print("  - Conclusion: Sentinel-2 imagery alone is UNSUITABLE for visual damage classification (Class 0, 1, 2).")
        print("  - S2 can identify regional inundation/floods, but cannot resolve roof collapse or debris piles.")
        print("-" * 60 + "\n")
        
    except Exception as e:
        print(f"  Warning: Building test failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 2015 Image Acquisition using Earth Engine L1C Harmonized.")
    parser.add_argument("--export", action="store_true", default=False, help="Enable Earth Engine Google Drive exports.")
    parser.add_argument("--test-building", action="store_true", default=False, help="Run Step 11 building-level test chip extraction.")
    args = parser.parse_args()
    
    # 1. Initialize Earth Engine
    ee = initialize_earth_engine()
    
    print("\n" + "="*80)
    print("QUERYING SENTINEL-2 L1C HARMONIZED (2015)")
    print("="*80)
    
    all_searches = {}
    selected_scenes = []
    
    # 2. Query and Evaluate Candidates
    for region, bbox in AOIS.items():
        all_searches[region] = {}
        for role, dates in DATE_RANGES.items():
            start_date, end_date = dates
            candidates = query_candidates(ee, region, role, bbox, start_date, end_date)
            
            all_searches[region][role] = {
                "candidates": candidates,
                "count": len(candidates)
            }
            
            print(f"Region: {region.upper()}")
            print(f"Role: {role}")
            print(f"Candidates: {len(candidates)}")
            
            # Sort candidates by coverage and date proximity
            ranked = rank_candidates(candidates, role)
            
            if ranked:
                best = ranked[0]
                selected_scenes.append({
                    "region": region,
                    "role": role,
                    "status": "AVAILABLE" if best["aoi_valid_pct"] >= 10.0 else "UNUSABLE",
                    "date": best["date"],
                    "image_id": best["image_id"],
                    "product_id": best["product_bundle_id"],
                    "cloud_pct": round(best["cloud_pct_metadata"], 2),
                    "aoi_cloud_pct": round(best["aoi_cloud_pct"], 2),
                    "aoi_valid_pct": round(best["aoi_valid_pct"], 2),
                    "aoi_masked_pct": round(best["aoi_masked_pct"], 2),
                    "image": best["image"],
                    "aoi": bbox
                })
                
                print(f"Selected Date: {best['date']}")
                print(f"Product ID   : {best['product_bundle_id']}")
                print(f"Cloud Metadata: {best['cloud_pct_metadata']:.2f}%")
                print(f"AOI Valid %   : {best['aoi_valid_pct']:.2f}%")
                print(f"AOI Cloud %   : {best['aoi_cloud_pct']:.2f}%")
            else:
                selected_scenes.append({
                    "region": region,
                    "role": role,
                    "status": "UNAVAILABLE",
                    "date": "N/A",
                    "image_id": "N/A",
                    "product_id": "N/A",
                    "cloud_pct": "N/A",
                    "aoi_cloud_pct": "N/A",
                    "aoi_valid_pct": 0.0,
                    "aoi_masked_pct": 100.0,
                    "image": None,
                    "aoi": bbox
                })
                print("Selected Date: N/A (NO USABLE SENTINEL-2 IMAGE FOUND)")
            print("-" * 50)
            
    # 3. Print Final Selected Table
    print("\n" + "="*80)
    print("CONSOLIDATED SENTINEL-2 2015 PRE/POST IMAGE ACQUISITION TABLE")
    print("="*80)
    print(f"{'Region':<15} {'Role':<6} {'Status':<12} {'Date':<12} {'Cloud %':<10} {'Valid %':<10}")
    print("-" * 80)
    
    for s in selected_scenes:
        cloud_str = f"{s['aoi_cloud_pct']}" if isinstance(s['aoi_cloud_pct'], float) else s['aoi_cloud_pct']
        valid_str = f"{s['aoi_valid_pct']}" if isinstance(s['aoi_valid_pct'], float) else s['aoi_valid_pct']
        print(f"{s['region'].capitalize():<15} {s['role']:<6} {s['status']:<12} {s['date']:<12} {cloud_str:<10} {valid_str:<10}")
    print("="*80 + "\n")
    
    # 4. Save CSV and MD Reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "gee_acquisition_results.csv"
    md_path = REPORT_DIR / "gee_acquisition_report.md"
    
    # Write CSV
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["region", "role", "status", "selected_date", "image_id", "product_id", "cloud_percentage", "valid_pixel_percentage", "masked_pixel_percentage", "export_task_name"])
        for s in selected_scenes:
            export_name = f"S2_2015_{s['region']}_{s['role']}_{s['date']}" if s["status"] == "AVAILABLE" else "N/A"
            writer.writerow([
                s["region"], s["role"], s["status"], s["date"], s["image_id"], s["product_id"],
                s["aoi_cloud_pct"], s["aoi_valid_pct"], s["aoi_masked_pct"], export_name
            ])
    print(f"Acquisition CSV report written to: {csv_path}")
    
    # Write MD Report
    report_content = f"""# Sentinel-2 2015 Image Acquisition Report (Google Earth Engine)

This report details the image query, ranking, and validation parameters used to acquire Sentinel-2 imagery for the Chennai, Cuddalore, and Nagapattinam 2015 study areas, conforming to the Phase 4 requirements.

## 1. Google Earth Engine Initialized Parameters
- **Collection Used:** `COPERNICUS/S2_HARMONIZED` (Level-1C Top-Of-Atmosphere)
- **Cloud Assessor Joined:** `COPERNICUS/S2_CLOUD_PROBABILITY` (s2cloudless)
- **Pixel-Level Masking logic:** Combined `probability > 50` threshold and `QA60` bits 10 and 11.

---

## 2. Image Selection & Suitability Results

| Region | Role | Status | Date | Product ID | Cloud % | Valid % |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
"""
    for s in selected_scenes:
        cloud_str = f"{s['aoi_cloud_pct']}%" if isinstance(s['aoi_cloud_pct'], float) else s['aoi_cloud_pct']
        valid_str = f"{s['aoi_valid_pct']}%" if isinstance(s['aoi_valid_pct'], float) else s['aoi_valid_pct']
        report_content += f"| **{s['region'].capitalize()}** | {s['role']} | `{s['status']}` | {s['date']} | `{s['product_id']}` | {cloud_str} | {valid_str} |\n"
        
    report_content += "\n---\n\n## 3. Detailed Region Suitability Notes\n"
    for s in selected_scenes:
        if s["role"] == "POST":
            suitability = "SUITABLE FOR REGIONAL INUNDATION ONLY" if s["status"] == "AVAILABLE" else "UNUSABLE"
            report_content += f"### {s['region'].capitalize()} (POST Event)\n"
            report_content += f"- **Selected Date:** {s['date']}\n"
            report_content += f"- **AOI Valid Coverage:** {s['aoi_valid_pct']}%\n"
            report_content += f"- **Visual Building Suitability Verdict:** **NOT SUITABLE for individual building-level damage classification.** Sentinel-2 10m pixels do not resolve structural roof details (only regional changes).\n\n"
            
    with open(md_path, "w") as f_md:
        f_md.write(report_content)
    print(f"Acquisition Markdown report written to: {md_path}")
    
    # 5. Export Tasks
    if args.export:
        print("\n" + "="*80)
        print("STARTING GOOGLE DRIVE EXPORT TASKS")
        print("="*80)
        
        for s in selected_scenes:
            if s["status"] == "AVAILABLE" and s["image"] is not None:
                img = s["image"]
                geom = ee.Geometry.BBox(*s["aoi"])
                
                # Select 10m bands (B2, B3, B4, B8)
                bands_10m = img.select(["B2", "B3", "B4", "B8"])
                
                export_name = f"S2_2015_{s['region']}_{s['role']}_{s['date']}"
                
                # Drive Export task
                print(f"Submitting export task for {s['region'].capitalize()} {s['role']}...")
                task = ee.batch.Export.image.toDrive(
                    image=bands_10m,
                    description=export_name,
                    folder="gee_sentinel2_2015",
                    fileNamePrefix=f"{s['region']}/{s['role']}_S2_2015",
                    scale=10,
                    region=geom,
                    maxPixels=1e9
                )
                task.start()
                print(f"  Submitted task '{export_name}'. Task ID: {task.id}")
            else:
                print(f"Skipping export for {s['region']} {s['role']} (Status: {s['status']})")
                
    else:
        print("\n[INFO] Running in SEARCH-ONLY mode. Exports were skipped. To enable, re-run with --export.")

    # 6. Step 11 Building Test
    if args.test_building:
        pre_chennai = [s for s in selected_scenes if s["region"] == "chennai" and s["role"] == "PRE"][0]
        post_chennai = [s for s in selected_scenes if s["region"] == "chennai" and s["role"] == "POST"][0]
        
        if pre_chennai["status"] == "AVAILABLE" and post_chennai["status"] == "AVAILABLE":
            perform_building_test(ee, pre_chennai["image"], post_chennai["image"])
        else:
            print("\n  Building test skipped: Chennai PRE and POST images must be available.")

if __name__ == "__main__":
    main()
