import os
import sys
import json
import argparse
from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12")
PROCESSED_ROOT = WORKSPACE_ROOT / "data/processed"

# AOI Coordinates [min_lon, min_lat, max_lon, max_lat]
AOIS = {
    "chennai": [80.0, 12.85, 80.35, 13.20],
    "cuddalore": [79.65, 11.65, 79.90, 11.90],
    "nagapattinam": [79.65, 10.65, 79.90, 10.90]
}

# Known disaster periods (Pre: Nov 2015, Post: Dec 2015)
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
        ee.Initialize()
        
        # Access the user project from internal state
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
        print("You can authenticate by running: gcloud auth application-default login")
        print(f"Details: {e}")
        print("="*80 + "\n")
        sys.exit(1)

def query_best_image(ee, region, role, bbox, start_date, end_date):
    """Query the Harmonized S2 SR collection for the best cloud-free image."""
    geom = ee.Geometry.BBox(*bbox)
    
    # Query Harmonized S2 L2A collection (Surface Reflectance)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start_date, end_date)
    )
    
    # Check if collection is empty
    count = int(collection.size().getInfo())
    if count == 0:
        return None
        
    # SCL Mask function to calculate valid (cloud-free) pixels
    # SCL Band Values: 0=no_data, 1=saturated/defective, 3=cloud_shadows, 8=cloud_medium, 9=cloud_high, 10=thin_cirrus
    # Valid = not in [0, 1, 3, 8, 9, 10]
    def calculate_valid_pct(image):
        scl = image.select("SCL")
        valid_mask = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
        )
        
        # Calculate mean valid pixel fraction over the AOI
        mean_dict = valid_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=20, # SCL resolution is 20m
            maxPixels=1e8
        )
        valid_fraction = mean_dict.get("SCL")
        
        # Set property on the image
        return image.set({
            "valid_pixel_percentage": ee.Number(valid_fraction).multiply(100.0),
            "acquisition_date": ee.Date(image.get("system:time_start")).format("YYYY-MM-DD")
        })
        
    # Apply valid pct calculation
    collection_with_stats = collection.map(calculate_valid_pct)
    
    # Sort: prefer highest valid pixel percentage over the AOI, then lowest cloudy pixel percentage metadata
    best_image = (
        collection_with_stats
        .sort("valid_pixel_percentage", False) # descending (highest valid pct first)
        .sort("CLOUDY_PIXEL_PERCENTAGE")       # ascending (lowest cloud metadata)
        .first()
    )
    
    # Fetch details
    try:
        info = best_image.getInfo()
        if not info:
            return None
            
        img_id = info["id"]
        props = info["properties"]
        
        return {
            "image": best_image,
            "id": img_id,
            "date": props.get("acquisition_date"),
            "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
            "valid_pct": props.get("valid_pixel_percentage")
        }
    except Exception as e:
        # If any getInfo fails (e.g. empty collection)
        return None

def main():
    parser = argparse.ArgumentParser(description="Acquire Sentinel-2 PRE/POST scenes using Google Earth Engine.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run search and statistics queries only, without starting exports.")
    args = parser.parse_args()
    
    # 1. Initialize Earth Engine
    ee = initialize_earth_engine()
    
    print("\n" + "="*80)
    print(f"SEARCHING SENTINEL-2 SURFACE REFLECTANCE IMAGERY (DRY-RUN={args.dry_run})")
    print("="*80)
    
    selected_scenes = []
    
    # 2. Search PRE and POST for each region
    for region, bbox in AOIS.items():
        for role, dates in DATE_RANGES.items():
            start_date, end_date = dates
            print(f"Searching {region.upper()} {role} (Range: {start_date} to {end_date})...")
            
            scene_info = query_best_image(ee, region, role, bbox, start_date, end_date)
            
            if scene_info:
                selected_scenes.append({
                    "region": region,
                    "role": role,
                    "status": "AVAILABLE",
                    "date": scene_info["date"],
                    "image_id": scene_info["id"],
                    "cloud_pct": round(scene_info["cloud_pct"], 2),
                    "valid_pct": round(scene_info["valid_pct"], 2),
                    "image": scene_info["image"],
                    "aoi": bbox
                })
            else:
                selected_scenes.append({
                    "region": region,
                    "role": role,
                    "status": "UNAVAILABLE",
                    "date": "N/A",
                    "image_id": "N/A",
                    "cloud_pct": "N/A",
                    "valid_pct": 0.0,
                    "image": None,
                    "aoi": bbox
                })

    # 3. Print Selected Scenes Table
    print("\nSELECTED IMAGES SUMMARY:")
    print("-" * 80)
    print(f"{'Region':<15} {'Role':<6} {'Status':<12} {'Date':<12} {'Cloud %':<10} {'Valid %':<10}")
    print("-" * 80)
    
    unsuitable_post = False
    
    for s in selected_scenes:
        cloud_str = f"{s['cloud_pct']}" if isinstance(s['cloud_pct'], float) else s['cloud_pct']
        valid_str = f"{s['valid_pct']}" if isinstance(s['valid_pct'], float) else s['valid_pct']
        
        print(f"{s['region'].capitalize():<15} {s['role']:<6} {s['status']:<12} {s['date']:<12} {cloud_str:<10} {valid_str:<10}")
        
        # Check if POST is unavailable or has very low valid percentage (e.g. < 10% valid pixels)
        if s["role"] == "POST" and (s["status"] == "UNAVAILABLE" or s["valid_pct"] < 10.0):
            unsuitable_post = True
            
    print("-" * 80 + "\n")
    
    # 4. Handle exports and reports
    report_data = []
    
    for s in selected_scenes:
        role_meta = {
            "region": s["region"],
            "role": s["role"],
            "status": s["status"],
            "selected_date": s["date"],
            "image_id": s["image_id"],
            "cloud_percentage": s["cloud_pct"],
            "valid_pixel_percentage": s["valid_pct"],
            "bands": ["B2", "B3", "B4", "B8", "B11", "SCL"],
            "scale": 10,
            "AOI": s["aoi"],
            "export_task_name": "N/A"
        }
        
        # Export logic
        if s["status"] == "AVAILABLE" and s["image"] is not None:
            img = s["image"]
            geom = ee.Geometry.BBox(*s["aoi"])
            
            # Select bands
            bands_10m = img.select(["B2", "B3", "B4", "B8", "SCL"])
            
            # Resample B11 (native 20m) to 10m using bilinear interpolation
            # S2 Surface Reflectance B11 is native 20m.
            # Using bilinear interpolation keeps transitions smooth when upsampled.
            b11_10m = img.select("B11").resample("bilinear")
            
            # Combine bands
            final_img = bands_10m.addBands(b11_10m).select(["B2", "B3", "B4", "B8", "B11", "SCL"])
            
            export_name = f"S2_{s['region']}_{s['role']}_{s['date']}"
            role_meta["export_task_name"] = export_name
            
            if not args.dry_run:
                # Check Google Drive destination
                # In python API we start the export task
                print(f"Starting Earth Engine Export for {s['region']} {s['role']}...")
                task = ee.batch.Export.image.toDrive(
                    image=final_img,
                    description=export_name,
                    folder="gee_sentinel2",
                    fileNamePrefix=f"{s['region']}/{s['role']}_S2",
                    scale=10,
                    region=geom,
                    maxPixels=1e9
                )
                task.start()
                print(f"  Export task '{export_name}' started successfully. Task ID: {task.id}")
            else:
                print(f"[DRY-RUN] Skipping Export for task: {export_name}")
                
        report_data.append(role_meta)
        
    # Write report to processed folder
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = PROCESSED_ROOT / "gee_sentinel2_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Acquisition report written to: {report_path}")
    
    # 5. Validation Warning for clouds
    if unsuitable_post:
        print("\n" + "!"*80)
        print("WARNING: POST Sentinel-2 imagery unsuitable for visual building labeling.")
        print("Reason: No cloud-free Harmonized L2A Surface Reflectance scenes are available in 2015.")
        print("!"*80 + "\n")
        
    print("Phase 4 Google Earth Engine acquisition search completed.")

if __name__ == "__main__":
    main()
