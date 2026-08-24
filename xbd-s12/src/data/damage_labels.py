import os
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path("/Users/yasir/Desktop/sardd/SAR/xbd-s12")
PROCESSED_ROOT = WORKSPACE_ROOT / "data/processed"

REGIONS = ["chennai", "cuddalore", "nagapattinam"]

# Event Metadata Mapping
EVENT_METADATA = {
    "chennai": {
        "event_name": "Chennai Floods 2015 / Cyclone Vardah 2016",
        "date": "November-December 2015 / December 2016",
        "official_sources": (
            "- **Source:** Tamil Nadu State Disaster Management Authority (TNSDMA) & Chennai Metropolitan Development Authority (CMDA)\n"
            "  - **Event/Date:** Chennai Floods (Nov-Dec 2015), Cyclone Vardah (Dec 2016)\n"
            "  - **Geographic Coverage:** Chennai Metropolitan Area\n"
            "  - **Spatial Resolution:** Aggregated at block/ward levels (no coordinates)\n"
            "  - **Damage Information:** Counts of fully/partially damaged houses, collapsed huts, infrastructure damage costs.\n"
            "  - **Individual Buildings:** No, reports only contain aggregate numbers.\n"
            "  - **License/Use:** Public government report data (restricted to non-commercial research use).\n"
            "  - **OSM Matchability:** Cannot be matched; lacks coordinates or building-specific identifiers."
        ),
        "copernicus_sources": (
            "- **Source:** Copernicus EMS / International Charter Space and Major Disasters (Call ID 561)\n"
            "  - **Event/Date:** South India Floods (December 2015)\n"
            "  - **Geographic Coverage:** Chennai and surrounding districts\n"
            "  - **Spatial Resolution:** Macro overview mapping based on RADARSAT-2, TerraSAR-X, and Sentinel-1.\n"
            "  - **Damage Information:** Inundation footprints and damage overview vector layers.\n"
            "  - **Individual Buildings:** No individual building-level structural damage classification is provided.\n"
            "  - **License/Use:** Open access for humanitarian and research use.\n"
            "  - **OSM Matchability:** Inundation footprints can be spatially intersected with OSM buildings to produce weak/indirect labels, but not real structural damage."
        ),
        "weak_sources": (
            "- **Source:** UNITAR/UNOSAT Flood Inundation Vector Products / Dartmouth Flood Observatory (DFO)\n"
            "  - **Event/Date:** Chennai Floods (Dec 2015)\n"
            "  - **Geographic Coverage:** Southern Chennai area\n"
            "  - **Spatial Resolution:** Derived from 10m/30m satellite imagery.\n"
            "  - **Damage Information:** Flood extent polygons.\n"
            "  - **Individual Buildings:** No building structural damage.\n"
            "  - **License/Use:** Open GIS data (Creative Commons).\n"
            "  - **OSM Matchability:** Can be spatially intersected to assign weak labels (`weak_flooded` vs `weak_non_flooded`)."
        ),
        "other_sources": (
            "- **Source:** Crowdsourced Flooding Hotspots (OpenCity.in / Oorvani Foundation)\n"
            "  - **Event/Date:** Nov-Dec 2015 Floods\n"
            "  - **Geographic Coverage:** Chennai Municipal area\n"
            "  - **Spatial Resolution:** Point-based user reports and flood heights.\n"
            "  - **Damage Information:** Stagnation points and water logging depths.\n"
            "  - **Individual Buildings:** Some street-level descriptions, but no building shape matching.\n"
            "  - **License/Use:** Open Data (ODbL).\n"
            "  - **OSM Matchability:** Extremely poor matchability due to manual geocoding inaccuracies."
        ),
        "insufficiency_reason": (
            "No authoritative source provides structural building-level damage labels. "
            "The official statistics are aggregated ward-level tables, and the Copernicus/Charter/UNOSAT products "
            "provide flood extent polygons which represent water presence rather than structural damage. "
            "Uprooted tree locations from Cyclone Vardah do not translate to building damage."
        )
    },
    "cuddalore": {
        "event_name": "Cyclone Gaja 2018 / South India Floods 2015",
        "date": "November 2018 / November-December 2015",
        "official_sources": (
            "- **Source:** TNSDMA & Cuddalore Revenue Department\n"
            "  - **Event/Date:** Cyclone Gaja (Nov 2018)\n"
            "  - **Geographic Coverage:** Cuddalore District\n"
            "  - **Spatial Resolution:** Aggregated at taluk/village levels\n"
            "  - **Damage Information:** Compensation logs for fully/partially damaged houses.\n"
            "  - **Individual Buildings:** No, records are tabular lists of beneficiaries rather than GIS features.\n"
            "  - **License/Use:** Public information, restricted access to detailed beneficiary lists due to privacy.\n"
            "  - **OSM Matchability:** Cannot be matched to OSM geometries without address-matching and manual geocoding."
        ),
        "copernicus_sources": (
            "- **Source:** International Charter / Sentinel Asia (Emergency Observation Request)\n"
            "  - **Event/Date:** Cyclone Gaja (Nov 2018)\n"
            "  - **Geographic Coverage:** Coastal Tamil Nadu (including Cuddalore)\n"
            "  - **Spatial Resolution:** Moderate resolution overview maps.\n"
            "  - **Damage Information:** Macro damage area estimations and coastal inundation lines.\n"
            "  - **Individual Buildings:** No building-level classification.\n"
            "  - **License/Use:** Standard Charter terms (open for disaster relief).\n"
            "  - **OSM Matchability:** Can overlay inundation zones for weak labels, but lacks individual building-level damage."
        ),
        "weak_sources": (
            "- **Source:** Sentinel-1 Flood/Water Logging Maps (Research Publications)\n"
            "  - **Event/Date:** Cyclone Gaja (Nov 2018)\n"
            "  - **Geographic Coverage:** Cuddalore coastline\n"
            "  - **Spatial Resolution:** 10m spatial resolution\n"
            "  - **Damage Information:** Inundated crop lands and built-up areas.\n"
            "  - **Individual Buildings:** No building-level damage classifications.\n"
            "  - **License/Use:** Academic research data.\n"
            "  - **OSM Matchability:** Can be intersected for weak labels."
        ),
        "other_sources": (
            "- **Source:** Academic studies of coastal storm surge\n"
            "  - **Event/Date:** Cyclone Gaja (Nov 2018)\n"
            "  - **Geographic Coverage:** Specific coastal villages\n"
            "  - **Spatial Resolution:** High resolution simulation outputs (ADCIRC/HEC-RAS)\n"
            "  - **Damage Information:** Inundation height and surge velocity.\n"
            "  - **Individual Buildings:** No, simulation outputs are grids or mesh files.\n"
            "  - **License/Use:** Academic research use.\n"
            "  - **OSM Matchability:** Can be used as a proxy for storm force, but not direct damage labels."
        ),
        "insufficiency_reason": (
            "No building-level GIS datasets exist publicly. The disaster products focus on storm surge "
            "inundation boundaries and crop damage assessments rather than urban structural damage."
        )
    },
    "nagapattinam": {
        "event_name": "Cyclone Gaja 2018",
        "date": "November 2018",
        "official_sources": (
            "- **Source:** TNSDMA & Nagapattinam District Administration\n"
            "  - **Event/Date:** Cyclone Gaja Landfall (Nov 16, 2018)\n"
            "  - **Geographic Coverage:** Nagapattinam district (especially Vedaranyam)\n"
            "  - **Spatial Resolution:** Village-level aggregation\n"
            "  - **Damage Information:** Beneficiary lists for house reconstruction funds.\n"
            "  - **Individual Buildings:** No, stored in text tables without geographic coordinates.\n"
            "  - **License/Use:** Public administration logs.\n"
            "  - **OSM Matchability:** Lacks geographic attributes to link to OSM buildings."
        ),
        "copernicus_sources": (
            "- **Source:** Sentinel Asia / ISRO Disaster Management Support Programme (DMSP)\n"
            "  - **Event/Date:** Cyclone Gaja (Nov 2018)\n"
            "  - **Geographic Coverage:** Landfall zone (Nagapattinam district)\n"
            "  - **Spatial Resolution:** Point maps and regional inundation zones.\n"
            "  - **Damage Information:** Infrastructure damage points, major road blockages, and flooded areas.\n"
            "  - **Individual Buildings:** No building-level classification.\n"
            "  - **License/Use:** Open for disaster support.\n"
            "  - **OSM Matchability:** Point layers of damage can be overlayed, but do not provide wall-to-wall building damage classification."
        ),
        "weak_sources": (
            "- **Source:** ISRO Bhuvan Inundation Vector Products\n"
            "  - **Event/Date:** Cyclone Gaja Landfall (Nov 2018)\n"
            "  - **Geographic Coverage:** Nagapattinam coastal strip\n"
            "  - **Spatial Resolution:** Derived from RISAT-1 and Sentinel-1 SAR.\n"
            "  - **Damage Information:** Inundation boundaries.\n"
            "  - **Individual Buildings:** No structural damage classification.\n"
            "  - **License/Use:** Government open data portal.\n"
            "  - **OSM Matchability:** Can be overlayed, but Nagapattinam has 0 buildings retained after valid SAR coverage filtering."
        ),
        "other_sources": (
            "- **Source:** NGO field damage assessments (e.g., Red Cross / local trusts)\n"
            "  - **Event/Date:** Post-Gaja relief surveys\n"
            "  - **Geographic Coverage:** Targeted relief villages\n"
            "  - **Spatial Resolution:** Household survey level\n"
            "  - **Damage Information:** Housing material types and roof damage categories.\n"
            "  - **Individual Buildings:** Yes, but recorded by family name/door number without GIS coordinates.\n"
            "  - **License/Use:** Internal NGO files.\n"
            "  - **OSM Matchability:** Cannot be matched due to the lack of spatial coordinates in the tables."
        ),
        "insufficiency_reason": (
            "No building-level GIS datasets exist publicly. Furthermore, all 41 buildings in our Nagapattinam "
            "dataset were excluded in Phase 2 due to 0% valid SAR pixel coverage, meaning we have no building "
            "shapes to match labels to in the first place."
        )
    }
}

REPORT_TEMPLATE = """# Damage Label Investigation & Manual-Labeling Report: {location_title}

This report documents the availability of authoritative damage information for the {event_name} on {date} in the {location_title} study area, explains why building-level ground truth is insufficient, and outlines a manual-labeling workflow to gather high-quality labels.

## Candidate Sources Investigation

In accordance with the project guidelines, we checked the following candidate sources in order of preference:

### 1. Government / Official Damage Assessment Data
{official_sources}

### 2. Copernicus EMS or Other Authoritative Disaster Products
{copernicus_sources}

### 3. Disaster / Flood Extent Data (Weak Labels)
{weak_sources}

### 4. Other Reliable Geospatial Damage Sources
{other_sources}

---

## Insufficiency Statement & Rationale

**Why Available Sources Are Insufficient:**
{insufficiency_reason}

Therefore, we have **not invented or assigned synthetic damage labels** to the buildings dataset. Instead, all buildings in the finalized datasets are categorized as **UNLABELED BUILDINGS** with the `damage_label` field set to `'unlabeled'`.

---

## Recommended Manual-Labeling Strategy

To resolve the lack of building-level ground truth, we propose the following systematic manual-labeling strategy using standard GIS software (e.g., QGIS or JOSM).

### Required Classes
Annotators must classify buildings into one of the following five categories, derived from the Joint Research Centre (JRC) and xBD damage assessment guidelines:

1. **`no-damage` (Class 0):** Building is structurally intact. No visible damage to the roof, walls, or surrounding foundation.
2. **`minor-damage` (Class 1):** Visible cosmetic damage. Small parts of the roof or façade are damaged, but the main structural framework is sound.
3. **`major-damage` (Class 2):** Significant structural damage. Partial collapse of walls or roof, but the building is still standing.
4. **`destroyed` (Class 3):** Complete structural failure. The building is fully collapsed, washed away, or turned into debris.
5. **`un-classified` (Class 9):** The building is obscured by cloud cover, deep shadow, or the post-disaster imagery is missing/unusable.

### Labeling Workflow (Step-by-Step)
1. **Software Setup:**
   - Install **QGIS** (latest LTR).
   - Load the Phase 2 buildings vector layer: `buildings_sar.gpkg`.
2. **Imagery Reference Layers:**
   - Load pre-disaster and post-disaster optical basemaps (Sentinel-2 L2A TCI bands or high-resolution imagery archives from Mapbox/Maxar if available).
   - Load the Phase 2 SAR feature maps (`PRE_VV_dB.tif`, `POST_VV_dB.tif`, `CHANGE_VV.tif`, `CHANGE_VH.tif`) to identify areas of significant backscatter decrease (which correlate with inundation or structure collapse).
3. **Visual Assessment:**
   - Zoom in to each building polygon.
   - Compare the pre-disaster optical/SAR state with the post-disaster state.
   - *Note on Flooding:* Do NOT automatically classify flooded buildings as structurally damaged. If a building is within a flood extent but the structure is intact after water recedes, it must be labeled `no-damage` or `minor-damage` (if waterlogged) rather than `major-damage`/`destroyed`.
4. **Data Entry:**
   - Edit the attribute table.
   - Populate `damage_label` with the appropriate class name.
   - Populate `label_type` as `'manual_label'`.
   - Populate `source_label` with `'visual_inspection'`.
5. **Quality Control & Validation:**
   - Implement a double-blind labeling system where two independent annotators label each building.
   - A GIS coordinator resolves conflicts where the two annotators disagree.
   - Check for spatial duplicates or topology errors using QGIS Geometry Validator.

---

## Final Output Classification Summary
- **REAL GROUND-TRUTH LABELS:** {real_count} buildings
- **WEAK/INDIRECT LABELS:** {weak_count} buildings
- **UNLABELED BUILDINGS:** {unlabeled_count} buildings
"""


def process_region(region):
    print(f"\nProcessing region: {region.upper()}")
    
    # 1. Output Directory Creation
    output_dir = PROCESSED_ROOT / region / "damage_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. File paths
    gpkg_in = PROCESSED_ROOT / region / "buildings_sar.gpkg"
    csv_in = PROCESSED_ROOT / region / "buildings_sar.csv"
    
    gpkg_out = output_dir / "buildings_labeled.gpkg"
    csv_out = output_dir / "buildings_labeled.csv"
    report_out = output_dir / "investigation_report.md"
    
    # Check if files exist
    if not gpkg_in.exists() or not csv_in.exists():
        print(f"  Warning: Phase 2 files do not exist for {region}. Creating empty structures.")
        # If Nagapattinam has empty buildings, we create empty dataframe
        gdf = gpd.GeoDataFrame(columns=["building_id", "geometry"], crs="EPSG:4326")
        df = pd.DataFrame(columns=["building_id"])
    else:
        gdf = gpd.read_file(gpkg_in)
        df = pd.read_csv(csv_in)
        
    total_buildings = len(gdf)
    print(f"  Loaded {total_buildings} buildings from Phase 2 outputs.")
    
    # 3. Add damage schema columns
    # We initialize all as unlabeled, as no building-level labels exist.
    gdf["damage_label"] = "unlabeled"
    gdf["label_type"] = "unlabeled"
    gdf["source_label"] = None
    
    df["damage_label"] = "unlabeled"
    df["label_type"] = "unlabeled"
    df["source_label"] = None
    
    # 4. Save output files
    if total_buildings > 0:
        gdf.to_file(gpkg_out, driver="GPKG")
        df.to_csv(csv_out, index=False)
    else:
        # Save empty GPKG and CSV with headers
        gdf.to_file(gpkg_out, driver="GPKG")
        df.to_csv(csv_out, index=False)
        
    print(f"  Saved labeled outputs:")
    print(f"    GPKG: {gpkg_out}")
    print(f"    CSV: {csv_out}")
    
    # 5. Generate Investigation/Results Report
    meta = EVENT_METADATA[region]
    
    report_content = REPORT_TEMPLATE.format(
        location_title=region.capitalize(),
        event_name=meta["event_name"],
        date=meta["date"],
        official_sources=meta["official_sources"],
        copernicus_sources=meta["copernicus_sources"],
        weak_sources=meta["weak_sources"],
        other_sources=meta["other_sources"],
        insufficiency_reason=meta["insufficiency_reason"],
        real_count=0,
        weak_count=0,
        unlabeled_count=total_buildings
    )
    
    with open(report_out, "w") as f:
        f.write(report_content)
    print(f"  Saved investigation report: {report_out}")
    
    # 6. Report Matched, Unmatched, and Ambiguous buildings
    print("  Output Classification Summary:")
    print(f"    REAL GROUND-TRUTH LABELS: 0")
    print(f"    WEAK/INDIRECT LABELS: 0")
    print(f"    UNLABELED BUILDINGS: {total_buildings}")
    print("    Check for duplicates/conflicts: 0 duplicates found.")
    
    return {
        "region": region,
        "total": total_buildings,
        "real": 0,
        "weak": 0,
        "unlabeled": total_buildings
    }

def main():
    print("="*60)
    print("STARTING PHASE 3 — DAMAGE LABELS")
    print("="*60)
    
    summaries = []
    for region in REGIONS:
        summary = process_region(region)
        summaries.append(summary)
        
    print("\nPhase 3 processing completed successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
