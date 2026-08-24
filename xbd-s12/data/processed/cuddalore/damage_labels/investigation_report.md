# Damage Label Investigation & Manual-Labeling Report: Cuddalore

This report documents the availability of authoritative damage information for the Cyclone Gaja 2018 / South India Floods 2015 on November 2018 / November-December 2015 in the Cuddalore study area, explains why building-level ground truth is insufficient, and outlines a manual-labeling workflow to gather high-quality labels.

## Candidate Sources Investigation

In accordance with the project guidelines, we checked the following candidate sources in order of preference:

### 1. Government / Official Damage Assessment Data
- **Source:** TNSDMA & Cuddalore Revenue Department
  - **Event/Date:** Cyclone Gaja (Nov 2018)
  - **Geographic Coverage:** Cuddalore District
  - **Spatial Resolution:** Aggregated at taluk/village levels
  - **Damage Information:** Compensation logs for fully/partially damaged houses.
  - **Individual Buildings:** No, records are tabular lists of beneficiaries rather than GIS features.
  - **License/Use:** Public information, restricted access to detailed beneficiary lists due to privacy.
  - **OSM Matchability:** Cannot be matched to OSM geometries without address-matching and manual geocoding.

### 2. Copernicus EMS or Other Authoritative Disaster Products
- **Source:** International Charter / Sentinel Asia (Emergency Observation Request)
  - **Event/Date:** Cyclone Gaja (Nov 2018)
  - **Geographic Coverage:** Coastal Tamil Nadu (including Cuddalore)
  - **Spatial Resolution:** Moderate resolution overview maps.
  - **Damage Information:** Macro damage area estimations and coastal inundation lines.
  - **Individual Buildings:** No building-level classification.
  - **License/Use:** Standard Charter terms (open for disaster relief).
  - **OSM Matchability:** Can overlay inundation zones for weak labels, but lacks individual building-level damage.

### 3. Disaster / Flood Extent Data (Weak Labels)
- **Source:** Sentinel-1 Flood/Water Logging Maps (Research Publications)
  - **Event/Date:** Cyclone Gaja (Nov 2018)
  - **Geographic Coverage:** Cuddalore coastline
  - **Spatial Resolution:** 10m spatial resolution
  - **Damage Information:** Inundated crop lands and built-up areas.
  - **Individual Buildings:** No building-level damage classifications.
  - **License/Use:** Academic research data.
  - **OSM Matchability:** Can be intersected for weak labels.

### 4. Other Reliable Geospatial Damage Sources
- **Source:** Academic studies of coastal storm surge
  - **Event/Date:** Cyclone Gaja (Nov 2018)
  - **Geographic Coverage:** Specific coastal villages
  - **Spatial Resolution:** High resolution simulation outputs (ADCIRC/HEC-RAS)
  - **Damage Information:** Inundation height and surge velocity.
  - **Individual Buildings:** No, simulation outputs are grids or mesh files.
  - **License/Use:** Academic research use.
  - **OSM Matchability:** Can be used as a proxy for storm force, but not direct damage labels.

---

## Insufficiency Statement & Rationale

**Why Available Sources Are Insufficient:**
No building-level GIS datasets exist publicly. The disaster products focus on storm surge inundation boundaries and crop damage assessments rather than urban structural damage.

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
- **REAL GROUND-TRUTH LABELS:** 0 buildings
- **WEAK/INDIRECT LABELS:** 0 buildings
- **UNLABELED BUILDINGS:** 212 buildings
