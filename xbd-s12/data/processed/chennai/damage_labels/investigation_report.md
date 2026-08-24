# Damage Label Investigation & Manual-Labeling Report: Chennai

This report documents the availability of authoritative damage information for the Chennai Floods 2015 / Cyclone Vardah 2016 on November-December 2015 / December 2016 in the Chennai study area, explains why building-level ground truth is insufficient, and outlines a manual-labeling workflow to gather high-quality labels.

## Candidate Sources Investigation

In accordance with the project guidelines, we checked the following candidate sources in order of preference:

### 1. Government / Official Damage Assessment Data
- **Source:** Tamil Nadu State Disaster Management Authority (TNSDMA) & Chennai Metropolitan Development Authority (CMDA)
  - **Event/Date:** Chennai Floods (Nov-Dec 2015), Cyclone Vardah (Dec 2016)
  - **Geographic Coverage:** Chennai Metropolitan Area
  - **Spatial Resolution:** Aggregated at block/ward levels (no coordinates)
  - **Damage Information:** Counts of fully/partially damaged houses, collapsed huts, infrastructure damage costs.
  - **Individual Buildings:** No, reports only contain aggregate numbers.
  - **License/Use:** Public government report data (restricted to non-commercial research use).
  - **OSM Matchability:** Cannot be matched; lacks coordinates or building-specific identifiers.

### 2. Copernicus EMS or Other Authoritative Disaster Products
- **Source:** Copernicus EMS / International Charter Space and Major Disasters (Call ID 561)
  - **Event/Date:** South India Floods (December 2015)
  - **Geographic Coverage:** Chennai and surrounding districts
  - **Spatial Resolution:** Macro overview mapping based on RADARSAT-2, TerraSAR-X, and Sentinel-1.
  - **Damage Information:** Inundation footprints and damage overview vector layers.
  - **Individual Buildings:** No individual building-level structural damage classification is provided.
  - **License/Use:** Open access for humanitarian and research use.
  - **OSM Matchability:** Inundation footprints can be spatially intersected with OSM buildings to produce weak/indirect labels, but not real structural damage.

### 3. Disaster / Flood Extent Data (Weak Labels)
- **Source:** UNITAR/UNOSAT Flood Inundation Vector Products / Dartmouth Flood Observatory (DFO)
  - **Event/Date:** Chennai Floods (Dec 2015)
  - **Geographic Coverage:** Southern Chennai area
  - **Spatial Resolution:** Derived from 10m/30m satellite imagery.
  - **Damage Information:** Flood extent polygons.
  - **Individual Buildings:** No building structural damage.
  - **License/Use:** Open GIS data (Creative Commons).
  - **OSM Matchability:** Can be spatially intersected to assign weak labels (`weak_flooded` vs `weak_non_flooded`).

### 4. Other Reliable Geospatial Damage Sources
- **Source:** Crowdsourced Flooding Hotspots (OpenCity.in / Oorvani Foundation)
  - **Event/Date:** Nov-Dec 2015 Floods
  - **Geographic Coverage:** Chennai Municipal area
  - **Spatial Resolution:** Point-based user reports and flood heights.
  - **Damage Information:** Stagnation points and water logging depths.
  - **Individual Buildings:** Some street-level descriptions, but no building shape matching.
  - **License/Use:** Open Data (ODbL).
  - **OSM Matchability:** Extremely poor matchability due to manual geocoding inaccuracies.

---

## Insufficiency Statement & Rationale

**Why Available Sources Are Insufficient:**
No authoritative source provides structural building-level damage labels. The official statistics are aggregated ward-level tables, and the Copernicus/Charter/UNOSAT products provide flood extent polygons which represent water presence rather than structural damage. Uprooted tree locations from Cyclone Vardah do not translate to building damage.

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
- **UNLABELED BUILDINGS:** 3022 buildings
