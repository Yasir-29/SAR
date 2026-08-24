# Damage Label Investigation & Manual-Labeling Report: Nagapattinam

This report documents the availability of authoritative damage information for the Cyclone Gaja 2018 on November 2018 in the Nagapattinam study area, explains why building-level ground truth is insufficient, and outlines a manual-labeling workflow to gather high-quality labels.

## Candidate Sources Investigation

In accordance with the project guidelines, we checked the following candidate sources in order of preference:

### 1. Government / Official Damage Assessment Data
- **Source:** TNSDMA & Nagapattinam District Administration
  - **Event/Date:** Cyclone Gaja Landfall (Nov 16, 2018)
  - **Geographic Coverage:** Nagapattinam district (especially Vedaranyam)
  - **Spatial Resolution:** Village-level aggregation
  - **Damage Information:** Beneficiary lists for house reconstruction funds.
  - **Individual Buildings:** No, stored in text tables without geographic coordinates.
  - **License/Use:** Public administration logs.
  - **OSM Matchability:** Lacks geographic attributes to link to OSM buildings.

### 2. Copernicus EMS or Other Authoritative Disaster Products
- **Source:** Sentinel Asia / ISRO Disaster Management Support Programme (DMSP)
  - **Event/Date:** Cyclone Gaja (Nov 2018)
  - **Geographic Coverage:** Landfall zone (Nagapattinam district)
  - **Spatial Resolution:** Point maps and regional inundation zones.
  - **Damage Information:** Infrastructure damage points, major road blockages, and flooded areas.
  - **Individual Buildings:** No building-level classification.
  - **License/Use:** Open for disaster support.
  - **OSM Matchability:** Point layers of damage can be overlayed, but do not provide wall-to-wall building damage classification.

### 3. Disaster / Flood Extent Data (Weak Labels)
- **Source:** ISRO Bhuvan Inundation Vector Products
  - **Event/Date:** Cyclone Gaja Landfall (Nov 2018)
  - **Geographic Coverage:** Nagapattinam coastal strip
  - **Spatial Resolution:** Derived from RISAT-1 and Sentinel-1 SAR.
  - **Damage Information:** Inundation boundaries.
  - **Individual Buildings:** No structural damage classification.
  - **License/Use:** Government open data portal.
  - **OSM Matchability:** Can be overlayed, but Nagapattinam has 0 buildings retained after valid SAR coverage filtering.

### 4. Other Reliable Geospatial Damage Sources
- **Source:** NGO field damage assessments (e.g., Red Cross / local trusts)
  - **Event/Date:** Post-Gaja relief surveys
  - **Geographic Coverage:** Targeted relief villages
  - **Spatial Resolution:** Household survey level
  - **Damage Information:** Housing material types and roof damage categories.
  - **Individual Buildings:** Yes, but recorded by family name/door number without GIS coordinates.
  - **License/Use:** Internal NGO files.
  - **OSM Matchability:** Cannot be matched due to the lack of spatial coordinates in the tables.

---

## Insufficiency Statement & Rationale

**Why Available Sources Are Insufficient:**
No building-level GIS datasets exist publicly. Furthermore, all 41 buildings in our Nagapattinam dataset were excluded in Phase 2 due to 0% valid SAR pixel coverage, meaning we have no building shapes to match labels to in the first place.

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
- **UNLABELED BUILDINGS:** 0 buildings
