# Building Damage Ground Truth & Labeling Report: Nagapattinam

This report documents the creation of the building damage dataset for the Nagapattinam study area using manual annotation of high-resolution visual evidence and spatial matching to independent disaster extent layers, conforming to the Phase 4 requirements.

## 1. Visual Evidence and Independent Disaster Data
- **Disaster Event:** Cyclone Gaja (2018)
- **Evidence Date Range:** Pre-disaster: October 2018 / Post-disaster: November 18-25, 2018
- **Platform/Sensor:** Pleiades-1A/1B (CNES / Airbus optical imagery)
- **Spatial Resolution:** 0.5m (optical)
- **Description:** No buildings retained after Phase 2 Sentinel-1 valid coverage filtering.

**Note on Dataset Availability:** Although Nagapattinam contains 41 mapped OSM building footprints, all 41 buildings were excluded during Phase 2 preprocessing due to 0% valid Sentinel-1 SAR coverage (no-data masking). Thus, they are not usable in the final building damage dataset. This does not mean there are zero buildings in Nagapattinam, but rather that no buildings met the valid SAR coverage threshold required for feature extraction.

---

## 2. Spatial Matching Methodology
1. **Real Ground-Truth (Class 0, 1, 2):** Since no authoritative building-level damage databases exist for these regions, the count of `REAL GROUND-TRUTH` is 0.
2. **Manual Labels (Class 0, 1, 2):** Selected representative samples of buildings were manually annotated based on visual inspection of high-resolution pre- and post-disaster optical imagery (representing a manual labeling session).
3. **Weak/Indirect Labels (Class unlabeled, source 'weak/indirect'):** 
   - **Chennai:** Spatial intersection with the official UNOSAT satellite-detected flood waters shapefile. If a building intersects with a flood polygon (and has no manual label), it is marked as `weak/indirect` (indicating it was in the affected/flooded area but not structurally damaged).
   - **Cuddalore:** Spatial proximity filter of 150m around the Gadilam River centerline. If a building lies in this inundation zone, it is marked as `weak/indirect`.
4. **Unlabeled Buildings:** All remaining buildings without sufficient visual evidence or inundation mapping overlap are left as `unlabeled`.

---

## 3. Labeling Statistics Breakdown

- **Total Buildings in Dataset:** 0
- **Intact Buildings (Class 0):** 0
- **Damaged Buildings (Class 1):** 0
- **Destroyed Buildings (Class 2):** 0
- **Weak/Indirect Labels:** 0
- **Unlabeled Buildings:** 0

### Confidence Levels
- **High-Confidence Labels:** 0
- **Medium-Confidence Labels:** 0
- **Low-Confidence Labels:** 0
- **Buildings without sufficient evidence:** 0

### Final Dataset Label Classification Summary
- **REAL GROUND-TRUTH:** 0
- **MANUAL LABEL:** 0
- **WEAK/INDIRECT LABEL:** 0
- **UNLABELED:** 0

---

## 4. Annotation Sample Table

For the manual labeling session, we selected a representative subset of buildings in the study area:

*No manual annotations were performed for this study area due to 0 retained buildings.*
