# xBD-S12 Dataset Scale and Data Selection

This document summarizes the dataset scale, sample selection, and spatial leakage prevention strategy for Phase 6.

## 1. xBD-S12 Dataset Scale
The official xBD-S12 dataset contains:
- **Total Patches:** 10,315 features in `xbd_s12_metadata.geojson`.
- **Total Buildings:** 372,052 building instances.
- **Disaster Events:** 16 disaster events (e.g. hurricane-michael, nepal-flooding).
- **Perils Covered:** Flooding, storm/wind, volcanic eruption, fire, tsunami, earthquake.

## 2. Training Sample Selection
To train a robust damage-detection model, we selected a balanced subset of **10,000 buildings**:
- **Class 0 (Intact):** 4,000 buildings
- **Class 1 (Damaged):** 3,000 buildings (combining minor + major damage)
- **Class 2 (Destroyed):** 3,000 buildings

### Selection Rules:
1. **Peril Prioritization:** Primary disasters representing flooding and storm/wind hazards (matching the South India monsoon context) were prioritized.
2. **Exclude Garbage Labels:** Excluded buildings with unclassified labels, missing geometries, or unusable Sentinel imagery.

## 3. Spatial Leakage Prevention Split
To prevent spatial data leakage:
- **Patch-level Split Assignment:** Splits are assigned at the patch/disaster-event level rather than randomly splitting buildings. This ensures that all buildings inside the same 128x128 pixel patch are grouped into the same split (either Train, Val, or Test).
- **Split Proportions:** The dataset is split into:
  - **Train:** 6,252 buildings (62.5%)
  - **Validation:** 1,509 buildings (15.1%)
  - **Test:** 2,239 buildings (22.4%)
- This distribution achieves the target splits while maintaining class balance.
