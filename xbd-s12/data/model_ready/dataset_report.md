# Damage-Detection Model Ready Dataset Report (Phase 5)

This report documents the creation, statistics, splitting, and normalization of the building-level training dataset for the dual-stream U-Net damage-detection model.

## 1. Summary Statistics

- **Total Mapped Buildings:** 3234
  - Chennai: 3022
  - Cuddalore: 212
- **Supervised Training Split (Labeled):** 30
- **Excluded (Unlabeled / Flood-only):** 3204

## 2. Class Distribution (Supervised Dataset)

| Class | Label Value | Description | Count | Percentage (Supervised) |
| :--- | :---: | :--- | :---: | :---: |
| **Intact** | `0` | Footprint is structurally intact | 23 | 76.67% |
| **Damaged** | `1` | Footprint shows structural damage | 4 | 13.33% |
| **Destroyed** | `2` | Building is completely collapsed | 3 | 10.00% |
| **Unlabeled** | `unlabeled` | Coarse resolution / flood-only (excluded) | 3204 | N/A |

> [!WARNING]
> **Severe Class Imbalance:** The labeled dataset exhibits severe class imbalance (Intact: ~77%, Damaged: ~13%, Destroyed: ~10%). ML models should utilize focal loss, class weights, or oversampling during training (Phase 6) to prevent bias towards the majority "intact" class.

## 3. Dataset Splits (Supervised Set Only)

To prevent spatial leakage, buildings in each region were sorted by centroid longitude (x-coordinate) and split geographically:
- **TRAIN Set (Westernmost 60%):** 18 samples (60.0%)
- **VALIDATION Set (Central 20%):** 6 samples (20.0%)
- **TEST Set (Easternmost 20%):** 6 samples (20.0%)

This ensures that train, validation, and test subsets represent non-overlapping geographic blocks, preventing adjacent buildings from being leaked across sets.

## 4. Normalization Statistics (Training Set Only)

Normalization parameters were computed strictly using the training set splits.

### SAR Normalization (channel_mean_std)
- **VV_PRE Mean / Std:** -1.4685 / 5.0658
- **VH_PRE Mean / Std:** -12.4117 / 3.1462

### Optical Normalization (channel_mean_std)
- **B2 (Blue) Mean / Std:** 0.1144 / 0.1189
- **B3 (Green) Mean / Std:** 0.1284 / 0.1053
- **B4 (Red) Mean / Std:** 0.1301 / 0.1007
- **B8 (NIR) Mean / Std:** 0.2215 / 0.1113

## 5. Visual Validation Samples

The following are sample visualizations extracted for the target building classes:

### Intact Class Sample
![Intact Building Sample](file:///Users/yasir/.gemini/antigravity-ide/brain/10d9e7ca-c93b-4981-983d-1c5bed52c20b/visualizations/intact.png)

### Damaged Class Sample
![Damaged Building Sample](file:///Users/yasir/.gemini/antigravity-ide/brain/10d9e7ca-c93b-4981-983d-1c5bed52c20b/visualizations/damaged.png)

### Destroyed Class Sample
![Destroyed Building Sample](file:///Users/yasir/.gemini/antigravity-ide/brain/10d9e7ca-c93b-4981-983d-1c5bed52c20b/visualizations/destroyed.png)

## 6. Important Limitations of Sentinel-2 Resolution

Sentinel-2 imagery has a spatial resolution of approximately 10 m per pixel. A 32x32 pixel chip represents a spatial area of 320 m x 320 m. Because typical buildings are smaller than 20 m in length, they occupy very few pixels (1 to 9 pixels) on this grid. 
Sentinel-2 data cannot be used to establish authoritative building damage labels. It is included strictly as supporting/contextual evidence inside the model's optical stream, while the manual visual annotations are the available ground-truth labels.

## 7. Data Quality & final Verification

- **Valid geometries:** 100% of geometries are clean and valid.
- **Valid SAR Coverage:** 92.42% of chips have 100% valid SAR pixels (no NaNs).
- **Valid Optical Coverage:** 100.00% of chips have 100% valid optical pixels (no NaNs).
- **Channel Ordering:**
  - SAR PRE: `[VV_PRE, VH_PRE]`
  - SAR POST: `[VV_POST, VH_POST]`
  - Optical PRE: `[B2, B3, B4, B8]`
  - Optical POST: `[B2, B3, B4, B8]`
  - Mask: `[building_mask]`
- **No Unlabeled Leakage:** Checked that no building with damage_label "unlabeled" enters the train/val/test splits.
