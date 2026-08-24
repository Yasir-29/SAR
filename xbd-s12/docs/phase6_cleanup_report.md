# Phase 6 Cleanup Report

This report documents the repository inspection, lists files that are safe to delete, files that must be preserved, credentials/secrets identification, and large files.

## 1. Files Safe to Delete / Obsolete Files
These files are either temporary, cache files, or duplicates and can be safely cleaned up:
- **`__pycache__/` directories:** Python compiled bytecode cache directories located under `src/` and its subfolders.
- **`.ipynb_checkpoints/`:** Hidden Jupyter Notebook auto-save directories.
- **`*.log` files:** Any debug or process logs from previous executions.
- **`xbd-s12/phase6_training_data/` (Temporary Subset):** Contains 162 temporary duplicate files of Chennai/Cuddalore npy chips and dataset configurations created during previous stages. They can be safely deleted or ignored since the official results are stored in `data/model_ready/`.

## 2. Files that MUST be Preserved
The following folders and files contain critical project code, verified labels, or dataset indices and **MUST NOT** be deleted:
- **Source Code (`src/`):** All Python files under `src/data/`, `src/models/`, `src/configs/`, and `src/training/`.
- **Manually Verified Labels & Ground-Truth Files:**
  - `data/processed/chennai/damage_labels/buildings_ground_truth.csv` and `.gpkg`
  - `data/processed/cuddalore/damage_labels/buildings_ground_truth.csv` and `.gpkg`
  - `data/processed/nagapattinam/damage_labels/buildings_ground_truth.csv` and `.gpkg`
- **Phase 4 & 5 Reports:**
  - `data/processed/chennai/damage_labels/optical_evidence_report.md`
  - `data/processed/chennai/damage_labels/labeling_report.md`
  - `data/processed/cuddalore/damage_labels/optical_evidence_report.md`
  - `data/processed/cuddalore/damage_labels/labeling_report.md`
  - `data/model_ready/dataset_report.md`
- **Model-Ready Metadata & Indices:**
  - `data/model_ready/dataset_index.csv`
  - `data/model_ready/normalization.json`
- **xBD-S12 Dataset Metadata:**
  - `src/data/xbd_s12_metadata.geojson`
  - `data/xbd_s12/xbd_s12_metadata.geojson`

## 3. Credentials and Secrets Protection
The following credential files were found in the workspace:
- **`.env`**: Contains local environment configurations.
- **`.s5cfg`**: Contains Copernicus Data Space Ecosystem (CDSE) / Earth Engine access configurations.
- **Status:** Both files are securely listed in the project-level `.gitignore` and **will never** be committed or pushed to GitHub.

## 4. Large Files (>50MB)
The following raw satellite GeoTIFF files and shapefiles are too large for Git and are ignored in `.gitignore`:

| File Path | Size | Classification |
| :--- | :---: | :--- |
| `xbd_s12.tar.gz` (Project Root) | 9.53 GB | External storage candidate |
| `data/optical_data/chennai/PRE/optical.tif` | 236.6 MB | External storage candidate |
| `data/optical_data/chennai/POST/optical.tif` | 236.6 MB | External storage candidate |
| `data/optical_data/cuddalore/PRE/optical.tif` | 121.6 MB | External storage candidate |
| `data/optical_data/cuddalore/POST/optical.tif` | 121.6 MB | External storage candidate |
| `data/processed/chennai/PRE_OPTICAL_FEATURES.tif` | 128.0 MB | External storage candidate |
| `data/processed/chennai/POST_OPTICAL_FEATURES.tif` | 128.0 MB | External storage candidate |
| `data/processed/cuddalore/PRE_OPTICAL_FEATURES.tif` | 128.0 MB | External storage candidate |
| `data/processed/cuddalore/POST_OPTICAL_FEATURES.tif` | 128.0 MB | External storage candidate |
| `data/processed/nagapattinam/PRE_OPTICAL_FEATURES.tif` | 128.0 MB | External storage candidate |
| `data/processed/nagapattinam/PRE_S2_OPTICAL.tif` | 80.0 MB | External storage candidate |
| `data/processed/chennai/PRE_S2_OPTICAL.tif` | 80.0 MB | External storage candidate |
| `data/processed/chennai/POST_S2_OPTICAL.tif` | 80.0 MB | External storage candidate |
| `data/processed/cuddalore/PRE_S2_OPTICAL.tif` | 80.0 MB | External storage candidate |
| `data/processed/cuddalore/POST_S2_OPTICAL.tif` | 80.0 MB | External storage candidate |
| `data/raw/FL20151123IND_shp.zip` | 67.3 MB | External storage candidate |
| `data/raw/unosat_chennai_2015/S1_20151112_Flood.shp` | 64.0 MB | External storage candidate |
| `data/raw/unosat_chennai_2015/S1_20151124_Flood.shp` | 80.0 MB | External storage candidate |
