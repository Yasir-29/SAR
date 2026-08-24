# GitHub and Google Colab Setup Documentation

This document describes the structure, backup plan, and setup procedure for training the Siamese / Dual-Stream U-Net damage-detection model in Google Colab using our processed building-level dataset.

## 1. Project Identifiers
- **GitHub Repository:** `https://github.com/Yasir-29/SAR.git`
- **Local Project Path:** `/Users/yasir/Desktop/sardd/SAR/xbd-s12`

## 2. Storage Architecture

We split project artifacts into code-repositories (GitHub) and external large-file stores (Google Drive or similar external storage) to avoid committing huge binary files to Git history.

### A. Saved in GitHub Repository
- **Source Code:** All Python scripts under `src/` (including pipeline tools `src/data/integrate_optical.py` and `src/data/prepare_model_dataset.py`).
- **Configuration Files:** All YAML configs under `src/configs/`.
- **Documentation & Reports:** Phase 3 labeling investigation, Phase 4 reports, and Phase 5 dataset reports.
- **Dataset Indices:** [dataset_index.csv](file:///Users/yasir/Desktop/sardd/SAR/xbd-s12/data/model_ready/dataset_index.csv) containing building records, splits, and file paths.
- **Normalization Metadata:** [normalization.json](file:///Users/yasir/Desktop/sardd/SAR/xbd-s12/data/model_ready/normalization.json) containing training set channel-wise mean and std values.
- **Environment Config:** `requirements.txt` and `pyproject.toml`.

### B. Stored Externally (Google Drive / External Storage)
The following directories and files are ignored by git in [.gitignore](file:///Users/yasir/Desktop/sardd/SAR/xbd-s12/.gitignore):

#### Raw & Processed Satellite Rasters (under `data/processed/` and `data/optical_data/`)
- `data/raw/` (Raw Sentinel-1 zip files and UNOSAT shapefile datasets)
- `data/optical_data/` (Sentinel-2 multi-band PRE/POST GeoTIFFs)
- `data/processed/**/*.tif` (Aligned SAR and optical bands in GeoTIFF format)
- `data/south_coastal/` (Auxiliary local datasets)

#### Generated ML Training Arrays (under `data/model_ready/`)
- `data/model_ready/chennai/` (numpy arrays of chips and building masks)
- `data/model_ready/cuddalore/` (numpy arrays of chips and building masks)

## 3. Large Files List (>50MB)

| File Path | File Size | Classification |
| :--- | :---: | :--- |
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

## 4. Google Colab Setup Execution Workflow

During Phase 6 model training, Google Colab will perform:
1. **Repository Cloning:**
   ```bash
   !git clone https://github.com/Yasir-29/SAR.git
   %cd SAR/xbd-s12
   ```
2. **Package Installation:**
   ```bash
   !pip install -r requirements.txt
   ```
3. **Data Mounting:**
   - Mount Google Drive using PyColab drive tools.
   - Access the `.npy` files from the drive mount (e.g. `/content/drive/MyDrive/SAR_model_ready/`).
4. **Data Verification:** Load [dataset_index.csv](file:///Users/yasir/Desktop/sardd/SAR/xbd-s12/data/model_ready/dataset_index.csv) locally, instantiate the PyTorch `Dataset`, and stream chips for train/val/test training loops.

## 5. Phase Status Summary
- **Phase 5 (Dataset Prep):** **COMPLETED**. Validation report generated and visual validation completed.
- **Phase 6 (Model Training):** **PREPARED**. PyTorch dataset and requirements are configured; model implementation and training will follow.
