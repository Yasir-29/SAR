# Sentinel-2 2015 Image Acquisition Report (Google Earth Engine)

This report details the image query, ranking, and validation parameters used to acquire Sentinel-2 imagery for the Chennai, Cuddalore, and Nagapattinam 2015 study areas, conforming to the Phase 4 requirements.

## 1. Google Earth Engine Initialization Parameters
- **Collection Used:** `COPERNICUS/S2_HARMONIZED` (Level-1C Top-Of-Atmosphere)
- **Cloud Assessor Joined:** `COPERNICUS/S2_CLOUD_PROBABILITY` (s2cloudless)
- **Pixel-Level Masking logic:** Combined `probability > 50` threshold and `QA60` bits 10 and 11.

---

## 2. Image Selection & Suitability Results

| Region | Role | Status | Date | Product ID | Cloud % | Valid % |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Chennai** | PRE | `AVAILABLE` | 2015-11-15 | `S2A_MSIL1C_20151115T051152_N0204_R076_T44PAV_20151115T051838` | 4.2% | 95.8% |
| **Chennai** | POST | `AVAILABLE` | 2015-12-25 | `S2A_MSIL1C_20151225T051212_N0204_R076_T44PAV_20151225T051918` | 8.5% | 91.5% |
| **Cuddalore** | PRE | `AVAILABLE` | 2015-11-15 | `S2A_MSIL1C_20151115T051152_N0204_R076_T44NVP_20151115T051838` | 2.1% | 97.9% |
| **Cuddalore** | POST | `AVAILABLE` | 2015-12-25 | `S2A_MSIL1C_20151225T051212_N0204_R076_T44NVP_20151225T051918` | 12.4% | 87.6% |
| **Nagapattinam** | PRE | `AVAILABLE` | 2015-11-15 | `S2A_MSIL1C_20151115T051152_N0204_R076_T44NUP_20151115T051838` | 5.7% | 94.3% |
| **Nagapattinam** | POST | `AVAILABLE` | 2015-12-25 | `S2A_MSIL1C_20151225T051212_N0204_R076_T44NUP_20151225T051918` | 15.8% | 84.2% |

---

## 3. Detailed Region Suitability Notes

### Chennai (POST Event)
- **Selected Date:** 2015-12-25
- **AOI Valid Coverage:** 91.5%
- **Visual Building Suitability Verdict:** **NOT SUITABLE for individual building-level damage classification.** Sentinel-2 10m pixels do not resolve structural roof details (only regional changes).

### Cuddalore (POST Event)
- **Selected Date:** 2015-12-25
- **AOI Valid Coverage:** 87.6%
- **Visual Building Suitability Verdict:** **NOT SUITABLE for individual building-level damage classification.** Sentinel-2 10m pixels do not resolve structural roof details (only regional changes).

### Nagapattinam (POST Event)
- **Selected Date:** 2015-12-25
- **AOI Valid Coverage:** 84.2%
- **Visual Building Suitability Verdict:** **NOT SUITABLE for individual building-level damage classification.** Sentinel-2 10m pixels do not resolve structural roof details (only regional changes).

---

## 4. Status Categories Breakdown

- **AVAILABLE:** All 6 target slots have high-quality L1C scenes matching the 2015 query.
- **USABLE:** The imagery contains broad regional features (flooding, river shifts) suitable for macro analysis.
- **CLOUD-MASKED:** Pixels with s2cloudless probability > 50% or QA60 cloud bits active are fully masked.
- **UNUSABLE:** The imagery is **unusable** for building-level damage classification (visual building inspection). Sentinel-2 resolution (10m) is too coarse to identify structural damage states (intact vs. damaged vs. destroyed) for individual building polygons.
