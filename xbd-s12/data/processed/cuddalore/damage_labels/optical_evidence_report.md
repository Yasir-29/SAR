# Sentinel-2 Optical & Flood Evidence Integration Report: CUDDALORE

This report summarizes the integration of Sentinel-2 optical data and spatial flood extents as supporting evidence for building-level damage analysis.

## 1. Summary Statistics

- **Total Input Buildings:** 212
- **Number covered by PRE imagery bounds:** 212
- **Number covered by POST imagery bounds:** 212
- **Number covered by both PRE and POST:** 212

## 2. Sentinel-2 Scene Information

- **Image/Product ID (PRE):** `S2A_MSIL2A_20151128T050142_N0500_R119_T44PLT_20231017T062052.SAFE, S2A_MSIL2A_20151128T050142_N0500_R119_T44PLU_20231017T062052.SAFE`
- **Image/Product ID (POST):** `S2A_MSIL2A_20151228T050222_N0500_R119_T44PLT_20231009T231712.SAFE, S2A_MSIL2A_20151228T050222_N0500_R119_T44PLU_20231009T231712.SAFE`
- **Acquisition Date (PRE):** `2015-11-28`
- **Acquisition Date (POST):** `2015-12-28`
- **Optical Source:** `Sentinel-2`
- **Optical Resolution:** `10.0 m`
- **CRS:** `EPSG:32644`

## 3. Pixel Statistics

- **PRE Valid-Pixel Statistics (per building):**
  - Min: 2.0
  - Max: 87.0
  - Mean: 11.33
  - Std Dev: 10.44
- **POST Valid-Pixel Statistics (per building):**
  - Min: 2.0
  - Max: 87.0
  - Mean: 11.33
  - Std Dev: 10.44

## 4. Evidence & Ground Truth Summary

- **Number of buildings with flood-overlap evidence:** 49
- **Number of buildings with other documented evidence (manual):** 10
- **Number of buildings with reliable damage labels (Class 0, 1, 2):** 10
- **Number of buildings remaining unlabeled:** 202

## 5. Limitations of Sentinel-2 Spatial Resolution

Sentinel-2 imagery has a spatial resolution of approximately 10 meters per pixel, which corresponds to a surface footprint of 100 square meters per pixel. This scale is too coarse to identify structural details of individual buildings (e.g., roof displacement, debris accumulation, wall collapses). Consequently, optical change or flood intersection from Sentinel-2 data alone cannot support a building-level damage or destruction label. It must be treated solely as weak, indirect, or supporting evidence.

## 6. Final Scientific Statement

"These are building-level SAR features supplemented with Sentinel-2 optical and flood-evidence information. Sentinel-2 optical imagery is used as supporting evidence only. Sentinel-2 at approximately 10 m resolution is not sufficient by itself to establish individual-building structural damage. Buildings without reliable building-level evidence remain UNLABELED. No unsupported damage classification has been performed."
