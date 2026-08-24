# Sentinel-2 Optical & Flood Evidence Integration Report: CHENNAI

This report summarizes the integration of Sentinel-2 optical data and spatial flood extents as supporting evidence for building-level damage analysis.

## 1. Summary Statistics

- **Total Input Buildings:** 3022
- **Number covered by PRE imagery bounds:** 3022
- **Number covered by POST imagery bounds:** 3022
- **Number covered by both PRE and POST:** 3022

## 2. Sentinel-2 Scene Information

- **Image/Product ID (PRE):** `S2A_MSIL2A_20151128T050142_N0500_R119_T44PLV_20231017T062052.SAFE, S2A_MSIL2A_20151128T050142_N0500_R119_T44PMV_20231017T062052.SAFE, S2A_MSIL2A_20151128T050142_N0500_R119_T44PLV_20231027T151725.SAFE, S2A_MSIL2A_20151128T050142_N0500_R119_T44PMV_20231027T151725.SAFE`
- **Image/Product ID (POST):** `S2A_MSIL2A_20151228T050222_N0500_R119_T44PLV_20231009T231712.SAFE, S2A_MSIL2A_20151228T050222_N0500_R119_T44PMV_20231009T231712.SAFE`
- **Acquisition Date (PRE):** `2015-11-28`
- **Acquisition Date (POST):** `2015-12-28`
- **Optical Source:** `Sentinel-2`
- **Optical Resolution:** `10.0 m`
- **CRS:** `EPSG:32644`

## 3. Pixel Statistics

- **PRE Valid-Pixel Statistics (per building):**
  - Min: 1.0
  - Max: 189.0
  - Mean: 7.01
  - Std Dev: 8.60
- **POST Valid-Pixel Statistics (per building):**
  - Min: 1.0
  - Max: 189.0
  - Mean: 7.01
  - Std Dev: 8.60

## 4. Evidence & Ground Truth Summary

- **Number of buildings with flood-overlap evidence:** 237
- **Number of buildings with other documented evidence (manual):** 20
- **Number of buildings with reliable damage labels (Class 0, 1, 2):** 20
- **Number of buildings remaining unlabeled:** 3002

## 5. Limitations of Sentinel-2 Spatial Resolution

Sentinel-2 imagery has a spatial resolution of approximately 10 meters per pixel, which corresponds to a surface footprint of 100 square meters per pixel. This scale is too coarse to identify structural details of individual buildings (e.g., roof displacement, debris accumulation, wall collapses). Consequently, optical change or flood intersection from Sentinel-2 data alone cannot support a building-level damage or destruction label. It must be treated solely as weak, indirect, or supporting evidence.

## 6. Final Scientific Statement

"These are building-level SAR features supplemented with Sentinel-2 optical and flood-evidence information. Sentinel-2 optical imagery is used as supporting evidence only. Sentinel-2 at approximately 10 m resolution is not sufficient by itself to establish individual-building structural damage. Buildings without reliable building-level evidence remain UNLABELED. No unsupported damage classification has been performed."
