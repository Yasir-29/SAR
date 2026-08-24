# Historical High-Resolution Imagery Investigation: Chennai Floods 2015

This report documents the availability, accessibility, licensing, and spatial resolution suitability of Cartosat-2 and other high-resolution remote sensing sources for the 2015 Chennai floods, conforming to the Phase 4 ground-truth building damage labeling requirements.

---

## 1. Remote Sensing Source Metadata
- **Source Agency:** National Remote Sensing Centre (NRSC) / Indian Space Research Organisation (ISRO)
- **Primary Satellite/Sensor:** Cartosat-2 Series (Panchromatic + Multispectral)
- **Target Acquisition Date:** November 21, 2015
- **Spatial Resolution:** 
  - Panchromatic: 0.8 meters
  - Multispectral: 3.2 meters
  - Combined Orthorectified Pan-Sharpened: ~1.0 meter
- **Spectral Bands:** Blue, Green, Red, Near-Infrared (NIR)
- **Spatial Coverage:** Partially covers the Chennai study area bounding box `[80.0, 12.85, 80.35, 13.20]`.

---

## 2. Legal and Technical Access Requirements
- **Portal URL:** [Bhoonidhi (NRSC Data Portal)](https://bhoonidhi.nrsc.gov.in/)
- **Registration Requirements:** **Mandatory**. An account must be registered to browse the full catalog, place orders, or request data.
- **Pricing & Licensing:** 
  - Under ISRO's data policy, free and open access is restricted to resolutions **coarser than 5 meters**.
  - All high-resolution data (finer than 5m, including Cartosat-2 ~1m data) is **priced** and distributed under strict End User License Agreements (EULA).
  - Private and general commercial users must purchase the data on a "work-order" basis by contacting the commercial arm of ISRO (NSIL) at `eodata@nsilindia.co.in`.
  - Authorized Indian government departments and approved academic/R&D institutions can request free/concessional access for disaster management, subject to official request submission and security clearance.
- **Acquisition/Download Procedure:**
  1. Log in to the Bhoonidhi portal.
  2. Define the geographic search AOI and date range (Nov 21, 2015).
  3. Locate the Cartosat-2 scene and add it to the cart.
  4. Submit an order request. If off-the-shelf online storage is available, download the product. If not, wait for the processing task to complete.

---

## 3. Scientific Feasibility of PRE/POST Pair (Temporal Availability)
A single post-event or early-event high-resolution image is **insufficient** for structural damage assessment. To perform building-level damage classification (Class 0, 1, 2), we require a matched, cloud-free PRE-disaster and POST-disaster high-resolution image pair over the exact same footprint.

- **Pre-Disaster Imagery:** Cartosat-2 or other high-resolution optical imagery acquired in October/early November 2015 is extremely scarce and heavily obscured by seasonal cloud cover.
- **Post-Disaster Imagery:** The peak flooding occurred during heavy rainfall on December 1–2, 2015. Optical sensors like Cartosat-2 were severely limited by persistent cloud cover during early December 2015. Consequently, radar imagery (such as RISAT-1) was primarily used by NRSC for inundation mapping.
- **Conclusion:** A matched, cloud-free, and legally accessible **PRE/POST high-resolution optical image pair does not exist** for this specific 2015 disaster footprint.

---

## 4. Quantitative One-Building Resolution Suitability Test

We loaded a representative building footprint from the Phase 2 Chennai dataset to analyze how spatial resolution impacts individual building pixel representation.

- **OSM Building ID:** `chennai_bld_000001`
- **Building Footprint Coordinates:** `[80.20558, 13.00649]`
- **Actual Footprint Area:** `5146.39 square meters`

### Building Footprint Pixel Coverage Comparison Table

| Sensor / Source | Spatial Resolution | Pixel Size | Estimated Pixels Covered | Suitability for Visual Damage Labeling |
| :--- | :---: | :---: | :---: | :--- |
| **Maxar / WorldView** | 0.3 m | 0.09 m² | **57182.1** | **EXCELLENT:** Fully resolves walls, roof outline, debris, and structural damage details. |
| **Cartosat-2 (ISRO)** | 1.0 m | 1.00 m² | **5146.4** | **SUITABLE:** Resolves the overall footprint shape and major collapses/damage. |
| **Sentinel-2 (ESA)** | 10.0 m | 100.00 m² | **51.5** | **UNSUITABLE:** Footprint is blurred into a fraction of a pixel; cannot resolve structural changes. |

### Visual Suitability Verdict
Sentinel-2 imagery is **completely unsuitable** for building-level labeling because the building is represented by only 51.46 pixels. High-resolution imagery (like 1m Cartosat-2 or 0.3m Maxar) is **technically suitable** because it resolves the footprint into 5146.4 to 57182.1 pixels. 

---

## 5. Final Decision and Action Plan

> [!WARNING]
> **Data Availability & Technical Constraints:**
> Because a matched, cloud-free, and legally open PRE-disaster and POST-disaster high-resolution image pair is **unavailable** for the 2015 Chennai event, we cannot proceed with building-level visual ground-truth labeling using Cartosat-2.
>
> **Action taken:**
> Conforming to the scientific rules, we will **not fabricate ground truth** and will **not assign damage labels based on flood/inundation extent alone** (since flood overlap is only evidence of water presence, not structural damage). We preserve the building-level labels as `unlabeled` (source `unlabeled`, confidence `none`) except for the designated validation samples.
