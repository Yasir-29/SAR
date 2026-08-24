# xBD-S12 Label Mapping Documentation

This document describes the building-level label mapping schema used for training the U-Net building damage detection model.

## 1. Class Mapping Definition

To ensure generalizability, reduce class noise, and focus on severe damage classes relevant for rapid disaster response, the original xBD damage labels are mapped to a simplified 3-class target system:

| Original xBD Label | Original ID | Mapped Class ID | Mapped Class Label | Description |
| :--- | :---: | :---: | :---: | :--- |
| `no-damage` | 1 | **0** | **INTACT** | Building is structurally intact. No visible structural damage. |
| `minor-damage` | 2 | **1** | **DAMAGED** | Building has minor cosmetic or roof damage but is structurally stable. |
| `major-damage` | 3 | **1** | **DAMAGED** | Building has major structural damage (partially collapsed walls or roof). |
| `destroyed` | 4 | **2** | **DESTROYED** | Building is fully collapsed, washed away, or completely ruined. |

## 2. Excluded Classes

The following classes are **EXCLUDED** from supervised training to prevent injecting noise or garbage labels:
- **`un-classified` (ID 5):** Reclassified or un-annotated buildings in the original dataset (e.g. under clouds, shadow, or new buildings that appeared post-disaster).
- **No data / Black borders (ID 6):** Regions containing missing or corrupt pixel values.

## 3. Scientific Justification

- **Pooling Minor & Major Damage:** In medium-resolution (10m) Sentinel-1 (SAR) and Sentinel-2 (optical) imagery, distinguishing between "minor" and "major" structural damage is extremely difficult due to pixel resolution limitations. Combining them into a single `DAMAGED` (Class 1) category drastically improves model robustness and training stability.
- **Background Separation:** Background pixels (non-building areas) are excluded from building-level loss calculations by using binary building footprint masks to constrain predictions to the building area only.
