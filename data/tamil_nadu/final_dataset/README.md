# Final Verified Dataset: Tamil Nadu Building Damage

This dataset contains structural building footprint masks paired with multi-temporal Sentinel-1 SAR and Sentinel-2 optical imagery covering 10 major disaster events across Tamil Nadu.

## Dataset Structure
- `train/`: Building folders containing pre/post chips and binary masks.
- `val/`: Geographic holdout split (TN_BUREVI_2020) for hyperparameter verification.
- `test/`: Geographically and temporally locked holdout split (TN_OCKHI_2017 & TN_MANDOUS_2022) for model testing.

## Files per Building Folder
- `sar_pre.npy`: shape (2, 32, 32), float32. VV/VH backscatter pre-event.
- `sar_post.npy`: shape (2, 32, 32), float32. VV/VH backscatter post-event.
- `optical_pre.npy`: shape (4, 32, 32), uint16. B2, B3, B4, B8 pre-event (where available).
- `optical_post.npy`: shape (4, 32, 32), uint16. B2, B3, B4, B8 post-event (where available).
- `building_mask.npy`: shape (32, 32), uint8. 0 = background, 1 = actual footprint.
- `metadata.json`: Contains bounding boxes, dates, CRS, areas, and verified unverified classifications.

## Metadata Mapping
Refer to `final_dataset.csv` for labels:
- `0` = INTACT
- `1` = DAMAGED
- `2` = DESTROYED

## Splitting Strategy
To prevent geographic leakage, train/val/test splits are geographically isolated by disaster zones. Duplicate structures between Vardah/Chennai Flood and Nivar/Nada are locked strictly to the train split.
