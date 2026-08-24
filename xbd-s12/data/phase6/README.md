# xBD-S12 Compact Training Archive

This archive contains the balanced training index, event summaries, and normalization parameters for training the building damage detection model.

## Contents
- `xbd_training_index.csv`: Index of ~10,000 building-level samples with splits and labels.
- `xbd_event_summary.csv`: Aggregated patch-level counts by disaster event.
- `normalization.json`: Normalization parameters.

## Code execution in Colab:
To run the dual-stream U-Net model:
1. Clone this repository in Colab.
2. Unpack this zip archive.
3. Access the Sentinel-1 and Sentinel-2 patches directly from `data/xbd_s12/` using the prepare script to extract building chips.
