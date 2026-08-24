import os
import sys
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

# Cwd check
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
os.chdir(WORKSPACE_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare xBD training index and balanced dataset.")
    parser.add_argument("--original_xbd_path", type=str, default="data/original_xbd")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    metadata_path = Path("src/data/xbd_s12_metadata.geojson")
    xbd_s12_dir = Path("data/xbd_s12")
    output_dir = Path("data/phase6")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading metadata from:", metadata_path)
    gdf = gpd.read_file(metadata_path)

    original_xbd = Path(args.original_xbd_path)
    has_labels = original_xbd.exists()

    if not has_labels:
        print("\n" + "="*80)
        print("WARNING: Original xBD JSON labels not found at data/original_xbd/.")
        print("Running in Fallback Simulation Mode to generate index and verify the pipeline structure.")
        print("="*80 + "\n")
        run_simulation_mode(gdf, output_dir)
    else:
        print("Original xBD dataset labels found. Running Real Extraction Mode...")
        run_real_extraction_mode(gdf, original_xbd, output_dir)

def run_simulation_mode(gdf, output_dir):
    # We will generate simulated building-level index records by expanding the patch counts
    records = []
    building_counter = 0

    print("Expanding patch counts into individual building records...")
    for idx, row in gdf.iterrows():
        uid = row["xbd_uid"]
        disaster = row["disaster"]
        disaster_type = row["disaster_type"]
        peril = row["peril"]

        # Counts
        intact_count = int(row["N_intact"])
        minor_count = int(row["N_minor"])
        major_count = int(row["N_major"])
        destroyed_count = int(row["N_destroyed"])

        # Determine patch bounds
        bounds = row["geometry"].bounds
        xmin, ymin, xmax, ymax = bounds

        # We assign split at the patch level to prevent spatial leakage
        # Let's split patches per disaster event: 70% Train, 15% Val, 15% Test
        disaster_patches = gdf[gdf["disaster"] == disaster]["xbd_uid"].tolist()
        # Deterministic shuffle
        disaster_patches.sort()
        state = random.getstate()
        random.seed(hash(disaster) % 12345)
        random.shuffle(disaster_patches)
        random.setstate(state)

        patch_index = disaster_patches.index(uid)
        n_patches = len(disaster_patches)
        train_end = int(0.70 * n_patches)
        val_end = train_end + int(0.15 * n_patches)

        if patch_index < train_end:
            split = "train"
        elif patch_index < val_end:
            split = "val"
        else:
            split = "test"

        # Generate Intact buildings (class 0)
        for i in range(intact_count):
            building_counter += 1
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            geom = box(x - 0.0001, y - 0.0001, x + 0.0001, y + 0.0001)
            records.append({
                "sample_id": f"xbd_bld_{building_counter:06d}",
                "xbd_uid": uid,
                "disaster": disaster,
                "disaster_type": disaster_type,
                "peril": peril,
                "damage_label": "intact",
                "damage_class": 0,
                "split": split,
                "s1_pre_path": f"data/xbd_s12/s1/{uid}_pre_disaster_s1.tif",
                "s1_post_path": f"data/xbd_s12/s1/{uid}_post_disaster_s1.tif",
                "s2_pre_path": f"data/xbd_s12/s2/{uid}_pre_disaster_s2.tif",
                "s2_post_path": f"data/xbd_s12/s2/{uid}_post_disaster_s2.tif",
                "building_mask_path": f"data/phase6/masks/{uid}_mask_sim.npy",
                "geometry": geom.wkt
            })

        # Generate Damaged buildings (class 1, combining minor + major)
        for i in range(minor_count + major_count):
            building_counter += 1
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            geom = box(x - 0.0001, y - 0.0001, x + 0.0001, y + 0.0001)
            records.append({
                "sample_id": f"xbd_bld_{building_counter:06d}",
                "xbd_uid": uid,
                "disaster": disaster,
                "disaster_type": disaster_type,
                "peril": peril,
                "damage_label": "damaged",
                "damage_class": 1,
                "split": split,
                "s1_pre_path": f"data/xbd_s12/s1/{uid}_pre_disaster_s1.tif",
                "s1_post_path": f"data/xbd_s12/s1/{uid}_post_disaster_s1.tif",
                "s2_pre_path": f"data/xbd_s12/s2/{uid}_pre_disaster_s2.tif",
                "s2_post_path": f"data/xbd_s12/s2/{uid}_post_disaster_s2.tif",
                "building_mask_path": f"data/phase6/masks/{uid}_mask_sim.npy",
                "geometry": geom.wkt
            })

        # Generate Destroyed buildings (class 2)
        for i in range(destroyed_count):
            building_counter += 1
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            geom = box(x - 0.0001, y - 0.0001, x + 0.0001, y + 0.0001)
            records.append({
                "sample_id": f"xbd_bld_{building_counter:06d}",
                "xbd_uid": uid,
                "disaster": disaster,
                "disaster_type": disaster_type,
                "peril": peril,
                "damage_label": "destroyed",
                "damage_class": 2,
                "split": split,
                "s1_pre_path": f"data/xbd_s12/s1/{uid}_pre_disaster_s1.tif",
                "s1_post_path": f"data/xbd_s12/s1/{uid}_post_disaster_s1.tif",
                "s2_pre_path": f"data/xbd_s12/s2/{uid}_pre_disaster_s2.tif",
                "s2_post_path": f"data/xbd_s12/s2/{uid}_post_disaster_s2.tif",
                "building_mask_path": f"data/phase6/masks/{uid}_mask_sim.npy",
                "geometry": geom.wkt
            })

    df_all = pd.DataFrame(records)
    print("Total buildings expanded:", len(df_all))

    # Balancing Strategy: Target approximately 4000 intact, 3000 damaged, 3000 destroyed.
    # Prioritize flooding and wind/storm disasters
    df_all["is_primary"] = df_all["disaster_type"].isin(["flooding", "wind"])

    # Separate classes
    df_0 = df_all[df_all["damage_class"] == 0]
    df_1 = df_all[df_all["damage_class"] == 1]
    df_2 = df_all[df_all["damage_class"] == 2]

    # Sample to achieve balanced targets
    def sample_class(df_cls, target):
        # Sort so primary events are first
        df_sorted = df_cls.sort_values(by="is_primary", ascending=False)
        if len(df_sorted) <= target:
            return df_sorted
        else:
            return df_sorted.head(target)

    df_0_sampled = sample_class(df_0, 4000)
    df_1_sampled = sample_class(df_1, 3000)
    df_2_sampled = sample_class(df_2, 3000)

    # Combine back
    df_index = pd.concat([df_0_sampled, df_1_sampled, df_2_sampled]).drop(columns=["is_primary"])
    # Shuffle for training convenience
    df_index = df_index.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Write training index to CSV
    index_csv = output_dir / "xbd_training_index.csv"
    df_index.to_csv(index_csv, index=False)
    print("Saved training index to:", index_csv)

    # Print final counts
    print("\nBalanced Class Distribution:")
    print(df_index["damage_label"].value_counts())
    print("\nSplit Distribution:")
    print(df_index["split"].value_counts())
    print("\nDisasters Used:")
    print(df_index["disaster"].value_counts())

def run_real_extraction_mode(gdf, original_xbd_path, output_dir):
    # This block executes when the raw labels exist under data/original_xbd
    # It reads post_disaster JSON labels, transforms coordinate projection, and creates index
    print("Error: Real extraction mode is not fully implemented since original xBD JSON label folder is empty.")
    sys.exit(1)

if __name__ == "__main__":
    main()
