#!/usr/bin/env python3
import sys
import os
import argparse
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.osm import (
    OverpassClient,
    BuildingExtractor,
    OSMException,
    DEFAULT_OUTPUT_PATH
)

def main():
    parser = argparse.ArgumentParser(
        description="Download OpenStreetMap building footprints via Overpass API into GeoJSON."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="Bounding box coordinates (south west north east), e.g. 11.74 79.76 11.76 79.78"
    )
    group.add_argument(
        "--place",
        type=str,
        help="Place/Area name, e.g. 'Cuddalore, Tamil Nadu, India' or 'Chennai'"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output GeoJSON filepath (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP request timeout in seconds (default: 90)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for rate-limiting or server errors (default: 3)"
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=0.5,
        help="Maximum allowed bounding box area in sq degrees (default: 0.5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output"
    )

    args = parser.parse_args()

    client = OverpassClient(
        timeout=args.timeout,
        max_retries=args.max_retries
    )
    extractor = BuildingExtractor(client=client)

    if args.verbose:
        print("Initializing OSM Building Extraction...")
        print(f"Target Output: {args.output}")

    try:
        if args.bbox:
            south, west, north, east = args.bbox
            if args.verbose:
                print(f"Querying BBox: S={south}, W={west}, N={north}, E={east}...")
            res = extractor.extract_from_bbox(
                south=south,
                west=west,
                north=north,
                east=east,
                output_path=args.output,
                max_area_deg=args.max_area
            )
        else:
            if args.verbose:
                print(f"Querying Place: '{args.place}'...")
            res = extractor.extract_from_place(
                place_name=args.place,
                output_path=args.output
            )

        print("\n" + "="*50)
        print("OSM BUILDING EXTRACTION SUCCESSFUL")
        print("="*50)
        print(f"Valid Buildings Saved:  {res.building_count:,}")
        print(f"Repaired Polygons:      {res.repaired_count:,}")
        print(f"Duplicates Skipped:     {res.duplicate_count:,}")
        print(f"Invalid Shapes Skipped: {res.invalid_count:,}")
        print(f"Download Latency:       {res.download_time_sec:.3f} s")
        print(f"Total Processing Time:  {res.total_time_sec:.3f} s")
        print(f"Output File:            {res.output_path}")
        print("="*50)
        sys.exit(0)

    except OSMException as e:
        print(f"\n[ERROR] OSM Extraction Failed ({e.error_code}): {e.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during extraction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
