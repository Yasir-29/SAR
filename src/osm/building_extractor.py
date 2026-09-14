import os
import time
import json
from typing import Dict, Any, List, Optional
from .models import BoundingBox, BuildingFootprint, ExtractionResult
from .geometry import way_coords_to_polygon
from .overpass_client import OverpassClient
from .exceptions import (
    OSMException,
    InvalidBBoxError,
    NoBuildingsFoundError,
    OutputWriteError
)

DEFAULT_OUTPUT_PATH = "data/tamil_nadu/osm/buildings.geojson"
MAX_BBOX_AREA_SQ_DEG = 0.5  # Protection against accidental massive downloads (e.g. ~50km x 50km)

def build_bbox_query(bbox: BoundingBox, timeout: int = 60) -> str:
    return f"""[out:json][timeout:{timeout}];
way["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});
out geom;"""

def build_place_query(place_name: str, timeout: int = 60) -> str:
    sanitized_name = place_name.replace('"', '\\"')
    return f"""[out:json][timeout:{timeout}];
area["name"="{sanitized_name}"]["boundary"="administrative"]->.searchArea;
(
  way["building"](area.searchArea);
);
out geom;"""

class BuildingExtractor:
    def __init__(self, client: Optional[OverpassClient] = None):
        self.client = client or OverpassClient()

    def extract_from_bbox(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        output_path: str = DEFAULT_OUTPUT_PATH,
        max_area_deg: float = MAX_BBOX_AREA_SQ_DEG
    ) -> ExtractionResult:
        t_start = time.time()
        bbox = BoundingBox(south=south, west=west, north=north, east=east)
        
        if bbox.area_sq_deg > max_area_deg:
            raise InvalidBBoxError(
                f"Requested bounding box area ({bbox.area_sq_deg:.4f} sq deg) exceeds limit ({max_area_deg} sq deg). "
                "Please select a smaller sub-region or tile the area."
            )

        query = build_bbox_query(bbox, timeout=self.client.timeout)
        
        t_query_start = time.time()
        data = self.client.execute_query(query)
        t_download = time.time() - t_query_start

        return self._process_and_save(data, output_path, t_download, t_start)

    def extract_from_place(
        self,
        place_name: str,
        output_path: str = DEFAULT_OUTPUT_PATH
    ) -> ExtractionResult:
        t_start = time.time()
        if not place_name or not place_name.strip():
            raise OSMException("Place name cannot be empty.", error_code="INVALID_PLACE")

        query = build_place_query(place_name.strip(), timeout=self.client.timeout)
        
        t_query_start = time.time()
        data = self.client.execute_query(query)
        t_download = time.time() - t_query_start

        return self._process_and_save(data, output_path, t_download, t_start)

    def _process_and_save(
        self,
        raw_data: Dict[str, Any],
        output_path: str,
        download_time: float,
        start_time: float
    ) -> ExtractionResult:
        elements = raw_data.get("elements", [])
        if not elements:
            raise NoBuildingsFoundError()

        seen_ids = set()
        features = []
        valid_count = 0
        invalid_count = 0
        repaired_count = 0
        duplicate_count = 0

        for elem in elements:
            if elem.get("type") != "way":
                continue

            osm_id = elem.get("id")
            if not osm_id:
                continue

            if osm_id in seen_ids:
                duplicate_count += 1
                continue

            tags = elem.get("tags", {})
            if "building" not in tags:
                continue

            points = elem.get("geometry", [])
            if not points:
                invalid_count += 1
                continue

            geom_json, was_repaired, err = way_coords_to_polygon(points)
            if not geom_json:
                invalid_count += 1
                continue

            if was_repaired:
                repaired_count += 1

            footprint = BuildingFootprint(
                osm_id=osm_id,
                osm_type="way",
                geometry=geom_json,
                properties=tags
            )
            features.append(footprint.to_geojson_feature())
            seen_ids.add(osm_id)
            valid_count += 1

        if valid_count == 0:
            raise NoBuildingsFoundError("No valid building polygons could be parsed from Overpass response.")

        # Construct GeoJSON FeatureCollection with required ODbL attribution
        geojson_collection = {
            "type": "FeatureCollection",
            "metadata": {
                "source": "OpenStreetMap",
                "source_license": "ODbL",
                "source_url": "https://www.openstreetmap.org/",
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "building_count": valid_count,
                "repaired_count": repaired_count,
                "duplicate_count": duplicate_count
            },
            "features": features
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson_collection, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise OutputWriteError(f"Failed to write GeoJSON to {output_path}: {str(e)}")

        total_time = time.time() - start_time
        query_time = total_time - download_time

        return ExtractionResult(
            success=True,
            output_path=output_path,
            building_count=valid_count,
            invalid_count=invalid_count,
            repaired_count=repaired_count,
            duplicate_count=duplicate_count,
            query_time_sec=max(query_time, 0.0),
            download_time_sec=download_time,
            total_time_sec=total_time
        )

# Public API functions
def extract_buildings_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    output_path: str = DEFAULT_OUTPUT_PATH,
    client: Optional[OverpassClient] = None
) -> ExtractionResult:
    extractor = BuildingExtractor(client=client)
    return extractor.extract_from_bbox(south, west, north, east, output_path)

def extract_buildings_place(
    place_name: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    client: Optional[OverpassClient] = None
) -> ExtractionResult:
    extractor = BuildingExtractor(client=client)
    return extractor.extract_from_place(place_name, output_path)

def load_building_footprints(path: str) -> List[Dict[str, Any]]:
    """
    Loads saved building polygons and returns structured dicts with building_id, geometry, and properties.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Building footprints file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    footprints = []
    for feat in data.get("features", []):
        footprints.append({
            "building_id": feat.get("id") or str(feat.get("properties", {}).get("osm_id")),
            "geometry": feat.get("geometry"),
            "properties": feat.get("properties", {})
        })
    return footprints
