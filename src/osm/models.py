from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .exceptions import InvalidBBoxError

@dataclass
class BoundingBox:
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not (-90.0 <= self.south <= 90.0):
            raise InvalidBBoxError(f"South latitude {self.south} out of range [-90, 90].")
        if not (-90.0 <= self.north <= 90.0):
            raise InvalidBBoxError(f"North latitude {self.north} out of range [-90, 90].")
        if not (-180.0 <= self.west <= 180.0):
            raise InvalidBBoxError(f"West longitude {self.west} out of range [-180, 180].")
        if not (-180.0 <= self.east <= 180.0):
            raise InvalidBBoxError(f"East longitude {self.east} out of range [-180, 180].")
        if self.south >= self.north:
            raise InvalidBBoxError(f"South latitude ({self.south}) must be less than North latitude ({self.north}).")
        if self.west >= self.east:
            raise InvalidBBoxError(f"West longitude ({self.west}) must be less than East longitude ({self.east}).")

    @property
    def area_sq_deg(self) -> float:
        return (self.north - self.south) * (self.east - self.west)

    def to_overpass_str(self) -> str:
        return f"{self.south},{self.west},{self.north},{self.east}"

@dataclass
class BuildingFootprint:
    osm_id: int
    osm_type: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_geojson_feature(self) -> Dict[str, Any]:
        props = {
            "osm_id": self.osm_id,
            "osm_type": self.osm_type,
            **self.properties
        }
        return {
            "type": "Feature",
            "id": f"osm_{self.osm_type}_{self.osm_id}",
            "properties": props,
            "geometry": self.geometry
        }

@dataclass
class ExtractionResult:
    success: bool
    output_path: str
    building_count: int
    invalid_count: int
    repaired_count: int
    duplicate_count: int
    query_time_sec: float
    download_time_sec: float
    total_time_sec: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "success" if self.success else "error",
            "output_path": self.output_path,
            "building_count": self.building_count,
            "invalid_count": self.invalid_count,
            "repaired_count": self.repaired_count,
            "duplicate_count": self.duplicate_count,
            "query_time_sec": round(self.query_time_sec, 3),
            "download_time_sec": round(self.download_time_sec, 3),
            "total_time_sec": round(self.total_time_sec, 3),
            "error_message": self.error_message
        }
