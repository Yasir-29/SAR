import math
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, mapping

def way_coords_to_polygon(points: List[Dict[str, float]]) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Converts a list of Overpass points [{'lat': y, 'lon': x}, ...] into a GeoJSON Polygon.
    Returns: (geometry_dict, was_repaired, error_reason)
    """
    if not points or len(points) < 3:
        return None, False, "Less than 3 points provided"

    coords = []
    for pt in points:
        lat = pt.get("lat")
        lon = pt.get("lon")
        if lat is None or lon is None:
            return None, False, "Missing lat or lon coordinate"
        if math.isnan(lat) or math.isnan(lon):
            return None, False, "NaN coordinate detected"
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return None, False, f"Coordinate ({lat}, {lon}) out of global range"
        coords.append((float(lon), float(lat)))

    # Ensure polygon is closed (first == last)
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:
        return None, False, "Polygon must have at least 4 coordinates (3 vertices + closing point)"

    # Check for duplicate-only geometry
    unique_coords = set(coords)
    if len(unique_coords) < 3:
        return None, False, "Polygon has less than 3 unique vertices (collapsed line/point)"

    try:
        poly = Polygon(coords)
    except Exception as e:
        return None, False, f"Polygon creation failed: {str(e)}"

    repaired = False
    if not poly.is_valid:
        try:
            poly_repaired = poly.buffer(0)
            if poly_repaired.is_valid and not poly_repaired.is_empty and poly_repaired.area > 0:
                if poly_repaired.geom_type == 'Polygon':
                    poly = poly_repaired
                    repaired = True
                elif poly_repaired.geom_type == 'MultiPolygon':
                    poly = max(poly_repaired.geoms, key=lambda p: p.area)
                    repaired = True
                else:
                    return None, False, f"Repaired geometry is not polygon: {poly_repaired.geom_type}"
            else:
                return None, False, "Polygon is invalid and buffer(0) repair failed"
        except Exception as e:
            return None, False, f"Validation repair exception: {str(e)}"

    if poly.is_empty:
        return None, False, "Empty polygon geometry"

    if poly.area <= 0:
        return None, False, "Zero-area polygon geometry"

    geom_json = mapping(poly)
    return geom_json, repaired, None
