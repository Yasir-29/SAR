import math
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from app.config import settings

def create_aoi_bbox(lat: float, lon: float, size_meters: float = 500.0) -> List[float]:
    """
    Creates a bounding box [min_lon, min_lat, max_lon, max_lat] approximately 
    size_meters x size_meters centered at (lat, lon).
    """
    half_size = size_meters / 2.0
    lat_deg_delta = half_size / 111320.0
    lon_deg_delta = half_size / (111320.0 * math.cos(math.radians(lat)))

    min_lat = round(lat - lat_deg_delta, 6)
    max_lat = round(lat + lat_deg_delta, 6)
    min_lon = round(lon - lon_deg_delta, 6)
    max_lon = round(lon + lon_deg_delta, 6)

    return [min_lon, min_lat, max_lon, max_lat]

def search_sentinel2_image(
    bbox: List[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 30.0,
    target_disaster_date: str = ""
) -> Dict[str, Any]:
    """
    Searches real Sentinel-2 Level-2A imagery via Copernicus Data Space Ecosystem (CDSE) Catalogue API.
    Prefers lowest cloud cover and acquisition date closest to target_disaster_date.
    """
    lon = (bbox[0] + bbox[2]) / 2.0
    lat = (bbox[1] + bbox[3]) / 2.0
    wkt = f"POINT({lon:.6f} {lat:.6f})"

    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and startswith(Name,'S2') and ContentDate/Start ge {start_date}T00:00:00.000Z and ContentDate/Start le {end_date}T23:59:59.000Z&$top=20"

    try:
        response = requests.get(url, headers={"User-Agent": "TamilNaduDisasterAssessment/1.0"}, timeout=15)
        if response.status_code == 200:
            items = response.json().get("value", [])
            if items:
                target_dt = datetime.strptime(target_disaster_date, "%Y-%m-%d") if target_disaster_date else datetime.now()
                candidates = []
                for it in items:
                    p_name = it.get("Name", "")
                    c_date = it.get("ContentDate", {}).get("Start", "")
                    if not c_date:
                        continue
                    try:
                        dt = datetime.fromisoformat(c_date.replace("Z", "+00:00")).replace(tzinfo=None)
                        diff_days = abs((dt - target_dt).total_seconds()) / 86400.0
                    except Exception:
                        diff_days = 999.0

                    cloud_cover = 10.0
                    attrs = it.get("Attributes", [])
                    for a in attrs:
                        if a.get("Name") == "cloudCover":
                            try:
                                cloud_cover = float(a.get("Value", 10.0))
                            except (ValueError, TypeError):
                                pass
                            break

                    candidates.append({
                        "product_id": p_name,
                        "date": c_date.split("T")[0],
                        "datetime": c_date,
                        "cloud_cover": round(cloud_cover, 1),
                        "days_diff": diff_days,
                        "id": it.get("Id"),
                        "bbox": bbox
                    })

                candidates.sort(key=lambda x: (x["days_diff"], x["cloud_cover"]))
                best = candidates[0]
                return {
                    "product_id": best["product_id"],
                    "date": best["datetime"],
                    "cloud_cover": best["cloud_cover"],
                    "source": "Sentinel-2 L2A",
                    "bbox": bbox
                }
    except Exception as e:
        pass

    raise Exception("No suitable Copernicus Sentinel image found for this event/location.")

