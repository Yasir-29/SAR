import pytest
from datetime import datetime
from app.copernicus.search import create_aoi_bbox, search_sentinel2_image
from app.copernicus.imagery import download_sentinel2_chip
from app.config import settings

def test_aoi_bbox_creation():
    lat, lon = 11.0168, 76.9558
    bbox = create_aoi_bbox(lat, lon, size_meters=500.0)
    assert len(bbox) == 4
    min_lon, min_lat, max_lon, max_lat = bbox
    assert min_lon < lon < max_lon
    assert min_lat < lat < max_lat

def test_stac_search_mock_mode():
    bbox = [76.9500, 11.0100, 76.9600, 11.0200]
    res_before = search_sentinel2_image(
        bbox=bbox,
        start_date="2025-01-01",
        end_date="2025-01-14",
        max_cloud_cover=30.0,
        target_disaster_date="2025-01-15"
    )
    assert "product_id" in res_before
    assert "date" in res_before
    assert res_before["cloud_cover"] <= 30.0

def test_imagery_download_mock_mode():
    bbox = [76.9500, 11.0100, 76.9600, 11.0200]
    dummy_meta = {"date": "2025-01-10T10:30:00Z", "product_id": "TEST_PRODUCT"}
    img_path, arr = download_sentinel2_chip(bbox, dummy_meta, is_before=True, request_id="test_run")
    assert img_path is not None
    assert arr.shape == (256, 256, 3)
