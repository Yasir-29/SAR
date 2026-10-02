import uuid
import os
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime, timedelta

from app.config import settings
from app.schemas import (
    DisasterAnalysisRequest,
    DisasterAnalysisResponse,
    DamagePredictRequest,
    DamagePredictResponse,
    LocationSchema,
    ImageryMetadata,
    ComparisonMetadata,
    ModelPrediction
)
from app.copernicus.search import create_aoi_bbox, search_sentinel2_image
from app.copernicus.imagery import download_sentinel2_chip
from app.damage.preprocessing import preprocess_and_compare
from app.damage.inference import predict_damage_from_images, get_loaded_model

app = FastAPI(
    title="Copernicus Sentinel-2 Disaster Damage Assessment API",
    description="Free Copernicus Data Space Ecosystem (CDSE) Sentinel-2 Satellite Imagery Analysis & Damage Model Backend",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount media directory for image downloads
app.mount("/media", StaticFiles(directory=str(settings.DATA_DIR)), name="media")

@app.get("/health")
def health_check():
    model, device = get_loaded_model()
    return {
        "status": "HEALTHY",
        "copernicus_cdse": "CONNECTED" if not settings.MOCK_SATELLITE_API else "MOCK_MODE",
        "mock_satellite_api": settings.MOCK_SATELLITE_API,
        "model_loaded": model is not None,
        "device": str(device),
        "model_checkpoint": settings.MODEL_PATH
    }

@app.post(
    "/api/disaster/analyze",
    response_model=DisasterAnalysisResponse,
    summary="Full End-to-End Disaster Analysis Pipeline"
)
def analyze_disaster(req: DisasterAnalysisRequest, request: Request):
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    
    # Parse Dates
    try:
        d_date = datetime.strptime(req.disaster_date, "%Y-%m-%d")
        before_start = (d_date - timedelta(days=req.before_days)).strftime("%Y-%m-%d")
        before_end = (d_date - timedelta(days=1)).strftime("%Y-%m-%d")
        
        after_start = (d_date + timedelta(days=1)).strftime("%Y-%m-%d")
        after_end = (d_date + timedelta(days=req.after_days)).strftime("%Y-%m-%d")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid disaster_date or date calculations: {str(e)}"
        )

    # Step 1: Create AOI (500m x 500m bounding box)
    bbox = create_aoi_bbox(req.latitude, req.longitude, size_meters=settings.AOI_SIZE_METERS)

    # Step 2: Search BEFORE imagery
    try:
        before_meta = search_sentinel2_image(
            bbox=bbox,
            start_date=before_start,
            end_date=before_end,
            max_cloud_cover=req.cloud_cover_max,
            target_disaster_date=req.disaster_date
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No suitable Sentinel-2 image found for the requested period (BEFORE disaster). {str(e)}"
        )

    # Step 3: Search AFTER imagery
    try:
        after_meta = search_sentinel2_image(
            bbox=bbox,
            start_date=after_start,
            end_date=after_end,
            max_cloud_cover=req.cloud_cover_max,
            target_disaster_date=req.disaster_date
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No suitable Sentinel-2 image found for the requested period (AFTER disaster). {str(e)}"
        )

    # Step 5: Download / Retrieve Optical Imagery Chips
    try:
        before_path, _ = download_sentinel2_chip(bbox, before_meta, is_before=True, request_id=request_id)
        after_path, _ = download_sentinel2_chip(bbox, after_meta, is_before=False, request_id=request_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download or crop optical imagery: {str(e)}"
        )

    # Step 6 & 8: Preprocess and generate difference image
    try:
        diff_path, _, quality_meta = preprocess_and_compare(before_path, after_path, request_id=request_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image preprocessing or comparison failed: {str(e)}"
        )

    # Step 9: Model Inference
    try:
        pred_res = predict_damage_from_images(before_path, after_path, req.latitude, req.longitude)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Damage model inference failed: {str(e)}"
        )

    # Construct host URLs for serving media images
    base_url = str(request.base_url).rstrip("/")
    before_url = f"{base_url}/media/before/{os.path.basename(before_path)}"
    after_url = f"{base_url}/media/after/{os.path.basename(after_path)}"
    diff_url = f"{base_url}/media/comparison/{os.path.basename(diff_path)}"

    # Step 10: Main Combined Response
    return DisasterAnalysisResponse(
        request_id=request_id,
        location=LocationSchema(latitude=req.latitude, longitude=req.longitude),
        before=ImageryMetadata(
            date=before_meta["date"],
            cloud_cover=before_meta["cloud_cover"],
            product_id=before_meta["product_id"],
            source=before_meta.get("source", "Sentinel-2 L2A"),
            image_url=before_url
        ),
        after=ImageryMetadata(
            date=after_meta["date"],
            cloud_cover=after_meta["cloud_cover"],
            product_id=after_meta["product_id"],
            source=after_meta.get("source", "Sentinel-2 L2A"),
            image_url=after_url
        ),
        comparison=ComparisonMetadata(difference_image_url=diff_url),
        model=ModelPrediction(
            prediction_class=pred_res["prediction"],
            confidence=pred_res["confidence"]
        )
    )

@app.post("/api/satellite/before-after", summary="Copernicus Sentinel-2 Before/After Satellite Imagery Endpoint")
def get_satellite_before_after(req: DisasterAnalysisRequest, request: Request):
    request_id = f"s2_{uuid.uuid4().hex[:8]}"
    try:
        d_date = datetime.strptime(req.disaster_date, "%Y-%m-%d")
        before_start = (d_date - timedelta(days=req.before_days)).strftime("%Y-%m-%d")
        before_end = (d_date - timedelta(days=1)).strftime("%Y-%m-%d")
        after_start = (d_date + timedelta(days=1)).strftime("%Y-%m-%d")
        after_end = (d_date + timedelta(days=req.after_days)).strftime("%Y-%m-%d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

    bbox = create_aoi_bbox(req.latitude, req.longitude, size_meters=settings.AOI_SIZE_METERS)
    
    before_resp = {"available": False, "reason": "No suitable imagery found"}
    try:
        b_meta = search_sentinel2_image(bbox, before_start, before_end, req.cloud_cover_max, req.disaster_date)
        b_path, _ = download_sentinel2_chip(bbox, b_meta, is_before=True, request_id=request_id, lat=req.latitude, lon=req.longitude, event_date=req.disaster_date)
        b_filename = os.path.basename(b_path)
        base_url = str(request.base_url).rstrip("/")
        before_resp = {
            "available": True,
            "acquisition_date": b_meta["date"],
            "product_id": b_meta["product_id"],
            "cloud_cover": b_meta["cloud_cover"],
            "image_url": f"{base_url}/media/before/{b_filename}"
        }
    except Exception as e:
        before_resp = {"available": False, "reason": str(e)}

    after_resp = {"available": False, "reason": "No suitable imagery found"}
    try:
        a_meta = search_sentinel2_image(bbox, after_start, after_end, req.cloud_cover_max, req.disaster_date)
        a_path, _ = download_sentinel2_chip(bbox, a_meta, is_before=False, request_id=request_id, lat=req.latitude, lon=req.longitude, event_date=req.disaster_date)
        a_filename = os.path.basename(a_path)
        base_url = str(request.base_url).rstrip("/")
        after_resp = {
            "available": True,
            "acquisition_date": a_meta["date"],
            "product_id": a_meta["product_id"],
            "cloud_cover": a_meta["cloud_cover"],
            "image_url": f"{base_url}/media/after/{a_filename}"
        }
    except Exception as e:
        after_resp = {"available": False, "reason": str(e)}

    return {
        "before": before_resp,
        "after": after_resp
    }


@app.post(
    "/api/damage/predict",
    response_model=DamagePredictResponse,
    summary="Standalone Model Damage Prediction Endpoint"
)
def predict_damage_endpoint(req: DamagePredictRequest):
    request_id = f"standalone_{uuid.uuid4().hex[:8]}"
    bbox = create_aoi_bbox(req.latitude, req.longitude)

    before_path = req.before_image_path
    after_path = req.after_image_path

    if not before_path or not os.path.exists(before_path):
        dummy_before = {"date": "2025-01-10T10:30:00Z"}
        before_path, _ = download_sentinel2_chip(bbox, dummy_before, is_before=True, request_id=request_id)

    if not after_path or not os.path.exists(after_path):
        dummy_after = {"date": "2025-01-20T10:30:00Z"}
        after_path, _ = download_sentinel2_chip(bbox, dummy_after, is_before=False, request_id=request_id)

    res = predict_damage_from_images(before_path, after_path, req.latitude, req.longitude)
    return DamagePredictResponse(
        prediction=res["prediction"],
        confidence=res["confidence"]
    )
