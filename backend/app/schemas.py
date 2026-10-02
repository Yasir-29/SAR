from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime

class DisasterAnalysisRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of location", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="Longitude of location", ge=-180.0, le=180.0)
    disaster_date: str = Field(..., description="Disaster date in YYYY-MM-DD format")
    before_days: int = Field(30, description="Days before disaster date to search", gt=0)
    after_days: int = Field(30, description="Days after disaster date to search", gt=0)
    cloud_cover_max: float = Field(30.0, description="Maximum allowed cloud coverage percentage", ge=0.0, le=100.0)

    @field_validator("disaster_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("disaster_date must be a valid ISO date in YYYY-MM-DD format.")

class LocationSchema(BaseModel):
    latitude: float
    longitude: float

class ImageryMetadata(BaseModel):
    date: str
    cloud_cover: float
    product_id: str
    source: str = "Sentinel-2 L2A"
    image_url: Optional[str] = None

class ComparisonMetadata(BaseModel):
    difference_image_url: str

class ModelPrediction(BaseModel):
    prediction_class: str = Field(..., alias="class")
    confidence: float

    model_config = ConfigDict(populate_by_name=True)

class DisasterAnalysisResponse(BaseModel):
    request_id: str
    location: LocationSchema
    before: ImageryMetadata
    after: ImageryMetadata
    comparison: ComparisonMetadata
    model: ModelPrediction

class DamagePredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    before_image_path: Optional[str] = None
    after_image_path: Optional[str] = None

class DamagePredictResponse(BaseModel):
    prediction: str
    confidence: float
