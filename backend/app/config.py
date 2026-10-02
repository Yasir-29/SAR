import sys
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class Settings(BaseSettings):
    CDSE_CLIENT_ID: str = ""
    CDSE_CLIENT_SECRET: str = ""
    MODEL_PATH: str = str(PROJECT_ROOT / "data" / "tamil_nadu" / "final" / "checkpoints" / "tamil_nadu_phase9_best.pt")
    CLOUD_COVER_MAX: float = 30.0
    AOI_SIZE_METERS: float = 500.0
    MOCK_SATELLITE_API: bool = False
    
    # Copernicus Data Space Ecosystem API URLs
    CDSE_TOKEN_URL: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CDSE_STAC_URL: str = "https://stac.dataspace.copernicus.eu/v1/search"
    CDSE_PROCESS_URL: str = "https://sh.dataspace.copernicus.eu/api/v1/process"
    
    # Media Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    BEFORE_DIR: Path = BASE_DIR / "data" / "before"
    AFTER_DIR: Path = BASE_DIR / "data" / "after"
    COMPARISON_DIR: Path = BASE_DIR / "data" / "comparison"
    
    # Project Root Path string attribute
    PROJECT_ROOT: str = str(PROJECT_ROOT)

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.BEFORE_DIR.mkdir(parents=True, exist_ok=True)
settings.AFTER_DIR.mkdir(parents=True, exist_ok=True)
settings.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
