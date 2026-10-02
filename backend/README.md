# Copernicus Sentinel-2 Disaster Building Damage Assessment Backend API

A complete, free satellite imagery integration for disaster building damage assessment built with Python, FastAPI, PyTorch, and the Copernicus Data Space Ecosystem (CDSE) APIs.

---

## 📁 Repository & Folder Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application & REST endpoints
│   ├── config.py             # System configuration & environment variables
│   ├── schemas.py            # Pydantic V2 request & response validation schemas
│   ├── copernicus/
│   │   ├── __init__.py
│   │   ├── auth.py           # OAuth2 client credentials & token caching manager
│   │   ├── search.py         # STAC search & Sentinel-2 L2A candidate selector
│   │   └── imagery.py        # Sentinel Hub Process API & optical chip downloader
│   └── damage/
│       ├── __init__.py
│       ├── preprocessing.py   # Spatial alignment, normalization & difference imagery
│       └── inference.py       # Certified PyTorch model inference integration
├── data/
│   ├── before/               # Cached BEFORE satellite imagery chips
│   ├── after/                # Cached AFTER satellite imagery chips
│   └── comparison/           # Generated optical difference heatmaps
├── tests/
│   ├── __init__.py
│   ├── test_copernicus.py    # Unit tests for search, AOI, & mock download
│   └── test_api.py           # Integration tests for FastAPI endpoints
├── .env                      # Local environment secrets (ignored by git)
├── .env.example              # Environment variables template
├── requirements.txt          # Production dependencies
└── README.md                 # Complete system documentation
```

---

## ⚙️ Copernicus Account & Credential Setup

1. Register for a free Copernicus Data Space Ecosystem (CDSE) account at [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/).
2. Navigate to **User Settings** -> **OAuth Clients** and create a new Client.
3. Copy your `Client ID` and `Client Secret`.
4. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
5. Set environment variables in `.env`:
   ```env
   CDSE_CLIENT_ID=your_actual_client_id
   CDSE_CLIENT_SECRET=your_actual_client_secret
   MODEL_PATH=data/tamil_nadu/final/checkpoints/tamil_nadu_phase9_best.pt
   CLOUD_COVER_MAX=30
   AOI_SIZE_METERS=500
   MOCK_SATELLITE_API=false
   ```
   *(Note: Set `MOCK_SATELLITE_API=true` if testing without consuming API quota).*

---

## 🚀 Installation & Startup Instructions

### Backend Startup

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Set Python path
export PYTHONPATH=.

# 4. Start FastAPI Uvicorn Server
uvicorn app.main:app --reload --port 8000
```

- **Backend Base URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Frontend Startup

```bash
# Navigate to frontend folder
cd frontend

# Install node dependencies
npm install

# Start Next.js development server
npm run dev
```

- **Frontend URL**: [http://localhost:3000](http://localhost:3000)
- **Copernicus Analysis Interface**: [http://localhost:3000/copernicus](http://localhost:3000/copernicus)

---

## 🧪 Testing

Execute automated unit and integration tests using pytest:

```bash
source venv/bin/activate
export PYTHONPATH=.
python -m pytest backend/tests
```

---

## 📡 API Usage & Example Payloads

### Main Combined Endpoint: `POST /api/disaster/analyze`

#### Request:
```json
{
  "latitude": 11.0168,
  "longitude": 76.9558,
  "disaster_date": "2025-01-15",
  "before_days": 30,
  "after_days": 30,
  "cloud_cover_max": 30.0
}
```

#### Response:
```json
{
  "request_id": "req_8f1a23b4cd",
  "location": {
    "latitude": 11.0168,
    "longitude": 76.9558
  },
  "before": {
    "date": "2025-01-10T10:30:15Z",
    "cloud_cover": 8.4,
    "product_id": "S2A_MSIL2A_20250110T103015_N0500_R062_T44PKA",
    "source": "Sentinel-2 L2A",
    "image_url": "http://127.0.0.1:8000/media/before/req_8f1a23b4cd_before.png"
  },
  "after": {
    "date": "2025-01-19T10:30:18Z",
    "cloud_cover": 12.1,
    "product_id": "S2B_MSIL2A_20250119T103018_N0500_R062_T44PKA",
    "source": "Sentinel-2 L2A",
    "image_url": "http://127.0.0.1:8000/media/after/req_8f1a23b4cd_after.png"
  },
  "comparison": {
    "difference_image_url": "http://127.0.0.1:8000/media/comparison/req_8f1a23b4cd_difference.png"
  },
  "model": {
    "class": "DAMAGED",
    "confidence": 0.785
  }
}
```

---

## 🔍 Imagery Selection & Resolution Limitations

### BEFORE & AFTER Imagery Selection Strategy
1. **AOI Generation**: Creates a ~500m × 500m bounding box centered on the target coordinate `(latitude, longitude)`.
2. **STAC Search Windows**:
   - **BEFORE Window**: `[disaster_date - before_days]` to `[disaster_date - 1 day]`.
   - **AFTER Window**: `[disaster_date + 1 day]` to `[disaster_date + after_days]`.
3. **Candidate Filtering & Ranking**:
   - Restricts to Sentinel-2 Level-2A (L2A) surface reflectance products.
   - Rejects scenes exceeding `cloud_cover_max` (default 30%).
   - Sorts scenes prioritizing acquisition date closest to the disaster date and lowest cloud cover.

### Sentinel-2 Resolution Limitations for Building-Level Assessment
- **Spatial Resolution**: Sentinel-2 optical bands (B02 Blue, B03 Green, B04 Red, B08 NIR) feature a **10-meter ground sample distance (GSD)** per pixel.
- **Building Footprint Size**: Average residential building footprints (~10m × 10m to 20m × 20m) span only **1 to 4 pixels** in Sentinel-2 imagery.
- **Operational Implication**:
  - 10m resolution optical imagery excels at detecting **area-level spectral shifts**, vegetation clearing, flood inundation, and severe structural collapse across neighborhoods.
  - Sub-building level roof structural damage or minor cracking requires sub-meter high-resolution imagery (e.g. WorldView, PlanetScope 3m, or Sentinel-1 SAR backscatter change).
