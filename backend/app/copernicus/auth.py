import time
import requests
from typing import Optional
from app.config import settings

_token_cache: Optional[str] = None
_token_expires_at: float = 0.0

def get_cdse_token() -> str:
    """
    Retrieves and caches OAuth2 access token for Copernicus Data Space Ecosystem.
    Implements token caching to avoid requesting a new token on every request.
    Uses exponential backoff for temporary rate-limit / server errors.
    """
    global _token_cache, _token_expires_at

    # Return cached token if valid
    if _token_cache and time.time() < (_token_expires_at - 60):
        return _token_cache

    if not settings.CDSE_CLIENT_ID or not settings.CDSE_CLIENT_SECRET:
        if settings.MOCK_SATELLITE_API:
            return "mock_cdse_oauth2_token"
        raise ValueError("CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables must be configured.")

    payload = {
        "grant_type": "client_credentials",
        "client_id": settings.CDSE_CLIENT_ID,
        "client_secret": settings.CDSE_CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    max_retries = 3
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            response = requests.post(settings.CDSE_TOKEN_URL, data=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                _token_cache = data["access_token"]
                expires_in = data.get("expires_in", 600)
                _token_expires_at = time.time() + expires_in
                return _token_cache
            elif response.status_code in (429, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
            else:
                raise Exception(f"CDSE Authentication failed with status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"CDSE Authentication connection error: {str(e)}")
            time.sleep(backoff)
            backoff *= 2

    raise Exception("CDSE Authentication failed after maximum retries.")
