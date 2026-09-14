import time
import requests
from typing import Dict, Any, Optional
from .exceptions import (
    OverpassTimeoutError,
    OverpassRateLimitError,
    OverpassHTTPError,
    OverpassInvalidJSONError
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_USER_AGENT = "TamilNaduDamageAssessment/1.0 (Disaster Resilience Research)"

class OverpassClient:
    def __init__(self, endpoint_url: str = OVERPASS_URL, timeout: int = 90, max_retries: int = 3, user_agent: str = DEFAULT_USER_AGENT):
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Executes an Overpass QL query using HTTP POST with exponential backoff retry.
        """
        headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint_url,
                    data={"data": query},
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as e:
                        raise OverpassInvalidJSONError(f"Malformed JSON: {str(e)}")
                        
                elif response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
                    time.sleep(sleep_time)
                    last_error = OverpassRateLimitError()
                    continue
                    
                elif response.status_code in [502, 503, 504]:
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                    last_error = OverpassHTTPError(response.text[:200], status_code=response.status_code)
                    continue
                    
                else:
                    raise OverpassHTTPError(response.text[:200], status_code=response.status_code)
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectTimeout):
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    last_error = OverpassTimeoutError()
                    continue
                raise OverpassTimeoutError()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    last_error = OverpassHTTPError(str(e))
                    continue
                raise OverpassHTTPError(str(e))
                
        if last_error:
            raise last_error
        raise OverpassHTTPError("Failed after retries")
