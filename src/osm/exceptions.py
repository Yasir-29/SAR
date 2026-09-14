class OSMException(Exception):
    """Base exception for OSM operations."""
    def __init__(self, message: str, error_code: str = "OSM_ERROR", retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable

    def to_dict(self):
        return {
            "status": "error",
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable
        }

class InvalidBBoxError(OSMException):
    def __init__(self, message: str):
        super().__init__(message, error_code="INVALID_BBOX", retryable=False)

class PlaceNotFoundError(OSMException):
    def __init__(self, message: str):
        super().__init__(message, error_code="PLACE_NOT_FOUND", retryable=False)

class PlaceAmbiguousError(OSMException):
    def __init__(self, message: str):
        super().__init__(message, error_code="PLACE_AMBIGUOUS", retryable=False)

class OverpassTimeoutError(OSMException):
    def __init__(self, message: str = "Overpass request timed out."):
        super().__init__(message, error_code="OVERPASS_TIMEOUT", retryable=True)

class OverpassRateLimitError(OSMException):
    def __init__(self, message: str = "Overpass rate limit exceeded (HTTP 429)."):
        super().__init__(message, error_code="OVERPASS_RATE_LIMIT", retryable=True)

class OverpassHTTPError(OSMException):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(f"Overpass HTTP {status_code}: {message}", error_code="OVERPASS_HTTP_ERROR", retryable=(status_code >= 500))

class OverpassInvalidJSONError(OSMException):
    def __init__(self, message: str = "Failed to parse JSON response from Overpass."):
        super().__init__(message, error_code="OVERPASS_INVALID_JSON", retryable=False)

class NoBuildingsFoundError(OSMException):
    def __init__(self, message: str = "No building footprints were returned for the requested area."):
        super().__init__(message, error_code="NO_BUILDINGS_FOUND", retryable=False)

class InvalidGeometryError(OSMException):
    def __init__(self, message: str):
        super().__init__(message, error_code="INVALID_GEOMETRY", retryable=False)

class OutputWriteError(OSMException):
    def __init__(self, message: str):
        super().__init__(message, error_code="OUTPUT_WRITE_ERROR", retryable=False)
