"""Test Copernicus Data Space user OAuth (password grant) for OData downloads.

Loads CDSE_USERNAME and CDSE_PASSWORD from the project-root .env via python-dotenv,
then from the process environment. Uses grant_type=password and client_id=cdse-public.
Does not print credentials or access tokens. Does not download imagery.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PUBLIC_CLIENT_ID = "cdse-public"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_cdse_user_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _safe_error_body(raw: bytes, limit: int = 300) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:limit].replace("\n", " ")
    parts = []
    for key in ("error", "error_description", "errorMessage"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)[:limit] if parts else "unrecognized error payload"


def main() -> int:
    _load_cdse_user_env()
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        missing = [
            name
            for name, val in (
                ("CDSE_USERNAME", username),
                ("CDSE_PASSWORD", password),
            )
            if not val
        ]
        print(
            "CDSE authentication: FAILED — "
            f"missing environment variable(s): {', '.join(missing)}"
        )
        return 1

    body = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": PUBLIC_CLIENT_ID,
            "username": username,
            "password": password,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", 200)
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(
            f"CDSE authentication: FAILED — HTTP {exc.code}; "
            f"{_safe_error_body(exc.read())}"
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"CDSE authentication: FAILED — network error: {exc.reason!s}")
        return 1

    token = payload.get("access_token")
    if status == 200 and isinstance(token, str) and token:
        print("CDSE authentication: SUCCESS")
        return 0

    print(
        f"CDSE authentication: FAILED — HTTP {status}; "
        "response did not contain an access token"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
