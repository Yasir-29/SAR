"""Download the four selected Chennai Flood 2015 Copernicus products via CDSE OData.

Uses CDSE_USERNAME / CDSE_PASSWORD with grant_type=password and client_id=cdse-public.
Never prints credentials or tokens. Does not resample, convert, train, or download other events.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PUBLIC_CLIENT_ID = "cdse-public"
CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"

PROJECT = Path(__file__).resolve().parents[2]
AOI_PATH = PROJECT / "data/south_coastal/events/chennai_flood_2015/aoi.geojson"
RAW_ROOT = PROJECT / "data/south_coastal/raw/chennai_flood_2015"

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S2_FILE_TOKENS = {
    "B1": "B01",
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B05",
    "B6": "B06",
    "B7": "B07",
    "B8": "B08",
    "B8A": "B8A",
    "B9": "B09",
    "B11": "B11",
    "B12": "B12",
}

# Exact catalog-selected products (validated AOI tile 44PMV; non-COG GRD).
PRODUCTS = [
    {
        "role": "s1_pre",
        "sensor": "Sentinel-1",
        "subdir": "sentinel1",
        "name": "S1A_IW_GRDH_1SDV_20151112T003120_20151112T003149_008564_00C241_37E8.SAFE",
        "expect_date": "2015-11-12",
        "expect_type": "IW_GRDH_1S",
        "expect_orbit": 92,
        "expect_dir": "DESCENDING",
        "expect_pol": "VV&VH",
    },
    {
        "role": "s1_post",
        "sensor": "Sentinel-1",
        "subdir": "sentinel1",
        "name": "S1A_IW_GRDH_1SDV_20151206T003120_20151206T003149_008914_00CC1F_BEF9.SAFE",
        "expect_date": "2015-12-06",
        "expect_type": "IW_GRDH_1S",
        "expect_orbit": 92,
        "expect_dir": "DESCENDING",
        "expect_pol": "VV&VH",
    },
    {
        "role": "s2_pre",
        "sensor": "Sentinel-2",
        "subdir": "sentinel2",
        "name": "S2A_MSIL2A_20151128T050142_N0500_R119_T44PMV_20231027T151725.SAFE",
        "expect_date": "2015-11-28",
        "expect_type": "S2MSI2A",
        "expect_tile": "44PMV",
    },
    {
        "role": "s2_post",
        "sensor": "Sentinel-2",
        "subdir": "sentinel2",
        "name": "S2A_MSIL2A_20151228T050222_N0500_R119_T44PMV_20231009T231712.SAFE",
        "expect_date": "2015-12-28",
        "expect_type": "S2MSI2A",
        "expect_tile": "44PMV",
    },
]


def load_cdse_env() -> bool:
    load_dotenv(PROJECT / ".env")
    return bool(os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD"))


def get_token() -> str:
    body = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": PUBLIC_CLIENT_ID,
            "username": os.environ["CDSE_USERNAME"],
            "password": os.environ["CDSE_PASSWORD"],
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("token endpoint did not return an access token")
    return token


def catalog_by_name(name: str) -> dict:
    filt = f"Name eq '{name}'"
    url = CATALOG + "?" + urllib.parse.urlencode(
        {"$filter": filt, "$top": "1", "$expand": "Attributes"}
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "sardd-chennai-dl/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        items = json.loads(resp.read().decode())["value"]
    if not items:
        raise RuntimeError(f"catalog miss: {name}")
    return items[0]


def attrs(item: dict) -> dict:
    return {a["Name"]: a["Value"] for a in (item.get("Attributes") or [])}


def coords_of(geom: dict | None) -> list[tuple[float, float]]:
    if not geom:
        return []
    out: list[tuple[float, float]] = []

    def walk(node):
        if not node:
            return
        if isinstance(node[0], (int, float)):
            out.append((float(node[0]), float(node[1])))
            return
        for child in node:
            walk(child)

    walk(geom.get("coordinates"))
    return out


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if denom != 0 and x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def load_aoi_ring() -> list[tuple[float, float]]:
    fc = json.loads(AOI_PATH.read_text())
    geom = fc["features"][0]["geometry"]
    return [(float(x), float(y)) for x, y in geom["coordinates"][0]]


def geoms_intersect(footprint: dict | None, aoi_ring: list[tuple[float, float]]) -> bool:
    pts = coords_of(footprint)
    if len(pts) < 3:
        return False
    aoi_xs = [p[0] for p in aoi_ring]
    aoi_ys = [p[1] for p in aoi_ring]
    fp_xs = [p[0] for p in pts]
    fp_ys = [p[1] for p in pts]
    if max(fp_xs) < min(aoi_xs) or max(aoi_xs) < min(fp_xs) or max(fp_ys) < min(aoi_ys) or max(aoi_ys) < min(fp_ys):
        return False
    aoi_c = (sum(aoi_xs) / len(aoi_xs), sum(aoi_ys) / len(aoi_ys))
    if point_in_ring(aoi_c[0], aoi_c[1], pts):
        return True
    for p in aoi_ring:
        if point_in_ring(p[0], p[1], pts):
            return True
    fp_c = (sum(fp_xs) / len(fp_xs), sum(fp_ys) / len(fp_ys))
    return point_in_ring(fp_c[0], fp_c[1], aoi_ring)


def download_product(product_id: str, dest: Path, token: str, expected_len: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = DOWNLOAD.format(id=product_id)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "*/*", "User-Agent": "sardd-chennai-dl/1.0"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=3600, context=ctx) as resp:
        total = 0
        with part.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                total += len(chunk)
                if total % (100 * 1024 * 1024) < 1024 * 1024:
                    print(f"  wrote {total / 1e6:.0f} MB -> {dest.name}", flush=True)
    part.replace(dest)
    size = dest.stat().st_size
    if expected_len and size != expected_len:
        print(f"  note: size {size} vs catalog {expected_len}", flush=True)


def zip_ok(path: Path) -> tuple[bool, str]:
    if not zipfile.is_zipfile(path):
        return False, "not a zip"
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                return False, f"crc failed: {bad}"
            names = zf.namelist()
            if not names:
                return False, "empty zip"
    except zipfile.BadZipFile as exc:
        return False, str(exc)
    return True, "zip crc ok"


def s2_bands_in_zip(path: Path) -> list[str]:
    found: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        names = [n.upper() for n in zf.namelist()]
    for band, token in S2_FILE_TOKENS.items():
        needle = f"{token}_"
        if any(needle in n and n.endswith(".JP2") and "IMG_DATA" in n and "TCI" not in n for n in names):
            found.add(band)
        elif any(f"_{token}." in n and n.endswith(".JP2") for n in names):
            found.add(band)
    return [b for b in S2_BANDS if b in found]


def s1_pols_in_zip(path: Path) -> list[str]:
    found = []
    with zipfile.ZipFile(path) as zf:
        joined = " ".join(zf.namelist()).upper()
    for pol in ("VV", "VH"):
        if f"-{pol}-" in joined or f"_{pol}_" in joined or f"-{pol}." in joined:
            found.append(pol)
    return found


def main() -> int:
    if not load_cdse_env():
        print("CDSE authentication: FAILED — missing environment variable(s): CDSE_USERNAME, CDSE_PASSWORD")
        return 1
    try:
        token = get_token()
    except Exception as exc:
        print(f"CDSE authentication: FAILED — {type(exc).__name__}")
        return 1
    print("CDSE authentication: SUCCESS", flush=True)

    aoi_ring = load_aoi_ring()
    results = []
    for spec in PRODUCTS:
        print(f"\n=== {spec['role']} {spec['name']} ===", flush=True)
        item = catalog_by_name(spec["name"])
        product_id = item["Id"]
        a = attrs(item)
        acq = item["ContentDate"]["Start"][:10]
        ptype = a.get("productType") or item.get("ProductType")
        dest = RAW_ROOT / spec["subdir"] / (spec["name"].replace(".SAFE", ".zip"))
        if dest.exists():
            print(f"exists, not overwriting: {dest}", flush=True)
        else:
            token = get_token()
            print("downloading...", flush=True)
            try:
                download_product(product_id, dest, token, item.get("ContentLength"))
            except urllib.error.HTTPError as exc:
                reason = f"HTTP {exc.code}"
                if exc.code == 401:
                    reason += " — download service rejected the Bearer token"
                print(f"DOWNLOAD FAILED {reason} for {spec['role']}", flush=True)
                results.append(
                    {
                        "role": spec["role"],
                        "name": spec["name"],
                        "path": str(dest),
                        "size": dest.stat().st_size if dest.exists() else 0,
                        "intersects_aoi": False,
                        "integrity": False,
                        "ok": False,
                        "error": reason,
                    }
                )
                continue
        size = dest.stat().st_size if dest.exists() else 0
        z_ok, z_msg = zip_ok(dest) if dest.exists() else (False, "missing")
        checks = []
        if acq != spec["expect_date"]:
            checks.append(f"date {acq} != {spec['expect_date']}")
        if spec["sensor"] == "Sentinel-1":
            orbit = a.get("relativeOrbitNumber")
            direction = str(a.get("orbitDirection") or "")
            polar = str(a.get("polarisationChannels") or a.get("polarisation") or "")
            if int(orbit) != spec["expect_orbit"]:
                checks.append(f"orbit {orbit}")
            if spec["expect_dir"] not in direction.upper():
                checks.append(f"dir {direction}")
            if "VV" not in polar.upper() or "VH" not in polar.upper():
                checks.append(f"polar {polar}")
            pols = s1_pols_in_zip(dest) if z_ok else []
            if pols != ["VV", "VH"] and set(pols) != {"VV", "VH"}:
                checks.append(f"zip pols {pols}")
            if spec["expect_type"] not in str(ptype):
                checks.append(f"type {ptype}")
        else:
            tile = str(a.get("tileId") or "")
            if tile != spec["expect_tile"] and spec["expect_tile"] not in spec["name"]:
                checks.append(f"tile {tile}")
            if spec["expect_type"] not in str(ptype):
                checks.append(f"type {ptype}")
            bands = s2_bands_in_zip(dest) if z_ok else []
            missing = [b for b in S2_BANDS if b not in bands]
            if missing:
                checks.append(f"missing bands {missing}")
            b10 = False
            if z_ok:
                with zipfile.ZipFile(dest) as zf:
                    b10 = any("B10" in n.upper() and n.upper().endswith(".JP2") and "IMG_DATA" in n.upper() for n in zf.namelist())
            # L2A typically has no B10; presence is noted but not a failure for availability of the 12 required bands.
            if b10:
                print("  note: B10 JP2 present in SAFE (not used as model input)", flush=True)
        intersects = geoms_intersect(item.get("GeoFootprint"), aoi_ring)
        if not intersects:
            checks.append("no AOI intersection")
        if not z_ok:
            checks.append(z_msg)
        ok = dest.exists() and z_ok and not checks
        results.append(
            {
                "role": spec["role"],
                "name": spec["name"],
                "id": product_id,
                "date": acq,
                "type": ptype,
                "path": str(dest),
                "size": size,
                "intersects_aoi": intersects,
                "integrity": z_ok,
                "ok": ok,
                "checks": checks,
                "orbit": a.get("relativeOrbitNumber"),
                "direction": a.get("orbitDirection"),
                "polar": a.get("polarisationChannels"),
                "tile": a.get("tileId"),
            }
        )
        print(f"  id={product_id} date={acq} type={ptype} size={size} zip={z_ok} aoi={intersects} ok={ok}", flush=True)
        if checks:
            print(f"  checks: {checks}", flush=True)

    print("\n===== SUMMARY =====", flush=True)
    all_ok = True
    for r in results:
        print(
            f"{r['role']}\t{r['name']}\n  path={r['path']}\n  size={r.get('size', 0)}\n  "
            f"intersects_aoi={r.get('intersects_aoi', False)} integrity={r.get('integrity', False)} "
            f"passed={r.get('ok', False)}"
            + (f"\n  error={r['error']}" if r.get("error") else ""),
            flush=True,
        )
        all_ok = all_ok and r.get("ok", False)
    print("ALL_FOUR_PASSED" if all_ok and len(results) == 4 else "SOME_FAILED", flush=True)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
