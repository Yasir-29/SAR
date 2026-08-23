"""Phase 1: zero-shot xBD-S12 inference on user Copernicus GeoTIFFs.

Does not train, fine-tune, or change the model architecture. Original input
files are never overwritten. New files are written only under --output-dir.

Usage (from the xbd-s12 repo root, with PYTHONPATH=. or `uv run`):

    python src/inference/phase1_copernicus.py \\
        --s2-pre PATH --s2-post PATH --s1-pre PATH --s1-post PATH \\
        --output-dir data/inference/south_coastal_phase1

If Sentinel-1 VV/VH are separate single-band files:

    python src/inference/phase1_copernicus.py \\
        --s2-pre PATH --s2-post PATH \\
        --s1-pre PATH_VV --s1-pre-vh PATH_VH \\
        --s1-post PATH_VV --s1-post-vh PATH_VH \\
        --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.constants import S1_BANDS, S2_BANDS
from src.data.sentinel.utils import save_raster
from src.visualization import plot_mask

TARGET_RESOLUTION_M = 4.0
DB_MAX_ABS = 80.0  # dB backscatter is typically within roughly [-50, 20]


class ContractError(Exception):
    """Raised when inputs do not match the xBD-S12 inference contract."""


@dataclass
class RasterReport:
    path: str
    exists: bool
    driver: str | None = None
    crs: str | None = None
    transform: str | None = None
    transform_tuple: tuple[float, ...] | None = None
    width: int | None = None
    height: int | None = None
    count: int | None = None
    dtype: str | None = None
    nodata: float | None = None
    resolution: tuple[float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None
    descriptions: list[str | None] | None = None
    sample_min: float | None = None
    sample_max: float | None = None
    sample_mean: float | None = None
    acquisition: str | None = None
    notes: list[str] | None = None


def _crs_str(crs) -> str | None:
    if crs is None:
        return None
    try:
        if crs.to_epsg():
            return f"EPSG:{crs.to_epsg()}"
    except Exception:
        pass
    return crs.to_string()


def inspect_raster(path: Path, sample_size: int = 1024) -> RasterReport:
    notes: list[str] = []
    if not path.exists():
        return RasterReport(path=str(path), exists=False, notes=["FILE NOT FOUND"])

    with rasterio.open(path) as src:
        tags = src.tags()
        acquisition = (
            tags.get("TIFFTAG_DATETIME")
            or tags.get("ACQUISITION_DATE")
            or tags.get("start_datetime")
            or src.tags().get("PRODUCT_START_TIME")
        )
        h, w = src.height, src.width
        ch, cw = min(sample_size, h), min(sample_size, w)
        row, col = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
        arr = src.read(window=Window(col, row, cw, ch)).astype(np.float64)
        nodata = src.nodata
        finite = arr[np.isfinite(arr)]
        if nodata is not None:
            finite = finite[finite != nodata]
        sample_min = sample_max = sample_mean = None
        if finite.size:
            sample_min = float(np.min(finite))
            sample_max = float(np.max(finite))
            sample_mean = float(np.mean(finite))
        else:
            notes.append("No finite pixels in center sample window")

        return RasterReport(
            path=str(path.resolve()),
            exists=True,
            driver=src.driver,
            crs=_crs_str(src.crs),
            transform=str(src.transform),
            transform_tuple=tuple(src.transform)[:6],
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=str(src.dtypes[0]) if src.dtypes else None,
            nodata=float(nodata) if nodata is not None else None,
            resolution=(float(src.res[0]), float(src.res[1])),
            bounds=(float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)),
            descriptions=list(src.descriptions),
            sample_min=sample_min,
            sample_max=sample_max,
            sample_mean=sample_mean,
            acquisition=acquisition,
            notes=notes,
        )


def print_report(title: str, r: RasterReport) -> None:
    print(f"\n{title}:")
    print(f"  path: {r.path}")
    if not r.exists:
        print("  exists: NO")
        return
    print(f"  CRS: {r.crs}")
    print(f"  resolution: {r.resolution}")
    print(f"  shape: {r.count} x {r.height} x {r.width}  (bands x H x W)")
    print(f"  bands/descriptions: {r.descriptions}")
    print(f"  dtype: {r.dtype}")
    print(f"  nodata: {r.nodata}")
    print(f"  transform: {r.transform}")
    print(f"  bounds: {r.bounds}")
    print(f"  min/max/mean (center sample): {r.sample_min} / {r.sample_max} / {r.sample_mean}")
    print(f"  acquisition (if tagged): {r.acquisition}")
    if r.notes:
        print(f"  notes: {r.notes}")


def _normalize_band_name(name: str | None, index: int) -> str:
    if not name:
        return f"BAND_{index}"
    n = name.strip().upper().replace(" ", "")
    aliases = {
        "B01": "B1",
        "B02": "B2",
        "B03": "B3",
        "B04": "B4",
        "B05": "B5",
        "B06": "B6",
        "B07": "B7",
        "B08": "B8",
        "B8A": "B8A",
        "B08A": "B8A",
        "B09": "B9",
        "B11": "B11",
        "B12": "B12",
        "VV": "VV",
        "VH": "VH",
        "SIGMA0_VV": "VV",
        "SIGMA0_VH": "VH",
    }
    return aliases.get(n, n)


def looks_like_linear_sigma0(r: RasterReport) -> bool:
    """Heuristic: SNAP Sigma0 intensity is non-negative and typically << 10 in linear units."""
    if r.sample_min is None or r.sample_max is None:
        return False
    return r.sample_min >= -1e-6 and r.sample_max > 0 and r.sample_max < 50 and (r.sample_mean or 0) > 0 and (r.sample_mean or 0) < 5


def looks_like_db(r: RasterReport) -> bool:
    if r.sample_min is None or r.sample_max is None:
        return False
    return r.sample_min < 0 and r.sample_max < DB_MAX_ABS and r.sample_min > -DB_MAX_ABS


def stack_vv_vh(vv_path: Path, vh_path: Path, out_path: Path) -> Path:
    """Write a new 2-band GeoTIFF (VV, VH). Does not modify inputs."""
    with rasterio.open(vv_path) as vv, rasterio.open(vh_path) as vh:
        if vv.count != 1 or vh.count != 1:
            raise ContractError(f"Expected single-band VV/VH files, got counts {vv.count} and {vh.count}")
        if (vv.crs, vv.transform, vv.width, vv.height) != (vh.crs, vh.transform, vh.width, vh.height):
            raise ContractError(f"VV and VH are not on the same grid:\n  VV={vv_path}\n  VH={vh_path}")
        profile = vv.profile.copy()
        profile.update(count=2, dtype="float32", BIGTIFF="YES")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(vv.read(1).astype(np.float32), 1)
            dst.write(vh.read(1).astype(np.float32), 2)
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")
    print(f"Wrote stacked S1 (VV, VH) → {out_path}")
    return out_path


def linear_sigma0_to_db(src_path: Path, out_path: Path) -> Path:
    """Convert linear intensity to dB in a NEW file: 10 * log10(max(x, eps))."""
    xa = rxr.open_rasterio(src_path)
    eps = 1e-10
    db = (10.0 * np.log10(xa.where(xa > 0, eps))).astype("float32")
    db.rio.write_crs(xa.rio.crs, inplace=True)
    save_raster(db, out_path, if_exists="replace")
    # preserve / set band names
    with rasterio.open(src_path) as src, rasterio.open(out_path, "r+") as dst:
        for i, desc in enumerate(src.descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)
    print(f"Wrote linear→dB conversion → {out_path}")
    return out_path


def reorder_s2_bands(src_path: Path, out_path: Path, src_names: list[str]) -> Path:
    """Write a new 12-band GeoTIFF in exact S2_BANDS order. Rejects B10."""
    names = [_normalize_band_name(n, i + 1) for i, n in enumerate(src_names)]
    if "B10" in names:
        raise ContractError("Sentinel-2 contains B10. xBD-S12 requires B10 to be excluded.")
    missing = [b for b in S2_BANDS if b not in names]
    extra = [b for b in names if b not in S2_BANDS]
    if missing:
        raise ContractError(f"Sentinel-2 missing required bands {missing}. Extra/unrecognized: {extra}. Names={names}")
    indices = [names.index(b) + 1 for b in S2_BANDS]  # rasterio 1-based
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=12, dtype=src.dtypes[0], BIGTIFF="YES")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for out_i, src_i in enumerate(indices, start=1):
                dst.write(src.read(src_i), out_i)
                dst.set_band_description(out_i, S2_BANDS[out_i - 1])
    print(f"Wrote S2 in xBD-S12 band order → {out_path}")
    return out_path


def grids_identical(a: RasterReport, b: RasterReport, atol: float = 1e-6) -> dict[str, bool]:
    crs_ok = a.crs == b.crs
    shape_ok = a.width == b.width and a.height == b.height
    transform_ok = False
    if a.transform_tuple and b.transform_tuple:
        transform_ok = all(abs(x - y) <= atol for x, y in zip(a.transform_tuple, b.transform_tuple, strict=True))
    return {"crs": crs_ok, "transform": transform_ok, "shape": shape_ok}


def validate_s1(r: RasterReport, role: str, allow_linear_to_db: bool) -> list[str]:
    errors: list[str] = []
    if not r.exists:
        errors.append(f"{role}: file not found")
        return errors
    if r.count != 2:
        errors.append(f"{role}: expected 2 bands (VV, VH), found {r.count}. If you have split files, pass --s1-*-vh.")
    names = [_normalize_band_name(n, i + 1) for i, n in enumerate(r.descriptions or [])]
    if r.count == 2 and names[0] not in ("VV", "BAND_1") and names[0] != "VV":
        if names[:2] == ["VH", "VV"]:
            errors.append(f"{role}: band order is VH, VV. xBD-S12 requires band 1=VV, band 2=VH.")
        elif "VV" not in names or "VH" not in names:
            errors.append(
                f"{role}: cannot confirm VV/VH from descriptions {r.descriptions}. "
                "Refusing to guess. Re-export with band descriptions or stack VV then VH."
            )
    if r.dtype not in ("float32", "float64"):
        errors.append(f"{role}: dtype {r.dtype} is not float32 (xBD-S12 expectation).")
    if looks_like_linear_sigma0(r) and not looks_like_db(r):
        msg = (
            f"{role}: values look like LINEAR Sigma0 intensity "
            f"(sample min/max/mean={r.sample_min:.4g}/{r.sample_max:.4g}/{r.sample_mean:.4g}), "
            "not dB. GEE GRD / xBD-S12 use log amplitude in dB."
        )
        if allow_linear_to_db:
            msg += " Will convert to a NEW dB file because --allow-s1-linear-to-db was set."
        else:
            errors.append(msg + " Refusing to convert silently. Re-run with --allow-s1-linear-to-db if you want 10*log10.")
        if allow_linear_to_db:
            print("WARNING:", msg)
    elif not looks_like_db(r):
        errors.append(
            f"{role}: sample range [{r.sample_min}, {r.sample_max}] does not look like dB backscatter. Stopping."
        )
    return errors


def validate_s2(r: RasterReport, role: str, assume_order: bool) -> list[str]:
    errors: list[str] = []
    if not r.exists:
        errors.append(f"{role}: file not found — Sentinel-2 is required. No S2 product was found on disk.")
        return errors
    names = [_normalize_band_name(n, i + 1) for i, n in enumerate(r.descriptions or [])]
    if "B10" in names or (r.descriptions and any((d or "").upper() in ("B10", "B010") for d in r.descriptions)):
        errors.append(f"{role}: B10 is present. xBD-S12 uses 12 L2A bands without B10.")
    if r.count != 12:
        errors.append(f"{role}: expected 12 bands {S2_BANDS}, found {r.count}. TCI/RGB cannot be used as model input.")
        return errors
    if names == S2_BANDS:
        return errors
    if all(n.startswith("BAND_") for n in names):
        if assume_order:
            print(f"WARNING: {role} has no band names; --assume-s2-band-order treats them as {S2_BANDS}.")
        else:
            errors.append(
                f"{role}: 12 bands but no names {r.descriptions}. "
                "Will not assume order. Re-export with descriptions or pass --assume-s2-band-order."
            )
        return errors
    if sorted(names) == sorted(S2_BANDS) and names != S2_BANDS:
        print(f"WARNING: {role} has the right bands in a different order {names}. A reordered copy will be written.")
        return errors
    missing = [b for b in S2_BANDS if b not in names]
    if missing:
        errors.append(f"{role}: missing {missing}. Found {names}.")
    return errors


def needs_4m_reproject(r: RasterReport) -> bool:
    if not r.crs or r.resolution is None:
        return True
    # Geographic CRS in degrees is never a 4 m projected grid
    if r.crs.upper().startswith("EPSG:4326") or "GEOGCS" in r.crs.upper() or r.crs == "WGS 84":
        return True
    rx, ry = abs(r.resolution[0]), abs(r.resolution[1])
    return abs(rx - TARGET_RESOLUTION_M) > 0.25 or abs(ry - TARGET_RESOLUTION_M) > 0.25


def reproject_to_4m(s2_pre: Path, others: dict[str, Path], out_dir: Path) -> dict[str, Path]:
    """Match Palisades notebook: S2 pre → 4 m, then reproject_match Lanczos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xa = rxr.open_rasterio(s2_pre)
    target_fp = out_dir / "s2_pre_4m.tif"
    if xa.rio.crs is None:
        raise ContractError("Sentinel-2 pre has no CRS; cannot build a 4 m reference grid.")
    # Projected meters vs geographic: if CRS is geographic, first convert to UTM
    crs = xa.rio.crs
    if crs.to_epsg() == 4326 or crs.is_geographic:
        from src.utils.geometry import get_best_utm_crs
        from shapely.geometry import box

        b = xa.rio.bounds()
        geo = box(b[0], b[1], b[2], b[3])
        utm = get_best_utm_crs(geo)
        print(f"S2 pre is geographic ({_crs_str(crs)}); reprojecting to {utm} at {TARGET_RESOLUTION_M} m")
        xa_proj = xa.rio.reproject(utm, resolution=TARGET_RESOLUTION_M, resampling=Resampling.lanczos)
    else:
        xa_proj = xa.rio.reproject(crs, resolution=TARGET_RESOLUTION_M, resampling=Resampling.lanczos)
    save_raster(xa_proj, target_fp, if_exists="replace")
    _copy_band_descriptions(s2_pre, target_fp)

    target = rxr.open_rasterio(target_fp, chunks=True)
    mapping = {"s2_pre": target_fp}
    name_map = {"s2_post": "s2_post_4m.tif", "s1_pre": "s1_pre_4m.tif", "s1_post": "s1_post_4m.tif"}
    for key, fp in others.items():
        new_fp = out_dir / name_map[key]
        xa_o = rxr.open_rasterio(fp, chunks=True)
        xa_m = xa_o.rio.reproject_match(target, resampling=Resampling.lanczos)
        xa_m = xa_m.fillna(0)
        xa_m = xa_m.assign_coords({"x": target.x, "y": target.y})
        save_raster(xa_m, new_fp, if_exists="replace")
        _copy_band_descriptions(fp, new_fp)
        mapping[key] = new_fp
        print(f"Aligned {key} → {new_fp}")
    return mapping


def _copy_band_descriptions(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src, rasterio.open(dst_path, "r+") as dst:
        n = min(src.count, dst.count)
        for i in range(1, n + 1):
            desc = src.descriptions[i - 1]
            if desc:
                dst.set_band_description(i, desc)


def save_preview(s2_post_4m: Path, pred_fp: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    with rasterio.open(s2_post_4m) as src:
        descs = [_normalize_band_name(d, i + 1) for i, d in enumerate(src.descriptions)]
        if "B4" in descs and "B3" in descs and "B2" in descs:
            rgb = src.read([descs.index("B4") + 1, descs.index("B3") + 1, descs.index("B2") + 1]).astype(np.float32)
        else:
            rgb = src.read([4, 3, 2]).astype(np.float32) if src.count >= 4 else src.read([1, 1, 1]).astype(np.float32)
    vis = np.nanpercentile(rgb, 98)
    rgb = np.clip(rgb / (vis + 1e-6), 0, 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(np.moveaxis(rgb, 0, -1))
    axs[0].set_title("Sentinel-2 post RGB (B4,B3,B2)")
    axs[0].axis("off")
    plot_mask(pred_fp, ax=axs[1], add_colorbar=True, use_simplified_classes=True)
    axs[1].set_title("Zero-shot xBD-S12 prediction (not validated)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved preview → {out_png}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 zero-shot xBD-S12 inference on Copernicus GeoTIFFs.")
    p.add_argument("--s2-pre", type=Path, help="Sentinel-2 L2A pre-disaster GeoTIFF (12 bands)")
    p.add_argument("--s2-post", type=Path, help="Sentinel-2 L2A post-disaster GeoTIFF (12 bands)")
    p.add_argument("--s1-pre", type=Path, help="Sentinel-1 pre-disaster GeoTIFF (2-band VV,VH) or VV-only")
    p.add_argument("--s1-post", type=Path, help="Sentinel-1 post-disaster GeoTIFF (2-band VV,VH) or VV-only")
    p.add_argument("--s1-pre-vh", type=Path, default=None, help="Optional VH file if --s1-pre is VV-only")
    p.add_argument("--s1-post-vh", type=Path, default=None, help="Optional VH file if --s1-post is VV-only")
    p.add_argument("--output-dir", type=Path, default=Path("data/inference/south_coastal_phase1"))
    p.add_argument("--inspect-only", action="store_true", help="Print validation report and exit (no inference)")
    p.add_argument(
        "--allow-s1-linear-to-db",
        action="store_true",
        help="If S1 looks like linear Sigma0, write a NEW dB GeoTIFF (10*log10). Never overwrites inputs.",
    )
    p.add_argument(
        "--assume-s2-band-order",
        action="store_true",
        help="If S2 has 12 unnamed bands, assume they are already B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12.",
    )
    p.add_argument("--skip-inference", action="store_true", help="Validate and preprocess only")
    return p.parse_args(argv)


def prepare_s1(path: Path, vh: Path | None, role: str, out_dir: Path) -> Path:
    with rasterio.open(path) as src:
        count = src.count
    if count == 2:
        return path
    if count == 1:
        if vh is None:
            raise ContractError(f"{role} has 1 band. Provide the matching VH file via --{role.replace('_', '-')}-vh.")
        return stack_vv_vh(path, vh, out_dir / f"{role}_vvvh.tif")
    raise ContractError(f"{role} has {count} bands; expected 1 (split) or 2 (VV,VH).")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    missing_required = args.s2_pre is None or args.s2_post is None or args.s1_pre is None or args.s1_post is None
    if missing_required:
        errors.append(
            "Missing required paths (--s2-pre --s2-post --s1-pre --s1-post). "
            "No Sentinel-2 product was found on this machine (searched for MSIL2A / S2A / S2B). "
            "Local Sentinel-1 GeoTIFFs exist under /Users/yasir/Desktop/SAR/processed_{pre,post}/."
        )
        if args.s1_pre is None and Path("/Users/yasir/Desktop/SAR/processed_pre/PRECHENNAI_VV.tif").exists():
            print("Discovered local S1 files (they do not pass the full xBD-S12 contract — see report):")
            print("  PRE VV ", "/Users/yasir/Desktop/SAR/processed_pre/PRECHENNAI_VV.tif")
            print("  PRE VH ", "/Users/yasir/Desktop/SAR/processed_pre/PRECHENNAI_VH.tif")
            print("  POST VV", "/Users/yasir/Desktop/SAR/processed_post/POST_Chennai_VV.tif")
            print("  POST VH", "/Users/yasir/Desktop/SAR/processed_post/POST_Chennai_VH.tif")

    # Inspect whatever was provided
    reports = {}
    for key, path in {
        "S2 PRE": args.s2_pre,
        "S2 POST": args.s2_post,
        "S1 PRE": args.s1_pre,
        "S1 POST": args.s1_post,
    }.items():
        if path is not None:
            reports[key] = inspect_raster(path)
            print_report(key, reports[key])
        else:
            print_report(key, RasterReport(path="(not provided)", exists=False))

    if args.s1_pre_vh:
        print_report("S1 PRE VH", inspect_raster(args.s1_pre_vh))
    if args.s1_post_vh:
        print_report("S1 POST VH", inspect_raster(args.s1_post_vh))

    if missing_required:
        print("\nCOREGISTRATION: not evaluated (incomplete inputs)")
        print("MODEL INPUT: S1 channels: 2 | S2 channels: 12 | expected per date: 14")
        print("\nSTOPPED: " + " ".join(errors))
        if args.inspect_only:
            print("(--inspect-only: no inference attempted)")
        return 2

    try:
        s1_pre = prepare_s1(args.s1_pre, args.s1_pre_vh, "s1_pre", out_dir)
        s1_post = prepare_s1(args.s1_post, args.s1_post_vh, "s1_post", out_dir)
    except ContractError as e:
        print(f"\nSTOPPED: {e}")
        return 2

    r_s1_pre, r_s1_post = inspect_raster(s1_pre), inspect_raster(s1_post)
    r_s2_pre, r_s2_post = inspect_raster(args.s2_pre), inspect_raster(args.s2_post)

    contract: list[str] = []
    contract += validate_s1(r_s1_pre, "S1 PRE", args.allow_s1_linear_to_db)
    contract += validate_s1(r_s1_post, "S1 POST", args.allow_s1_linear_to_db)
    contract += validate_s2(r_s2_pre, "S2 PRE", args.assume_s2_band_order)
    contract += validate_s2(r_s2_post, "S2 POST", args.assume_s2_band_order)

    if contract:
        print("\nCOREGISTRATION: not started (contract failed)")
        print("MODEL INPUT: S1 channels: 2 | S2 channels: 12 | expected per date: 14")
        print("\nSTOPPED before inference. Contract failures:")
        for e in contract:
            print(" -", e)
        return 2

    # Optional linear → dB (new files only)
    if args.allow_s1_linear_to_db and looks_like_linear_sigma0(r_s1_pre):
        s1_pre = linear_sigma0_to_db(s1_pre, out_dir / "s1_pre_db.tif")
        r_s1_pre = inspect_raster(s1_pre)
    if args.allow_s1_linear_to_db and looks_like_linear_sigma0(r_s1_post):
        s1_post = linear_sigma0_to_db(s1_post, out_dir / "s1_post_db.tif")
        r_s1_post = inspect_raster(s1_post)

    # Reorder S2 if needed
    s2_pre, s2_post = args.s2_pre, args.s2_post
    names_pre = [_normalize_band_name(n, i + 1) for i, n in enumerate(r_s2_pre.descriptions or [])]
    if r_s2_pre.count == 12 and names_pre != S2_BANDS and not all(n.startswith("BAND_") for n in names_pre):
        s2_pre = reorder_s2_bands(s2_pre, out_dir / "s2_pre_ordered.tif", r_s2_pre.descriptions or [])
    names_post = [_normalize_band_name(n, i + 1) for i, n in enumerate(r_s2_post.descriptions or [])]
    if r_s2_post.count == 12 and names_post != S2_BANDS and not all(n.startswith("BAND_") for n in names_post):
        s2_post = reorder_s2_bands(s2_post, out_dir / "s2_post_ordered.tif", r_s2_post.descriptions or [])

    aligned = {"s2_pre": s2_pre, "s2_post": s2_post, "s1_pre": s1_pre, "s1_post": s1_post}
    r_s2_pre = inspect_raster(s2_pre)
    if any(needs_4m_reproject(inspect_raster(p)) for p in aligned.values()) or not all(
        grids_identical(inspect_raster(s2_pre), inspect_raster(p))["crs"]
        and grids_identical(inspect_raster(s2_pre), inspect_raster(p))["shape"]
        and grids_identical(inspect_raster(s2_pre), inspect_raster(p))["transform"]
        for p in (s2_post, s1_pre, s1_post)
    ):
        print("\nPreprocessing: building a common ~4 m grid from Sentinel-2 pre (Lanczos, same as Palisades notebook).")
        mapping = reproject_to_4m(s2_pre, {"s2_post": s2_post, "s1_pre": s1_pre, "s1_post": s1_post}, out_dir)
        aligned = mapping
    else:
        print("\nInputs already share a ~4 m grid; skipping reprojection.")

    reports_4m = {k: inspect_raster(v) for k, v in aligned.items()}
    ref = reports_4m["s2_pre"]
    coreg = {name: grids_identical(ref, reports_4m[name]) for name in ("s2_post", "s1_pre", "s1_post")}
    print("\nCOREGISTRATION:")
    all_ok = True
    for name, flags in coreg.items():
        line = f"  vs {name}: CRS identical={'YES' if flags['crs'] else 'NO'}  transform identical={'YES' if flags['transform'] else 'NO'}  shape identical={'YES' if flags['shape'] else 'NO'}"
        print(line)
        all_ok = all_ok and all(flags.values())
    print("MODEL INPUT: S1 channels: 2 | S2 channels: 12 | expected channels per date: 14")

    report_json = out_dir / "phase1_validation_report.json"
    report_json.write_text(json.dumps({k: asdict(v) for k, v in reports_4m.items()}, indent=2))
    print(f"Wrote {report_json}")

    if not all_ok:
        print("\nSTOPPED: 4 m files are not on an identical grid.")
        return 2

    if args.inspect_only or args.skip_inference:
        print("\nInspect/preprocess complete. Inference not run.")
        return 0

    pred_fp = out_dir / "phase1_damage_prediction.tif"
    if pred_fp.exists():
        print(f"\nSTOPPED: {pred_fp} already exists (xBD-S12 InferenceFromHub will not overwrite). Remove it to re-run.")
        return 2

    from src.inference.from_hub import InferenceFromHub

    print("\nLoading Hugging Face ensembles (3 loc + 3 dmg). Zero-shot only — no labels, no metrics.")
    infer = InferenceFromHub()
    infer.run_inference(
        aligned["s2_pre"],
        aligned["s2_post"],
        aligned["s1_pre"],
        aligned["s1_post"],
        pred_fp,
    )
    preview = out_dir / "phase1_damage_preview.png"
    save_preview(aligned["s2_post"], pred_fp, preview)
    print("\nPhase 1 complete (zero-shot). Do not interpret this as accuracy.")
    print(f"  GeoTIFF: {pred_fp.resolve()}")
    print(f"  Preview: {preview.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
