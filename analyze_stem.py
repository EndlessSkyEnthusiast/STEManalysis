#!/usr/bin/env python3
"""Systematic analysis pipeline for STEM cross-section TIFF images.

This script is intentionally written as a first, extensible version for one
specified test image while already providing the data structures needed for a
future batch workflow over all sample folders, detector modes, and zoom/HFW
settings.

Important scientific caveat
---------------------------
The sliding FFT analysis implemented here estimates lateral intensity/textural
periodicities in horizontal STEM image stripes. These distances are *not*
automatically equivalent to true grain sizes. True grain-size measurements should
later be validated with manual line-intercept analysis, Weka/ilastik-style
segmentation, or complementary diffraction methods. GIXRD Scherrer sizes and
STEM intercept sizes are different measurement quantities.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage, signal
from skimage import color, exposure, feature, filters

TIFF_EXTENSIONS = {".tif", ".tiff"}
LAYER_ORDER_BOTTOM_TO_TOP = ["Si", "TiN", "TiOxNx", "Au", "W"]
SCIENTIFIC_CAVEAT = (
    "Sliding FFT periods are lateral intensity/textural correlation distances, "
    "not automatically true grain sizes. Validate true grain sizes with manual "
    "line-intercept analysis, Weka/ilastik segmentation, or complementary "
    "methods. GIXRD Scherrer size and STEM intercept size are different metrics."
)


@dataclass
class ImageLoadResult:
    """Container for TIFF pixel data and extracted metadata."""

    gray: np.ndarray
    original: np.ndarray
    metadata: dict[str, Any]
    pixel_size_nm_from_metadata: float | None
    metadata_pixel_source: str | None


@dataclass
class DatabarCropResult:
    """Result of bottom databar detection and cropping."""

    cropped: np.ndarray
    crop: tuple[int, int, int, int]
    crop_y_bottom: int
    method: str
    confidence: float
    diagnostics: dict[str, Any]


@dataclass
class ValidRegionResult:
    """Result of valid detector/sample-region detection."""

    crop: tuple[int, int, int, int]
    candidate_edges: list[int]
    warnings: list[str]
    diagnostics: dict[str, Any]


@dataclass
class LayerResult:
    """Layer boundary candidates and optional assigned layer intervals."""

    boundaries_px: list[int]
    layers: dict[str, dict[str, int]]
    method: str
    warnings: list[str]
    diagnostics: dict[str, Any]


@dataclass
class FFTResult:
    """Sliding FFT result table and spectra for diagnostic plotting."""

    table: pd.DataFrame
    periods_nm: np.ndarray
    power_matrix: np.ndarray
    roi: tuple[int, int, int, int]
    reference_y_px: float | None
    warnings: list[str]


def safe_json_value(value: Any) -> Any:
    """Convert metadata values to JSON-serializable representations."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "preview": value.ravel()[:20].tolist()}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    return str(value)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with UTF-8 support for µ and × in paths/metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json_value(data), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_sample_folder_name(folder_name: str) -> dict[str, Any]:
    """Parse sample folder names such as ``#1_550C_50min``.

    Returns a dictionary with ``None`` values for fields that cannot be parsed;
    callers should warn but keep processing.
    """

    result: dict[str, Any] = {
        "sample_folder": folder_name,
        "sample_id": None,
        "sputter_temperature_C": None,
        "sputter_duration_min": None,
    }
    match = re.match(r"^#?(?P<id>\d+)[_\-](?P<temp>\d+(?:\.\d+)?)C[_\-](?P<dur>\d+(?:\.\d+)?)min$", folder_name)
    if match:
        result["sample_id"] = int(float(match.group("id")))
        result["sputter_temperature_C"] = float(match.group("temp"))
        result["sputter_duration_min"] = float(match.group("dur"))
    return result


def parse_stem_filename(filename: str) -> dict[str, Any]:
    """Parse common FEI/Thermo-style STEM filename tokens.

    The parser is deliberately permissive: all fields default to ``None`` and
    are filled when recognizable tokens appear. Unknown future tokens therefore
    do not break the pipeline.
    """

    stem = Path(filename).stem
    parts = stem.split("_")
    parsed: dict[str, Any] = {
        "filename": filename,
        "stem_basename": stem,
        "image_index": None,
        "date": None,
        "voltage_kV": None,
        "beam_current_nA": None,
        "detector_or_stem_channel": None,
        "mode": None,
        "dwell_time_us": None,
        "working_distance_mm": None,
        "hfw_um": None,
        "magnification": None,
        "pixel_size_nm_from_filename": None,
        "tilt_deg": None,
        "unparsed_tokens": [],
    }
    if parts and re.fullmatch(r"\d+", parts[0]):
        parsed["image_index"] = int(parts[0])
    for token in parts:
        normalized = token.replace("μ", "µ")
        if re.fullmatch(r"\d{8}", normalized):
            parsed["date"] = normalized
        elif m := re.fullmatch(r"(?P<v>\d+(?:\.\d+)?)kV", normalized, flags=re.IGNORECASE):
            parsed["voltage_kV"] = float(m.group("v"))
        elif m := re.fullmatch(r"(?P<i>\d+(?:\.\d+)?)nA", normalized, flags=re.IGNORECASE):
            parsed["beam_current_nA"] = float(m.group("i"))
        elif re.fullmatch(r"STEM[^_]*", normalized, flags=re.IGNORECASE):
            parsed["detector_or_stem_channel"] = normalized
        elif normalized.upper() in {"BF", "DF1", "DF2", "DF3", "DF4", "HAADF", "ADF", "SE", "BSE"}:
            parsed["mode"] = normalized.upper()
        elif m := re.fullmatch(r"(?P<d>\d+(?:\.\d+)?)µs", normalized, flags=re.IGNORECASE):
            parsed["dwell_time_us"] = float(m.group("d"))
        elif m := re.fullmatch(r"(?P<wd>\d+(?:\.\d+)?)mm", normalized, flags=re.IGNORECASE):
            parsed["working_distance_mm"] = float(m.group("wd"))
        elif m := re.fullmatch(r"(?P<hfw>\d+(?:\.\d+)?)µm", normalized, flags=re.IGNORECASE):
            parsed["hfw_um"] = float(m.group("hfw"))
        elif m := re.fullmatch(r"(?P<mag>\d+(?:\.\d+)?)[×xX]", normalized):
            parsed["magnification"] = int(float(m.group("mag")))
        elif m := re.fullmatch(r"(?P<px>\d+(?:\.\d+)?)nm", normalized, flags=re.IGNORECASE):
            parsed["pixel_size_nm_from_filename"] = float(m.group("px"))
        elif m := re.fullmatch(r"(?P<px>\d+(?:\.\d+)?)pm", normalized, flags=re.IGNORECASE):
            parsed["pixel_size_nm_from_filename"] = float(m.group("px")) / 1000.0
        elif m := re.fullmatch(r"(?P<t>[+\-]?\d+(?:\.\d+)?)deg", normalized, flags=re.IGNORECASE):
            parsed["tilt_deg"] = float(m.group("t"))
        elif token not in {parts[0]}:
            parsed["unparsed_tokens"].append(token)
    return parsed


def _extract_pixel_size_from_metadata(metadata: dict[str, Any]) -> tuple[float | None, str | None]:
    """Find PixelWidth/PixelHeight-like values in TIFF metadata and return nm/px."""

    candidates: list[tuple[str, Any]] = []

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if str(key).lower() in {"pixelwidth", "pixelheight", "pixelsize", "pixel_size"}:
                    candidates.append((path, val))
                walk(path, val)
        elif isinstance(obj, (list, tuple)) and len(obj) < 50:
            for idx, val in enumerate(obj):
                walk(f"{prefix}[{idx}]", val)

    walk("", metadata)
    for source, value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        # FEI metadata commonly stores meters per pixel, e.g. 1.378e-09 m.
        if numeric < 1e-3:
            return numeric * 1e9, source
        # Some metadata may already be nm/px.
        if numeric < 1e4:
            return numeric, source
    return None, None


def load_tiff_with_metadata(path: Path) -> ImageLoadResult:
    """Load TIFF image as float32 grayscale and collect tifffile metadata.

    RGB data are preserved in ``original`` for red-marker detection and converted
    to grayscale for analysis. Pixel size is extracted from metadata when a
    recognizable PixelWidth/PixelHeight field exists.
    """

    metadata: dict[str, Any] = {"path": str(path)}
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        page = tif.pages[0]
        metadata["series"] = [{"shape": list(s.shape), "dtype": str(s.dtype)} for s in tif.series]
        metadata["page_shape"] = list(page.shape)
        metadata["page_dtype"] = str(page.dtype)
        metadata["imagej_metadata"] = tif.imagej_metadata or {}
        metadata["ome_metadata"] = tif.ome_metadata
        metadata["fei_metadata"] = getattr(tif, "fei_metadata", None)
        metadata["sem_metadata"] = getattr(tif, "sem_metadata", None)
        metadata["tags"] = {tag.name: tag.value for tag in page.tags.values() if tag.name not in {"StripOffsets", "StripByteCounts"}}

    original = np.asarray(arr)
    if original.ndim == 3 and original.shape[-1] in (3, 4):
        rgb = original[..., :3]
        gray = color.rgb2gray(rgb).astype(np.float32)
    elif original.ndim == 3:
        gray = np.mean(original, axis=0).astype(np.float32)
    else:
        gray = original.astype(np.float32)
    gray = np.nan_to_num(gray)
    pixel_size_nm, source = _extract_pixel_size_from_metadata(metadata)
    return ImageLoadResult(gray=gray, original=original, metadata=metadata, pixel_size_nm_from_metadata=pixel_size_nm, metadata_pixel_source=source)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Robustly scale image to 8-bit for PNG diagnostics."""

    lo, hi = np.percentile(image, [0.5, 99.5]) if image.size else (0, 1)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image) + 1e-6)
    scaled = exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0))
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def detect_red_marker_image(image: np.ndarray, threshold: float = 0.001) -> dict[str, Any]:
    """Detect RGB red annotation/marker pixels that can corrupt analysis."""

    if image.ndim != 3 or image.shape[-1] < 3:
        return {"has_red_marker": False, "red_pixel_fraction": 0.0, "threshold": threshold, "method": "grayscale_no_rgb"}
    rgb = image[..., :3].astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_mask = (r > 120) & (r > 1.5 * (g + 1e-6)) & (r > 1.5 * (b + 1e-6))
    fraction = float(np.mean(red_mask))
    return {"has_red_marker": fraction > threshold, "red_pixel_fraction": fraction, "threshold": threshold, "method": "rgb_threshold"}


def _target_databar_color_mask(rgb: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
    """Return pixels matching the known STEM databar palette."""

    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    rgb = rgb[..., :3].astype(np.float32)
    palette = np.asarray(
        [
            [0x2E, 0x2E, 0x2E],
            [0x8C, 0xC9, 0xE7],
            [0xFF, 0xFF, 0xFF],
        ],
        dtype=np.float32,
    )
    distances = np.max(np.abs(rgb[:, :, None, :] - palette[None, None, :, :]), axis=-1)
    return np.min(distances, axis=-1) <= tolerance


def detect_and_crop_databar(
    image: np.ndarray,
    metadata: dict[str, Any],
    crop_bottom: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    color_image: np.ndarray | None = None,
    databar_palette_fraction: float = 0.95,
) -> DatabarCropResult:
    """Detect and remove the bottom acquisition databar/infobox.

    Manual ``--crop`` has highest priority, followed by ``--crop-bottom``. The
    automatic detector first searches for the uppermost row whose pixels are at
    least 95% drawn from the known databar palette (#2e2e2e, #8cc9e7, #ffffff).
    This directly identifies the top edge of the databar instead of guessing from
    generic bottom-image statistics. If no RGB palette row is available, the
    older row-statistic detector remains as a conservative fallback.
    """

    height, width = image.shape[:2]
    if crop is not None:
        x0, y0, x1, y1 = crop
        return DatabarCropResult(image[y0:y1, x0:x1], crop, y1, "manual_crop", 1.0, {})
    if crop_bottom is not None:
        y = int(np.clip(crop_bottom, 1, height))
        return DatabarCropResult(image[:y, :], (0, 0, width, y), y, "manual_crop_bottom", 1.0, {})

    if color_image is not None and color_image.ndim == 3 and color_image.shape[-1] >= 3:
        rgb = color_image[..., :3]
        if rgb.shape[:2] == image.shape[:2]:
            palette_mask = _target_databar_color_mask(rgb)
            row_fraction = np.mean(palette_mask, axis=1)
            candidates = np.flatnonzero(row_fraction >= databar_palette_fraction)
            if candidates.size:
                candidate = int(candidates[0])
                confidence = float(row_fraction[candidate])
                diagnostics = {
                    "target_palette_hex": ["#2e2e2e", "#8cc9e7", "#ffffff"],
                    "row_palette_fraction": row_fraction.tolist(),
                    "threshold_fraction": databar_palette_fraction,
                    "matched_rows": int(candidates.size),
                }
                return DatabarCropResult(
                    image[:candidate, :],
                    (0, 0, width, candidate),
                    candidate,
                    "auto_databar_palette_top_row",
                    confidence,
                    diagnostics,
                )

    # Metadata fallback: if ImageLength/Page shape suggests a smaller true data
    # region than the loaded raster, use it. This is rare but robust when present.
    meta_height = None
    for key in ("ImageLength", "ImageHeight"):
        value = metadata.get("tags", {}).get(key)
        if isinstance(value, (int, float)) and 0 < value < height:
            meta_height = int(value)
            break
    if meta_height is not None:
        return DatabarCropResult(image[:meta_height, :], (0, 0, width, meta_height), meta_height, "metadata_true_height", 0.95, {})

    row_mean = np.mean(image, axis=1)
    row_std = np.std(image, axis=1)
    smooth_mean = ndimage.gaussian_filter1d(row_mean, sigma=max(2, height / 400))
    smooth_std = ndimage.gaussian_filter1d(row_std, sigma=max(2, height / 400))
    z_mean = np.abs(np.gradient(smooth_mean)) / (np.std(np.gradient(smooth_mean)) + 1e-9)
    z_std = np.abs(np.gradient(smooth_std)) / (np.std(np.gradient(smooth_std)) + 1e-9)
    score = z_mean + 0.7 * z_std
    search_start = int(height * 0.55)
    search_end = int(height * 0.98)
    candidate = int(search_start + np.argmax(score[search_start:search_end])) if search_end > search_start else height
    confidence = float(np.clip(score[candidate] / 12.0, 0, 1))
    if confidence < 0.35 or candidate > height * 0.92:
        candidate = height
        method = "auto_none_low_confidence"
    else:
        method = "auto_row_stat_change_fallback"
    diagnostics = {"row_mean": row_mean.tolist(), "row_std": row_std.tolist(), "score_max": float(score[candidate - 1 if candidate == height else candidate])}
    return DatabarCropResult(image[:candidate, :], (0, 0, width, candidate), candidate, method, confidence, diagnostics)


def detect_valid_image_region(image: np.ndarray, apply_crop: bool = False) -> ValidRegionResult:
    """Diagnose likely vertical detector/sample-edge artifacts.

    The returned crop is conservative. Unless ``apply_crop`` is requested by a
    future caller, the full image region is kept and candidate edge positions are
    only reported/visualized.
    """

    h, w = image.shape[:2]
    col_mean = np.mean(image, axis=0)
    col_var = np.var(image, axis=0)
    smooth = ndimage.gaussian_filter1d(col_mean, sigma=max(2, w / 500))
    jump = np.abs(np.gradient(smooth))
    vertical_edges = np.mean(np.abs(filters.sobel_v(image)), axis=0)
    score = (jump / (np.std(jump) + 1e-9)) + (vertical_edges / (np.std(vertical_edges) + 1e-9))
    threshold = np.percentile(score, 99.0)
    raw_candidates = np.flatnonzero(score >= threshold)
    candidates: list[int] = []
    for group in np.split(raw_candidates, np.where(np.diff(raw_candidates) > 3)[0] + 1):
        if group.size:
            candidates.append(int(np.round(np.mean(group))))
    warnings_list: list[str] = []
    x0, x1 = 0, w
    margin = max(5, int(0.03 * w))
    left_edges = [x for x in candidates if x < 0.25 * w]
    right_edges = [x for x in candidates if x > 0.75 * w]
    if apply_crop and left_edges:
        x0 = max(left_edges) + margin
    if apply_crop and right_edges:
        x1 = min(right_edges) - margin
    if candidates:
        warnings_list.append("Possible vertical detector/sample-edge artifacts detected; not cropped unless explicitly enabled.")
    return ValidRegionResult((x0, 0, x1, h), candidates, warnings_list, {"col_mean": col_mean.tolist(), "col_var": col_var.tolist(), "edge_score": score.tolist()})


def _find_nonblack_content_start(image: np.ndarray) -> int:
    """Find the first row containing real sample signal above a top black void."""

    h = image.shape[0]
    if h == 0:
        return 0
    row_median = np.median(image, axis=1)
    row_std = np.std(image, axis=1)
    dynamic = float(np.percentile(image, 99.0) - np.percentile(image, 1.0))
    black_level = float(np.percentile(image, 1.0) + max(dynamic * 0.03, 1e-6))
    texture_level = float(max(np.percentile(row_std, 75.0) * 0.05, 1e-6))
    void_rows = (row_median <= black_level) & (row_std <= texture_level)
    first_content = 0
    for y in range(h):
        if not void_rows[y]:
            first_content = y
            break
    else:
        return 0
    # Require that the void touches the top and is not just a dark line.
    if first_content >= max(5, int(0.01 * h)) and np.all(void_rows[:first_content]):
        return int(first_content)
    return 0


def estimate_layer_boundaries(image: np.ndarray, layer_json: Path | None = None) -> LayerResult:
    """Estimate coarse horizontal layer boundaries or load manual JSON layers.

    The automatic path uses a standard Canny/Sobel-style edge workflow: ignore a
    continuous black top void, detect edges, collapse horizontal edge evidence by
    image row, and pick prominent row peaks as layer-interface candidates.
    """

    h, _w = image.shape[:2]
    warnings_list: list[str] = []
    if layer_json and layer_json.exists():
        data = json.loads(layer_json.read_text(encoding="utf-8"))
        layers = {name: {"y_min": int(v["y_min"]), "y_max": int(v["y_max"])} for name, v in data.get("layers", {}).items()}
        boundaries = sorted({v["y_min"] for v in layers.values()} | {v["y_max"] for v in layers.values() if v["y_max"] < h})
        return LayerResult(boundaries, layers, "manual_json", warnings_list, {"source": str(layer_json)})

    if h < 3 or image.size == 0:
        warnings_list.append("Automatic layer detection is uncertain because the image is too small.")
        return LayerResult([], {}, "auto_canny_sobel_edge", warnings_list, {})

    content_start = _find_nonblack_content_start(image)
    work = image[content_start:, :]
    if content_start:
        warnings_list.append(f"Ignored top black void from y=0 to y={content_start} px for layer detection.")
    if work.shape[0] < 3:
        warnings_list.append("Automatic layer detection is uncertain after removing the top black void.")
        return LayerResult([], {}, "auto_canny_sobel_edge", warnings_list, {"content_start_px": content_start})

    row_mean = np.mean(image, axis=1)
    row_median = np.median(image, axis=1)
    row_std = np.std(image, axis=1)
    work_norm = exposure.rescale_intensity(work.astype(np.float32), in_range="image", out_range=(0.0, 1.0))
    work_smooth = filters.gaussian(work_norm, sigma=1.0, preserve_range=True)
    canny_edges = feature.canny(work_smooth, sigma=1.5)
    horizontal_sobel = np.abs(filters.sobel_h(work_smooth))
    edge_density = np.mean(canny_edges, axis=1)
    sobel_score = np.mean(horizontal_sobel, axis=1)
    combined = edge_density / (np.std(edge_density) + 1e-9) + sobel_score / (np.std(sobel_score) + 1e-9)
    combined = ndimage.gaussian_filter1d(combined, sigma=max(1.0, work.shape[0] / 300))

    distance = max(8, work.shape[0] // 30)
    prominence = max(float(np.std(combined) * 0.8), 1e-9)
    peaks, props = signal.find_peaks(combined, distance=distance, prominence=prominence)
    ranked = sorted(peaks, key=lambda y: combined[y], reverse=True)[:4]
    boundaries = sorted(int(y + content_start) for y in ranked if 0 < y + content_start < h)

    if len(boundaries) < 2:
        warnings_list.append("Automatic edge-based layer detection is uncertain; use --layer-json for quantitative thicknesses.")

    intervals = [content_start] + boundaries + [h]
    interval_pairs = [(y0, y1) for y0, y1 in zip(intervals[:-1], intervals[1:]) if y1 > y0]
    top_to_bottom_labels = list(reversed(LAYER_ORDER_BOTTOM_TO_TOP))
    labels = top_to_bottom_labels[-len(interval_pairs) :] if len(interval_pairs) > len(top_to_bottom_labels) else top_to_bottom_labels[: len(interval_pairs)]
    layers: dict[str, dict[str, int]] = {}
    for label, (y0, y1) in zip(labels, interval_pairs):
        layers[label] = {"y_min": int(y0), "y_max": int(y1)}

    diagnostics = {
        "content_start_px": content_start,
        "row_mean": row_mean.tolist(),
        "row_median": row_median.tolist(),
        "row_std": row_std.tolist(),
        "canny_edge_density": np.pad(edge_density, (content_start, 0), constant_values=np.nan).tolist(),
        "horizontal_sobel_score": np.pad(sobel_score, (content_start, 0), constant_values=np.nan).tolist(),
        "edge_score": np.pad(combined, (content_start, 0), constant_values=np.nan).tolist(),
        "peak_prominences": props.get("prominences", np.array([])).tolist(),
    }
    return LayerResult(boundaries, layers, "auto_canny_sobel_edge", warnings_list, diagnostics)


def write_layer_template(path: Path, image_height: int, boundaries: Iterable[int]) -> None:
    """Write an editable layer JSON template with current candidate boundaries."""

    b = sorted({0, image_height, *[int(x) for x in boundaries if 0 < int(x) < image_height]})
    pairs = list(zip(b[:-1], b[1:]))
    labels = LAYER_ORDER_BOTTOM_TO_TOP[-len(pairs) :]
    layers = {label: {"y_min": int(y0), "y_max": int(y1)} for label, (y0, y1) in zip(reversed(labels), pairs)}
    template = {"comment": "Edit y_min/y_max values and rerun with --layer-json. Coordinates refer to databar-cropped image pixels.", "layers": layers}
    write_json(path, template)


def compute_layer_summary(layers: dict[str, dict[str, int]], sample: str, filename: str, pixel_size_nm: float | None, method: str) -> pd.DataFrame:
    """Create layer thickness summary table."""

    rows = []
    for layer, bounds in layers.items():
        y_min, y_max = int(bounds["y_min"]), int(bounds["y_max"])
        thickness_px = max(0, y_max - y_min)
        rows.append({
            "sample": sample,
            "file": filename,
            "layer": layer,
            "y_min_px": y_min,
            "y_max_px": y_max,
            "thickness_px": thickness_px,
            "thickness_nm": thickness_px * pixel_size_nm if pixel_size_nm else None,
            "pixel_size_nm": pixel_size_nm,
            "method": method,
        })
    return pd.DataFrame(rows)


def choose_fft_roi(image: np.ndarray, layers: dict[str, dict[str, int]], cli_roi: tuple[int, int, int, int] | None) -> tuple[tuple[int, int, int, int], list[str]]:
    """Choose FFT ROI from CLI, TiN layer, or conservative central fallback."""

    h, w = image.shape[:2]
    warnings_list: list[str] = []
    if cli_roi is not None:
        x0, y0, x1, y1 = cli_roi
        return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), warnings_list
    if "TiN" in layers:
        y0, y1 = layers["TiN"]["y_min"], layers["TiN"]["y_max"]
        return (0, max(0, y0), w, min(h, y1)), warnings_list
    warnings_list.append("No TiN layer found; using central 40% image height as fallback FFT ROI. Prefer --fft-roi or --layer-json.")
    return (0, int(0.35 * h), w, int(0.75 * h)), warnings_list


def compute_sliding_fft_grain_metrics(
    image: np.ndarray,
    layer_roi: tuple[int, int, int, int],
    pixel_size_nm: float,
    stripe_height_px: int = 4,
    stripe_step_px: int = 1,
    min_period_nm: float = 5.0,
    max_period_nm: float = 500.0,
    detrend_profile: bool = True,
    window: str = "hann",
    reference_y_px: float | None = None,
) -> FFTResult:
    """Compute sliding-stripe lateral FFT metrics.

    For each horizontal stripe, row intensities are averaged into I(x), detrended,
    windowed, transformed by FFT, and searched for dominant periods in the
    requested range. The dominant period is a textural/correlation distance, not
    automatically a grain size.
    """

    x0, y0, x1, y1 = layer_roi
    roi_img = image[y0:y1, x0:x1]
    warnings_list: list[str] = []
    if roi_img.size == 0 or roi_img.shape[1] < 8:
        return FFTResult(pd.DataFrame(), np.array([]), np.empty((0, 0)), layer_roi, reference_y_px, ["FFT ROI is empty or too narrow."])
    if stripe_height_px > roi_img.shape[0]:
        stripe_height_px = max(2, roi_img.shape[0])
        warnings_list.append("Stripe height reduced because ROI is shallow.")
    stripe_step_px = max(1, int(stripe_step_px))
    win = signal.get_window(window, roi_img.shape[1]) if window else np.ones(roi_img.shape[1])
    freqs = np.fft.rfftfreq(roi_img.shape[1], d=pixel_size_nm)
    valid = freqs > 0
    periods = np.zeros_like(freqs)
    periods[valid] = 1.0 / freqs[valid]
    period_mask = valid & (periods >= min_period_nm) & (periods <= max_period_nm)
    rows: list[dict[str, Any]] = []
    spectra: list[np.ndarray] = []
    period_axis = periods[period_mask]
    for local_y in range(0, max(1, roi_img.shape[0] - stripe_height_px + 1), stripe_step_px):
        stripe = roi_img[local_y : local_y + stripe_height_px, :]
        profile = np.mean(stripe, axis=0).astype(np.float64)
        if detrend_profile:
            profile = signal.detrend(profile, type="linear")
        profile = profile - np.mean(profile)
        fft = np.fft.rfft(profile * win)
        power = np.abs(fft) ** 2
        band_power = power[period_mask]
        y_center = y0 + local_y + stripe.shape[0] / 2.0
        if band_power.size == 0 or np.all(band_power <= 0):
            dominant_period = np.nan
            peak_power = np.nan
            centroid = np.nan
            top_periods: list[float] = []
        else:
            peak_idx = int(np.argmax(band_power))
            dominant_period = float(period_axis[peak_idx])
            peak_power = float(band_power[peak_idx])
            centroid = float(np.sum(period_axis * band_power) / np.sum(band_power))
            peak_indices, _ = signal.find_peaks(band_power, prominence=np.max(band_power) * 0.05)
            if peak_indices.size:
                top = peak_indices[np.argsort(band_power[peak_indices])[-3:]][::-1]
                top_periods = [float(period_axis[i]) for i in top]
            else:
                top_periods = [dominant_period]
        spectra.append(band_power)
        rows.append({
            "y_center_px": y_center,
            "y_center_nm_relative_to_reference": (reference_y_px - y_center) * pixel_size_nm if reference_y_px is not None else np.nan,
            "dominant_period_nm": dominant_period,
            "peak_power": peak_power,
            "spectral_centroid_nm": centroid,
            "top_periods_nm_json": json.dumps(top_periods),
            "stripe_height_px": stripe_height_px,
            "stripe_step_px": stripe_step_px,
            "fft_roi": str(layer_roi),
            "scientific_caveat": SCIENTIFIC_CAVEAT,
        })
    return FFTResult(pd.DataFrame(rows), period_axis, np.vstack(spectra) if spectra else np.empty((0, period_axis.size)), layer_roi, reference_y_px, warnings_list)


def save_diagnostic_plots(
    out_dir: Path,
    original_gray: np.ndarray,
    cropped: np.ndarray,
    databar: DatabarCropResult,
    valid: ValidRegionResult,
    layers: LayerResult,
    fft: FFTResult,
    parsed: dict[str, Any],
    pixel_size_nm: float | None,
) -> None:
    """Save all diagnostic plots and previews for one analyzed image."""

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalize_for_display(original_gray)).save(out_dir / "original_preview.png")
    Image.fromarray(normalize_for_display(cropped)).save(out_dir / "cropped_image.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(original_gray, cmap="gray")
    ax.axhline(databar.crop_y_bottom, color="red", lw=2, label=f"crop y={databar.crop_y_bottom} ({databar.method})")
    ax.set_title("Databar crop diagnostic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "crop_databar_diagnostic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cropped, cmap="gray")
    for x in valid.candidate_edges:
        ax.axvline(x, color="orange", ls="--", lw=1)
    x0, y0, x1, y1 = valid.crop
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", lw=2))
    ax.set_title("Valid image region diagnostic")
    fig.tight_layout()
    fig.savefig(out_dir / "valid_region_diagnostic.png", dpi=180)
    plt.close(fig)

    row_median = np.asarray(layers.diagnostics.get("row_median", np.median(cropped, axis=1)))
    gradient = np.asarray(layers.diagnostics.get("gradient", np.gradient(row_median)))
    fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax_img.imshow(cropped, cmap="gray")
    for y in layers.boundaries_px:
        ax_img.axhline(y, color="cyan", ls="--" if layers.method == "auto_candidate" else "-", lw=1.5)
    for layer, bounds in layers.layers.items():
        yc = 0.5 * (bounds["y_min"] + bounds["y_max"])
        ax_img.text(5, yc, layer, color="yellow", va="center", fontsize=9, bbox={"facecolor": "black", "alpha": 0.4, "pad": 1})
    ax_img.set_title(f"Layer candidates ({layers.method})")
    y_axis = np.arange(len(row_median))
    ax_prof.plot(row_median, y_axis, label="row median")
    ax_prof.plot(gradient * (np.std(row_median) / (np.std(gradient) + 1e-9)) + np.mean(row_median), y_axis, label="scaled gradient")
    ax_prof.invert_yaxis()
    ax_prof.legend()
    ax_prof.set_xlabel("Intensity / scaled gradient")
    fig.tight_layout()
    fig.savefig(out_dir / "layer_candidates.png", dpi=180)
    plt.close(fig)

    if not fft.table.empty and fft.power_matrix.size:
        log_power = np.log10(fft.power_matrix + 1e-12)
        yvals = fft.table["y_center_nm_relative_to_reference"].to_numpy()
        if np.all(np.isnan(yvals)):
            yvals = fft.table["y_center_px"].to_numpy()
            y_label = "y center (px)"
        else:
            y_label = "distance from reference interface (nm)"
        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(fft.periods_nm, yvals, log_power, shading="auto", cmap="magma")
        ax.set_xscale("log")
        ax.set_xlabel("Lateral period / textural spacing (nm)")
        ax.set_ylabel(y_label)
        ax.set_title("Sliding FFT period heatmap")
        fig.colorbar(mesh, ax=ax, label="log10(power)")
        fig.tight_layout()
        fig.savefig(out_dir / "fft_depth_period_heatmap.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fft.table["dominant_period_nm"], yvals, "o-", ms=3)
        ax.set_xscale("log")
        ax.set_xlabel("Dominant lateral period (nm)")
        ax.set_ylabel(y_label)
        ax.set_title("Dominant FFT period vs depth")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fft_dominant_period_vs_depth.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cropped, cmap="gray")
        x0, y0, x1, y1 = fft.roi
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2, label="FFT ROI"))
        for y in fft.table["y_center_px"]:
            ax.axhline(y, color="red", alpha=0.15, lw=0.5)
        info = f"mode={parsed.get('mode')} HFW={parsed.get('hfw_um')} µm mag={parsed.get('magnification')} px={pixel_size_nm:.4g} nm" if pixel_size_nm else "pixel size unknown"
        ax.set_title("FFT stripe overlay\n" + info)
        fig.tight_layout()
        fig.savefig(out_dir / "fft_roi_overlay.png", dpi=180)
        plt.close(fig)


def write_results_csv(out_dir: Path, layer_summary: pd.DataFrame, fft_result: FFTResult) -> None:
    """Write per-image CSV outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    layer_summary.to_csv(out_dir / "layer_summary.csv", index=False)
    fft_result.table.to_csv(out_dir / "fft_stripe_results.csv", index=False)


def parse_crop_arg(value: str | None) -> tuple[int, int, int, int] | None:
    """Parse x0,y0,x1,y1 CLI crop tuple."""

    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be x0,y0,x1,y1")
    return tuple(parts)  # type: ignore[return-value]


def build_images_index(root: Path) -> pd.DataFrame:
    """Index TIFF files below root for future sample/mode/HFW grouping."""

    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TIFF_EXTENSIONS:
            continue
        sample_folder = path.parent.name
        sample_info = parse_sample_folder_name(sample_folder)
        parsed = parse_stem_filename(path.name)
        rows.append({
            "full_path": str(path),
            "sample_folder": sample_folder,
            "sample_id": sample_info.get("sample_id"),
            "sputter_temperature_C": sample_info.get("sputter_temperature_C"),
            "sputter_duration_min": sample_info.get("sputter_duration_min"),
            "filename": path.name,
            "mode": parsed.get("mode"),
            "detector": parsed.get("detector_or_stem_channel"),
            "hfw_um": parsed.get("hfw_um"),
            "magnification": parsed.get("magnification"),
            "pixel_size_nm": parsed.get("pixel_size_nm_from_filename"),
            "red_marker_fraction": None,
            "has_red_marker": None,
            "analysis_status": "indexed_not_analyzed",
        })
    return pd.DataFrame(rows)


def find_images_to_analyze(index: pd.DataFrame, args: argparse.Namespace) -> list[Path]:
    """Select one requested test image or filtered batch images."""

    if index.empty:
        return []
    df = index.copy()
    if args.mode_filter:
        df = df[df["mode"].astype(str).str.upper() == args.mode_filter.upper()]
    if args.hfw_filter is not None:
        df = df[np.isclose(pd.to_numeric(df["hfw_um"], errors="coerce"), args.hfw_filter)]
    if not args.analyze_all:
        target = args.root / args.sample / args.file
        return [target]
    return [Path(p) for p in df["full_path"].tolist()]


def analyze_one_image(path: Path, args: argparse.Namespace, index_row_update: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the complete analysis pipeline for one TIFF image."""

    warnings_list: list[str] = []
    sample_info = parse_sample_folder_name(path.parent.name)
    parsed = parse_stem_filename(path.name)
    out_dir = args.out / path.parent.name / parsed["stem_basename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    load = load_tiff_with_metadata(path)
    write_json(out_dir / "metadata.json", load.metadata)
    write_json(out_dir / "parsed_filename.json", parsed)

    red = detect_red_marker_image(load.original)
    if red["has_red_marker"]:
        msg = f"Red markers detected (fraction={red['red_pixel_fraction']:.6g})."
        warnings_list.append(msg)
        warnings.warn(msg)
        if args.skip_red_marker:
            summary = {"file": str(path), "analysis_status": "skipped_red_marker", "red_marker": red, "warnings": warnings_list}
            write_json(out_dir / "analysis_summary.json", summary)
            return summary

    databar = detect_and_crop_databar(load.gray, load.metadata, args.crop_bottom, args.crop, color_image=load.original)
    valid = detect_valid_image_region(databar.cropped, apply_crop=False)
    warnings_list.extend(valid.warnings)

    pixel_size_nm = load.pixel_size_nm_from_metadata or parsed.get("pixel_size_nm_from_filename")
    pixel_source = load.metadata_pixel_source if load.pixel_size_nm_from_metadata else "filename"
    if pixel_size_nm is None:
        pixel_size_nm = 1.0
        pixel_source = "fallback_1_nm_per_px"
        warnings_list.append("Pixel size unavailable; using fallback 1.0 nm/px for plots and CSVs.")

    layer_json = Path(args.layer_json) if args.layer_json else None
    layers = estimate_layer_boundaries(databar.cropped, layer_json)
    warnings_list.extend(layers.warnings)
    if args.write_layer_template:
        write_layer_template(out_dir / "layer_template.json", databar.cropped.shape[0], layers.boundaries_px)

    layer_summary = compute_layer_summary(layers.layers, path.parent.name, path.name, pixel_size_nm, layers.method)

    fft_roi, roi_warnings = choose_fft_roi(databar.cropped, layers.layers, args.fft_roi)
    warnings_list.extend(roi_warnings)
    reference_y = layers.layers.get("Si", {}).get("y_min")
    if reference_y is None and "TiN" in layers.layers:
        reference_y = layers.layers["TiN"].get("y_max")
    fft = compute_sliding_fft_grain_metrics(
        databar.cropped,
        fft_roi,
        pixel_size_nm,
        stripe_height_px=args.stripe_height,
        stripe_step_px=args.stripe_step,
        min_period_nm=args.min_period_nm,
        max_period_nm=args.max_period_nm,
        reference_y_px=float(reference_y) if reference_y is not None else None,
    )
    warnings_list.extend(fft.warnings)

    save_diagnostic_plots(out_dir, load.gray, databar.cropped, databar, valid, layers, fft, parsed, pixel_size_nm)
    write_results_csv(out_dir, layer_summary, fft)

    summary = {
        "analysis_status": "ok",
        "scientific_caveat": SCIENTIFIC_CAVEAT,
        "file": str(path),
        "sample": sample_info,
        "parsed_filename": parsed,
        "pixel_size_nm": pixel_size_nm,
        "pixel_size_source": pixel_source,
        "original_shape": list(load.gray.shape),
        "cropped_shape": list(databar.cropped.shape),
        "red_marker": red,
        "databar_crop": asdict(databar) | {"cropped": f"array shape {databar.cropped.shape}"},
        "valid_region": asdict(valid),
        "layer_boundaries_px": layers.boundaries_px,
        "layer_method": layers.method,
        "fft_roi": fft_roi,
        "fft_stripes": int(len(fft.table)),
        "warnings": warnings_list,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "analysis_summary.json", summary)

    if index_row_update is not None:
        index_row_update.update({
            "red_marker_fraction": red["red_pixel_fraction"],
            "has_red_marker": red["has_red_marker"],
            "analysis_status": summary["analysis_status"],
            "pixel_size_nm": pixel_size_nm,
        })

    print("\nAnalysis summary")
    print("----------------")
    print(f"Analyzed file: {path}")
    print(f"Pixel size: {pixel_size_nm:.6g} nm/px ({pixel_source})")
    print(f"Original/cropped shape: {load.gray.shape} -> {databar.cropped.shape}")
    print(f"Mode: {parsed.get('mode')} | detector: {parsed.get('detector_or_stem_channel')}")
    print(f"HFW: {parsed.get('hfw_um')} µm | magnification: {parsed.get('magnification')}")
    print(f"Red marker: {red['has_red_marker']} (fraction={red['red_pixel_fraction']:.6g})")
    print(f"Databar crop: y={databar.crop_y_bottom}, method={databar.method}, confidence={databar.confidence:.2f}")
    print(f"Layer candidates: {layers.boundaries_px} ({layers.method})")
    print(f"FFT ROI: {fft_roi}")
    print(f"FFT stripes analyzed: {len(fft.table)}")
    print(f"Results: {out_dir}")
    return summary


def make_arg_parser() -> argparse.ArgumentParser:
    """Create command-line interface for the STEM analysis pipeline."""

    parser = argparse.ArgumentParser(description="Analyze STEM cross-section TIFF images with diagnostics and sliding FFT metrics.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root folder containing sample subfolders.")
    parser.add_argument("--sample", default="#1_550C_50min", help="Sample folder for single-image mode.")
    parser.add_argument("--file", default="1_20260306_14_30.00kV_0.10nA_STEM3+_DF3_30.00µs_5.1mm_1.06µm_120000×_689pm_38.0deg.tif", help="Filename for single-image mode.")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output results directory.")
    parser.add_argument("--analyze-all", action="store_true", help="Analyze all indexed TIFFs after optional filters.")
    parser.add_argument("--skip-red-marker", action="store_true", help="Skip RGB images with red annotation markers.")
    parser.add_argument("--crop-bottom", type=int, default=None, help="Manual bottom crop y pixel, e.g. 1024.")
    parser.add_argument("--crop", type=parse_crop_arg, default=None, help="Manual crop rectangle x0,y0,x1,y1.")
    parser.add_argument("--layer-json", type=Path, default=None, help="Manual editable layer JSON path.")
    parser.add_argument("--write-layer-template", action="store_true", help="Write editable layer_template.json in the image result folder.")
    parser.add_argument("--fft-roi", type=parse_crop_arg, default=None, help="FFT ROI x0,y0,x1,y1 in databar-cropped image coordinates.")
    parser.add_argument("--stripe-height", type=int, default=4, help="Sliding FFT stripe height in pixels.")
    parser.add_argument("--stripe-step", type=int, default=1, help="Sliding FFT stripe step in pixels; default 1 gives a wandering 4-row field (1-4, 2-5, 3-6, ...).")
    parser.add_argument("--min-period-nm", type=float, default=5.0, help="Minimum FFT period to evaluate in nm.")
    parser.add_argument("--max-period-nm", type=float, default=500.0, help="Maximum FFT period to evaluate in nm.")
    parser.add_argument("--mode-filter", default=None, help="Batch mode filter, e.g. BF.")
    parser.add_argument("--hfw-filter", type=float, default=None, help="Batch HFW filter in µm, e.g. 2.12.")
    parser.add_argument("--debug", action="store_true", help="Print additional debug information.")
    return parser


def main() -> int:
    """Entry point: build index, analyze requested image(s), and write outputs."""

    args = make_arg_parser().parse_args()
    args.root = args.root.resolve()
    args.out = args.out.resolve()
    index = build_images_index(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "images_index.csv"
    index.to_csv(index_path, index=False)
    if args.debug:
        print(f"Indexed {len(index)} TIFF file(s); wrote {index_path}")

    paths = find_images_to_analyze(index, args)
    if not paths:
        print("No TIFF images found for analysis.")
        return 1

    failures = 0
    summaries = []
    for path in paths:
        if not path.exists():
            print(f"WARNING: requested image does not exist: {path}")
            failures += 1
            continue
        try:
            summaries.append(analyze_one_image(path, args))
        except Exception as exc:  # Keep batch runs alive; record explicit failure.
            failures += 1
            fail_dir = args.out / path.parent.name / path.stem
            fail_dir.mkdir(parents=True, exist_ok=True)
            summary = {"analysis_status": "failed", "file": str(path), "error": repr(exc), "scientific_caveat": SCIENTIFIC_CAVEAT}
            write_json(fail_dir / "analysis_summary.json", summary)
            print(f"ERROR analyzing {path}: {exc}")
            if args.debug:
                raise
    # Update index with analyzed status where possible.
    if summaries:
        summary_by_path = {s.get("file"): s for s in summaries}
        for idx, row in index.iterrows():
            summary = summary_by_path.get(row.get("full_path"))
            if summary:
                index.at[idx, "analysis_status"] = summary.get("analysis_status")
                red_marker = summary.get("red_marker", {})
                index.at[idx, "red_marker_fraction"] = red_marker.get("red_pixel_fraction")
                index.at[idx, "has_red_marker"] = red_marker.get("has_red_marker")
                index.at[idx, "pixel_size_nm"] = summary.get("pixel_size_nm")
        index.to_csv(index_path, index=False)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
