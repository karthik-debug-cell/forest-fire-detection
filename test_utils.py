"""
tests/test_utils.py
===================
Unit tests for the Forest Fire Detection System utilities.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=utils --cov-report=term-missing
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    compute_severity,
    estimate_burn_area,
    file_md5,
    generate_heatmap,
    iou,
    is_image_file,
    pixel_to_latlon,
    soft_nms,
    timestamp_str,
    xyxy_to_xywh,
    xywh_to_xyxy,
)


# ─────────────────────────────────────────────────────────────
# Bounding box utilities
# ─────────────────────────────────────────────────────────────

def test_xyxy_to_xywh_basic():
    box = [10, 20, 50, 60]
    cx, cy, w, h = xyxy_to_xywh(box)
    assert cx == 30.0
    assert cy == 40.0
    assert w  == 40.0
    assert h  == 40.0


def test_xywh_to_xyxy_roundtrip():
    original = [10.0, 20.0, 50.0, 60.0]
    converted = xywh_to_xyxy(xyxy_to_xywh(original))
    for a, b in zip(original, converted):
        assert abs(a - b) < 1e-6


def test_iou_identical_boxes():
    box = [0.0, 0.0, 10.0, 10.0]
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping():
    boxA = [0.0, 0.0, 5.0, 5.0]
    boxB = [10.0, 10.0, 20.0, 20.0]
    assert iou(boxA, boxB) == 0.0


def test_iou_partial_overlap():
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [5.0, 0.0, 15.0, 10.0]
    # Intersection = 5×10 = 50; Union = 100+100-50 = 150
    assert iou(boxA, boxB) == pytest.approx(50 / 150, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# Soft-NMS
# ─────────────────────────────────────────────────────────────

def make_det(box, score, cls=0):
    return {"bbox": box, "score": score, "class": cls}


def test_soft_nms_empty():
    assert soft_nms([]) == []


def test_soft_nms_single():
    dets = [make_det([0, 0, 10, 10], 0.9)]
    result = soft_nms(dets)
    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(0.9)


def test_soft_nms_suppresses_overlapping():
    dets = [
        make_det([0, 0, 10, 10], 0.9),
        make_det([1, 1,  9,  9], 0.85),   # Heavily overlapping
        make_det([50, 50, 60, 60], 0.7),   # Separate
    ]
    result = soft_nms(dets, score_threshold=0.5)
    # The heavily overlapping box should have reduced score or be dropped
    scores = {tuple(d["bbox"]): d["score"] for d in result}
    # Separate box should survive
    assert any(d["bbox"] == [50, 50, 60, 60] for d in result)
    # Best box should survive with original score
    assert any(abs(d["score"] - 0.9) < 0.01 for d in result)


# ─────────────────────────────────────────────────────────────
# Severity scoring
# ─────────────────────────────────────────────────────────────

def test_severity_empty():
    result = compute_severity([])
    assert result["severity_level"] == "NONE"
    assert result["hotspot_count"] == 0
    assert result["fire_spread_risk"] == 0.0


def test_severity_low():
    dets = [make_det([0, 0, 5, 5], 0.4, cls=0)] * 2
    result = compute_severity(dets)
    assert result["severity_level"] == "LOW"
    assert result["hotspot_count"] == 2


def test_severity_extreme():
    dets = [make_det([i*10, i*10, i*10+5, i*10+5], 0.95, cls=0) for i in range(40)]
    result = compute_severity(dets)
    assert result["severity_level"] == "EXTREME"
    assert result["severity_score"] > 50


def test_spread_risk_bounded():
    dets = [make_det([0, 0, 5, 5], 0.99, cls=0)] * 100
    result = compute_severity(dets)
    assert 0.0 <= result["fire_spread_risk"] <= 1.0


# ─────────────────────────────────────────────────────────────
# Burn area estimation
# ─────────────────────────────────────────────────────────────

def test_burn_area_no_detections():
    result = estimate_burn_area([], 640, 640)
    assert result["burn_fraction"] == 0.0
    assert result["burn_pixel_area"] == 0


def test_burn_area_full_image():
    dets = [make_det([0, 0, 640, 640], 0.9, cls=0)]
    result = estimate_burn_area(dets, 640, 640)
    assert result["burn_fraction"] == pytest.approx(1.0, abs=0.001)


def test_burn_area_hectares():
    dets = [make_det([0, 0, 640, 640], 0.9, cls=0)]
    # 10km × 10km scene → 10,000 ha, full burn → 10,000 ha
    result = estimate_burn_area(dets, 640, 640, geo_width_m=10_000, geo_height_m=10_000)
    assert "burn_area_ha" in result
    assert result["burn_area_ha"] == pytest.approx(10_000.0, rel=0.01)


# ─────────────────────────────────────────────────────────────
# Heatmap generation
# ─────────────────────────────────────────────────────────────

def test_heatmap_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    dets  = [make_det([100, 100, 200, 200], 0.85, cls=0)]
    hm = generate_heatmap(image, dets, alpha=0.5)
    assert hm.shape == (480, 640, 3)
    assert hm.dtype == np.uint8


def test_heatmap_no_detections():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    hm = generate_heatmap(image, [], alpha=0.5)
    assert hm.shape == (100, 100, 3)


# ─────────────────────────────────────────────────────────────
# Pixel-to-latlon
# ─────────────────────────────────────────────────────────────

def test_pixel_to_latlon_wgs84():
    # Simple affine: origin at (10°E, 50°N), 1 pixel = 0.01°
    # rasterio affine: [c, a, b, f, d, e] = [0, res, 0, 0, 0, -res]
    # transform list from rasterio: [a, b, c, d, e, f]
    # transform[2]=c (x origin), transform[5]=f (y origin)
    transform = [0.01, 0.0, 10.0, 0.0, -0.01, 50.0]  # [a,b,c,d,e,f]
    lat, lon = pixel_to_latlon(0, 0, transform, crs_str="EPSG:4326")
    assert lat == pytest.approx(50.0, abs=0.001)
    assert lon == pytest.approx(10.0, abs=0.001)


# ─────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────

def test_is_image_file():
    assert is_image_file("satellite.tif")   is True
    assert is_image_file("photo.jpg")       is True
    assert is_image_file("data.csv")        is False
    assert is_image_file("model.pt")        is False


def test_timestamp_str_format():
    ts = timestamp_str()
    assert len(ts) == 15  # YYYYmmdd_HHMMSS
    assert "_" in ts


def test_file_md5_consistency(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    h1 = file_md5(str(f))
    h2 = file_md5(str(f))
    assert h1 == h2
    assert len(h1) == 32  # MD5 hex digest length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
