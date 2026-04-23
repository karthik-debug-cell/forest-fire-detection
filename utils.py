"""
utils.py
========
Shared utility functions for the Forest Fire Detection System.

Covers:
  - Logging setup
  - Device detection (GPU / CPU / MPS)
  - GeoTIFF metadata extraction
  - Pixel-to-latlon conversion
  - NMS helpers
  - JSON / CSV result I/O
  - Fire severity scoring
  - Burn area estimation
  - Webhook / alert helpers
"""

import os
import json
import csv
import math
import datetime
import hashlib
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from loguru import logger


# ─────────────────────────────────────────────────────────────
# 1. LOGGING
# ─────────────────────────────────────────────────────────────

def setup_logger(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure loguru with file + console sinks."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"firedet_{datetime.date.today()}.log"

    logger.remove()          # Remove default sink
    logger.add(
        log_path,
        level=level,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.info(f"Logger initialised → {log_path}")


# ─────────────────────────────────────────────────────────────
# 2. DEVICE DETECTION
# ─────────────────────────────────────────────────────────────

def get_device() -> str:
    """Return best available PyTorch device string."""
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {name} ({mem:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS (Metal) device detected")
    else:
        device = "cpu"
        logger.warning("No GPU detected – running on CPU (inference will be slow)")
    return device


# ─────────────────────────────────────────────────────────────
# 3. GEOTIFF / RASTERIO HELPERS
# ─────────────────────────────────────────────────────────────

def get_geotiff_metadata(tif_path: str) -> Dict[str, Any]:
    """
    Extract CRS, transform, bounds, band count and nodata from a GeoTIFF.
    Returns a metadata dict; returns empty dict if rasterio not available.
    """
    try:
        import rasterio
        from rasterio.crs import CRS

        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            meta = {
                "driver": src.driver,
                "crs": str(src.crs),
                "transform": list(src.transform),
                "width": src.width,
                "height": src.height,
                "count": src.count,          # Number of bands
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "bounds": {
                    "left":   bounds.left,
                    "bottom": bounds.bottom,
                    "right":  bounds.right,
                    "top":    bounds.top,
                },
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
            }
        logger.debug(f"GeoTIFF metadata: {tif_path} → {meta['crs']}, {meta['width']}x{meta['height']}")
        return meta
    except ImportError:
        logger.warning("rasterio not installed – cannot read GeoTIFF metadata")
        return {}
    except Exception as exc:
        logger.error(f"Failed to read GeoTIFF metadata: {exc}")
        return {}


def pixel_to_latlon(
    pixel_x: float,
    pixel_y: float,
    transform: List[float],
    crs_str: str = "EPSG:4326",
) -> Tuple[float, float]:
    """
    Convert pixel (col, row) coordinates to geographic (lat, lon).

    Parameters
    ----------
    pixel_x : column index
    pixel_y : row index
    transform : affine transform list [c, a, b, f, d, e] (rasterio convention)
    crs_str   : source CRS string (e.g. "EPSG:32643")

    Returns
    -------
    (latitude, longitude) in WGS84
    """
    try:
        from pyproj import Transformer

        # Affine: x_geo = c + a*col + b*row,  y_geo = f + d*col + e*row
        c, a, b, f, d, e = transform[2], transform[0], transform[1], transform[5], transform[3], transform[4]
        x_geo = c + a * pixel_x + b * pixel_y
        y_geo = f + d * pixel_x + e * pixel_y

        if crs_str in ("EPSG:4326", "WGS84"):
            return y_geo, x_geo   # lat, lon

        transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x_geo, y_geo)
        return lat, lon
    except Exception as exc:
        logger.error(f"pixel_to_latlon error: {exc}")
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────
# 4. BOUNDING BOX / NMS UTILITIES
# ─────────────────────────────────────────────────────────────

def xyxy_to_xywh(box: List[float]) -> List[float]:
    """Convert [x1,y1,x2,y2] to [cx,cy,w,h]."""
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert [cx,cy,w,h] to [x1,y1,x2,y2]."""
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou(boxA: List[float], boxB: List[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def soft_nms(
    detections: List[Dict],
    sigma: float = 0.5,
    score_threshold: float = 0.01,
) -> List[Dict]:
    """
    Soft-NMS (Bodla et al. 2017) on detection list.
    Each detection: {"bbox": [x1,y1,x2,y2], "score": float, "class": int}
    Returns filtered list sorted by score descending.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            overlap = iou(best["bbox"], d["bbox"])
            # Gaussian penalty
            d["score"] *= math.exp(-(overlap ** 2) / sigma)
            if d["score"] >= score_threshold:
                remaining.append(d)
        dets = sorted(remaining, key=lambda d: d["score"], reverse=True)

    return keep


# ─────────────────────────────────────────────────────────────
# 5. FIRE SEVERITY SCORING
# ─────────────────────────────────────────────────────────────

SEVERITY_THRESHOLDS = {
    "LOW":      (1, 3),
    "MODERATE": (4, 10),
    "HIGH":     (11, 30),
    "EXTREME":  (31, float("inf")),
}


def compute_severity(detections: List[Dict]) -> Dict[str, Any]:
    """
    Compute fire severity score based on detection count and confidence.

    Returns dict with:
      - hotspot_count
      - severity_level (LOW / MODERATE / HIGH / EXTREME)
      - severity_score  (0–100)
      - avg_confidence
      - max_confidence
      - fire_spread_risk (0.0–1.0)
    """
    if not detections:
        return {
            "hotspot_count": 0,
            "severity_level": "NONE",
            "severity_score": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "fire_spread_risk": 0.0,
        }

    fire_dets = [d for d in detections if d.get("class") == 0]
    smoke_dets = [d for d in detections if d.get("class") == 1]
    burn_dets  = [d for d in detections if d.get("class") == 2]

    n = len(detections)
    confs = [d["score"] for d in detections]
    avg_conf = float(np.mean(confs))
    max_conf = float(np.max(confs))

    # Weighted score: fires count double
    weighted_count = len(fire_dets) * 2 + len(smoke_dets) + len(burn_dets) * 0.5

    # Severity level
    level = "LOW"
    for lvl, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= weighted_count <= hi:
            level = lvl
            break

    # Normalised score 0–100
    severity_score = min(100, int(weighted_count / 60 * 100 + avg_conf * 30))

    # Fire spread risk: proximity + smoke presence
    spread_risk = min(1.0, (len(fire_dets) * 0.05 + len(smoke_dets) * 0.03 + avg_conf * 0.3))

    return {
        "hotspot_count": len(fire_dets),
        "smoke_count":   len(smoke_dets),
        "burn_count":    len(burn_dets),
        "severity_level": level,
        "severity_score": severity_score,
        "avg_confidence": round(avg_conf, 4),
        "max_confidence": round(max_conf, 4),
        "fire_spread_risk": round(spread_risk, 4),
    }


# ─────────────────────────────────────────────────────────────
# 6. BURN AREA ESTIMATION
# ─────────────────────────────────────────────────────────────

def estimate_burn_area(
    detections: List[Dict],
    image_width: int,
    image_height: int,
    geo_width_m: Optional[float] = None,
    geo_height_m: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate the total burn area from bounding boxes.

    If geo_width_m / geo_height_m are provided (scene real-world size in metres),
    returns area in hectares; otherwise returns fraction of image area.

    Parameters
    ----------
    detections    : list of detection dicts with 'bbox' [x1,y1,x2,y2]
    image_width   : pixel width of the full image
    image_height  : pixel height of the full image
    geo_width_m   : real-world scene width in metres (optional)
    geo_height_m  : real-world scene height in metres (optional)
    """
    pixel_area_total = image_width * image_height
    burn_pixel_area = 0.0

    for d in detections:
        if d.get("class") in (0, 2):   # fire_hotspot + burn_region
            x1, y1, x2, y2 = d["bbox"]
            burn_pixel_area += (x2 - x1) * (y2 - y1)

    fraction = burn_pixel_area / pixel_area_total if pixel_area_total > 0 else 0.0

    result = {
        "burn_pixel_area": round(burn_pixel_area),
        "image_pixel_area": pixel_area_total,
        "burn_fraction": round(fraction, 6),
    }

    if geo_width_m and geo_height_m:
        scene_area_m2 = geo_width_m * geo_height_m
        burn_area_m2  = fraction * scene_area_m2
        burn_area_ha  = burn_area_m2 / 10_000
        result["scene_area_ha"] = round(scene_area_m2 / 10_000, 2)
        result["burn_area_ha"]  = round(burn_area_ha, 4)
        result["burn_area_km2"] = round(burn_area_ha / 100, 6)

    return result


# ─────────────────────────────────────────────────────────────
# 7. RESULT I/O
# ─────────────────────────────────────────────────────────────

def save_results_json(
    results: Dict,
    output_path: str,
) -> None:
    """Serialize detection results to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {output_path}")


def save_results_csv(
    detections: List[Dict],
    output_path: str,
    extra_fields: Optional[Dict] = None,
) -> None:
    """
    Save flat detection list to CSV.
    Columns: image, class_id, class_name, confidence, x1, y1, x2, y2, lat, lon
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "lat", "lon",
    ]
    if extra_fields:
        fieldnames += list(extra_fields.keys())

    class_names = {0: "fire_hotspot", 1: "smoke", 2: "burn_region"}

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for det in detections:
            row = {
                "image":      det.get("image", ""),
                "class_id":   det.get("class", -1),
                "class_name": class_names.get(det.get("class", -1), "unknown"),
                "confidence": round(det.get("score", 0.0), 4),
                "x1": round(det["bbox"][0], 2) if "bbox" in det else "",
                "y1": round(det["bbox"][1], 2) if "bbox" in det else "",
                "x2": round(det["bbox"][2], 2) if "bbox" in det else "",
                "y2": round(det["bbox"][3], 2) if "bbox" in det else "",
                "lat": round(det.get("lat", 0.0), 6),
                "lon": round(det.get("lon", 0.0), 6),
            }
            if extra_fields:
                row.update(extra_fields)
            writer.writerow(row)

    logger.info(f"CSV saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# 8. HEATMAP GENERATION
# ─────────────────────────────────────────────────────────────

def generate_heatmap(
    image: np.ndarray,
    detections: List[Dict],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a confidence-weighted heatmap on the image.

    Parameters
    ----------
    image      : BGR numpy array (H, W, 3)
    detections : list of dicts with 'bbox' [x1,y1,x2,y2] and 'score'
    alpha      : blend weight for the heatmap overlay
    """
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        score = det.get("score", 1.0)
        # Draw Gaussian blob centred on box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(10, int(max(x2 - x1, y2 - y1) * 0.8))
        cv2.circle(heat, (cx, cy), radius, score, -1)

    # Smooth + normalise
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=radius // 2 or 10)
    if heat.max() > 0:
        heat = heat / heat.max()

    # Map to colour
    heat_uint8 = (heat * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlayed


# ─────────────────────────────────────────────────────────────
# 9. ALERT SYSTEM
# ─────────────────────────────────────────────────────────────

def send_alert(
    message: str,
    severity: str,
    method: str = "print",
    config: Optional[Dict] = None,
) -> bool:
    """
    Dispatch a fire alert via the specified method.

    Parameters
    ----------
    message  : alert text
    severity : e.g. "HIGH"
    method   : "print" | "email" | "sms" | "webhook"
    config   : dict with credentials / endpoint

    Returns True on success.
    """
    config = config or {}
    logger.warning(f"🔥 FIRE ALERT [{severity}] – {message}")

    if method == "print":
        print(f"\n{'='*60}")
        print(f"  🔥 FIRE ALERT – SEVERITY: {severity}")
        print(f"  {message}")
        print(f"{'='*60}\n")
        return True

    elif method == "email":
        try:
            smtp_host = config.get("smtp_host", "smtp.gmail.com")
            smtp_port = config.get("smtp_port", 587)
            sender    = config["sender"]
            password  = config["password"]
            recipient = config["recipient"]

            msg = MIMEMultipart()
            msg["From"]    = sender
            msg["To"]      = recipient
            msg["Subject"] = f"[FIRE ALERT] Severity: {severity}"
            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())

            logger.info(f"Email alert sent to {recipient}")
            return True
        except Exception as exc:
            logger.error(f"Email alert failed: {exc}")
            return False

    elif method == "sms":
        try:
            from twilio.rest import Client
            client = Client(config["account_sid"], config["auth_token"])
            client.messages.create(
                body=f"[FIRE ALERT] {severity}: {message}",
                from_=config["from_number"],
                to=config["to_number"],
            )
            logger.info("SMS alert sent via Twilio")
            return True
        except ImportError:
            logger.error("twilio not installed – pip install twilio")
            return False
        except Exception as exc:
            logger.error(f"SMS alert failed: {exc}")
            return False

    elif method == "webhook":
        try:
            payload = {
                "severity": severity,
                "message":  message,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
            resp = requests.post(
                config["url"],
                json=payload,
                timeout=10,
                headers=config.get("headers", {}),
            )
            resp.raise_for_status()
            logger.info(f"Webhook alert sent → {config['url']} (status {resp.status_code})")
            return True
        except Exception as exc:
            logger.error(f"Webhook alert failed: {exc}")
            return False

    else:
        logger.error(f"Unknown alert method: {method}")
        return False


# ─────────────────────────────────────────────────────────────
# 10. MISC HELPERS
# ─────────────────────────────────────────────────────────────

def file_md5(path: str) -> str:
    """Return MD5 hash of a file for deduplication / caching."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image_file(path: str) -> bool:
    """True if extension is a supported raster image format."""
    return Path(path).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    }


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if not exists, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_str() -> str:
    """Return compact UTC timestamp string safe for filenames."""
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
