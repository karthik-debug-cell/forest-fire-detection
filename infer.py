"""
infer.py
========
Inference pipeline for the Forest Fire Detection System.

Supports:
  - Single satellite image (JPG / PNG / GeoTIFF)
  - Folder of images (batch)
  - GeoTIFF large scene with automatic tiling
  - Near-real-time folder watcher (polls for new images)

Uses YOLOv8 + SAHI sliced inference for superior small-object detection.

Usage
-----
    # Single image
    python infer.py --source sample.tif

    # Folder
    python infer.py --source ./test_images --save_csv

    # Large GeoTIFF scene
    python infer.py --source big_scene.tif --large_scene

    # Near-real-time folder watch
    python infer.py --source ./incoming --watch --interval 30

    # Custom confidence threshold + model
    python infer.py --source sample.tif --conf 0.35 --weights runs/train/fire_det/weights/best.pt
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from utils import (
    compute_severity,
    ensure_dir,
    estimate_burn_area,
    generate_heatmap,
    get_device,
    get_geotiff_metadata,
    is_image_file,
    pixel_to_latlon,
    save_results_csv,
    save_results_json,
    send_alert,
    setup_logger,
    soft_nms,
    timestamp_str,
)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES  = {0: "fire_hotspot", 1: "smoke", 2: "burn_region"}
CLASS_COLORS = {
    0: (0,   0,   255),   # Red   – fire
    1: (128, 128, 128),   # Grey  – smoke
    2: (0,   140, 255),   # Orange – burn
}
DEFAULT_WEIGHTS = "runs/train/fire_det/weights/best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"


# ─────────────────────────────────────────────────────────────
# 1. MODEL LOADER
# ─────────────────────────────────────────────────────────────

def load_yolo_model(weights: str, device: str):
    """Load a YOLOv8 model. Falls back to pretrained base weights if file missing."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not Path(weights).exists():
        logger.warning(f"Weights not found: {weights} – using {FALLBACK_WEIGHTS}")
        weights = FALLBACK_WEIGHTS

    model = YOLO(weights)
    logger.info(f"Model loaded: {weights} on {device}")
    return model


def load_sahi_model(weights: str, device: str):
    """Load a SAHI AutoDetectionModel wrapping YOLOv8."""
    try:
        from sahi import AutoDetectionModel
    except ImportError:
        logger.error("sahi not installed. Run: pip install sahi")
        sys.exit(1)

    if not Path(weights).exists():
        weights = FALLBACK_WEIGHTS

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=0.25,  # Will be overridden per call
        device=device,
    )
    logger.info(f"SAHI model loaded: {weights}")
    return model


# ─────────────────────────────────────────────────────────────
# 2. SINGLE IMAGE INFERENCE (SAHI)
# ─────────────────────────────────────────────────────────────

def run_sahi_inference(
    sahi_model,
    image_path: str,
    conf: float = 0.30,
    iou_thresh: float = 0.45,
    slice_height: int = 640,
    slice_width:  int = 640,
    overlap_h:    float = 0.2,
    overlap_w:    float = 0.2,
) -> List[Dict]:
    """
    Run SAHI sliced inference on a single image.

    Returns
    -------
    List of detection dicts:
        {"bbox": [x1,y1,x2,y2], "score": float, "class": int,
         "class_name": str, "image": str}
    """
    try:
        from sahi.predict import get_sliced_prediction
    except ImportError:
        logger.error("sahi not installed.")
        return []

    # Update model confidence threshold
    sahi_model.confidence_threshold = conf

    result = get_sliced_prediction(
        image=image_path,
        detection_model=sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        postprocess_type="NMM",          # Non-max merging across slices
        postprocess_match_threshold=iou_thresh,
        verbose=0,
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        class_id = pred.category.id
        detections.append({
            "bbox":       [bbox[0], bbox[1], bbox[2], bbox[3]],
            "score":      float(pred.score.value),
            "class":      class_id,
            "class_name": CLASS_NAMES.get(class_id, "unknown"),
            "image":      str(image_path),
        })

    logger.debug(f"SAHI detections: {len(detections)} on {Path(image_path).name}")
    return detections


# ─────────────────────────────────────────────────────────────
# 3. SINGLE IMAGE INFERENCE (YOLO only, no SAHI)
# ─────────────────────────────────────────────────────────────

def run_yolo_inference(
    yolo_model,
    image_path: str,
    conf: float = 0.30,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
) -> List[Dict]:
    """
    Standard YOLOv8 inference (no slicing).
    Useful for small images that don't need tiling.
    """
    results = yolo_model.predict(
        source=image_path,
        conf=conf,
        iou=iou_thresh,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes, scores, classes):
            detections.append({
                "bbox":       box.tolist(),
                "score":      float(score),
                "class":      int(cls_id),
                "class_name": CLASS_NAMES.get(int(cls_id), "unknown"),
                "image":      str(image_path),
            })

    return detections


# ─────────────────────────────────────────────────────────────
# 4. GEO-COORDINATE ENRICHMENT
# ─────────────────────────────────────────────────────────────

def enrich_with_geocoords(
    detections: List[Dict],
    image_path: str,
) -> List[Dict]:
    """
    Add lat/lon centre coordinates to each detection if the image is a GeoTIFF.
    """
    if not Path(image_path).suffix.lower() in {".tif", ".tiff"}:
        return detections   # No georeferencing for plain images

    meta = get_geotiff_metadata(image_path)
    if not meta:
        return detections

    transform = meta.get("transform", [])
    crs       = meta.get("crs", "EPSG:4326")

    if len(transform) < 6:
        return detections

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        lat, lon = pixel_to_latlon(cx, cy, transform, crs)
        det["lat"] = lat
        det["lon"] = lon

    return detections


# ─────────────────────────────────────────────────────────────
# 5. ANNOTATION DRAWING
# ─────────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_labels: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id   = det.get("class", 0)
        score    = det.get("score", 0.0)
        name     = det.get("class_name", "?")
        color    = CLASS_COLORS.get(cls_id, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        if show_labels:
            label = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return annotated


# ─────────────────────────────────────────────────────────────
# 6. FULL SINGLE-IMAGE PIPELINE
# ─────────────────────────────────────────────────────────────

def process_image(
    image_path: str,
    sahi_model=None,
    yolo_model=None,
    use_sahi: bool = True,
    conf: float = 0.30,
    iou_thresh: float = 0.45,
    device: str = "cpu",
    output_dir: str = "outputs",
    save_heatmap: bool = True,
    alert_method: str = "print",
    alert_config: Optional[Dict] = None,
    alert_severity_threshold: str = "MODERATE",
) -> Dict:
    """
    Full pipeline for one image: infer → enrich → score → save.

    Returns a result dict.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)

    if img is None:
        # Try rasterio for GeoTIFF
        try:
            from preprocess import SatelliteImageReader, make_fire_composite, apply_clahe
            reader = SatelliteImageReader(image_path)
            composite = make_fire_composite(reader)
            composite = apply_clahe(composite)
            img = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.error(f"Cannot open image {image_path}: {exc}")
            return {"error": str(exc), "image": image_path}

    h, w = img.shape[:2]
    stem = Path(image_path).stem
    ts   = timestamp_str()

    # ── Inference ─────────────────────────────────────────────
    if use_sahi and sahi_model is not None:
        detections = run_sahi_inference(sahi_model, image_path, conf=conf, iou_thresh=iou_thresh)
    elif yolo_model is not None:
        detections = run_yolo_inference(yolo_model, image_path, conf=conf, iou_thresh=iou_thresh, device=device)
    else:
        logger.error("No model loaded!")
        return {}

    # ── Post-processing NMS (extra safety pass) ───────────────
    detections = soft_nms(detections, sigma=0.5, score_threshold=0.15)

    # ── Geocoordinate enrichment ──────────────────────────────
    detections = enrich_with_geocoords(detections, image_path)

    # ── Severity scoring ──────────────────────────────────────
    severity = compute_severity(detections)

    # ── Burn area estimation ──────────────────────────────────
    meta = get_geotiff_metadata(image_path)
    geo_w = geo_h = None
    if meta and "resolution_x" in meta:
        geo_w = meta["resolution_x"] * w
        geo_h = meta["resolution_y"] * h
    burn_area = estimate_burn_area(detections, w, h, geo_w, geo_h)

    # ── Draw annotations ──────────────────────────────────────
    annotated = draw_detections(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), detections)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # ── Save outputs ──────────────────────────────────────────
    annotated_dir = ensure_dir(Path(output_dir) / "annotated")
    reports_dir   = ensure_dir(Path(output_dir) / "reports")
    heatmap_dir   = ensure_dir(Path(output_dir) / "heatmaps")

    ann_path = annotated_dir / f"{stem}_{ts}.jpg"
    cv2.imwrite(str(ann_path), annotated_bgr)
    logger.info(f"Annotated image saved → {ann_path}")

    if save_heatmap and detections:
        heatmap_img = generate_heatmap(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), detections)
        hm_path = heatmap_dir / f"{stem}_{ts}_heatmap.jpg"
        cv2.imwrite(str(hm_path), cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        logger.info(f"Heatmap saved → {hm_path}")

    result = {
        "image":       image_path,
        "timestamp":   ts,
        "width":       w,
        "height":      h,
        "num_detections": len(detections),
        "detections":  detections,
        "severity":    severity,
        "burn_area":   burn_area,
        "annotated_path": str(ann_path),
    }

    json_path = reports_dir / f"{stem}_{ts}.json"
    save_results_json(result, str(json_path))

    # ── Alert ─────────────────────────────────────────────────
    sev_order = ["NONE", "LOW", "MODERATE", "HIGH", "EXTREME"]
    sev_level = severity["severity_level"]
    if (
        len(detections) > 0 and
        sev_order.index(sev_level) >= sev_order.index(alert_severity_threshold)
    ):
        msg = (
            f"Fire detected in {Path(image_path).name} | "
            f"{severity['hotspot_count']} hotspots | "
            f"Severity: {sev_level} (score {severity['severity_score']}) | "
            f"Spread risk: {severity['fire_spread_risk']:.0%}"
        )
        send_alert(msg, sev_level, method=alert_method, config=alert_config)

    return result


# ─────────────────────────────────────────────────────────────
# 7. BATCH / FOLDER INFERENCE
# ─────────────────────────────────────────────────────────────

def process_folder(
    source_dir: str,
    sahi_model=None,
    yolo_model=None,
    use_sahi: bool = True,
    conf: float = 0.30,
    output_dir: str = "outputs",
    save_csv: bool = True,
    device: str = "cpu",
) -> List[Dict]:
    """Process all images in a folder and optionally save combined CSV."""
    source_dir = Path(source_dir)
    image_files = sorted([p for p in source_dir.rglob("*") if is_image_file(str(p))])

    if not image_files:
        logger.warning(f"No images found in {source_dir}")
        return []

    logger.info(f"Processing {len(image_files)} images from {source_dir}…")
    all_results   = []
    all_detections = []

    from tqdm import tqdm
    for img_path in tqdm(image_files, desc="Inferring"):
        result = process_image(
            image_path=str(img_path),
            sahi_model=sahi_model,
            yolo_model=yolo_model,
            use_sahi=use_sahi,
            conf=conf,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(result)
        all_detections.extend(result.get("detections", []))

    if save_csv:
        csv_path = Path(output_dir) / f"detections_{timestamp_str()}.csv"
        save_results_csv(all_detections, str(csv_path))

    return all_results


# ─────────────────────────────────────────────────────────────
# 8. REAL-TIME FOLDER WATCHER
# ─────────────────────────────────────────────────────────────

def watch_folder(
    folder: str,
    sahi_model=None,
    yolo_model=None,
    use_sahi: bool = True,
    conf: float = 0.30,
    output_dir: str = "outputs",
    interval: int = 30,
    device: str = "cpu",
) -> None:
    """
    Poll a folder for new images every `interval` seconds and run inference.
    Maintains a set of already-processed files to avoid re-inference.
    """
    folder = Path(folder)
    processed: set = set()

    logger.info(f"👁  Watching {folder} every {interval}s for new satellite images…")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            new_files = [
                p for p in folder.iterdir()
                if is_image_file(str(p)) and str(p) not in processed
            ]
            for img_path in new_files:
                logger.info(f"New image detected: {img_path.name}")
                process_image(
                    image_path=str(img_path),
                    sahi_model=sahi_model,
                    yolo_model=yolo_model,
                    use_sahi=use_sahi,
                    conf=conf,
                    device=device,
                    output_dir=output_dir,
                )
                processed.add(str(img_path))

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Folder watcher stopped.")


# ─────────────────────────────────────────────────────────────
# 9. GIS EXPORT (GeoJSON)
# ─────────────────────────────────────────────────────────────

def export_geojson(
    detections: List[Dict],
    output_path: str,
    image_path: str = "",
) -> None:
    """
    Export detections with lat/lon as a GeoJSON FeatureCollection.
    Only detections with non-zero lat/lon are included.
    """
    features = []
    for det in detections:
        lat = det.get("lat", 0.0)
        lon = det.get("lon", 0.0)
        if lat == 0.0 and lon == 0.0:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "class_id":   det.get("class"),
                "class_name": det.get("class_name"),
                "confidence": det.get("score"),
                "image":      image_path,
                "bbox":       det.get("bbox"),
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    ensure_dir(Path(output_path).parent)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"GeoJSON exported → {output_path} ({len(features)} points)")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forest Fire Detection – Inference")
    p.add_argument("--source",    required=True, help="Image path, folder, or GeoTIFF")
    p.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    p.add_argument("--conf",      type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou",       type=float, default=0.45, help="IoU threshold")
    p.add_argument("--device",    default=None)
    p.add_argument("--use_sahi",  action="store_true", default=True)
    p.add_argument("--no_sahi",   action="store_true", help="Disable SAHI, use plain YOLOv8")
    p.add_argument("--output_dir",default="outputs")
    p.add_argument("--save_csv",  action="store_true")
    p.add_argument("--save_geojson", action="store_true")
    p.add_argument("--watch",     action="store_true", help="Enable real-time folder watch")
    p.add_argument("--interval",  type=int, default=30, help="Watch interval in seconds")
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--slice_size",type=int, default=640)
    p.add_argument("--overlap",   type=float, default=0.2)
    p.add_argument("--alert",     default="print", choices=["print", "email", "sms", "webhook"])
    return p.parse_args()


if __name__ == "__main__":
    setup_logger()
    args = parse_args()

    device   = args.device or get_device()
    use_sahi = args.use_sahi and not args.no_sahi

    # Load models
    if use_sahi:
        sahi_model = load_sahi_model(args.weights, device)
        yolo_model = None
    else:
        sahi_model = None
        yolo_model = load_yolo_model(args.weights, device)

    source = Path(args.source)

    if args.watch and source.is_dir():
        watch_folder(
            folder=str(source),
            sahi_model=sahi_model,
            yolo_model=yolo_model,
            use_sahi=use_sahi,
            conf=args.conf,
            output_dir=args.output_dir,
            interval=args.interval,
            device=device,
        )

    elif source.is_dir():
        results = process_folder(
            source_dir=str(source),
            sahi_model=sahi_model,
            yolo_model=yolo_model,
            use_sahi=use_sahi,
            conf=args.conf,
            output_dir=args.output_dir,
            save_csv=args.save_csv,
            device=device,
        )

    else:
        # Single image
        result = process_image(
            image_path=str(source),
            sahi_model=sahi_model,
            yolo_model=yolo_model,
            use_sahi=use_sahi,
            conf=args.conf,
            iou_thresh=args.iou,
            device=device,
            output_dir=args.output_dir,
            alert_method=args.alert,
        )

        if args.save_geojson and result.get("detections"):
            gj_path = Path(args.output_dir) / "reports" / f"{source.stem}_detections.geojson"
            export_geojson(result["detections"], str(gj_path), str(source))

        print(json.dumps(
            {k: v for k, v in result.items() if k != "detections"},
            indent=2, default=str,
        ))
