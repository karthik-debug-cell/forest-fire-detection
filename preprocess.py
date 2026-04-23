"""
preprocess.py
=============
Satellite image preprocessing pipeline for the Forest Fire Detection System.

Supports:
  - Reading GeoTIFF multispectral data (Sentinel-2, MODIS, VIIRS, Landsat)
  - RGB + false-colour composite generation (including SWIR fire composite)
  - CLAHE contrast enhancement
  - Cloud masking (simple NIR/SWIR heuristic)
  - Large-image tiling (512/640/1024 px with overlap)
  - YOLO-format annotation generation from FIRMS hotspot CSVs
  - 70/15/15 train-val-test splitting

Usage
-----
    python preprocess.py --source_dir /path/to/geotiffs \
                         --output_dir ./dataset \
                         --tile_size 640 \
                         --overlap 0.2

    python preprocess.py --firms_csv firms_hotspots.csv \
                         --image_dir /path/to/images \
                         --output_dir ./dataset \
                         --generate_labels
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from utils import ensure_dir, is_image_file, setup_logger, timestamp_str


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SENTINEL2_BAND_INDICES = {
    # Key Sentinel-2 band names → zero-based rasterio band index (band order varies)
    "B2":  1,   # Blue  (490 nm)
    "B3":  2,   # Green (560 nm)
    "B4":  3,   # Red   (665 nm)
    "B8":  7,   # NIR   (842 nm)
    "B8A": 8,   # Red Edge / NIR narrow (865 nm)
    "B11": 10,  # SWIR-1 (1610 nm) – excellent fire indicator
    "B12": 11,  # SWIR-2 (2190 nm) – best fire / hotspot band
}

CLOUD_THRESHOLD = 0.75   # NIR reflectance above this → likely cloud (0–1 scale)
VALID_PIXEL_MIN = 0.40   # Discard tiles where < 40 % pixels are valid (non-cloud, non-nodata)


# ─────────────────────────────────────────────────────────────
# 1. GEOTIFF READER
# ─────────────────────────────────────────────────────────────

class SatelliteImageReader:
    """
    Load a GeoTIFF (or multi-band TIFF) and expose band data as numpy arrays.
    Falls back gracefully to cv2 for plain JPG/PNG.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.is_geotiff = self.path.suffix.lower() in {".tif", ".tiff"}
        self._data: Optional[np.ndarray] = None   # (bands, H, W) float32 [0..1]
        self._meta: Dict = {}
        self._load()

    def _load(self) -> None:
        if self.is_geotiff:
            self._load_geotiff()
        else:
            self._load_rgb()

    def _load_geotiff(self) -> None:
        try:
            import rasterio

            with rasterio.open(self.path) as src:
                self._meta = {
                    "crs":       str(src.crs),
                    "transform": list(src.transform),
                    "width":     src.width,
                    "height":    src.height,
                    "count":     src.count,
                    "nodata":    src.nodata,
                    "bounds":    src.bounds,
                }
                # Read all bands as float32 normalised to [0, 1]
                raw = src.read().astype(np.float32)   # (bands, H, W)

                # Normalise per-band (handle uint16 Sentinel data etc.)
                for b in range(raw.shape[0]):
                    band_max = np.nanmax(raw[b])
                    if band_max > 1.0:
                        raw[b] /= band_max if band_max > 0 else 1.0

                self._data = raw
                logger.debug(
                    f"Loaded GeoTIFF: {self.path.name} | "
                    f"{src.count} bands | {src.width}×{src.height}"
                )
        except ImportError:
            logger.error("rasterio not installed. Install: pip install rasterio")
            raise
        except Exception as exc:
            logger.error(f"GeoTIFF read error for {self.path}: {exc}")
            raise

    def _load_rgb(self) -> None:
        """Load plain image via OpenCV as 3-band float [0,1]."""
        img = cv2.imread(str(self.path))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {self.path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._data = np.moveaxis(img, -1, 0)   # (3, H, W)
        self._meta = {"width": img.shape[1], "height": img.shape[0], "count": 3}

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def meta(self) -> Dict:
        return self._meta

    @property
    def height(self) -> int:
        return self._meta.get("height", self._data.shape[1])

    @property
    def width(self) -> int:
        return self._meta.get("width", self._data.shape[2])

    @property
    def num_bands(self) -> int:
        return self._data.shape[0]


# ─────────────────────────────────────────────────────────────
# 2. COMPOSITE GENERATORS
# ─────────────────────────────────────────────────────────────

def make_rgb_composite(reader: SatelliteImageReader) -> np.ndarray:
    """
    Generate a true-colour RGB composite (uint8 HxWx3).
    Uses bands B4/B3/B2 for Sentinel-2 (indices 2,1,0 if 3-band input).
    """
    n = reader.num_bands

    if n >= 4:
        # Assume Sentinel-2 band order: B2(0), B3(1), B4(2), B8(3), ...
        r = reader.data[2]  # B4 Red
        g = reader.data[1]  # B3 Green
        b = reader.data[0]  # B2 Blue
    elif n == 3:
        r, g, b = reader.data[0], reader.data[1], reader.data[2]
    else:
        r = g = b = reader.data[0]

    rgb = np.stack([r, g, b], axis=-1)  # HxWx3 float [0,1]
    return _float_to_uint8(rgb)


def make_fire_composite(reader: SatelliteImageReader) -> np.ndarray:
    """
    SWIR false-colour composite optimised for fire / hotspot detection.
    Formula: R=B12(SWIR2), G=B11(SWIR1), B=B4(Red)
    Active fire pixels appear bright red/orange in this composite.

    Falls back to standard RGB if insufficient bands.
    """
    n = reader.num_bands

    if n >= 12:
        r = reader.data[11]   # B12 SWIR-2
        g = reader.data[10]   # B11 SWIR-1
        b = reader.data[2]    # B4 Red
    elif n >= 3:
        logger.debug("< 12 bands – using standard RGB composite (no SWIR)")
        return make_rgb_composite(reader)
    else:
        return make_rgb_composite(reader)

    rgb = np.stack([r, g, b], axis=-1)
    return _float_to_uint8(rgb)


def make_nir_composite(reader: SatelliteImageReader) -> np.ndarray:
    """
    Near-infrared false-colour (Colour Infrared): R=NIR, G=Red, B=Green.
    Vegetation appears red; burn scars appear dark.
    """
    n = reader.num_bands

    if n >= 4:
        r = reader.data[3]   # B8 NIR
        g = reader.data[2]   # B4 Red
        b = reader.data[1]   # B3 Green
    else:
        return make_rgb_composite(reader)

    rgb = np.stack([r, g, b], axis=-1)
    return _float_to_uint8(rgb)


def _float_to_uint8(arr: np.ndarray, percentile_clip: float = 2.0) -> np.ndarray:
    """
    Convert float HxWx3 [0,1] to uint8, applying percentile stretch for contrast.
    """
    lo = np.percentile(arr, percentile_clip)
    hi = np.percentile(arr, 100 - percentile_clip)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    return (arr * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# 3. CLAHE ENHANCEMENT
# ─────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) per channel.

    Parameters
    ----------
    image      : uint8 HxWx3 BGR or RGB
    clip_limit : contrast limit for CLAHE
    tile_grid  : grid size for local histogram computation

    Returns
    -------
    uint8 HxWx3 enhanced image (same channel order as input)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────
# 4. CLOUD MASKING
# ─────────────────────────────────────────────────────────────

def compute_cloud_fraction(reader: SatelliteImageReader) -> float:
    """
    Estimate cloud fraction using a simple NIR + visible brightness heuristic.
    Returns fraction in [0, 1].
    """
    n = reader.num_bands

    if n >= 4:
        nir  = reader.data[3]   # B8
        red  = reader.data[2]   # B4
        green = reader.data[1]  # B3
        # Bright in all channels → cloud
        bright_mask = (nir > CLOUD_THRESHOLD) & (red > 0.6) & (green > 0.6)
    elif n >= 3:
        # RGB only: use luminance
        lum = 0.299 * reader.data[0] + 0.587 * reader.data[1] + 0.114 * reader.data[2]
        bright_mask = lum > CLOUD_THRESHOLD
    else:
        bright_mask = reader.data[0] > CLOUD_THRESHOLD

    cloud_frac = float(bright_mask.mean())
    return cloud_frac


def is_cloudy(reader: SatelliteImageReader, max_cloud_fraction: float = 0.5) -> bool:
    """Return True if image has too many clouds."""
    cf = compute_cloud_fraction(reader)
    logger.debug(f"Cloud fraction: {cf:.2%}")
    return cf > max_cloud_fraction


# ─────────────────────────────────────────────────────────────
# 5. TILING
# ─────────────────────────────────────────────────────────────

def tile_image(
    image: np.ndarray,
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
) -> List[Dict]:
    """
    Slice a large image into overlapping tiles.

    Parameters
    ----------
    image        : uint8 HxWx3
    tile_size    : square tile side length in pixels
    overlap_ratio: fractional overlap between adjacent tiles

    Returns
    -------
    List of dicts:
        {"tile": np.ndarray, "x_off": int, "y_off": int,
         "tile_w": int, "tile_h": int}
    """
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap_ratio))
    tiles = []

    for y_off in range(0, h, stride):
        for x_off in range(0, w, stride):
            x_end = min(x_off + tile_size, w)
            y_end = min(y_off + tile_size, h)
            tile = image[y_off:y_end, x_off:x_end]

            # Pad to tile_size × tile_size if needed (edge tiles)
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            tiles.append({
                "tile":   tile,
                "x_off":  x_off,
                "y_off":  y_off,
                "tile_w": x_end - x_off,
                "tile_h": y_end - y_off,
            })

    logger.debug(f"Tiled {w}×{h} image into {len(tiles)} patches ({tile_size}px, overlap={overlap_ratio:.0%})")
    return tiles


# ─────────────────────────────────────────────────────────────
# 6. DATASET BUILDER
# ─────────────────────────────────────────────────────────────

def build_dataset_from_folder(
    source_dir: str,
    output_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    max_cloud: float = 0.5,
    use_fire_composite: bool = True,
    apply_clahe_flag: bool = True,
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> None:
    """
    Full pipeline: read GeoTIFFs → preprocess → tile → split → save.

    Output layout (YOLO-compatible):
        output_dir/
          images/train|val|test/*.jpg
          labels/train|val|test/*.txt   ← placeholder empty labels (fill manually)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Find all images
    all_images = [p for p in source_dir.rglob("*") if is_image_file(str(p))]
    logger.info(f"Found {len(all_images)} source images in {source_dir}")

    if not all_images:
        logger.error("No images found. Exiting.")
        return

    processed_tiles: List[Tuple[np.ndarray, str]] = []   # (tile, base_name)

    for img_path in tqdm(all_images, desc="Processing images"):
        try:
            reader = SatelliteImageReader(str(img_path))

            # Skip cloudy images
            if is_cloudy(reader, max_cloud):
                logger.debug(f"Skipping cloudy image: {img_path.name}")
                continue

            # Generate composite
            if use_fire_composite:
                composite = make_fire_composite(reader)
            else:
                composite = make_rgb_composite(reader)

            # CLAHE enhancement
            if apply_clahe_flag:
                composite = apply_clahe(composite)

            # Tile
            tiles = tile_image(composite, tile_size=tile_size, overlap_ratio=overlap)

            for i, tile_info in enumerate(tiles):
                tile_img = tile_info["tile"]
                # Skip mostly black tiles (nodata)
                if tile_img.mean() < 5:
                    continue
                base = f"{img_path.stem}_tile{i:04d}"
                processed_tiles.append((tile_img, base))

        except Exception as exc:
            logger.error(f"Failed to process {img_path.name}: {exc}")
            continue

    logger.info(f"Total valid tiles: {len(processed_tiles)}")

    if not processed_tiles:
        logger.warning("No valid tiles generated.")
        return

    # Shuffle + split
    random.shuffle(processed_tiles)
    n = len(processed_tiles)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    splits = {
        "train": processed_tiles[:n_train],
        "val":   processed_tiles[n_train : n_train + n_val],
        "test":  processed_tiles[n_train + n_val :],
    }

    CLASS_NAMES = ["fire_hotspot", "smoke", "burn_region"]

    for split_name, tile_list in splits.items():
        img_dir = ensure_dir(output_dir / "images" / split_name)
        lbl_dir = ensure_dir(output_dir / "labels" / split_name)

        for tile_img, base in tqdm(tile_list, desc=f"Saving {split_name}"):
            # Save image as JPEG
            out_img = img_dir / f"{base}.jpg"
            save_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img), save_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write empty label file (annotations added manually or via FIRMS)
            lbl_file = lbl_dir / f"{base}.txt"
            lbl_file.touch()

        logger.info(f"{split_name}: {len(tile_list)} tiles → {img_dir}")

    logger.success("Dataset build complete.")


# ─────────────────────────────────────────────────────────────
# 7. FIRMS CSV → YOLO LABEL GENERATOR
# ─────────────────────────────────────────────────────────────

def firms_to_yolo_labels(
    firms_csv: str,
    image_dir: str,
    label_dir: str,
    image_size: int = 640,
    box_size_px: int = 12,
    confidence_col: str = "confidence",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> None:
    """
    Convert NASA FIRMS CSV hotspot data to YOLO annotation files.

    This function is a *helper starter* – full use requires matching the
    FIRMS lat/lon to image pixel coordinates via the image GeoTransform.
    For demonstration, it creates centred boxes for each hotspot.

    Parameters
    ----------
    firms_csv    : path to FIRMS active fire CSV
    image_dir    : directory of georeferenced tile images
    label_dir    : where to write .txt label files
    image_size   : assumed square tile size (pixels)
    box_size_px  : bounding box half-size in pixels for a single hotspot
    """
    import pandas as pd

    df = pd.read_csv(firms_csv)
    label_dir = Path(label_dir)
    ensure_dir(label_dir)

    logger.info(f"FIRMS CSV: {len(df)} hotspots from {firms_csv}")

    # Group by image filename if present
    if "image_file" in df.columns:
        groups = df.groupby("image_file")
    else:
        # Without image_file column, create a single combined label
        groups = {"all_hotspots.txt": df}.items()

    for img_name, group in groups:
        stem = Path(img_name).stem
        label_path = label_dir / f"{stem}.txt"

        lines = []
        for _, row in group.iterrows():
            # For matched images, pixel coords are in 'pixel_x', 'pixel_y' columns
            if "pixel_x" in row and "pixel_y" in row:
                px, py = row["pixel_x"], row["pixel_y"]
            else:
                # Fallback: place in image centre (replace with proper reprojection)
                px, py = image_size / 2, image_size / 2

            # Normalise
            cx = px / image_size
            cy = py / image_size
            bw = (box_size_px * 2) / image_size
            bh = (box_size_px * 2) / image_size

            # Clamp
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            bw = min(bw, 1.0)
            bh = min(bh, 1.0)

            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines))

    logger.info(f"YOLO labels written to {label_dir}")


# ─────────────────────────────────────────────────────────────
# 8. DATASET SPLIT UTILITY
# ─────────────────────────────────────────────────────────────

def split_existing_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """
    Re-split an existing flat images+labels folder into train/val/test.
    Expects paired image and label files with same stem.
    """
    random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    image_files = sorted([p for p in images_dir.iterdir() if is_image_file(str(p))])
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])

    buckets = {
        "train": image_files[:n_train],
        "val":   image_files[n_train : n_train + n_val],
        "test":  image_files[n_train + n_val :],
    }

    for bucket, files in buckets.items():
        img_out = ensure_dir(output_dir / "images" / bucket)
        lbl_out = ensure_dir(output_dir / "labels" / bucket)

        for img_path in files:
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            shutil.copy2(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            else:
                (lbl_out / f"{img_path.stem}.txt").touch()

        logger.info(f"{bucket}: {len(files)} samples")

    logger.success(f"Dataset split complete → {output_dir}")


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Satellite wildfire dataset preprocessor")
    p.add_argument("--source_dir",  default="raw_images",  help="Raw satellite images folder")
    p.add_argument("--output_dir",  default="./dataset",   help="Output dataset folder")
    p.add_argument("--tile_size",   type=int, default=640, choices=[512, 640, 1024])
    p.add_argument("--overlap",     type=float, default=0.2, help="Tile overlap ratio")
    p.add_argument("--max_cloud",   type=float, default=0.5, help="Max cloud fraction")
    p.add_argument("--no_clahe",    action="store_true",   help="Disable CLAHE enhancement")
    p.add_argument("--rgb_only",    action="store_true",   help="Use RGB composite (not SWIR fire)")
    p.add_argument("--firms_csv",   default=None,          help="FIRMS hotspot CSV for label gen")
    p.add_argument("--generate_labels", action="store_true", help="Generate labels from FIRMS CSV")
    p.add_argument("--split_only",  action="store_true",   help="Only re-split existing flat dataset")
    p.add_argument("--images_dir",  default=None,          help="Flat images dir (for split_only)")
    p.add_argument("--labels_dir",  default=None,          help="Flat labels dir (for split_only)")
    return p.parse_args()


if __name__ == "__main__":
    setup_logger()
    args = parse_args()

    if args.split_only:
        if not args.images_dir or not args.labels_dir:
            logger.error("--split_only requires --images_dir and --labels_dir")
        else:
            split_existing_dataset(args.images_dir, args.labels_dir, args.output_dir)

    elif args.generate_labels and args.firms_csv:
        label_out = Path(args.output_dir) / "labels" / "raw"
        firms_to_yolo_labels(
            firms_csv=args.firms_csv,
            image_dir=args.output_dir,
            label_dir=str(label_out),
        )

    else:
        build_dataset_from_folder(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            max_cloud=args.max_cloud,
            use_fire_composite=not args.rgb_only,
            apply_clahe_flag=not args.no_clahe,
        )
