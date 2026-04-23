"""
main.py
=======
Streamlit dashboard for the Early Forest Fire Detection System.

Features:
  - Upload satellite image (JPG / PNG / GeoTIFF / TIFF)
  - Run YOLOv8 + SAHI inference
  - Interactive detection visualisation
  - Severity gauge + burn area metrics
  - Heatmap overlay
  - Downloadable JSON / CSV report
  - GeoJSON map view (folium)

Run
---
    streamlit run main.py
    # or
    python main.py   (launches streamlit programmatically)
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Bootstrap path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

try:
    import streamlit as st
except ImportError:
    print("streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from utils import (
    compute_severity,
    estimate_burn_area,
    generate_heatmap,
    get_device,
    get_geotiff_metadata,
    save_results_csv,
    save_results_json,
    setup_logger,
    timestamp_str,
)
from infer import (
    CLASS_NAMES,
    CLASS_COLORS,
    draw_detections,
    enrich_with_geocoords,
    load_sahi_model,
    load_yolo_model,
    process_image,
    run_sahi_inference,
    run_yolo_inference,
    soft_nms,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Forest Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

    :root {
        --fire-orange: #FF5E1A;
        --fire-red:    #D91A1A;
        --smoke-gray:  #8E9BAF;
        --burn-brown:  #7C4A1E;
        --safe-green:  #1ACE74;
        --dark-bg:     #0D1117;
        --card-bg:     #161B22;
        --border:      #30363D;
    }

    html, body, .main { background-color: var(--dark-bg) !important; color: #E6EDF3; }

    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; }
    .stMetricValue { font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; }

    .fire-header {
        background: linear-gradient(135deg, #1A0500 0%, #3D0F00 50%, #1A0500 100%);
        border: 1px solid var(--fire-orange);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }

    .severity-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sev-none     { background: #1A2A1A; color: #1ACE74; border: 1px solid #1ACE74; }
    .sev-low      { background: #2A2A1A; color: #F0C040; border: 1px solid #F0C040; }
    .sev-moderate { background: #2A1A0A; color: #FF9A3C; border: 1px solid #FF9A3C; }
    .sev-high     { background: #2A0A0A; color: #FF5E1A; border: 1px solid #FF5E1A; }
    .sev-extreme  { background: #1A0000; color: #D91A1A; border: 1px solid #D91A1A; }

    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.3rem 0;
    }

    .detection-item {
        background: var(--card-bg);
        border-left: 3px solid var(--fire-orange);
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        border-radius: 0 6px 6px 0;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--fire-orange), var(--fire-red));
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 1px;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    .stFileUploader { border: 1px dashed var(--fire-orange) !important; border-radius: 8px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_cached_model(weights: str, use_sahi: bool, device: str):
    """Cache model across Streamlit reruns."""
    if use_sahi:
        return {"sahi": load_sahi_model(weights, device), "yolo": None}
    else:
        return {"sahi": None, "yolo": load_yolo_model(weights, device)}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()

        weights = st.text_input(
            "Model weights path",
            value="runs/train/fire_det/weights/best.pt",
            help="Path to trained YOLOv8 .pt file",
        )

        use_sahi = st.toggle("Use SAHI sliced inference", value=True,
                             help="Recommended for large satellite images")

        st.markdown("#### Detection parameters")
        conf     = st.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05)
        iou      = st.slider("IoU / NMS threshold",  0.10, 0.90, 0.45, 0.05)

        st.markdown("#### SAHI settings (if enabled)")
        slice_sz = st.selectbox("Slice size (px)", [512, 640, 1024], index=1)
        overlap  = st.slider("Overlap ratio", 0.0, 0.5, 0.2, 0.05)

        st.markdown("#### Output")
        save_hm  = st.toggle("Generate heatmap", value=True)
        show_lbl = st.toggle("Show labels on image", value=True)

        st.divider()
        device = get_device()
        st.caption(f"🖥  Device: **{device.upper()}**")
        st.caption("YOLOv8 + SAHI | Satellite Fire Detection")

    return {
        "weights":   weights,
        "use_sahi":  use_sahi,
        "conf":      conf,
        "iou":       iou,
        "slice_sz":  slice_sz,
        "overlap":   overlap,
        "save_hm":   save_hm,
        "show_lbl":  show_lbl,
        "device":    device,
    }


# ─────────────────────────────────────────────────────────────
# SEVERITY BADGE
# ─────────────────────────────────────────────────────────────

SEV_CLASS = {
    "NONE":     "sev-none",
    "LOW":      "sev-low",
    "MODERATE": "sev-moderate",
    "HIGH":     "sev-high",
    "EXTREME":  "sev-extreme",
    "N/A":      "sev-none",
}

def severity_badge(level: str) -> str:
    cls = SEV_CLASS.get(level.upper(), "sev-none")
    return f'<span class="severity-badge {cls}">{level}</span>'


# ─────────────────────────────────────────────────────────────
# MAP VIEW
# ─────────────────────────────────────────────────────────────

def render_map(detections: list) -> None:
    """Render folium map of detections with lat/lon."""
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.warning("Install streamlit-folium for map view: pip install streamlit-folium folium")
        return

    geo_dets = [d for d in detections if d.get("lat") and d.get("lon")]
    if not geo_dets:
        st.info("No geographic coordinates available. Image must be a georeferenced GeoTIFF.")
        return

    lats = [d["lat"] for d in geo_dets]
    lons = [d["lon"] for d in geo_dets]
    center = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB dark_matter")

    color_map = {0: "red", 1: "gray", 2: "orange"}
    icon_map  = {0: "fire", 1: "cloud", 2: "leaf"}

    for det in geo_dets:
        cls_id = det.get("class", 0)
        folium.CircleMarker(
            location=[det["lat"], det["lon"]],
            radius=8,
            color=color_map.get(cls_id, "red"),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{det.get('class_name','?')}</b><br>"
                f"Confidence: {det.get('score', 0):.2%}<br>"
                f"Lat: {det['lat']:.5f}<br>Lon: {det['lon']:.5f}",
                max_width=200,
            ),
        ).add_to(m)

    st_folium(m, width=700, height=400)


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────

def main():
    # ── Header ─────────────────────────────────────────────
    st.markdown("""
    <div class="fire-header">
        <h1 style="margin:0; color: #FF5E1A; font-size: 2rem;">🛰 Early Forest Fire Detection System</h1>
        <p style="margin: 0.5rem 0 0; color: #8E9BAF; font-family: 'Space Mono', monospace; font-size: 0.85rem;">
            YOLOv8 + SAHI Sliced Inference  ·  Sentinel-2 / MODIS / VIIRS / Landsat
        </p>
    </div>
    """, unsafe_allow_html=True)

    cfg = render_sidebar()

    # ── Upload ─────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload satellite image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Supports Sentinel-2 GeoTIFF, MODIS, VIIRS, plain JPG/PNG",
        )

    with col_info:
        st.markdown("""
        <div class="metric-card">
        <b>Supported formats</b><br>
        🛰 GeoTIFF (multispectral)<br>
        📷 JPG / PNG (RGB)<br><br>
        <b>Best sources</b><br>
        • Sentinel-2 (ESA Copernicus)<br>
        • MODIS Terra/Aqua<br>
        • VIIRS NOAA/NASA<br>
        • Landsat 8/9
        </div>
        """, unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center; padding: 4rem; color: #555; font-family: 'Space Mono', monospace;">
            ↑ Upload a satellite image to begin detection
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Save upload to temp file ────────────────────────────
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── Show original image ─────────────────────────────────
    st.subheader("📡 Uploaded Image")
    img_bytes = open(tmp_path, "rb").read()
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image(img_bytes, caption=f"Original: {uploaded.name}", use_container_width=True)
        except Exception:
            st.warning("Preview not available for this image format.")

    # ── Run detection ───────────────────────────────────────
    run_btn = st.button("🔥  Run Fire Detection", use_container_width=False)

    if not run_btn:
        with col2:
            st.markdown("""
            <div style="height:300px; display:flex; align-items:center; justify-content:center;
                        border: 1px dashed #30363D; border-radius: 8px; color: #555;
                        font-family: 'Space Mono', monospace;">
                Detection results will appear here
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Load model ──────────────────────────────────────────
    with st.spinner("Loading model…"):
        try:
            models = get_cached_model(cfg["weights"], cfg["use_sahi"], cfg["device"])
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            st.info("Make sure you have trained a model or the weights file exists.")
            return

    # ── Inference ───────────────────────────────────────────
    with st.spinner("🛰 Running inference with YOLOv8" + (" + SAHI…" if cfg["use_sahi"] else "…")):
        import time
        t0 = time.time()

        if cfg["use_sahi"] and models["sahi"]:
            detections = run_sahi_inference(
                sahi_model=models["sahi"],
                image_path=tmp_path,
                conf=cfg["conf"],
                iou_thresh=cfg["iou"],
                slice_height=cfg["slice_sz"],
                slice_width=cfg["slice_sz"],
                overlap_h=cfg["overlap"],
                overlap_w=cfg["overlap"],
            )
        elif models["yolo"]:
            detections = run_yolo_inference(
                yolo_model=models["yolo"],
                image_path=tmp_path,
                conf=cfg["conf"],
                iou_thresh=cfg["iou"],
                device=cfg["device"],
            )
        else:
            st.error("No model loaded.")
            return

        elapsed = time.time() - t0

    # ── Post-processing ─────────────────────────────────────
    detections = soft_nms(detections, score_threshold=0.15)
    detections = enrich_with_geocoords(detections, tmp_path)
    severity   = compute_severity(detections)

    # ── Read image for drawing ──────────────────────────────
    img_cv = cv2.imread(tmp_path)
    if img_cv is None:
        # GeoTIFF fallback
        try:
            from preprocess import SatelliteImageReader, make_fire_composite, apply_clahe
            reader = SatelliteImageReader(tmp_path)
            composite = make_fire_composite(reader)
            composite = apply_clahe(composite)
            img_cv = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            st.error(f"Cannot render image: {exc}")
            return

    h, w = img_cv.shape[:2]
    burn_area = estimate_burn_area(detections, w, h)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    annotated = draw_detections(img_rgb, detections, show_labels=cfg["show_lbl"])

    with col2:
        st.image(annotated, caption="Detection result", use_container_width=True)

    # ── Metrics row ─────────────────────────────────────────
    st.divider()
    st.subheader("📊 Detection Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    sev = severity["severity_level"]

    m1.metric("🔥 Hotspots",    severity.get("hotspot_count", 0))
    m2.metric("💨 Smoke",       severity.get("smoke_count",   0))
    m3.metric("🪨 Burn Regions",severity.get("burn_count",    0))
    m4.metric("📡 Confidence",  f"{severity.get('avg_confidence', 0):.1%}")
    m5.metric("⏱ Inference",   f"{elapsed:.2f}s")

    # Severity badge
    st.markdown(
        f"**Severity Level:** {severity_badge(sev)}  "
        f"&nbsp;&nbsp; **Score:** `{severity.get('severity_score', 0)}/100`  "
        f"&nbsp;&nbsp; **Spread Risk:** `{severity.get('fire_spread_risk', 0):.0%}`",
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────
    tab_det, tab_heat, tab_map, tab_report = st.tabs(
        ["🎯 Detections", "🌡 Heatmap", "🗺 Map View", "📋 Report"]
    )

    with tab_det:
        if not detections:
            st.success("✅ No fire detected in this image.")
        else:
            st.markdown(f"**{len(detections)} detection(s) found:**")
            for i, det in enumerate(detections, 1):
                x1,y1,x2,y2 = [int(v) for v in det["bbox"]]
                lat = det.get("lat", 0)
                lon = det.get("lon", 0)
                geo_str = f" | Lat {lat:.4f}, Lon {lon:.4f}" if lat else ""
                st.markdown(
                    f'<div class="detection-item">'
                    f'#{i} &nbsp; <b>{det["class_name"].upper()}</b> &nbsp; '
                    f'conf={det["score"]:.2%} &nbsp; '
                    f'bbox=[{x1},{y1},{x2},{y2}]'
                    f'{geo_str}</div>',
                    unsafe_allow_html=True,
                )

    with tab_heat:
        if detections:
            hm = generate_heatmap(img_rgb, detections, alpha=0.55)
            st.image(hm, caption="Fire probability heatmap", use_container_width=True)
        else:
            st.info("No detections – heatmap not available.")

    with tab_map:
        render_map(detections)

    with tab_report:
        # Burn area
        st.markdown("#### 🌲 Burn Area Estimate")
        ba_cols = st.columns(3)
        ba_cols[0].metric("Burn fraction",   f"{burn_area['burn_fraction']:.4%}")
        if "burn_area_ha" in burn_area:
            ba_cols[1].metric("Burn area (ha)", f"{burn_area['burn_area_ha']:.2f}")
            ba_cols[2].metric("Burn area (km²)", f"{burn_area['burn_area_km2']:.4f}")

        # Download report
        st.markdown("#### 📥 Download")
        report = {
            "image":         uploaded.name,
            "timestamp":     timestamp_str(),
            "model":         cfg["weights"],
            "use_sahi":      cfg["use_sahi"],
            "confidence":    cfg["conf"],
            "inference_time_s": round(elapsed, 3),
            "num_detections": len(detections),
            "severity":      severity,
            "burn_area":     burn_area,
            "detections":    detections,
        }
        json_str = json.dumps(report, indent=2, default=str)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇ Download JSON Report",
            data=json_str,
            file_name=f"fire_report_{timestamp_str()}.json",
            mime="application/json",
        )

        # CSV
        csv_buf = io.StringIO()
        import csv as csv_mod
        writer = csv_mod.DictWriter(
            csv_buf,
            fieldnames=["class_name", "confidence", "x1", "y1", "x2", "y2", "lat", "lon"],
        )
        writer.writeheader()
        for det in detections:
            x1, y1, x2, y2 = det.get("bbox", [0,0,0,0])
            writer.writerow({
                "class_name":  det.get("class_name"),
                "confidence":  round(det.get("score", 0), 4),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "lat": det.get("lat", ""),
                "lon": det.get("lon", ""),
            })
        dl2.download_button(
            "⬇ Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"detections_{timestamp_str()}.csv",
            mime="text/csv",
        )

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess

    # Check if already inside streamlit
    if os.environ.get("STREAMLIT_SERVER_PORT"):
        # Running inside streamlit context – just call main()
        main()
    else:
        # Launch streamlit from CLI
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", __file__,
             "--server.port", "8501",
             "--server.headless", "false"],
            check=True,
        )
else:
    # Called by streamlit run main.py
    main()
