"""
Visual Attention Heatmap Tool – Main Streamlit Application
===========================================================
AI-powered saliency & gaze prediction for UI/UX and marketing.

Run with:
    streamlit run app.py

Author  : Visual Attention Heatmap Tool
License : MIT
"""

import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.saliency import SaliencyEngine
from utils.visualization import VisualizationEngine
from utils.insights import InsightEngine
from utils.report_generator import ReportGenerator, REPORTLAB_AVAILABLE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
CSS_FILE = ASSETS_DIR / "style.css"

COLORMAPS = {
    "Jet": cv2.COLORMAP_JET,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Hot": cv2.COLORMAP_HOT,
    "Turbo": cv2.COLORMAP_TURBO,
    "Magma": cv2.COLORMAP_MAGMA,
    "Plasma": cv2.COLORMAP_PLASMA,
}

MAX_DIM = 1200  # max pixel dimension for processing


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Visual Attention Heatmap Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Inject custom CSS."""
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)


load_css()


# ---------------------------------------------------------------------------
# Cached engine initialisation
# ---------------------------------------------------------------------------
@st.cache_resource
def get_saliency_engine():
    return SaliencyEngine()


@st.cache_resource
def get_viz_engine():
    return VisualizationEngine()


@st.cache_resource
def get_insight_engine():
    return InsightEngine()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def load_image(uploaded_file) -> np.ndarray:
    """Load uploaded file into BGR ndarray, resize if too large."""
    pil = Image.open(uploaded_file).convert("RGB")
    img = np.array(pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR ndarray to PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def render_metric_card(icon, value, label, desc, color="#4F8BF9"):
    """Render a styled metric card using HTML."""
    st.markdown(f"""
    <div class="metric-card animate-in">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="background:linear-gradient(135deg,{color},#7EAAFF);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            {value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


def render_recommendation(rec):
    """Render a recommendation card."""
    icons = {"info": "✅", "warning": "⚠️", "critical": "🚨"}
    st.markdown(f"""
    <div class="rec-card severity-{rec.severity} animate-in">
        <div class="rec-header">
            <span class="rec-category">{rec.category}</span>
            <span>{icons.get(rec.severity, '•')}</span>
        </div>
        <div class="rec-message">{rec.message}</div>
        <div class="rec-details">{rec.details}</div>
    </div>""", unsafe_allow_html=True)


def render_progress_bar(value, max_val=100, color="#4F8BF9"):
    """Render a custom progress bar."""
    pct = min(value / max_val * 100, 100)
    st.markdown(f"""
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},
            {color}88);"></div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Build the sidebar UI. Returns dict of settings."""
    with st.sidebar:
        st.markdown("## 🔍 Attention Heatmap")
        st.markdown("---")

        # Upload
        st.markdown("### 📤 Upload Image")
        uploaded = st.file_uploader(
            "Upload an image, screenshot, or ad creative",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="uploader",
        )

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Analysis Settings")
        opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.55, 0.05, key="opacity")
        n_hotspots = st.slider("Number of Hotspots", 1, 10, 3, key="n_hotspots")
        colormap_name = st.selectbox("Heatmap Colormap", list(COLORMAPS.keys()), key="cmap")
        center_bias = st.checkbox("Apply Center Bias", True, key="center_bias")

        st.markdown("---")

        # Analysis mode
        st.markdown("### 🎯 Analysis Mode")
        mode = st.radio(
            "Select mode",
            ["Auto Detect", "UI/UX Audit", "Ad Creative", "General"],
            key="mode",
        )
        mode_map = {
            "Auto Detect": "auto",
            "UI/UX Audit": "ui_ux",
            "Ad Creative": "ad_creative",
            "General": "general",
        }

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Visual Attention Heatmap Tool** predicts where human eyes
            are most likely to look using multi-layer saliency modelling.

            **Models used:**
            - Deep Learning (if PyTorch available)
            - OpenCV Spectral Residual
            - Multi-scale Fusion (always available)

            Built for designers, marketers, and UX researchers.
            """)

    # Resolve image source
    image_source = uploaded

    return {
        "image_source": image_source,
        "opacity": opacity,
        "n_hotspots": n_hotspots,
        "colormap": COLORMAPS[colormap_name],
        "colormap_name": colormap_name,
        "center_bias": center_bias,
        "mode": mode_map[mode],
    }


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
def render_hero():
    """Render the hero header when no image is loaded."""
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>🔍 Visual Attention Heatmap Tool</h1>
        <p>AI-powered saliency & gaze prediction for UI/UX design and marketing.
        Upload any image to discover where human eyes look first.</p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    features = [
        ("🎯", "Fixation Prediction", "Ranked gaze hotspots"),
        ("🔥", "Heatmap Overlay", "Colour-coded attention maps"),
        ("📊", "Design Insights", "Actionable UX metrics"),
        ("📄", "PDF Reports", "Export professional reports"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            render_metric_card(icon, title, "", desc)


def run_analysis(image_bgr, settings):
    """Run the full saliency pipeline and return all results."""
    saliency_eng = get_saliency_engine()
    saliency_eng.center_bias = settings["center_bias"]

    viz_eng = get_viz_engine()
    insight_eng = get_insight_engine()

    # Compute saliency
    result = saliency_eng.compute(image_bgr, n_fixations=settings["n_hotspots"])

    # Generate visualisations
    heatmap = viz_eng.generate_heatmap_overlay(
        image_bgr, result.smoothed_saliency,
        opacity=settings["opacity"], colormap=settings["colormap"],
    )
    saliency_cmap = viz_eng.generate_saliency_colormap(result.smoothed_saliency)
    focus_map = viz_eng.generate_focus_intensity_map(image_bgr, result.smoothed_saliency)
    annotated = viz_eng.annotate_fixations(image_bgr, result.fixation_points)
    scanpath_frames = viz_eng.generate_scanpath_frames(image_bgr, result.fixation_points)

    # Generate insights
    report = insight_eng.analyse(
        image_bgr, result.smoothed_saliency,
        result.fixation_points, mode=settings["mode"],
    )

    return {
        "saliency_result": result,
        "heatmap": heatmap,
        "saliency_cmap": saliency_cmap,
        "focus_map": focus_map,
        "annotated": annotated,
        "scanpath_frames": scanpath_frames,
        "insight_report": report,
    }


def render_dashboard(image_bgr, settings):
    """Render the full analysis dashboard."""
    # Progress indicator
    progress = st.progress(0, text="Analysing visual attention...")
    t0 = time.time()

    results = run_analysis(image_bgr, settings)

    elapsed = time.time() - t0
    progress.progress(100, text=f"✅ Analysis complete in {elapsed:.1f}s")
    time.sleep(0.5)
    progress.empty()

    sal = results["saliency_result"]
    report = results["insight_report"]

    # Model badge
    model_labels = {
        "deep_learning": ("🧠 Deep Learning", "badge-good"),
        "spectral_residual": ("📡 Spectral Residual", "badge-good"),
        "custom_fusion": ("🔬 Multi-Scale Fusion", "badge-warning"),
    }
    m_label, m_class = model_labels.get(sal.model_used, ("Unknown", "badge-warning"))
    st.markdown(
        f'<span class="badge {m_class}">Model: {m_label}</span>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    tab_overview, tab_heatmap, tab_insights, tab_export = st.tabs([
        "📋 Overview", "🔥 Heatmap", "💡 Insights", "📄 Export Report",
    ])

    # ── TAB: Overview ──
    with tab_overview:
        st.markdown("### Original Image")
        st.image(bgr_to_pil(image_bgr), use_container_width=True)

        st.markdown("### 🎯 Attention Ranking Overlay")
        st.image(bgr_to_pil(results["annotated"]), use_container_width=True)

        # Fixation table
        if sal.fixation_points:
            st.markdown("### Fixation Points")
            for fp in sal.fixation_points:
                c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
                ordinal_icons = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣"}
                c1.markdown(f"**{ordinal_icons.get(fp.rank, f'#{fp.rank}')}**")
                c2.markdown(f"Position: ({fp.x}, {fp.y})")
                c3.markdown(f"Score: **{fp.score:.1f}%**")
                c4.markdown(f"Radius: {fp.radius}px")

        # Scanpath simulation
        st.markdown("### 👁️ Eye-Tracking Scanpath Simulation")
        frames = results["scanpath_frames"]
        if frames:
            frame_idx = st.slider(
                "Scanpath Frame", 0, len(frames) - 1, 0, key="scanpath_slider"
            )
            st.image(bgr_to_pil(frames[frame_idx]), use_container_width=True)

    # ── TAB: Heatmap ──
    with tab_heatmap:
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown("### Saliency Heatmap")
            st.image(bgr_to_pil(results["heatmap"]), use_container_width=True)
        with h_col2:
            st.markdown("### Saliency Map")
            st.image(bgr_to_pil(results["saliency_cmap"]), use_container_width=True)

        st.markdown("### Focus Intensity Map")
        st.image(bgr_to_pil(results["focus_map"]), use_container_width=True)

    # ── TAB: Insights ──
    with tab_insights:
        # Metric cards
        st.markdown("### 📊 Key Metrics")
        cols = st.columns(len(report.metric_cards))
        for col, card in zip(cols, report.metric_cards):
            with col:
                render_metric_card(card.icon, card.value, card.title, card.description, card.color)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Visual hierarchy score
        st.markdown("### 📈 Visual Hierarchy Score")
        score = report.visual_hierarchy_score
        score_color = "#10B981" if score > 65 else "#F59E0B" if score > 40 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card animate-in" style="text-align:center;">
            <div class="metric-value" style="font-size:3rem;background:linear-gradient(135deg,
                {score_color},{score_color}88);-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;background-clip:text;">
                {score:.0f}/100</div>
            <div class="metric-label">Overall Visual Hierarchy Quality</div>
        </div>""", unsafe_allow_html=True)
        render_progress_bar(score, 100, score_color)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Zone analysis
        if report.zone_scores:
            st.markdown("### 🗺️ Attention Zone Distribution")
            zone_cols = st.columns(len(report.zone_scores))
            for col, (zone, pct) in zip(zone_cols, report.zone_scores.items()):
                with col:
                    st.metric(zone.replace("_", " ").title(), f"{pct:.1f}%")

        # UI/UX Audit
        if report.ui_audit:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 🖥️ UI/UX Audit")
            audit_cols = st.columns(3)
            audit_items = [
                ("Navbar Attention", report.ui_audit.get("navbar", 0), "%"),
                ("Hero Dominance", report.ui_audit.get("hero", 0), "%"),
                ("CTA Discoverability", report.ui_audit.get("cta_discoverability", 0), "%"),
            ]
            for col, (label, val, unit) in zip(audit_cols, audit_items):
                with col:
                    st.metric(label, f"{val:.1f}{unit}")

        # Ad Creative Audit
        if report.ad_creative:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 📢 Ad Creative Audit")
            ad_cols = st.columns(3)
            ad_items = [
                ("Product Prominence", report.ad_creative.get("product_prominence", 0), "%"),
                ("Text Competition", report.ad_creative.get("text_competition", 0), "%"),
                ("Face Attention", report.ad_creative.get("face_attention_pct", 0), "%"),
            ]
            for col, (label, val, unit) in zip(ad_cols, ad_items):
                with col:
                    st.metric(label, f"{val:.1f}{unit}")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Recommendations
        st.markdown("### 💡 Designer & Marketer Recommendations")
        for rec in report.recommendations:
            render_recommendation(rec)

    # ── TAB: Export ──
    with tab_export:
        st.markdown("### 📥 Download Results")

        dl_cols = st.columns(3)
        with dl_cols[0]:
            st.download_button(
                "⬇️ Original Image",
                pil_to_bytes(bgr_to_pil(image_bgr)),
                "original.png", "image/png",
            )
        with dl_cols[1]:
            st.download_button(
                "⬇️ Heatmap Overlay",
                pil_to_bytes(bgr_to_pil(results["heatmap"])),
                "heatmap.png", "image/png",
            )
        with dl_cols[2]:
            st.download_button(
                "⬇️ Annotated Image",
                pil_to_bytes(bgr_to_pil(results["annotated"])),
                "annotated.png", "image/png",
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # PDF report
        st.markdown("### 📄 PDF Report")
        if REPORTLAB_AVAILABLE:
            if st.button("🔄 Generate PDF Report", key="gen_pdf"):
                with st.spinner("Generating PDF report..."):
                    gen = ReportGenerator()
                    pdf_bytes = gen.generate(
                        original_image=image_bgr,
                        heatmap_image=results["heatmap"],
                        annotated_image=results["annotated"],
                        fixations=sal.fixation_points,
                        report=report,
                    )
                st.download_button(
                    "⬇️ Download PDF Report",
                    pdf_bytes,
                    "attention_report.pdf",
                    "application/pdf",
                    key="dl_pdf",
                )
                st.success("PDF report generated successfully!")
        else:
            st.warning("Install `reportlab` for PDF export: `pip install reportlab`")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Saliency & focus map downloads
        st.markdown("### Additional Exports")
        dl2 = st.columns(2)
        with dl2[0]:
            st.download_button(
                "⬇️ Saliency Map",
                pil_to_bytes(bgr_to_pil(results["saliency_cmap"])),
                "saliency_map.png", "image/png",
            )
        with dl2[1]:
            st.download_button(
                "⬇️ Focus Intensity Map",
                pil_to_bytes(bgr_to_pil(results["focus_map"])),
                "focus_intensity.png", "image/png",
            )


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def render_batch_mode():
    """Batch analysis of multiple images."""
    st.markdown("### 📦 Batch Image Analysis")
    batch_files = st.file_uploader(
        "Upload multiple images for batch analysis",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if batch_files:
        saliency_eng = get_saliency_engine()
        viz_eng = get_viz_engine()
        insight_eng = get_insight_engine()

        for i, f in enumerate(batch_files):
            with st.expander(f"📷 {f.name}", expanded=(i == 0)):
                img = load_image(f)
                result = saliency_eng.compute(img, n_fixations=3)
                heatmap = viz_eng.generate_heatmap_overlay(img, result.smoothed_saliency)
                report = insight_eng.analyse(img, result.smoothed_saliency, result.fixation_points)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(bgr_to_pil(img), caption="Original", use_container_width=True)
                with c2:
                    st.image(bgr_to_pil(heatmap), caption="Heatmap", use_container_width=True)

                st.markdown(f"**Visual Hierarchy Score:** {report.visual_hierarchy_score:.0f}/100")
                st.markdown(f"**Primary Focus:** {report.primary_focus_strength:.0f}%")


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------
def render_ab_comparison():
    """A/B design comparison mode."""
    st.markdown("### 🔄 A/B Design Comparison")
    c1, c2 = st.columns(2)
    with c1:
        file_a = st.file_uploader("Design A", type=["png", "jpg", "jpeg", "webp"], key="ab_a")
    with c2:
        file_b = st.file_uploader("Design B", type=["png", "jpg", "jpeg", "webp"], key="ab_b")

    if file_a and file_b:
        saliency_eng = get_saliency_engine()
        viz_eng = get_viz_engine()
        insight_eng = get_insight_engine()

        img_a, img_b = load_image(file_a), load_image(file_b)
        res_a = saliency_eng.compute(img_a, 3)
        res_b = saliency_eng.compute(img_b, 3)

        heat_a = viz_eng.generate_heatmap_overlay(img_a, res_a.smoothed_saliency)
        heat_b = viz_eng.generate_heatmap_overlay(img_b, res_b.smoothed_saliency)

        rep_a = insight_eng.analyse(img_a, res_a.smoothed_saliency, res_a.fixation_points)
        rep_b = insight_eng.analyse(img_b, res_b.smoothed_saliency, res_b.fixation_points)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Design A")
            st.image(bgr_to_pil(heat_a), use_container_width=True)
            st.metric("Hierarchy Score", f"{rep_a.visual_hierarchy_score:.0f}/100")
            st.metric("Focus Strength", f"{rep_a.primary_focus_strength:.0f}%")
        with col2:
            st.markdown("#### Design B")
            st.image(bgr_to_pil(heat_b), use_container_width=True)
            st.metric("Hierarchy Score", f"{rep_b.visual_hierarchy_score:.0f}/100")
            st.metric("Focus Strength", f"{rep_b.primary_focus_strength:.0f}%")

        # Winner
        winner = "A" if rep_a.visual_hierarchy_score >= rep_b.visual_hierarchy_score else "B"
        diff = abs(rep_a.visual_hierarchy_score - rep_b.visual_hierarchy_score)
        st.success(f"🏆 Design **{winner}** has a stronger visual hierarchy (+{diff:.0f} points)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    settings = render_sidebar()

    # Main page mode selector
    page_mode = st.radio(
        "",
        ["🔍 Single Analysis", "📦 Batch Mode", "🔄 A/B Comparison"],
        horizontal=True,
        key="page_mode",
        label_visibility="collapsed",
    )

    if page_mode == "📦 Batch Mode":
        render_batch_mode()
        return

    if page_mode == "🔄 A/B Comparison":
        render_ab_comparison()
        return

    # Single analysis mode
    if settings["image_source"] is None:
        render_hero()
        return

    # Load image
    try:
        image_bgr = load_image(settings["image_source"])
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return

    render_dashboard(image_bgr, settings)


if __name__ == "__main__":
    main()
