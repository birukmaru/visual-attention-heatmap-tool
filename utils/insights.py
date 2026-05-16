"""
Insight Engine – Design & Marketing Intelligence
==================================================

Analyses saliency data and image properties to generate actionable
insights for UI/UX designers, marketers, and creative teams.

Insight categories:
    1. Visual Hierarchy Score (0–100)
    2. CTA Visibility Assessment
    3. Banner Blindness Risk
    4. Attention Distribution Analysis
    5. UI/UX Audit Mode (navbar, hero, CTA zones)
    6. Ad Creative Mode (product prominence, face bias, text competition)
    7. Designer / Marketer Recommendations

Author  : Visual Attention Heatmap Tool
License : MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from utils.saliency import FixationPoint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricCard:
    """A single dashboard metric card."""
    title: str
    value: str
    description: str
    icon: str               # Emoji icon
    color: str = "#4F8BF9"  # Accent colour (hex)
    trend: str = ""         # e.g. "good", "warning", "critical"


@dataclass
class Recommendation:
    """Actionable design/marketing recommendation."""
    category: str       # "UX", "Marketing", "Visual", "Content"
    severity: str       # "info", "warning", "critical"
    message: str
    details: str = ""


@dataclass
class InsightReport:
    """Complete insight report for a single image analysis."""
    # Core metrics
    visual_hierarchy_score: float = 0.0
    primary_focus_strength: float = 0.0
    distraction_score: float = 0.0
    cta_visibility: float = 0.0
    attention_spread: float = 0.0
    contrast_score: float = 0.0

    # Zone analysis
    zone_scores: Dict[str, float] = field(default_factory=dict)

    # Cards for dashboard display
    metric_cards: List[MetricCard] = field(default_factory=list)

    # Recommendations
    recommendations: List[Recommendation] = field(default_factory=list)

    # Mode-specific data
    ui_audit: Optional[Dict[str, float]] = None
    ad_creative: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Insight Engine
# ---------------------------------------------------------------------------

class InsightEngine:
    """
    Generates design and marketing insights from saliency analysis.

    The engine divides the image into semantic zones (top, centre, bottom,
    thirds grid) and measures attention distribution across each zone
    to produce actionable metrics.
    """

    # Zone definitions as (y_start%, y_end%, x_start%, x_end%) of image
    UI_ZONES = {
        "navbar":     (0.00, 0.12, 0.00, 1.00),
        "hero":       (0.12, 0.50, 0.00, 1.00),
        "mid_section":(0.50, 0.80, 0.00, 1.00),
        "footer":     (0.80, 1.00, 0.00, 1.00),
    }

    AD_ZONES = {
        "top_third":    (0.00, 0.33, 0.00, 1.00),
        "center_third": (0.33, 0.67, 0.00, 1.00),
        "bottom_third": (0.67, 1.00, 0.00, 1.00),
        "left_half":    (0.00, 1.00, 0.00, 0.50),
        "right_half":   (0.00, 1.00, 0.50, 1.00),
    }

    # CTA-like region (lower-center area where buttons typically appear)
    CTA_ZONE = (0.55, 0.85, 0.25, 0.75)

    def analyse(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        fixations: List[FixationPoint],
        mode: str = "auto",
    ) -> InsightReport:
        """
        Run full insight analysis.

        Parameters
        ----------
        image : np.ndarray
            Original image (BGR uint8).
        saliency : np.ndarray
            Smoothed saliency map (float32, [0,1]).
        fixations : list of FixationPoint
            Ranked fixation points.
        mode : str
            ``"auto"`` | ``"ui_ux"`` | ``"ad_creative"`` | ``"general"``
        """
        h, w = saliency.shape[:2]
        report = InsightReport()

        # --- Core metrics ---
        report.primary_focus_strength = self._primary_focus_strength(saliency, fixations)
        report.distraction_score = self._distraction_score(saliency, fixations)
        report.attention_spread = self._attention_spread(saliency)
        report.contrast_score = self._contrast_score(image)
        report.cta_visibility = self._zone_attention(saliency, self.CTA_ZONE)
        report.visual_hierarchy_score = self._visual_hierarchy_score(
            report.primary_focus_strength,
            report.distraction_score,
            report.attention_spread,
            report.contrast_score,
        )

        # --- Zone scores ---
        for name, zone in self.UI_ZONES.items():
            report.zone_scores[name] = round(self._zone_attention(saliency, zone), 1)

        # --- Build metric cards ---
        report.metric_cards = self._build_metric_cards(report)

        # --- Mode-specific analysis ---
        if mode == "auto":
            mode = self._detect_mode(image, saliency)

        if mode == "ui_ux":
            report.ui_audit = self._ui_ux_audit(saliency)
        elif mode == "ad_creative":
            report.ad_creative = self._ad_creative_audit(image, saliency)

        # --- Recommendations ---
        report.recommendations = self._generate_recommendations(report, fixations, mode)

        return report

    # ------------------------------------------------------------------
    # Core metric computations
    # ------------------------------------------------------------------

    def _primary_focus_strength(
        self,
        saliency: np.ndarray,
        fixations: List[FixationPoint],
    ) -> float:
        """
        How strong is the primary attention anchor?
        High value = clear focal point. Low = ambiguous attention.
        Returns 0–100.
        """
        if not fixations:
            return 0.0
        top = fixations[0]
        h, w = saliency.shape[:2]
        # Measure saliency concentration around top fixation
        mask = np.zeros_like(saliency)
        radius = max(int(min(h, w) * 0.1), 30)
        cv2.circle(mask, (top.x, top.y), radius, 1.0, -1)
        region_sum = (saliency * mask).sum()
        total_sum = saliency.sum() + 1e-8
        concentration = region_sum / total_sum
        return round(min(concentration * 300, 100.0), 1)

    def _distraction_score(
        self,
        saliency: np.ndarray,
        fixations: List[FixationPoint],
    ) -> float:
        """
        Measures visual noise / competing elements.
        High value = many distracting elements. Low = clean focus.
        Returns 0–100.
        """
        # Count high-saliency regions
        threshold = 0.4
        binary = (saliency > threshold).astype(np.float32)
        labeled, n_regions = ndimage.label(binary)
        # Many small hot regions = distraction
        if n_regions <= 1:
            return 5.0
        elif n_regions <= 3:
            return 25.0
        elif n_regions <= 6:
            return 50.0
        elif n_regions <= 10:
            return 70.0
        else:
            return min(90.0, 50.0 + n_regions * 3)

    def _attention_spread(self, saliency: np.ndarray) -> float:
        """
        Entropy-like measure of how distributed attention is.
        Low = focused. High = spread out.
        Returns 0–100.
        """
        # Flatten and treat as probability distribution
        flat = saliency.flatten().astype(np.float64)
        flat = flat / (flat.sum() + 1e-8)
        # Compute entropy
        flat = flat[flat > 1e-10]
        entropy = -np.sum(flat * np.log2(flat))
        max_entropy = np.log2(len(saliency.flatten()))
        normalised = (entropy / max_entropy) * 100
        return round(normalised, 1)

    def _contrast_score(self, image: np.ndarray) -> float:
        """
        Overall contrast quality of the image.
        Returns 0–100.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Standard deviation of pixel intensities
        std = gray.std()
        # Map to 0–100 (std of 60+ is very high contrast)
        return round(min((std / 60.0) * 100, 100.0), 1)

    def _visual_hierarchy_score(
        self,
        focus_strength: float,
        distraction: float,
        spread: float,
        contrast: float,
    ) -> float:
        """
        Composite Visual Hierarchy Score (0–100).
        High = clear, well-structured visual hierarchy.
        """
        score = (
            focus_strength * 0.35
            + (100 - distraction) * 0.25
            + (100 - spread) * 0.20
            + contrast * 0.20
        )
        return round(max(0, min(100, score)), 1)

    # ------------------------------------------------------------------
    # Zone analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _zone_attention(
        saliency: np.ndarray,
        zone: Tuple[float, float, float, float],
    ) -> float:
        """
        Compute percentage of total saliency within a zone.
        Zone is defined as (y_start%, y_end%, x_start%, x_end%).
        """
        h, w = saliency.shape[:2]
        y1 = int(zone[0] * h)
        y2 = int(zone[1] * h)
        x1 = int(zone[2] * w)
        x2 = int(zone[3] * w)
        region = saliency[y1:y2, x1:x2]
        total = saliency.sum() + 1e-8
        return (region.sum() / total) * 100

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    def _detect_mode(self, image: np.ndarray, saliency: np.ndarray) -> str:
        """
        Heuristic mode detection based on image aspect ratio and
        content characteristics.
        """
        h, w = image.shape[:2]
        aspect = w / h

        # Landscape with certain aspect ratios suggest web UI
        if 1.2 < aspect < 2.5 and h > 400:
            return "ui_ux"
        # Portrait or square with high contrast suggests ad creative
        elif aspect < 1.0 or (0.8 < aspect < 1.2):
            return "ad_creative"
        return "general"

    # ------------------------------------------------------------------
    # UI/UX Audit
    # ------------------------------------------------------------------

    def _ui_ux_audit(self, saliency: np.ndarray) -> Dict[str, float]:
        """
        Compute UI/UX zone attention percentages.
        """
        audit = {}
        for name, zone in self.UI_ZONES.items():
            audit[name] = round(self._zone_attention(saliency, zone), 1)

        # CTA discoverability
        audit["cta_discoverability"] = round(
            self._zone_attention(saliency, self.CTA_ZONE), 1
        )
        return audit

    # ------------------------------------------------------------------
    # Ad Creative Audit
    # ------------------------------------------------------------------

    def _ad_creative_audit(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
    ) -> Dict[str, float]:
        """
        Ad-specific metrics: product prominence, text competition, etc.
        """
        audit = {}
        for name, zone in self.AD_ZONES.items():
            audit[name] = round(self._zone_attention(saliency, zone), 1)

        # Product prominence (center region attention)
        product_zone = (0.20, 0.80, 0.20, 0.80)
        audit["product_prominence"] = round(
            self._zone_attention(saliency, product_zone), 1
        )

        # Text competition: measure edge density (text = many edges)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0
        audit["text_competition"] = round(edge_density * 100, 1)

        # Face bias: detect faces using Haar cascades
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            audit["face_detected"] = len(faces) > 0
            if len(faces) > 0:
                # Check what % of attention is on faces
                face_mask = np.zeros_like(saliency)
                for (x, y, fw, fh) in faces:
                    face_mask[y:y+fh, x:x+fw] = 1.0
                face_attention = (saliency * face_mask).sum() / (saliency.sum() + 1e-8) * 100
                audit["face_attention_pct"] = round(face_attention, 1)
            else:
                audit["face_attention_pct"] = 0.0
        except Exception:
            audit["face_detected"] = False
            audit["face_attention_pct"] = 0.0

        return audit

    # ------------------------------------------------------------------
    # Metric cards builder
    # ------------------------------------------------------------------

    def _build_metric_cards(self, report: InsightReport) -> List[MetricCard]:
        """Build dashboard metric cards from computed report values."""
        cards = []

        # Primary Focus Strength
        trend = "good" if report.primary_focus_strength > 60 else (
            "warning" if report.primary_focus_strength > 35 else "critical"
        )
        cards.append(MetricCard(
            title="Primary Focus Strength",
            value=f"{report.primary_focus_strength:.0f}%",
            description="How strongly the main element captures attention",
            icon="🎯",
            color="#10B981" if trend == "good" else "#F59E0B" if trend == "warning" else "#EF4444",
            trend=trend,
        ))

        # Distraction Score
        trend = "good" if report.distraction_score < 30 else (
            "warning" if report.distraction_score < 60 else "critical"
        )
        cards.append(MetricCard(
            title="Distraction Score",
            value=f"{report.distraction_score:.0f}%",
            description="Level of visual noise competing for attention",
            icon="🔀",
            color="#10B981" if trend == "good" else "#F59E0B" if trend == "warning" else "#EF4444",
            trend=trend,
        ))

        # Visual Hierarchy Quality
        trend = "good" if report.visual_hierarchy_score > 65 else (
            "warning" if report.visual_hierarchy_score > 40 else "critical"
        )
        cards.append(MetricCard(
            title="Visual Hierarchy",
            value=f"{report.visual_hierarchy_score:.0f}/100",
            description="Overall quality of the visual structure and flow",
            icon="📊",
            color="#10B981" if trend == "good" else "#F59E0B" if trend == "warning" else "#EF4444",
            trend=trend,
        ))

        # CTA Visibility
        trend = "good" if report.cta_visibility > 15 else (
            "warning" if report.cta_visibility > 8 else "critical"
        )
        cards.append(MetricCard(
            title="CTA Visibility",
            value=f"{report.cta_visibility:.0f}%",
            description="Attention directed to call-to-action region",
            icon="👁️",
            color="#10B981" if trend == "good" else "#F59E0B" if trend == "warning" else "#EF4444",
            trend=trend,
        ))

        return cards

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        report: InsightReport,
        fixations: List[FixationPoint],
        mode: str,
    ) -> List[Recommendation]:
        """Generate actionable recommendations based on analysis."""
        recs = []

        # --- Focus Strength ---
        if report.primary_focus_strength < 40:
            recs.append(Recommendation(
                category="Visual",
                severity="warning",
                message="Weak primary focal point detected",
                details=(
                    "The main visual element does not strongly anchor attention. "
                    "Consider increasing its size, contrast, or adding whitespace "
                    "around it to create a clearer visual hierarchy."
                ),
            ))
        elif report.primary_focus_strength > 75:
            recs.append(Recommendation(
                category="Visual",
                severity="info",
                message="Strong focal anchor – excellent attention capture",
                details=(
                    "The primary element dominates visual attention effectively. "
                    "Ensure secondary information (e.g., sub-headlines, CTAs) "
                    "still receives adequate visibility."
                ),
            ))

        # --- Distraction ---
        if report.distraction_score > 60:
            recs.append(Recommendation(
                category="UX",
                severity="critical",
                message="High visual noise – too many competing elements",
                details=(
                    "Multiple hot-spots are competing for attention, creating "
                    "cognitive overload. Simplify the layout, reduce the number "
                    "of visual elements, and group related content."
                ),
            ))
        elif report.distraction_score > 35:
            recs.append(Recommendation(
                category="UX",
                severity="warning",
                message="Moderate distraction – consider simplifying",
                details=(
                    "There are several secondary attention points that may "
                    "dilute the primary message. Review element placement "
                    "and visual weight distribution."
                ),
            ))

        # --- CTA Visibility ---
        if report.cta_visibility < 8:
            recs.append(Recommendation(
                category="Marketing",
                severity="critical",
                message="CTA button is visually weak – low discoverability",
                details=(
                    "Very little attention is directed to the typical CTA region "
                    "(lower-centre). Increase button size, use contrasting colours, "
                    "or reposition the CTA near the primary focal point."
                ),
            ))
        elif report.cta_visibility < 15:
            recs.append(Recommendation(
                category="Marketing",
                severity="warning",
                message="CTA visibility is moderate – room for improvement",
                details=(
                    "The call-to-action region receives some attention but could "
                    "benefit from improved visual prominence through colour "
                    "contrast, size increase, or strategic whitespace."
                ),
            ))

        # --- Attention Spread ---
        if report.attention_spread > 80:
            recs.append(Recommendation(
                category="Content",
                severity="warning",
                message="Text overload reduces focus – attention is too diffuse",
                details=(
                    "Attention is spread very evenly across the image, suggesting "
                    "lack of visual hierarchy. Use larger headings, bold imagery, "
                    "and strategic whitespace to guide the eye."
                ),
            ))

        # --- Visual Hierarchy ---
        if report.visual_hierarchy_score > 70:
            recs.append(Recommendation(
                category="Visual",
                severity="info",
                message="Strong visual hierarchy – well-structured layout",
                details=(
                    "The design has a clear visual flow with well-differentiated "
                    "importance levels. Maintain this balance in future iterations."
                ),
            ))

        # --- Fixation alignment ---
        if fixations and len(fixations) >= 2:
            top = fixations[0]
            h_img = report.zone_scores.get("hero", 0)
            # Check if top fixation is in an unexpected region
            img_h = 1000  # Approximate; proportional check
            if top.score < 60:
                recs.append(Recommendation(
                    category="UX",
                    severity="warning",
                    message="Top attention is misaligned from the primary content area",
                    details=(
                        "The strongest fixation point may not align with the "
                        "intended hero or primary content zone. Review element "
                        "placement and ensure the key message is in the "
                        "visual path."
                    ),
                ))

        # --- Contrast ---
        if report.contrast_score < 40:
            recs.append(Recommendation(
                category="Visual",
                severity="warning",
                message="Low overall contrast – may reduce readability",
                details=(
                    "The image has relatively low contrast, which can make "
                    "text and UI elements harder to distinguish. Consider "
                    "increasing contrast between foreground and background."
                ),
            ))

        # --- Mode-specific ---
        if mode == "ui_ux" and report.ui_audit:
            navbar_attn = report.ui_audit.get("navbar", 0)
            if navbar_attn > 25:
                recs.append(Recommendation(
                    category="UX",
                    severity="info",
                    message="Navbar is drawing significant attention",
                    details=(
                        "The navigation bar captures a large share of attention. "
                        "While this aids navigation, ensure it doesn't overshadow "
                        "the hero content below."
                    ),
                ))
            hero_attn = report.ui_audit.get("hero", 0)
            if hero_attn < 25:
                recs.append(Recommendation(
                    category="UX",
                    severity="warning",
                    message="Hero section has low visual dominance",
                    details=(
                        "The hero/above-the-fold area is not capturing enough "
                        "attention. Strengthen it with a compelling image, "
                        "larger headline, or animation."
                    ),
                ))

        if mode == "ad_creative" and report.ad_creative:
            prod = report.ad_creative.get("product_prominence", 0)
            if prod < 30:
                recs.append(Recommendation(
                    category="Marketing",
                    severity="warning",
                    message="Product is not visually prominent in the creative",
                    details=(
                        "The central product/service area receives low attention. "
                        "Increase its visual weight through size, lighting, or "
                        "contrast against the background."
                    ),
                ))
            face_pct = report.ad_creative.get("face_attention_pct", 0)
            if face_pct > 40:
                recs.append(Recommendation(
                    category="Marketing",
                    severity="info",
                    message="Strong face bias detected in attention",
                    details=(
                        "Human faces are drawing a large share of visual attention. "
                        "This is effective for emotional connection but may "
                        "divert attention from the product or CTA."
                    ),
                ))
            text_comp = report.ad_creative.get("text_competition", 0)
            if text_comp > 40:
                recs.append(Recommendation(
                    category="Content",
                    severity="warning",
                    message="High text density – text elements compete with imagery",
                    details=(
                        "The creative contains a high density of text edges, "
                        "suggesting multiple text blocks. Reduce copy or "
                        "increase font hierarchy to improve scannability."
                    ),
                ))

        # Ensure at least one positive recommendation
        if not any(r.severity == "info" for r in recs):
            recs.append(Recommendation(
                category="Visual",
                severity="info",
                message="Balanced visual composition detected",
                details=(
                    "The overall attention distribution suggests a reasonably "
                    "balanced design. Fine-tune individual elements to "
                    "optimise for your specific conversion goals."
                ),
            ))

        return recs
