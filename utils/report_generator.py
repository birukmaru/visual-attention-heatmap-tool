"""
Report Generator – PDF Export with ReportLab
==============================================

Generates professional PDF reports containing:
    - Project branding header
    - Original image + heatmap overlay
    - Fixation point table
    - Key metrics dashboard
    - Recommendations section
    - Footer with timestamp

Author  : Visual Attention Heatmap Tool
License : MIT
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from utils.saliency import FixationPoint
from utils.insights import InsightReport, MetricCard, Recommendation

# ReportLab imports
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """
    Generates polished PDF reports from saliency analysis results.

    Usage
    -----
    >>> gen = ReportGenerator()
    >>> pdf_bytes = gen.generate(
    ...     original_image=img,
    ...     heatmap_image=heatmap,
    ...     annotated_image=annotated,
    ...     fixations=fixations,
    ...     report=insight_report,
    ...     filename="analysis_report.pdf",
    ... )
    """

    # Brand colours
    PRIMARY = colors.HexColor("#4F8BF9")
    DARK = colors.HexColor("#1E293B")
    LIGHT = colors.HexColor("#F8FAFC")
    ACCENT = colors.HexColor("#10B981")
    WARNING = colors.HexColor("#F59E0B")
    DANGER = colors.HexColor("#EF4444")

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF export. "
                "Install it with: pip install reportlab"
            )

    def generate(
        self,
        original_image: np.ndarray,
        heatmap_image: np.ndarray,
        annotated_image: np.ndarray,
        fixations: List[FixationPoint],
        report: InsightReport,
        filename: str = "attention_report.pdf",
    ) -> bytes:
        """
        Generate a PDF report and return it as bytes.

        Parameters
        ----------
        original_image : np.ndarray
            Original uploaded image (BGR).
        heatmap_image : np.ndarray
            Heatmap overlay image (BGR).
        annotated_image : np.ndarray
            Image with fixation annotations (BGR).
        fixations : list of FixationPoint
            Ranked fixation data.
        report : InsightReport
            Complete insight analysis report.
        filename : str
            PDF filename (for metadata only).

        Returns
        -------
        bytes
            PDF file content.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            title="Visual Attention Analysis Report",
            author="Visual Attention Heatmap Tool",
        )

        styles = getSampleStyleSheet()
        story = []

        # --- Custom styles ---
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            textColor=self.DARK,
            spaceAfter=6,
            alignment=TA_CENTER,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.gray,
            spaceAfter=20,
            alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=self.DARK,
            spaceBefore=16,
            spaceAfter=8,
            borderPadding=4,
        )
        body_style = ParagraphStyle(
            "BodyText",
            parent=styles["Normal"],
            fontSize=10,
            textColor=self.DARK,
            spaceAfter=6,
            leading=14,
        )
        metric_label_style = ParagraphStyle(
            "MetricLabel",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.gray,
        )
        metric_value_style = ParagraphStyle(
            "MetricValue",
            parent=styles["Normal"],
            fontSize=14,
            textColor=self.DARK,
            alignment=TA_CENTER,
        )

        # ============================================================
        # PAGE 1: Header + Original + Heatmap
        # ============================================================

        # Title
        story.append(Paragraph(
            "🔍 Visual Attention Analysis Report", title_style
        ))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            subtitle_style,
        ))
        story.append(Spacer(1, 8))

        # Images: Original + Heatmap side by side
        story.append(Paragraph("Original Image & Attention Heatmap", heading_style))
        story.append(Spacer(1, 4))

        orig_path = self._save_temp_image(original_image)
        heat_path = self._save_temp_image(heatmap_image)

        img_width = 85 * mm
        img_height = self._calc_height(original_image, img_width)

        image_table = Table(
            [[
                RLImage(orig_path, width=img_width, height=img_height),
                RLImage(heat_path, width=img_width, height=img_height),
            ]],
            colWidths=[img_width + 5 * mm, img_width + 5 * mm],
        )
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 4))

        # Captions
        cap_table = Table(
            [[
                Paragraph("Original Image", metric_label_style),
                Paragraph("Saliency Heatmap", metric_label_style),
            ]],
            colWidths=[img_width + 5 * mm, img_width + 5 * mm],
        )
        cap_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(cap_table)
        story.append(Spacer(1, 12))

        # Annotated image
        story.append(Paragraph("Fixation Points Overlay", heading_style))
        ann_path = self._save_temp_image(annotated_image)
        ann_width = 140 * mm
        ann_height = self._calc_height(annotated_image, ann_width)
        story.append(RLImage(ann_path, width=ann_width, height=ann_height))
        story.append(Spacer(1, 12))

        # ============================================================
        # PAGE 2: Metrics + Fixation Table
        # ============================================================
        story.append(PageBreak())

        # Key Metrics
        story.append(Paragraph("Key Metrics", heading_style))
        story.append(Spacer(1, 4))

        metric_data = []
        metric_headers = []
        metric_values = []
        for card in report.metric_cards:
            metric_headers.append(Paragraph(
                f"{card.icon} {card.title}", metric_label_style
            ))
            color = self.ACCENT if card.trend == "good" else (
                self.WARNING if card.trend == "warning" else self.DANGER
            )
            metric_values.append(Paragraph(
                f'<font color="{color.hexval()}">{card.value}</font>',
                metric_value_style,
            ))

        if metric_headers:
            metrics_table = Table(
                [metric_headers, metric_values],
                colWidths=[45 * mm] * len(metric_headers),
            )
            metrics_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), self.LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 16))

        # Fixation Points Table
        story.append(Paragraph("Fixation Points (Ranked)", heading_style))
        story.append(Spacer(1, 4))

        fix_data = [["Rank", "Label", "X", "Y", "Score", "Radius"]]
        for fp in fixations:
            fix_data.append([
                str(fp.rank),
                fp.label,
                str(fp.x),
                str(fp.y),
                f"{fp.score:.1f}%",
                f"{fp.radius}px",
            ])

        fix_table = Table(fix_data, colWidths=[20 * mm, 35 * mm, 22 * mm, 22 * mm, 25 * mm, 25 * mm])
        fix_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(fix_table)
        story.append(Spacer(1, 16))

        # Zone Analysis
        if report.zone_scores:
            story.append(Paragraph("Zone Attention Distribution", heading_style))
            zone_data = [["Zone", "Attention %"]]
            for zone_name, score in report.zone_scores.items():
                zone_data.append([zone_name.replace("_", " ").title(), f"{score:.1f}%"])
            zone_table = Table(zone_data, colWidths=[60 * mm, 40 * mm])
            zone_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), self.PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(zone_table)
            story.append(Spacer(1, 16))

        # ============================================================
        # Recommendations
        # ============================================================
        story.append(Paragraph("Design & Marketing Recommendations", heading_style))
        story.append(Spacer(1, 4))

        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        severity_colors = {
            "info": self.ACCENT,
            "warning": self.WARNING,
            "critical": self.DANGER,
        }

        for rec in report.recommendations:
            icon = severity_icons.get(rec.severity, "•")
            color = severity_colors.get(rec.severity, self.DARK)

            story.append(Paragraph(
                f'<font color="{color.hexval()}">{icon} [{rec.category}] {rec.message}</font>',
                ParagraphStyle(
                    "RecTitle",
                    parent=body_style,
                    fontSize=10,
                    spaceBefore=8,
                    spaceAfter=2,
                    textColor=color,
                ),
            ))
            if rec.details:
                story.append(Paragraph(rec.details, body_style))

        story.append(Spacer(1, 20))

        # Footer
        story.append(Paragraph(
            "— Generated by Visual Attention Heatmap Tool —",
            ParagraphStyle(
                "Footer", parent=body_style,
                fontSize=8, textColor=colors.gray,
                alignment=TA_CENTER,
            ),
        ))

        # Build PDF
        doc.build(story)

        # Cleanup temp files
        for path in [orig_path, heat_path, ann_path]:
            try:
                os.unlink(path)
            except OSError:
                pass

        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _save_temp_image(image: np.ndarray) -> str:
        """Save a BGR ndarray to a temporary PNG and return its path."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_img.save(path, "PNG")
        return path

    @staticmethod
    def _calc_height(image: np.ndarray, target_width_points: float) -> float:
        """Calculate proportional height for a given width in points."""
        h, w = image.shape[:2]
        aspect = h / w
        return target_width_points * aspect
