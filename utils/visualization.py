"""
Visualization Engine – Heatmap Overlays & Annotated Outputs
============================================================

Provides high-quality visualization of saliency maps including:
    - Heatmap overlay on original image
    - Raw saliency colourmap
    - Focus intensity (contour) map
    - Fixation marker annotations with numbered labels
    - Comparison / side-by-side layouts

All outputs are returned as NumPy BGR arrays ready for display
or export.

Author  : Visual Attention Heatmap Tool
License : MIT
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.saliency import FixationPoint


# ---------------------------------------------------------------------------
# Colour palettes for fixation markers
# ---------------------------------------------------------------------------
FIXATION_COLORS: List[Tuple[int, int, int]] = [
    (0, 120, 255),   # Rank 1 – vivid orange (BGR)
    (0, 200, 80),    # Rank 2 – green
    (255, 180, 0),   # Rank 3 – cyan-blue
    (180, 50, 255),  # Rank 4 – magenta
    (0, 255, 255),   # Rank 5 – yellow
]

# Emoji-style labels for overlay
RANK_LABELS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]


class VisualizationEngine:
    """
    Generates publication-quality visual outputs from saliency data.

    Parameters
    ----------
    heatmap_colormap : int
        OpenCV colormap constant (default: ``cv2.COLORMAP_JET``).
    heatmap_opacity : float
        Alpha blending factor for heatmap overlay [0, 1].
    marker_thickness : int
        Stroke thickness for fixation circles.
    """

    def __init__(
        self,
        heatmap_colormap: int = cv2.COLORMAP_JET,
        heatmap_opacity: float = 0.55,
        marker_thickness: int = 3,
    ):
        self.colormap = heatmap_colormap
        self.opacity = heatmap_opacity
        self.marker_thickness = marker_thickness

    # ------------------------------------------------------------------
    # Heatmap overlay
    # ------------------------------------------------------------------

    def generate_heatmap_overlay(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        opacity: Optional[float] = None,
        colormap: Optional[int] = None,
    ) -> np.ndarray:
        """
        Overlay a coloured heatmap on the original image.

        Parameters
        ----------
        image : np.ndarray
            Original image (BGR, uint8).
        saliency : np.ndarray
            Saliency map (float32, [0, 1]).
        opacity : float, optional
            Override default opacity.
        colormap : int, optional
            Override default colormap.

        Returns
        -------
        np.ndarray
            Blended BGR image (uint8).
        """
        alpha = opacity if opacity is not None else self.opacity
        cmap = colormap if colormap is not None else self.colormap

        # Ensure saliency is uint8 [0, 255]
        sal_u8 = (np.clip(saliency, 0, 1) * 255).astype(np.uint8)
        # Apply colourmap
        heatmap = cv2.applyColorMap(sal_u8, cmap)
        # Resize heatmap to match image dimensions
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        # Blend
        blended = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
        return blended

    # ------------------------------------------------------------------
    # Raw saliency colourmap
    # ------------------------------------------------------------------

    def generate_saliency_colormap(
        self,
        saliency: np.ndarray,
        colormap: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert a raw saliency map into a standalone colour-mapped image.
        """
        cmap = colormap if colormap is not None else cv2.COLORMAP_INFERNO
        sal_u8 = (np.clip(saliency, 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(sal_u8, cmap)

    # ------------------------------------------------------------------
    # Focus intensity (contour) map
    # ------------------------------------------------------------------

    def generate_focus_intensity_map(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a contour-style focus-intensity visualisation showing
        iso-attention curves overlaid on a dimmed version of the original.
        """
        # Dim the original image
        dimmed = (image.astype(np.float32) * 0.35).astype(np.uint8)

        sal_u8 = (np.clip(saliency, 0, 1) * 255).astype(np.uint8)
        if sal_u8.shape[:2] != image.shape[:2]:
            sal_u8 = cv2.resize(sal_u8, (image.shape[1], image.shape[0]))

        # Draw contours at multiple threshold levels
        output = dimmed.copy()
        levels = [50, 100, 150, 200, 230]
        colors_contour = [
            (80, 60, 20),
            (140, 100, 30),
            (200, 160, 40),
            (80, 200, 255),
            (0, 255, 255),
        ]
        for level, color in zip(levels, colors_contour):
            _, thresh = cv2.threshold(sal_u8, level, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, color, 2)

        # Spotlight: brighten high-saliency regions
        mask = sal_u8.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask] * 3, axis=-1)
        output = (output.astype(np.float32) * (0.6 + 0.4 * mask_3ch)).clip(0, 255).astype(np.uint8)

        return output

    # ------------------------------------------------------------------
    # Fixation marker annotations
    # ------------------------------------------------------------------

    def annotate_fixations(
        self,
        image: np.ndarray,
        fixations: List[FixationPoint],
        show_labels: bool = True,
        show_scores: bool = True,
    ) -> np.ndarray:
        """
        Draw numbered fixation markers (circles + labels) on the image.
        """
        output = image.copy()
        h, w = output.shape[:2]
        font_scale = max(0.6, min(h, w) / 800.0)
        thickness = max(2, int(min(h, w) / 300))

        for fp in fixations:
            idx = min(fp.rank - 1, len(FIXATION_COLORS) - 1)
            color = FIXATION_COLORS[idx]

            # Outer glow circle (semi-transparent)
            overlay = output.copy()
            cv2.circle(overlay, (fp.x, fp.y), fp.radius + 8, color, -1)
            cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

            # Main circle
            cv2.circle(output, (fp.x, fp.y), fp.radius, color, thickness)
            # Inner dot
            cv2.circle(output, (fp.x, fp.y), 5, color, -1)

            if show_labels:
                # Draw rank number inside a filled circle
                label_radius = int(font_scale * 18)
                label_x = fp.x + fp.radius + 10
                label_y = fp.y - fp.radius - 5
                # Keep within bounds
                label_x = min(label_x, w - label_radius - 5)
                label_y = max(label_y, label_radius + 5)

                cv2.circle(output, (label_x, label_y), label_radius, color, -1)
                text = str(fp.rank)
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, 2
                )
                cv2.putText(
                    output, text,
                    (label_x - tw // 2, label_y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if show_scores:
                score_text = f"{fp.score:.0f}%"
                text_x = fp.x - 20
                text_y = fp.y + fp.radius + int(25 * font_scale)
                text_y = min(text_y, h - 10)

                # Background rectangle for readability
                (tw, th), baseline = cv2.getTextSize(
                    score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, 1
                )
                cv2.rectangle(
                    output,
                    (text_x - 4, text_y - th - 6),
                    (text_x + tw + 4, text_y + baseline + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    output, score_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return output

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------

    def side_by_side(
        self,
        left: np.ndarray,
        right: np.ndarray,
        labels: Tuple[str, str] = ("Before", "After"),
        gap: int = 4,
    ) -> np.ndarray:
        """
        Create a side-by-side comparison image with labels.
        """
        h = max(left.shape[0], right.shape[0])
        # Resize both to same height
        left_r = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        right_r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))

        # Create gap
        separator = np.full((h, gap, 3), 40, dtype=np.uint8)

        combined = np.hstack([left_r, separator, right_r])

        # Add labels
        font_scale = max(0.6, h / 600.0)
        for i, (label, x_off) in enumerate(
            zip(labels, [10, left_r.shape[1] + gap + 10])
        ):
            cv2.putText(
                combined, label,
                (x_off, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return combined

    # ------------------------------------------------------------------
    # Scanpath animation frames
    # ------------------------------------------------------------------

    def generate_scanpath_frames(
        self,
        image: np.ndarray,
        fixations: List[FixationPoint],
        n_frames: int = 30,
    ) -> List[np.ndarray]:
        """
        Generate animation frames simulating an eye-tracking scanpath
        that moves between fixation points.
        """
        if not fixations:
            return [image.copy()]

        frames: List[np.ndarray] = []
        h, w = image.shape[:2]
        gaze_radius = max(int(min(h, w) * 0.12), 40)

        # Create darkened background
        dark = (image.astype(np.float32) * 0.15).astype(np.uint8)

        all_points = [(fp.x, fp.y) for fp in fixations]

        for i in range(len(all_points)):
            fx, fy = all_points[i]
            # Create spotlight mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (fx, fy), gaze_radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=gaze_radius * 0.4)
            mask_3ch = np.stack([mask] * 3, axis=-1)

            # Blend: dark background + spotlight on original
            frame = (dark.astype(np.float32) * (1 - mask_3ch) +
                     image.astype(np.float32) * mask_3ch).clip(0, 255).astype(np.uint8)

            # Draw fixation marker
            idx = min(i, len(FIXATION_COLORS) - 1)
            cv2.circle(frame, (fx, fy), 8, FIXATION_COLORS[idx], -1)
            cv2.circle(frame, (fx, fy), 12, FIXATION_COLORS[idx], 2)

            # Draw connecting lines to previous fixations
            for j in range(i):
                px, py = all_points[j]
                cv2.line(frame, (px, py), (fx, fy), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, FIXATION_COLORS[min(j, len(FIXATION_COLORS) - 1)], -1)

            # Add rank label
            label = f"Fixation #{i+1}"
            cv2.putText(
                frame, label,
                (fx + 15, fy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Hold this frame for multiple frames for animation pacing
            hold = max(n_frames // len(all_points), 5)
            frames.extend([frame] * hold)

        return frames

    # ------------------------------------------------------------------
    # Utility: BGR ↔ RGB conversion
    # ------------------------------------------------------------------

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR (OpenCV) to RGB (Pillow/Streamlit)."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR."""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
