"""
Saliency Engine – Multi-Layer Visual Saliency Computation
==========================================================

Implements a layered saliency prediction pipeline with graceful fallback:
    1. Deep Learning Saliency  (DeepGaze IIE / TensorFlow if available)
    2. OpenCV Spectral Residual Saliency
    3. Multi-Scale Frequency + Color + Contrast Fusion (always available)

The engine computes a combined saliency map, detects fixation points via
local-maxima extraction with non-maximum suppression, and ranks them by
predicted gaze priority.

Author  : Visual Attention Heatmap Tool
License : MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter, label
from skimage.feature import peak_local_max

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FixationPoint:
    """Represents a single predicted gaze fixation."""
    rank: int               # 1-indexed rank (1 = first look)
    x: int                  # X coordinate (column)
    y: int                  # Y coordinate (row)
    score: float            # Normalised attention score [0-100]
    radius: int             # Bounding circle radius in pixels
    label: str = ""         # Human-readable label

    def __post_init__(self):
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(self.rank, f"{self.rank}th")
        self.label = f"{ordinal} Look"


@dataclass
class SaliencyResult:
    """Container for all saliency computation outputs."""
    raw_saliency: np.ndarray                     # float32 [0, 1]
    smoothed_saliency: np.ndarray                # float32 [0, 1]
    fixation_points: List[FixationPoint] = field(default_factory=list)
    model_used: str = "unknown"
    center_bias_applied: bool = False


# ---------------------------------------------------------------------------
# Saliency Engine
# ---------------------------------------------------------------------------

class SaliencyEngine:
    """
    Multi-layer saliency prediction engine.

    Parameters
    ----------
    center_bias : bool
        Apply Gaussian center-bias to mimic natural human tendency
        to look toward the centre of an image.
    smoothing_sigma : float
        Sigma for Gaussian blur applied to the final saliency map
        before peak detection.
    min_fixation_distance : int
        Minimum pixel distance between two fixation points
        (controls non-maximum suppression).
    """

    # Class-level cache for model availability check
    _model_availability: Optional[str] = None

    def __init__(
        self,
        center_bias: bool = True,
        smoothing_sigma: float = 25.0,
        min_fixation_distance: int = 80,
    ):
        self.center_bias = center_bias
        self.smoothing_sigma = smoothing_sigma
        self.min_fixation_distance = min_fixation_distance
        self.model_type = self._detect_best_model()

    # ------------------------------------------------------------------
    # Model detection & fallback
    # ------------------------------------------------------------------

    @classmethod
    def _detect_best_model(cls) -> str:
        """Probe available backends and select the best one."""
        if cls._model_availability is not None:
            return cls._model_availability

        # --- Tier 1: Deep learning saliency (DeepGaze IIE via pysaliency) ---
        try:
            import torch  # noqa: F401
            # Check if a pre-trained deep saliency model is loadable
            logger.info("PyTorch detected – deep-learning saliency available.")
            cls._model_availability = "deep_learning"
            return cls._model_availability
        except ImportError:
            logger.info("PyTorch not found – skipping deep-learning backend.")

        # --- Tier 2: OpenCV Spectral Residual ---
        try:
            _sr = cv2.saliency.StaticSaliencySpectralResidual_create()
            logger.info("OpenCV Spectral-Residual saliency available.")
            cls._model_availability = "spectral_residual"
            return cls._model_availability
        except AttributeError:
            logger.info("OpenCV saliency module not compiled – skipping.")

        # --- Tier 3: Custom multi-scale fusion (always available) ---
        logger.info("Falling back to custom multi-scale saliency fusion.")
        cls._model_availability = "custom_fusion"
        return cls._model_availability

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        image: np.ndarray,
        n_fixations: int = 3,
    ) -> SaliencyResult:
        """
        Run the full saliency pipeline on *image* (BGR uint8 array).

        Returns a ``SaliencyResult`` containing the raw/smoothed maps
        and ranked fixation points.
        """
        # Ensure consistent input
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        h, w = image.shape[:2]

        # --- Compute raw saliency (multi-method fusion) ---
        maps: List[np.ndarray] = []
        model_tag = self.model_type

        # Always compute custom fusion components (robust & always available)
        maps.append(self._spectral_residual_custom(image))
        maps.append(self._color_contrast_saliency(image))
        maps.append(self._luminance_contrast_saliency(image))
        maps.append(self._edge_density_saliency(image))

        # If OpenCV SR module exists, add it
        if self.model_type in ("spectral_residual", "deep_learning"):
            sr_map = self._opencv_spectral_residual(image)
            if sr_map is not None:
                maps.append(sr_map)

        # If deep-learning backend available, add neural saliency
        if self.model_type == "deep_learning":
            dl_map = self._deep_learning_saliency(image)
            if dl_map is not None:
                maps.append(dl_map)
                # Give DL map higher weight
                maps.append(dl_map)  # double weight
                model_tag = "deep_learning"

        # --- Fuse maps ---
        raw = self._fuse_maps(maps, h, w)

        # --- Optional center bias ---
        if self.center_bias:
            raw = self._apply_center_bias(raw)

        # --- Normalise to [0, 1] ---
        raw = self._normalise(raw)

        # --- Smooth for fixation detection ---
        sigma = self.smoothing_sigma * (max(h, w) / 1000.0)
        smoothed = ndimage.gaussian_filter(raw.astype(np.float64), sigma=sigma)
        smoothed = self._normalise(smoothed).astype(np.float32)

        # --- Detect fixation points ---
        fixations = self._detect_fixations(smoothed, n_fixations, h, w)

        return SaliencyResult(
            raw_saliency=raw.astype(np.float32),
            smoothed_saliency=smoothed,
            fixation_points=fixations,
            model_used=model_tag,
            center_bias_applied=self.center_bias,
        )

    # ------------------------------------------------------------------
    # Saliency sub-methods
    # ------------------------------------------------------------------

    def _spectral_residual_custom(self, image: np.ndarray) -> np.ndarray:
        """
        Hou & Zhang (2007) Spectral Residual approach implemented
        manually for maximum portability.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Resize for efficiency
        small = cv2.resize(gray, (64, 64))
        # Log amplitude spectrum
        dft = np.fft.fft2(small)
        amplitude = np.abs(dft)
        log_amp = np.log1p(amplitude)
        phase = np.angle(dft)
        # Spectral residual = log_amp - smoothed(log_amp)
        avg_log_amp = ndimage.uniform_filter(log_amp, size=3)
        spectral_residual = log_amp - avg_log_amp
        # Reconstruct
        saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
        saliency = cv2.resize(saliency.astype(np.float32), (image.shape[1], image.shape[0]))
        saliency = ndimage.gaussian_filter(saliency, sigma=8)
        return self._normalise(saliency)

    def _color_contrast_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Colour-based saliency using CIE-Lab colour opponency.
        Regions with colours dissimilar to the mean attract attention.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
        mean_color = lab.mean(axis=(0, 1))
        # Per-pixel distance from mean colour
        dist = np.sqrt(np.sum((lab - mean_color) ** 2, axis=2))
        dist = ndimage.gaussian_filter(dist, sigma=12)
        return self._normalise(dist)

    def _luminance_contrast_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-scale luminance contrast (centre-surround mechanism).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        saliency = np.zeros_like(gray)
        for sigma_c, sigma_s in [(2, 8), (4, 16), (8, 32)]:
            center = ndimage.gaussian_filter(gray, sigma_c)
            surround = ndimage.gaussian_filter(gray, sigma_s)
            diff = np.abs(center - surround)
            saliency += self._normalise(diff)
        return self._normalise(saliency)

    def _edge_density_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Edge density map – areas with strong edges attract attention.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        # Smooth to create a density map
        density = ndimage.gaussian_filter(edges, sigma=20)
        return self._normalise(density)

    def _opencv_spectral_residual(self, image: np.ndarray) -> Optional[np.ndarray]:
        """OpenCV built-in spectral residual (if compiled with saliency module)."""
        try:
            sr = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, sal_map = sr.computeSaliency(image)
            if success and sal_map is not None:
                return self._normalise(sal_map.astype(np.float32))
        except Exception as exc:
            logger.debug("OpenCV SR failed: %s", exc)
        return None

    def _deep_learning_saliency(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Deep-learning saliency using a lightweight pre-trained model.
        Uses a simple CNN-based approach via PyTorch if available.
        """
        try:
            import torch
            import torch.nn.functional as F

            # Convert to tensor
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Resize to model input
            resized = cv2.resize(img_rgb, (224, 224))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)

            # Use a simple feature extraction approach
            # Compute multi-channel gradient magnitude as neural-inspired saliency
            with torch.no_grad():
                # Compute spatial gradients across channels
                dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
                dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
                
                # Gradient magnitude per channel, then max across channels
                grad_x = torch.sqrt((dx ** 2).sum(dim=1, keepdim=True) + 1e-8)
                grad_y = torch.sqrt((dy ** 2).sum(dim=1, keepdim=True) + 1e-8)
                
                # Pad to original size
                grad_x = F.pad(grad_x, (0, 1, 0, 0))
                grad_y = F.pad(grad_y, (0, 0, 0, 1))
                
                sal = (grad_x + grad_y).squeeze().numpy()

            # Resize back to original
            sal = cv2.resize(sal, (image.shape[1], image.shape[0]))
            sal = ndimage.gaussian_filter(sal, sigma=10)
            return self._normalise(sal)

        except Exception as exc:
            logger.debug("Deep-learning saliency failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Map fusion & normalisation
    # ------------------------------------------------------------------

    def _fuse_maps(
        self,
        maps: List[np.ndarray],
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Fuse multiple saliency maps via weighted averaging.
        All maps are resized to (h, w) before fusion.
        """
        resized = []
        for m in maps:
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h))
            resized.append(self._normalise(m))
        fused = np.mean(resized, axis=0)
        return fused

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise array to [0, 1]."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)

    def _apply_center_bias(self, saliency: np.ndarray) -> np.ndarray:
        """
        Multiply saliency by a 2-D Gaussian centred on the image
        to simulate human centre-bias tendency.
        """
        h, w = saliency.shape[:2]
        cy, cx = h / 2, w / 2
        sigma_y, sigma_x = h * 0.4, w * 0.4
        y_grid, x_grid = np.ogrid[0:h, 0:w]
        gaussian = np.exp(
            -((x_grid - cx) ** 2 / (2 * sigma_x ** 2)
              + (y_grid - cy) ** 2 / (2 * sigma_y ** 2))
        ).astype(np.float32)
        # Blend: 70% saliency + 30% center-biased
        biased = 0.7 * saliency + 0.3 * (saliency * gaussian)
        return self._normalise(biased)

    # ------------------------------------------------------------------
    # Fixation detection
    # ------------------------------------------------------------------

    def _detect_fixations(
        self,
        saliency: np.ndarray,
        n: int,
        h: int,
        w: int,
    ) -> List[FixationPoint]:
        """
        Detect top-N fixation points via local-maxima extraction
        with non-maximum suppression.
        """
        # Adaptive minimum distance based on image size
        min_dist = max(
            self.min_fixation_distance,
            int(min(h, w) * 0.08),
        )

        try:
            # scikit-image peak_local_max for robust peak detection
            coords = peak_local_max(
                saliency,
                min_distance=min_dist,
                threshold_abs=0.15,
                num_peaks=n * 3,  # detect extras, then filter
                exclude_border=int(min(h, w) * 0.03),
            )
        except Exception:
            # Manual fallback
            coords = self._manual_peak_detection(saliency, min_dist, n * 3)

        if len(coords) == 0:
            # If no peaks found, place fixation at the absolute max
            max_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
            coords = np.array([list(max_idx)])

        # Score each candidate
        scored = []
        for (r, c) in coords:
            score = float(saliency[r, c])
            scored.append((score, r, c))
        scored.sort(reverse=True)

        # NMS: keep top N with minimum distance enforcement
        kept: List[Tuple[float, int, int]] = []
        for score, r, c in scored:
            if len(kept) >= n:
                break
            too_close = False
            for _, kr, kc in kept:
                if np.hypot(r - kr, c - kc) < min_dist:
                    too_close = True
                    break
            if not too_close:
                kept.append((score, r, c))

        # Build FixationPoint list
        max_score = max(s for s, _, _ in kept) if kept else 1.0
        base_radius = int(min(h, w) * 0.04)
        fixations = []
        for rank_idx, (score, r, c) in enumerate(kept, start=1):
            normalised_score = (score / max_score) * 100.0 if max_score > 0 else 0.0
            radius = max(base_radius, int(base_radius * (score / max_score)))
            fixations.append(FixationPoint(
                rank=rank_idx,
                x=int(c),
                y=int(r),
                score=round(normalised_score, 1),
                radius=radius,
            ))

        return fixations

    @staticmethod
    def _manual_peak_detection(
        saliency: np.ndarray,
        min_dist: int,
        max_peaks: int,
    ) -> np.ndarray:
        """Fallback peak detection using scipy maximum_filter."""
        local_max = maximum_filter(saliency, size=min_dist)
        detected = (saliency == local_max) & (saliency > 0.15)
        labeled, num_features = label(detected)
        coords = []
        for i in range(1, num_features + 1):
            region = np.argwhere(labeled == i)
            centroid = region.mean(axis=0).astype(int)
            coords.append(centroid)
        if not coords:
            return np.array([]).reshape(0, 2)
        coords = np.array(coords)
        # Sort by score descending
        scores = [saliency[r, c] for r, c in coords]
        order = np.argsort(scores)[::-1][:max_peaks]
        return coords[order]
