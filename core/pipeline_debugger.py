"""Pipeline debugging utilities with image source abstractions.

Provides abstract and concrete image sources for testing and debugging
the detection pipeline with both synthetic and real images.

Also provides :class:`PipelineResult` and :class:`PipelineRunner` for
encapsulating and executing the full detection pipeline.
"""

from __future__ import annotations

import abc
import itertools
import os
import random
from dataclasses import dataclass, field

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.classifier import classify_grain
from core.detector import detect_grains, DetectionResult, FlocculationConfig
from core.morphology import (
    compute_morphology,
    compute_statistics,
    GrainContour,
    GrainMorphology,
    GrainStatistics,
)
from core.preprocessor import PreprocessConfig, preprocess


__all__ = [
    "ImageSource",
    "SyntheticImageSource",
    "RealImageSource",
    "PipelineResult",
    "PipelineRunner",
]

# ---------------------------------------------------------------------------
# Confidence constants for classification
# ---------------------------------------------------------------------------
FLOCCULATION_CONFIDENCE = 0.9
NORMAL_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# ImageSource classes (Task 1)
# ---------------------------------------------------------------------------

class ImageSource(abc.ABC):
    """Abstract base class for image sources used in pipeline debugging."""

    @abc.abstractmethod
    def load(self) -> np.ndarray:
        """Load and return the image as a grayscale numpy array."""
        ...

    @classmethod
    def from_path(cls, path: str | None, **kwargs) -> ImageSource:
        """Factory that returns an ImageSource from a file path.

        If *path* is ``None``, a :class:`SyntheticImageSource` is returned.
        Otherwise a :class:`RealImageSource` pointing to *path* is returned.

        Args:
            path: File path to a real image, or ``None`` for synthetic data.
            **kwargs: Forwarded to the concrete class constructor.

        Returns:
            An instance of a concrete ImageSource subclass.
        """
        if path is None:
            return SyntheticImageSource(**kwargs)
        return RealImageSource(path, **kwargs)


class SyntheticImageSource(ImageSource):
    """Generates reproducible synthetic images with elliptical grain-like shapes.

    The generated images follow the same convention as the existing test fixtures
    in ``tests/conftest.py``: a medium-gray background (128) with bright white
    ellipses (255) and added Gaussian noise for realism.
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 400,
        num_grains: int = 10,
        seed: int = 42,
    ) -> None:
        """Configure the synthetic image generator.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            num_grains: Number of elliptical grains to draw.
            seed: Random seed for reproducibility.
        """
        self.width = width
        self.height = height
        self.num_grains = num_grains
        self.seed = seed

    def load(self) -> np.ndarray:
        """Generate and return a synthetic grayscale image.

        Returns:
            Grayscale image (uint8) with elliptical shapes on a noisy background.
        """
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        # Medium-gray background
        img = np.full((self.height, self.width), 128, dtype=np.uint8)

        for _ in range(self.num_grains):
            # Random center within image bounds with margin
            cx = rng.randint(30, self.width - 30)
            cy = rng.randint(30, self.height - 30)

            # Random semi-axes (ensuring at least 5 px each)
            a = rng.randint(8, 25)
            b = rng.randint(5, max(a - 1, 5))

            # Random rotation angle
            angle = rng.randint(0, 180)

            # Draw bright white ellipse
            cv2.ellipse(
                img,
                (cx, cy),
                (a, b),
                angle,
                0,
                360,
                255,
                thickness=-1,
            )

        # Add Gaussian noise for realism
        noise = np_rng.normal(loc=0.0, scale=10.0, size=img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img


class RealImageSource(ImageSource):
    """Loads real images from disk."""

    def __init__(self, path: str) -> None:
        """Initialise with a file path.

        Args:
            path: Path to an image file readable by OpenCV.
        """
        self.path = path

    def load(self) -> np.ndarray:
        """Load and return the image as grayscale.

        Returns:
            Grayscale image (uint8).

        Raises:
            FileNotFoundError: If the image cannot be read.
            ValueError: If the loaded image is empty or invalid.
        """
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {self.path!r}")
        if img.size == 0:
            raise ValueError(f"Loaded image from {self.path!r} is empty.")
        return img


# ---------------------------------------------------------------------------
# PipelineResult dataclass (Task 2)
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Container for the complete output of a detection pipeline run.

    Attributes:
        mask: Final binary mask from preprocessing.
        grains: List of :class:`GrainContour` objects.
        morphologies: List of :class:`GrainMorphology` objects.
        statistics: Aggregate :class:`GrainStatistics` across all grains, or None.
    """

    mask: np.ndarray
    grains: list[GrainContour]
    morphologies: list[GrainMorphology]
    statistics: GrainStatistics | None


# ---------------------------------------------------------------------------
# PipelineRunner class (Task 2)
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Wraps the full detection pipeline for a given image source.

    The runner loads an image from an :class:`ImageSource`, runs the
    preprocessing, detects grains, computes morphology, classifies grains,
    and produces aggregate statistics.
    """

    def __init__(self, source: ImageSource) -> None:
        """Initialise with an image source.

        Args:
            source: ImageSource instance that provides the raw image.
        """
        self.source = source
        self._last_result: PipelineResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        config: PreprocessConfig | None = None,
        min_area: int = 1000,
        max_area: int = 15000,
        border_margin: int = 5,
        hull_expansion_ratio: float = 1.5,
        floc_config: FlocculationConfig | None = None,
        crop_black_background: bool = True,
    ) -> PipelineResult:
        """Run the full detection pipeline.

        Steps:
            1. Load image from :attr:`source`.
            2. Run preprocessing via :func:`core.preprocessor.preprocess`.
            3. Detect grains via :func:`core.detector.detect_grains`.
            4. Build :class:`GrainContour` objects.
            5. Compute morphology for each grain.
            6. Classify each grain (Zingg + flocculation).
            7. Compute aggregate statistics.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
            min_area: Minimum grain area.
            max_area: Maximum grain area.
            border_margin: Distance from border to filter.
            hull_expansion_ratio: Threshold for using convex hull vs mask filling.
            floc_config: Flocculation detection config. Uses defaults if None.
            crop_black_background: Whether to crop black background before processing.

        Returns:
            :class:`PipelineResult` with mask, grains, morphologies, and stats.
        """
        if config is None:
            config = PreprocessConfig()

        image = self.source.load()

        # Run preprocessing to get binary mask
        mask = preprocess(image, config)

        # Detect grains from the original image using the config
        results = detect_grains(
            image=image,
            config=config,
            min_area=min_area,
            max_area=max_area,
            border_margin=border_margin,
            hull_expansion_ratio=hull_expansion_ratio,
            floc_config=floc_config,
            crop_black_background=crop_black_background,
        )

        grains, morphologies = self._build_pipeline_results(results)
        statistics = self._compute_statistics(morphologies)

        pipeline_result = PipelineResult(
            mask=mask,
            grains=grains,
            morphologies=morphologies,
            statistics=statistics,
        )
        self._last_result = pipeline_result
        return pipeline_result

    def visualize_branches(self, save_path: str) -> None:
        """Create a figure showing the original image and branch masks.

        Must call :meth:`run` first (or use a cached last result).

        Creates a matplotlib figure with subplots showing:
        - Original image
        - Preprocessed mask

        Saves the figure to *save_path* as a PNG.

        Args:
            save_path: Path where the PNG figure will be saved.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        if self._last_result is None:
            raise RuntimeError("Must call run() before visualize_branches()")

        result = self._last_result
        image = self.source.load()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        titles = ["Original", "Preprocessed Mask"]
        images = [
            image,
            result.mask,
        ]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()

        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def compare_configs(
        self,
        config_a: PreprocessConfig,
        config_b: PreprocessConfig,
        save_path: str,
    ) -> None:
        """Run the pipeline with two configs and create a side-by-side comparison.

        Creates a figure showing:
        - Config A mask
        - Config B mask
        - Absolute difference |B - A|
        - Overlay on original image for both configs

        Saves the figure to *save_path* as a PNG.

        Args:
            config_a: First preprocessing configuration.
            config_b: Second preprocessing configuration.
            save_path: Path where the PNG figure will be saved.
        """
        # Preserve the original _last_result so compare_configs is non-destructive
        original_result = self._last_result

        try:
            image = self.source.load()

            result_a = self.run(config=config_a)
            result_b = self.run(config=config_b)

            mask_a = result_a.mask
            mask_b = result_b.mask
            diff = cv2.absdiff(mask_b, mask_a)

            # Create overlay: original with mask in green
            overlay_a = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay_b = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            green = np.array([0, 255, 0], dtype=np.uint8)
            overlay_a[mask_a > 0] = green
            overlay_b[mask_b > 0] = green

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            ax_a, ax_b, ax_diff, ax_overlay = axes.flatten()

            ax_a.imshow(mask_a, cmap="gray")
            ax_a.set_title("Config A Mask")
            ax_a.axis("off")

            ax_b.imshow(mask_b, cmap="gray")
            ax_b.set_title("Config B Mask")
            ax_b.axis("off")

            ax_diff.imshow(diff, cmap="gray")
            ax_diff.set_title("Absolute Difference |B - A|")
            ax_diff.axis("off")

            # Show both overlays side-by-side in the same subplot
            ax_overlay.imshow(np.hstack([overlay_a, overlay_b]))
            ax_overlay.set_title("Overlay: Config A (left) | Config B (right)")
            ax_overlay.axis("off")

            plt.tight_layout()

            dirname = os.path.dirname(save_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        finally:
            # Restore the original _last_result regardless of success or failure
            self._last_result = original_result

    def grid_search(
        self,
        param_ranges: dict[str, list],
        metric: str = "grain_count",
        min_area: int = 50,
        max_area: int = 15000,
    ) -> pd.DataFrame:
        """Run the pipeline over a grid of parameter combinations.

        For every combination of values in *param_ranges*, a new
        :class:`PreprocessConfig` is created, the pipeline is executed,
        and the requested metric(s) are recorded.  Exceptions during a
        run are caught and stored in the ``error`` column so the whole
        grid search never crashes on a single bad combination.

        Args:
            param_ranges: Mapping from parameter name to a list of values.
                Supported keys: ``adaptive_block_size``, ``adaptive_c``,
                ``blur_kernel``, ``morph_kernel_size``, ``morph_open_iter``,
                ``morph_close_iter``, ``min_area``.
            metric: Metric to extract from each run. One of
                ``"grain_count"``, ``"mask_pixels"``, ``"circularity_mean"``.
            min_area: Minimum grain area passed to :meth:`run`.
            max_area: Maximum grain area passed to :meth:`run`.

        Returns:
            A :class:`pandas.DataFrame` with one row per parameter
            combination.  Columns include each parameter key plus
            ``grain_count``, ``mask_pixels``, ``circularity_mean``,
            and ``error``.
        """
        import dataclasses

        if not param_ranges:
            return pd.DataFrame()

        # Validate param_ranges keys
        _SUPPORTED_KEYS = {
            "adaptive_block_size",
            "adaptive_c",
            "blur_kernel",
            "morph_kernel_size",
            "morph_open_iter",
            "morph_close_iter",
            "min_area",
        }
        unsupported = set(param_ranges.keys()) - _SUPPORTED_KEYS
        if unsupported:
            import warnings

            warnings.warn(
                f"Unsupported parameter keys will be ignored: {sorted(unsupported)}",
                stacklevel=2,
            )

        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]

        rows: list[dict] = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            row: dict = params.copy()

            base = PreprocessConfig()
            config = dataclasses.replace(
                base,
                **{
                    k: v
                    for k, v in params.items()
                    if k in _SUPPORTED_KEYS and hasattr(base, k)
                }
            )

            try:
                result = self.run(
                    config=config,
                    min_area=min_area,
                    max_area=max_area,
                )
                row["grain_count"] = len(result.grains)
                row["mask_pixels"] = int(np.count_nonzero(result.mask))
                if metric == "circularity_mean":
                    if result.statistics is not None:
                        row["circularity_mean"] = result.statistics.circularity_mean
                    else:
                        row["circularity_mean"] = None
                row["error"] = None
            except Exception as exc:  # noqa: BLE001
                row["grain_count"] = None
                row["mask_pixels"] = None
                if metric == "circularity_mean":
                    row["circularity_mean"] = None
                row["error"] = str(exc)

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline_results(
        self, results: list[DetectionResult]
    ) -> tuple[list[GrainContour], list[GrainMorphology]]:
        """Build GrainContour and GrainMorphology objects from detection results.

        Returns:
            Tuple of (grains, morphologies).
        """
        grains: list[GrainContour] = []
        morphologies: list[GrainMorphology] = []

        for result in results:
            grain = GrainContour(contour=result.contour, mask=result.mask)
            grains.append(grain)

            morph = compute_morphology(result.contour, result.mask)
            morph.is_flocculation = result.is_flocculation
            morph.shape_class = classify_grain(
                morph.aspect_ratio, result.is_flocculation, circularity=morph.circularity
            )
            morph.confidence = (
                FLOCCULATION_CONFIDENCE
                if result.is_flocculation
                else NORMAL_CONFIDENCE
            )
            morphologies.append(morph)

        return grains, morphologies

    def _compute_statistics(
        self, morphologies: list[GrainMorphology]
    ) -> GrainStatistics | None:
        """Compute aggregate statistics from a list of morphologies.

        Returns:
            GrainStatistics if there are morphologies, otherwise None.
        """
        if not morphologies:
            return None
        return compute_statistics(morphologies)
