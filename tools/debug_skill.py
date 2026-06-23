"""Agent-facing debug skill for sand grain detection parameter tuning.

Provides a high-level :class:`DebugSkill` interface that wraps the lower-level
:class:`PipelineRunner` and :class:`ImageSource` classes from
:mod:`core.pipeline_debugger`.  This skill is designed to be used by an AI
agent when debugging or optimising detection parameters.

Example::

    from tools.debug_skill import DebugSkill

    skill = DebugSkill()

    # Visualise pipeline output
    skill.visualize_branches("image.jpg", save_path="debug.png")

    # Compare two configurations
    from core.preprocessor import PreprocessConfig
    cfg_a = PreprocessConfig(adaptive_c=2)
    cfg_b = PreprocessConfig(adaptive_c=5)
    skill.compare_configs("image.jpg", cfg_a, cfg_b, save_path="compare.png")

    # Grid search over parameters
    df = skill.grid_search(
        "image.jpg",
        param_ranges={"adaptive_c": [2, 5, 10], "blur_kernel": [3, 5]},
    )

    # Regression test
    results = skill.regression_test(
        "image.jpg",
        config=PreprocessConfig(),
        expected_counts={"default": 10, "macro_sand": 8},
    )
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from core.pipeline_debugger import ImageSource, PipelineRunner
from core.preprocessor import PreprocessConfig


class DebugSkill:
    """High-level debugging interface for sand grain detection parameters.

    This class wraps :class:`PipelineRunner` and provides self-contained
    methods that create an image source, run the pipeline, and return
    results in an agent-friendly format.
    """

    def visualize_branches(
        self,
        image_path: str | None,
        config: PreprocessConfig | None = None,
        save_path: str = "debug_branches.png",
        **source_kwargs: Any,
    ) -> str:
        """Load an image, run the pipeline, and visualise the results.

        Args:
            image_path: Path to a real image, or ``None`` to use a synthetic image.
            config: Preprocessing configuration. Uses defaults if ``None``.
            save_path: Path where the PNG figure will be saved.
            **source_kwargs: Forwarded to :meth:`ImageSource.from_path`.
                For synthetic images, useful kwargs include ``width``,
                ``height``, ``num_grains``, and ``seed``.

        Returns:
            Absolute path to the saved PNG file.
        """
        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        runner.run(config=config)
        runner.visualize_branches(save_path)
        return os.path.abspath(save_path)

    def compare_configs(
        self,
        image_path: str | None,
        config_a: PreprocessConfig,
        config_b: PreprocessConfig,
        save_path: str = "debug_compare.png",
        **source_kwargs: Any,
    ) -> str:
        """Load an image and compare two preprocessing configurations side-by-side.

        Args:
            image_path: Path to a real image, or ``None`` to use a synthetic image.
            config_a: First preprocessing configuration.
            config_b: Second preprocessing configuration.
            save_path: Path where the PNG comparison figure will be saved.
            **source_kwargs: Forwarded to :meth:`ImageSource.from_path`.

        Returns:
            Absolute path to the saved PNG file.
        """
        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        runner.compare_configs(config_a, config_b, save_path)
        return os.path.abspath(save_path)

    def grid_search(
        self,
        image_path: str | None,
        param_ranges: dict[str, list],
        metric: str = "grain_count",
        save_path: str | None = None,
        **source_kwargs: Any,
    ) -> pd.DataFrame:
        """Run a grid search over preprocessing parameter ranges.

        For every combination of values in *param_ranges*, the pipeline is
        executed and the requested metric is recorded.

        Args:
            image_path: Path to a real image, or ``None`` to use a synthetic image.
            param_ranges: Mapping from parameter name to a list of values.
                Supported keys: ``adaptive_block_size``, ``adaptive_c``,
                ``blur_kernel``, ``morph_kernel_size``, ``morph_open_iter``,
                ``morph_close_iter``, ``min_area``.
            metric: Metric to extract from each run. One of
                ``"grain_count"``, ``"mask_pixels"``, ``"circularity_mean"``.
            save_path: Optional path to save the results as a CSV file.
                If ``None``, no CSV is written.
            **source_kwargs: Forwarded to :meth:`ImageSource.from_path`.

        Returns:
            A :class:`pandas.DataFrame` with one row per parameter combination.
        """
        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        df = runner.grid_search(param_ranges, metric=metric)

        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df

    def regression_test(
        self,
        image_path: str | None,
        config: PreprocessConfig,
        expected_counts: dict[str, int],
        tolerance: float = 0.1,
        min_area: int = 1000,
        max_area: int = 15000,
        **source_kwargs: Any,
    ) -> dict[str, bool]:
        """Run the pipeline and compare grain counts against expected values.

        Args:
            image_path: Path to a real image, or ``None`` to use a synthetic image.
            config: Preprocessing configuration to use.
            expected_counts: Mapping from test name to expected grain count.
            tolerance: Relative tolerance for pass/fail (e.g. ``0.1`` = 10%).
            min_area: Minimum grain area passed to :meth:`PipelineRunner.run`.
            max_area: Maximum grain area passed to :meth:`PipelineRunner.run`.
            **source_kwargs: Forwarded to :meth:`ImageSource.from_path`.

        Returns:
            Dictionary mapping each test name to ``True`` (passed) or
            ``False`` (failed).
        """
        source = ImageSource.from_path(image_path, **source_kwargs)
        runner = PipelineRunner(source)
        result = runner.run(config=config, min_area=min_area, max_area=max_area)
        actual_count = len(result.grains)

        outcomes: dict[str, bool] = {}
        for name, expected in expected_counts.items():
            lower = expected * (1 - tolerance)
            upper = expected * (1 + tolerance)
            outcomes[name] = lower <= actual_count <= upper

        return outcomes
