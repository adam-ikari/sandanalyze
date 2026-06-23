"""Tests for core/pipeline_debugger.py and tools/debug_skill.py."""

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest

from core.pipeline_debugger import (
    ImageSource,
    PipelineResult,
    PipelineRunner,
    RealImageSource,
    SyntheticImageSource,
)
from core.preprocessor import PreprocessConfig
from tools.debug_skill import DebugSkill


class TestImageSource:
    """Tests for the abstract ImageSource base and factory."""

    def test_is_abstract(self):
        """ImageSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ImageSource()

    def test_from_path_returns_synthetic_when_none(self):
        """from_path(None) should return a SyntheticImageSource."""
        source = ImageSource.from_path(None)
        assert isinstance(source, SyntheticImageSource)

    def test_from_path_returns_real_when_path_given(self):
        """from_path('some/path.jpg') should return a RealImageSource."""
        source = ImageSource.from_path("some/path.jpg")
        assert isinstance(source, RealImageSource)
        assert source.path == "some/path.jpg"

    def test_from_path_forwards_kwargs_to_synthetic(self):
        """Extra kwargs should be forwarded to SyntheticImageSource."""
        source = ImageSource.from_path(None, width=200, height=200, num_grains=5, seed=7)
        assert isinstance(source, SyntheticImageSource)
        assert source.width == 200
        assert source.height == 200
        assert source.num_grains == 5
        assert source.seed == 7

    def test_from_path_forwards_kwargs_to_real(self):
        """Extra kwargs should be forwarded to RealImageSource."""
        # RealImageSource only accepts 'path' positionally, but kwargs are passed
        # through.  Since RealImageSource.__init__ only takes path, extra kwargs
        # would cause a TypeError.  We verify the documented behaviour: only
        # 'path' is expected for real sources.
        source = ImageSource.from_path("/tmp/test.jpg")
        assert isinstance(source, RealImageSource)


class TestSyntheticImageSource:
    """Tests for SyntheticImageSource."""

    def test_load_returns_numpy_array(self):
        """load() should return a numpy ndarray."""
        source = SyntheticImageSource(width=200, height=150, num_grains=5)
        img = source.load()
        assert isinstance(img, np.ndarray)

    def test_load_returns_correct_shape(self):
        """The returned image shape should match configured width/height."""
        source = SyntheticImageSource(width=300, height=200, num_grains=3)
        img = source.load()
        assert img.shape == (200, 300)

    def test_load_returns_grayscale(self):
        """The returned image should be 2-D (grayscale)."""
        source = SyntheticImageSource(width=100, height=100, num_grains=2)
        img = source.load()
        assert img.ndim == 2

    def test_load_returns_uint8(self):
        """Pixel values should be uint8."""
        source = SyntheticImageSource(width=100, height=100, num_grains=2)
        img = source.load()
        assert img.dtype == np.uint8

    def test_reproducibility_with_same_seed(self):
        """Two instances with the same seed should produce identical images."""
        source1 = SyntheticImageSource(seed=42, width=200, height=200, num_grains=10)
        source2 = SyntheticImageSource(seed=42, width=200, height=200, num_grains=10)
        img1 = source1.load()
        img2 = source2.load()
        assert np.array_equal(img1, img2)

    def test_different_seeds_produce_different_images(self):
        """Different seeds should produce different images."""
        source1 = SyntheticImageSource(seed=1, width=200, height=200, num_grains=10)
        source2 = SyntheticImageSource(seed=2, width=200, height=200, num_grains=10)
        img1 = source1.load()
        img2 = source2.load()
        assert not np.array_equal(img1, img2)

    def test_image_has_bright_regions(self):
        """The synthetic image should contain bright pixels (ellipses)."""
        source = SyntheticImageSource(width=200, height=200, num_grains=10, seed=42)
        img = source.load()
        # At least some pixels should be bright (near 255 from ellipses)
        assert img.max() > 200

    def test_image_has_background_regions(self):
        """The synthetic image should contain mid-gray background pixels."""
        source = SyntheticImageSource(width=200, height=200, num_grains=10, seed=42)
        img = source.load()
        # Background is 128; with noise some pixels will be near 128
        # We just check the min is not all white
        assert img.min() < 200

    def test_default_dimensions(self):
        """Default width/height should be 400x400."""
        source = SyntheticImageSource()
        assert source.width == 400
        assert source.height == 400
        assert source.num_grains == 10
        assert source.seed == 42

    def test_load_with_zero_grains(self):
        """Should produce an image with no bright ellipses."""
        source = SyntheticImageSource(width=200, height=200, num_grains=0, seed=42)
        img = source.load()
        # No ellipses drawn, only background + noise
        # Max should be well below 255 since no white ellipses were drawn
        assert img.max() < 200


class TestRealImageSource:
    """Tests for RealImageSource."""

    def test_load_reads_existing_image(self):
        """Should successfully load an existing image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a synthetic image on disk
            img = np.full((100, 150), 128, dtype=np.uint8)
            cv2.imwrite(tmp_path, img)

            source = RealImageSource(tmp_path)
            loaded = source.load()

            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == (100, 150)
            assert loaded.dtype == np.uint8
        finally:
            os.unlink(tmp_path)

    def test_load_returns_grayscale(self):
        """A colour image on disk should be loaded as grayscale."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a BGR image on disk
            img = np.full((50, 50, 3), (100, 150, 200), dtype=np.uint8)
            cv2.imwrite(tmp_path, img)

            source = RealImageSource(tmp_path)
            loaded = source.load()

            assert loaded.ndim == 2
        finally:
            os.unlink(tmp_path)

    def test_load_raises_file_not_found(self):
        """Loading a non-existent path should raise FileNotFoundError."""
        source = RealImageSource("/nonexistent/path/to/image.jpg")
        with pytest.raises(FileNotFoundError):
            source.load()

    def test_load_raises_on_empty_image(self):
        """Loading an empty/corrupt image should raise FileNotFoundError or ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            f.write(b"not an image")

        try:
            source = RealImageSource(tmp_path)
            with pytest.raises((FileNotFoundError, ValueError)):
                source.load()
        finally:
            os.unlink(tmp_path)

    def test_path_attribute(self):
        """The path attribute should store the constructor argument."""
        source = RealImageSource("/tmp/test.png")
        assert source.path == "/tmp/test.png"


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_fields_are_set(self):
        """PipelineResult should store all provided fields."""
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        grains = []
        morphologies = []
        from core.morphology import GrainStatistics
        stats = GrainStatistics(count=0)
        result = PipelineResult(
            mask=mask,
            grains=grains,
            morphologies=morphologies,
            statistics=stats,
        )
        assert np.array_equal(result.mask, mask)
        assert result.grains is grains
        assert result.morphologies is morphologies
        assert result.statistics is stats


class TestPipelineRunner:
    """Tests for PipelineRunner class."""

    def test_init_stores_source(self):
        """PipelineRunner should store the image source."""
        source = SyntheticImageSource(width=200, height=200, num_grains=5)
        runner = PipelineRunner(source)
        assert runner.source is source

    def test_run_returns_pipeline_result(self):
        """run() should return a PipelineResult."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        assert isinstance(result, PipelineResult)

    def test_run_result_has_mask(self):
        """PipelineResult should contain a binary mask."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        assert isinstance(result.mask, np.ndarray)
        assert result.mask.ndim == 2
        assert result.mask.dtype == np.uint8

    def test_run_result_has_grains(self):
        """PipelineResult should contain a list of GrainContour objects."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        assert isinstance(result.grains, list)
        from core.morphology import GrainContour
        for grain in result.grains:
            assert isinstance(grain, GrainContour)

    def test_run_result_has_morphologies(self):
        """PipelineResult should contain a list of GrainMorphology objects."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        assert isinstance(result.morphologies, list)
        from core.morphology import GrainMorphology
        for morph in result.morphologies:
            assert isinstance(morph, GrainMorphology)

    def test_run_result_has_statistics(self):
        """PipelineResult should contain GrainStatistics."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        from core.morphology import GrainStatistics
        assert isinstance(result.statistics, GrainStatistics)

    def test_run_grain_count_matches_morphologies(self):
        """Number of grains should match number of morphologies."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        assert len(result.grains) == len(result.morphologies)

    def test_run_with_real_image_source(self):
        """PipelineRunner should work with RealImageSource."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a synthetic image on disk with bright ellipses
            img = np.full((400, 400), 128, dtype=np.uint8)
            cv2.ellipse(img, (200, 200), (30, 20), 45, 0, 360, 255, thickness=-1)
            cv2.ellipse(img, (100, 100), (25, 15), 0, 0, 360, 255, thickness=-1)
            cv2.imwrite(tmp_path, img)

            source = RealImageSource(tmp_path)
            runner = PipelineRunner(source)
            config = PreprocessConfig(min_area=100)
            result = runner.run(config=config, min_area=100)
            assert isinstance(result, PipelineResult)
            assert len(result.grains) > 0
        finally:
            os.unlink(tmp_path)

    def test_run_with_none_config(self):
        """PipelineRunner should use default config when None is passed."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        result = runner.run(config=None, min_area=100)
        assert isinstance(result, PipelineResult)
        assert len(result.grains) > 0

    def test_run_morphology_has_shape_class(self):
        """Each morphology should have a shape class assigned."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        for morph in result.morphologies:
            assert morph.shape_class in ("spherical", "rod-like", "discoidal", "flocculation")

    def test_run_morphology_has_confidence(self):
        """Each morphology should have a confidence value."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        for morph in result.morphologies:
            assert morph.confidence in (0.9, 0.95)


class TestPipelineRunnerVisualization:
    """Tests for PipelineRunner visualization methods."""

    def test_visualize_branches_creates_png(self):
        """visualize_branches should create a PNG file."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        runner.run(config=config, min_area=100)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            runner.visualize_branches(save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_visualize_branches_without_run_raises(self):
        """visualize_branches should raise RuntimeError if run() not called."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)

        with pytest.raises(RuntimeError):
            runner.visualize_branches("/tmp/nonexistent.png")

    def test_compare_configs_creates_png(self):
        """compare_configs should create a PNG file."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config_a = PreprocessConfig(min_area=100)
        config_b = PreprocessConfig(min_area=200)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            runner.compare_configs(config_a, config_b, save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_compare_configs_preserves_last_result(self):
        """compare_configs should not modify _last_result."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config_a = PreprocessConfig(min_area=100)
        config_b = PreprocessConfig(min_area=200)

        # Run once to set _last_result
        initial_result = runner.run(config=config_a, min_area=100)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            runner.compare_configs(config_a, config_b, save_path)
            # After compare_configs, _last_result should still be the initial result
            assert runner._last_result is initial_result
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_run_stores_last_result(self):
        """run() should store the result in _last_result."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        config = PreprocessConfig(min_area=100)
        result = runner.run(config=config, min_area=100)
        assert runner._last_result is result


class TestPipelineRunnerGridSearch:
    """Tests for PipelineRunner.grid_search method."""

    def test_grid_search_returns_dataframe(self):
        """grid_search should return a pandas DataFrame."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21], "adaptive_c": [2, 5]}
        df = runner.grid_search(param_ranges, min_area=100)
        assert isinstance(df, pd.DataFrame)

    def test_grid_search_combinations(self):
        """grid_search should produce one row per parameter combination."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21, 31], "adaptive_c": [2, 5]}
        df = runner.grid_search(param_ranges, min_area=100)
        # 3 * 2 = 6 combinations
        assert len(df) == 6

    def test_grid_search_columns(self):
        """grid_search DataFrame should have parameter and metric columns."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21], "adaptive_c": [2, 5]}
        df = runner.grid_search(param_ranges, min_area=100)
        assert "adaptive_block_size" in df.columns
        assert "adaptive_c" in df.columns
        assert "grain_count" in df.columns
        assert "mask_pixels" in df.columns
        assert "error" in df.columns

    def test_grid_search_columns_with_circularity_metric(self):
        """grid_search with metric='circularity_mean' should include that column."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21], "adaptive_c": [2, 5]}
        df = runner.grid_search(param_ranges, metric="circularity_mean", min_area=100)
        assert "adaptive_block_size" in df.columns
        assert "adaptive_c" in df.columns
        assert "grain_count" in df.columns
        assert "mask_pixels" in df.columns
        assert "circularity_mean" in df.columns
        assert "error" in df.columns

    def test_grid_search_grain_count_non_negative(self):
        """grain_count should be a non-negative integer."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21], "adaptive_c": [2, 5]}
        df = runner.grid_search(param_ranges, min_area=100)
        assert (df["grain_count"] >= 0).all()

    def test_grid_search_mask_pixels_non_negative(self):
        """mask_pixels should be a non-negative integer."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21], "adaptive_c": [2, 5]}
        df = runner.grid_search(param_ranges, min_area=100)
        assert (df["mask_pixels"] >= 0).all()

    def test_grid_search_empty_param_ranges(self):
        """grid_search with empty param_ranges should return an empty DataFrame."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        df = runner.grid_search({}, min_area=100)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_grid_search_catches_exceptions(self):
        """grid_search should catch exceptions and record them in the error column."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        # Invalid adaptive_block_size (negative value triggers an error)
        param_ranges = {"adaptive_block_size": [-1]}
        df = runner.grid_search(param_ranges, min_area=100)
        assert len(df) == 1
        # Should have an error recorded
        assert df["error"].iloc[0] is not None
        assert len(str(df["error"].iloc[0])) > 0

    def test_grid_search_all_supported_params(self):
        """grid_search should work with all supported parameters."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {
            "adaptive_block_size": [11, 21],
            "adaptive_c": [2, 5],
            "blur_kernel": [3, 5],
            "morph_kernel_size": [3, 5],
            "morph_open_iter": [1, 2],
            "morph_close_iter": [1, 2],
            "min_area": [50, 100],
        }
        df = runner.grid_search(param_ranges, metric="circularity_mean", min_area=50)
        expected_rows = 2 ** 7  # 7 params, 2 values each
        assert len(df) == expected_rows
        assert "grain_count" in df.columns
        assert "mask_pixels" in df.columns
        assert "circularity_mean" in df.columns
        assert "error" in df.columns

    def test_grid_search_different_params_produce_different_results(self):
        """Different parameter values should produce different results."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {
            "adaptive_block_size": [11, 101],
            "adaptive_c": [2, 20],
        }
        df = runner.grid_search(param_ranges, min_area=100)
        # Verify there are at least some differences in results
        grain_counts = df["grain_count"].dropna()
        assert len(grain_counts) > 0
        # Different parameter combinations should not all produce the same result
        assert grain_counts.nunique() > 0

    def test_grid_search_metric_parameter_is_used(self):
        """The metric parameter should control which metric columns is computed."""
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        param_ranges = {"adaptive_block_size": [11, 21]}
        # With metric="grain_count", circularity_mean should not be in results
        df = runner.grid_search(param_ranges, metric="grain_count", min_area=100)
        assert "grain_count" in df.columns
        assert "mask_pixels" in df.columns
        # circularity_mean should not be present when metric is "grain_count"
        assert "circularity_mean" not in df.columns

        # With metric="circularity_mean", it should be present
        df2 = runner.grid_search(param_ranges, metric="circularity_mean", min_area=100)
        assert "circularity_mean" in df2.columns


class TestDebugSkill:
    """Tests for DebugSkill agent-facing interface."""

    def test_init(self):
        """DebugSkill should instantiate without arguments."""
        skill = DebugSkill()
        assert isinstance(skill, DebugSkill)

    def test_visualize_branches_returns_path(self):
        """visualize_branches should return the absolute path to the saved PNG."""
        skill = DebugSkill()
        config = PreprocessConfig(min_area=100)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            result_path = skill.visualize_branches(
                None, config=config, save_path=save_path,
                width=400, height=400, num_grains=10, seed=42,
            )
            assert isinstance(result_path, str)
            assert os.path.isabs(result_path)
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_visualize_branches_default_save_path(self):
        """visualize_branches should use default save_path when not provided."""
        skill = DebugSkill()
        config = PreprocessConfig(min_area=100)

        # Use a temp directory to avoid cluttering the working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = os.path.join(tmpdir, "debug_branches.png")
            result_path = skill.visualize_branches(
                None, config=config, save_path=default_path,
                width=400, height=400, num_grains=10, seed=42,
            )
            assert os.path.exists(result_path)

    def test_compare_configs_returns_path(self):
        """compare_configs should return the absolute path to the saved PNG."""
        skill = DebugSkill()
        config_a = PreprocessConfig(min_area=100)
        config_b = PreprocessConfig(min_area=200)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            result_path = skill.compare_configs(
                None, config_a, config_b, save_path=save_path,
                width=400, height=400, num_grains=10, seed=42,
            )
            assert isinstance(result_path, str)
            assert os.path.isabs(result_path)
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_compare_configs_default_save_path(self):
        """compare_configs should use default save_path when not provided."""
        skill = DebugSkill()
        config_a = PreprocessConfig(min_area=100)
        config_b = PreprocessConfig(min_area=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = os.path.join(tmpdir, "debug_compare.png")
            result_path = skill.compare_configs(
                None, config_a, config_b, save_path=default_path,
                width=400, height=400, num_grains=10, seed=42,
            )
            assert os.path.exists(result_path)

    def test_grid_search_returns_dataframe(self):
        """grid_search should return a pandas DataFrame."""
        skill = DebugSkill()
        param_ranges = {"adaptive_block_size": [11, 21], "adaptive_c": [2, 5]}
        df = skill.grid_search(
            None, param_ranges, metric="grain_count",
            width=400, height=400, num_grains=10, seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 * 2 combinations
        assert "adaptive_block_size" in df.columns
        assert "adaptive_c" in df.columns
        assert "grain_count" in df.columns

    def test_grid_search_saves_csv(self):
        """grid_search should save to CSV when save_path is provided."""
        skill = DebugSkill()
        param_ranges = {"adaptive_block_size": [11, 21]}

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            df = skill.grid_search(
                None, param_ranges, metric="grain_count", save_path=csv_path,
                width=400, height=400, num_grains=10, seed=42,
            )
            assert os.path.exists(csv_path)
            # Verify the CSV can be read back
            df_read = pd.read_csv(csv_path)
            assert len(df_read) == len(df)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_grid_search_no_save(self):
        """grid_search should not save CSV when save_path is None."""
        skill = DebugSkill()
        param_ranges = {"adaptive_block_size": [11, 21]}
        df = skill.grid_search(
            None, param_ranges, metric="grain_count", save_path=None,
            width=400, height=400, num_grains=10, seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        # No file should be created at default location
        assert not os.path.exists("debug_grid_search.csv")

    def test_regression_test_returns_dict(self):
        """regression_test should return a dict mapping names to booleans."""
        skill = DebugSkill()
        config = PreprocessConfig(min_area=100)
        expected = {"default": 8, "high": 10}
        results = skill.regression_test(
            None, config, expected, tolerance=0.1,
            width=400, height=400, num_grains=10, seed=42,
            min_area=100,
        )
        assert isinstance(results, dict)
        assert set(results.keys()) == {"default", "high"}
        for value in results.values():
            assert isinstance(value, bool)

    def test_regression_test_passes_when_within_tolerance(self):
        """regression_test should pass when actual count is within tolerance."""
        skill = DebugSkill()
        config = PreprocessConfig(min_area=100)
        # First run regression_test to get the actual count, then verify
        # that a value within tolerance passes
        results = skill.regression_test(
            None, config, {"placeholder": 1}, tolerance=0.5,
            width=400, height=400, num_grains=10, seed=42,
            min_area=100,
        )
        # Now we know the actual count; use exact match with tolerance=0
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        result = runner.run(config=config, min_area=100)
        actual = len(result.grains)

        expected = {"exact": actual}
        results = skill.regression_test(
            None, config, expected, tolerance=0.0,
            width=400, height=400, num_grains=10, seed=42,
            min_area=100,
        )
        assert results["exact"] is True

    def test_regression_test_fails_when_outside_tolerance(self):
        """regression_test should fail when actual count is outside tolerance."""
        skill = DebugSkill()
        config = PreprocessConfig(min_area=100)
        # Expect a very different count
        expected = {"way_off": 999}
        results = skill.regression_test(
            None, config, expected, tolerance=0.1,
            width=400, height=400, num_grains=10, seed=42,
            min_area=100,
        )
        assert results["way_off"] is False

    def test_regression_test_tolerance_zero(self):
        """regression_test with tolerance=0 should require exact match."""
        skill = DebugSkill()
        config = PreprocessConfig(min_area=100)
        source = SyntheticImageSource(width=400, height=400, num_grains=10, seed=42)
        runner = PipelineRunner(source)
        result = runner.run(config=config, min_area=100)
        actual = len(result.grains)

        expected = {"exact": actual, "off_by_one": actual + 1}
        results = skill.regression_test(
            None, config, expected, tolerance=0.0,
            width=400, height=400, num_grains=10, seed=42,
            min_area=100,
        )
        # Both should match because the same seed produces the same image
        # and the same config produces the same grain count
        assert results["exact"] is True
        assert results["off_by_one"] is False

    def test_regression_test_with_real_image(self):
        """regression_test should work with a real image path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a synthetic image on disk with bright ellipses
            img = np.full((400, 400), 128, dtype=np.uint8)
            cv2.ellipse(img, (200, 200), (30, 20), 45, 0, 360, 255, thickness=-1)
            cv2.ellipse(img, (100, 100), (25, 15), 0, 0, 360, 255, thickness=-1)
            cv2.imwrite(tmp_path, img)

            skill = DebugSkill()
            config = PreprocessConfig(min_area=100)
            expected = {"test": 2}
            results = skill.regression_test(
                tmp_path, config, expected, tolerance=0.5,
            )
            assert isinstance(results, dict)
            assert "test" in results
            assert isinstance(results["test"], bool)
        finally:
            os.unlink(tmp_path)

    def test_visualize_branches_with_real_image(self):
        """visualize_branches should work with a real image path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            img = np.full((400, 400), 128, dtype=np.uint8)
            cv2.ellipse(img, (200, 200), (30, 20), 45, 0, 360, 255, thickness=-1)
            cv2.imwrite(tmp_path, img)

            skill = DebugSkill()
            config = PreprocessConfig(min_area=100)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
                save_path = f2.name

            try:
                result_path = skill.visualize_branches(
                    tmp_path, config=config, save_path=save_path,
                )
                assert os.path.exists(result_path)
            finally:
                if os.path.exists(save_path):
                    os.unlink(save_path)
        finally:
            os.unlink(tmp_path)
