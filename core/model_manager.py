"""YOLO model management for offline deployment.

Handles model download, caching, and bundling with the application.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME = "yolov8n-seg.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt"


def get_model_dir() -> Path:
    """Get the directory where models are stored.

    Priority:
    1. Bundled models/ directory (for PyInstaller builds)
    2. Project models/ directory (for development)
    3. User cache directory

    Returns
    -------
    Path
        Path to the models directory.
    """
    # Check for bundled models (PyInstaller)
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        bundled = Path(sys._MEIPASS) / "models"  # type: ignore[attr-defined]
        if bundled.exists():
            return bundled

    # Development: project models/ directory
    project_models = Path(__file__).parent.parent / "models"
    if project_models.exists():
        return project_models

    # Fallback: user cache directory
    cache_dir = Path.home() / ".cache" / "sandanalyze" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path | None:
    """Get the path to a model file.

    Parameters
    ----------
    model_name : str
        Name of the model file.

    Returns
    -------
    Path | None
        Path to the model file, or None if not found.
    """
    model_dir = get_model_dir()
    model_path = model_dir / model_name

    if model_path.exists():
        return model_path

    return None


def download_model(model_name: str = DEFAULT_MODEL_NAME, force: bool = False) -> Path:
    """Download a YOLO model if not already present.

    Parameters
    ----------
    model_name : str
        Name of the model file to download.
    force : bool
        If True, re-download even if the file exists.

    Returns
    -------
    Path
        Path to the downloaded model file.

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    model_dir = get_model_dir()
    model_path = model_dir / model_name

    if model_path.exists() and not force:
        logger.info("Model already exists: %s", model_path)
        return model_path

    logger.info("Downloading model %s...", model_name)

    try:
        # Try to use ultralytics to download the model
        from ultralytics import YOLO

        # This will download the model to the default location
        model = YOLO(model_name)

        # Move/copy to our models directory if needed
        if not model_path.exists():
            # Find where ultralytics downloaded it
            default_path = Path.home() / ".ultralytics" / "models" / model_name
            if default_path.exists():
                import shutil

                shutil.copy2(default_path, model_path)
                logger.info("Model copied to: %s", model_path)

        return model_path

    except Exception as exc:
        logger.error("Failed to download model: %s", exc)
        raise RuntimeError(f"Failed to download model {model_name}: {exc}") from exc


def ensure_model_available(model_name: str = DEFAULT_MODEL_NAME) -> Path | None:
    """Ensure a model is available, downloading if necessary.

    Parameters
    ----------
    model_name : str
        Name of the model file.

    Returns
    -------
    Path | None
        Path to the model file, or None if unavailable.
    """
    # Check if model already exists
    model_path = get_model_path(model_name)
    if model_path is not None:
        return model_path

    # Try to download
    try:
        return download_model(model_name)
    except Exception:
        logger.warning("Model %s not available offline", model_name)
        return None
