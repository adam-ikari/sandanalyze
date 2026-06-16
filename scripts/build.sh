#!/usr/bin/env bash
# Build script for SandAnalyze
# Supports: Linux, macOS
#
# Usage:
#   ./scripts/build.sh          # Build for current platform
#   ./scripts/build.sh --clean  # Clean build directories first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="sandanalyze"
BUILD_DIR="${PROJECT_ROOT}/build"
DIST_DIR="${PROJECT_ROOT}/dist"

# Parse arguments
CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean]"
            exit 1
            ;;
    esac
done

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Linux*)     TARGET="linux";;
    Darwin*)    TARGET="macos";;
    *)
        echo "Unsupported OS: ${OS}"
        echo "This script supports Linux and macOS. For Windows, use scripts/build.bat"
        exit 1;;
esac

echo "========================================"
echo "Building ${PROJECT_NAME}"
echo "Platform: ${TARGET} (${ARCH})"
echo "========================================"

# Clean previous builds if requested
if [[ "${CLEAN}" == true ]]; then
    echo "Cleaning previous builds..."
    rm -rf "${BUILD_DIR}" "${DIST_DIR}"
fi

# Create directories
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "${PROJECT_ROOT}"

# Install dependencies
echo "Installing dependencies..."
uv sync --extra dev

# Install PyInstaller if not present
if ! uv pip show pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    uv pip install pyinstaller
fi

# Download YOLO model if not present
echo "Checking for YOLO model..."
MODEL_DIR="${PROJECT_ROOT}/models"
MODEL_FILE="${MODEL_DIR}/yolov8n-seg.pt"
mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_FILE}" ]; then
    echo "Downloading YOLO model..."
    uv run python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" 2>/dev/null || true

    # Check if model was downloaded to default location
    DEFAULT_MODEL="${HOME}/.ultralytics/models/yolov8n-seg.pt"
    if [ -f "${DEFAULT_MODEL}" ]; then
        cp "${DEFAULT_MODEL}" "${MODEL_FILE}"
        echo "Model downloaded to: ${MODEL_FILE}"
    else
        echo "Warning: Could not download YOLO model. Build will continue without it."
    fi
else
    echo "YOLO model already exists: ${MODEL_FILE}"
fi

# Build executable
echo "Building executable with PyInstaller..."
uv run pyinstaller \
    --onefile \
    --name "${PROJECT_NAME}" \
    --add-data "core:core" \
    --add-data "gui:gui" \
    --add-data "models:models" \
    --hidden-import PyQt6.sip \
    --hidden-import cv2 \
    --hidden-import numpy \
    --hidden-import matplotlib \
    --hidden-import scipy \
    --hidden-import ultralytics \
    --distpath "${DIST_DIR}" \
    --workpath "${BUILD_DIR}" \
    main.py

# Package
echo "Packaging..."
cd "${DIST_DIR}"
tar -czf "${PROJECT_NAME}-${TARGET}-${ARCH}.tar.gz" "${PROJECT_NAME}"

echo ""
echo "========================================"
echo "Build complete!"
echo "Executable: ${DIST_DIR}/${PROJECT_NAME}"
echo "Package:    ${DIST_DIR}/${PROJECT_NAME}-${TARGET}-${ARCH}.tar.gz"
echo "========================================"
