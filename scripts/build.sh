#!/usr/bin/env bash
# Build script for SandAnalyze
# Supports: Linux, macOS, Windows (via WSL or cross-compilation)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="sandanalyze"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Linux*)     TARGET="linux";;
    Darwin*)    TARGET="macos";;
    CYGWIN*|MINGW*|MSYS*)
        TARGET="windows";;
    *)
        echo "Unsupported OS: ${OS}"
        exit 1;;
esac

echo "Building ${PROJECT_NAME} for ${TARGET} (${ARCH})..."

# Clean previous builds
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install dependencies
echo "Installing dependencies..."
uv sync --extra dev

# Install PyInstaller if not present
if ! uv pip show pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    uv pip install pyinstaller
fi

# Build executable
echo "Building executable with PyInstaller..."
uv run pyinstaller \
    --onefile \
    --name "${PROJECT_NAME}" \
    --add-data "core:core" \
    --add-data "gui:gui" \
    --hidden-import PyQt6.sip \
    --hidden-import cv2 \
    --hidden-import numpy \
    --hidden-import matplotlib \
    --hidden-import scipy \
    --hidden-import ultralytics \
    main.py

# Package
echo "Packaging..."
case "${TARGET}" in
    linux|macos)
        cd dist
        tar -czf "${PROJECT_NAME}-${TARGET}-${ARCH}.tar.gz" "${PROJECT_NAME}"
        echo "Created: dist/${PROJECT_NAME}-${TARGET}-${ARCH}.tar.gz"
        ;;
    windows)
        cd dist
        7z a "${PROJECT_NAME}-${TARGET}-${ARCH}.zip" "${PROJECT_NAME}.exe"
        echo "Created: dist/${PROJECT_NAME}-${TARGET}-${ARCH}.zip"
        ;;
esac

echo "Build complete!"
