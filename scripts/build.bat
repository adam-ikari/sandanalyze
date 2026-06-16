@echo off
REM Build script for SandAnalyze on Windows
REM Requires: Python 3.13, uv
REM
REM Usage:
REM   scripts\build.bat          - Build for current platform
REM   scripts\build.bat --clean   - Clean build directories first

setlocal enabledelayedexpansion

set "PROJECT_NAME=sandanalyze"
set "PROJECT_ROOT=%~dp0.."
set "BUILD_DIR=%PROJECT_ROOT%\build"
set "DIST_DIR=%PROJECT_ROOT%\dist"

REM Parse arguments
set "CLEAN=false"
if "%~1"=="--clean" set "CLEAN=true"

REM Detect architecture
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if "%ARCH%"=="AMD64" set "ARCH=x86_64"

set "TARGET=windows"

echo ========================================
echo Building %PROJECT_NAME%
echo Platform: %TARGET% (%ARCH%)
echo ========================================

REM Clean previous builds if requested
if "%CLEAN%"=="true" (
    echo Cleaning previous builds...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
)

REM Create directories
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"

REM Check for uv
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -Command "& { irm https://astral.sh/uv/install.ps1 | iex }"
    set "PATH=%PATH%;%USERPROFILE%\.local\bin"
)

cd /d "%PROJECT_ROOT%"

REM Install dependencies
echo Installing dependencies...
uv sync --extra dev

REM Install PyInstaller
uv pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    uv pip install pyinstaller
)

REM Download YOLO model if not present
echo Checking for YOLO model...
set "MODEL_DIR=%PROJECT_ROOT%\models"
set "MODEL_FILE=%MODEL_DIR%\yolov8n-seg.pt"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if not exist "%MODEL_FILE%" (
    echo Downloading YOLO model...
    uv run python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" 2>nul

    REM Check if model was downloaded to default location
    set "DEFAULT_MODEL=%USERPROFILE%\.ultralytics\models\yolov8n-seg.pt"
    if exist "%DEFAULT_MODEL%" (
        copy "%DEFAULT_MODEL%" "%MODEL_FILE%"
        echo Model downloaded to: %MODEL_FILE%
    ) else (
        echo Warning: Could not download YOLO model. Build will continue without it.
    )
) else (
    echo YOLO model already exists: %MODEL_FILE%
)

REM Build executable
echo Building executable with PyInstaller...
uv run pyinstaller ^
    --onefile ^
    --name %PROJECT_NAME% ^
    --add-data "core;core" ^
    --add-data "gui;gui" ^
    --add-data "models;models" ^
    --hidden-import PyQt6.sip ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import matplotlib ^
    --hidden-import scipy ^
    --hidden-import ultralytics ^
    --distpath "%DIST_DIR%" ^
    --workpath "%BUILD_DIR%" ^
    main.py

REM Package
echo Packaging...
cd /d "%DIST_DIR%"
powershell -Command "Compress-Archive -Path '%PROJECT_NAME%.exe' -DestinationPath '%PROJECT_NAME%-%TARGET%-%ARCH%.zip'"

echo.
echo ========================================
echo Build complete!
echo Executable: %DIST_DIR%\%PROJECT_NAME%.exe
echo Package:    %DIST_DIR%\%PROJECT_NAME%-%TARGET%-%ARCH%.zip
echo ========================================
