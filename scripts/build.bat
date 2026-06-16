@echo off
REM Build script for SandAnalyze on Windows
REM Requires: Python 3.13, uv

echo Building SandAnalyze for Windows...

REM Check for uv
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -Command "& { $env:UV_INSTALL_DIR = 'C:\uv'; irm https://astral.sh/uv/install.ps1 | iex }"
    set "PATH=%PATH%;C:\uv"
)

REM Install dependencies
echo Installing dependencies...
uv sync --extra dev

REM Install PyInstaller
uv pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    uv pip install pyinstaller
)

REM Build executable
echo Building executable with PyInstaller...
uv run pyinstaller ^
    --onefile ^
    --name sandanalyze ^
    --add-data "core;core" ^
    --add-data "gui;gui" ^
    --hidden-import PyQt6.sip ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import matplotlib ^
    --hidden-import scipy ^
    --hidden-import ultralytics ^
    main.py

REM Package
echo Packaging...
cd dist
powershell -Command "Compress-Archive -Path sandanalyze.exe -DestinationPath sandanalyze-windows-x86_64.zip"
cd ..

echo Build complete!
echo Created: dist/sandanalyze-windows-x86_64.zip
