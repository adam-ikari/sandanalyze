# SandAnalyze - Sand Grain Morphology Analysis System

A sand grain detection, shape analysis, and statistical counting tool based on traditional OpenCV image processing methods.

## Features

- **Grain Detection**: Traditional OpenCV contour detection with edge filtering and flocculation detection
- **Multi-dimensional Morphology**: Circularity, sphericity, aspect ratio, convexity, Feret diameter, etc.
- **Four-Class Classification**: Zingg shape classification (spherical, rod-like, discoidal) + flocculation detection
- **Statistical Analysis**: Size distribution, circularity-sphericity scatter plots, classification distribution
- **Web Interface**: Streamlit interactive interface, runs in browser
- **Export**: CSV data, annotated PNG images, PDF reports
- **Batch Processing**: Process multiple images in a directory with summary statistics

## Installation

```bash
uv sync --extra dev
```

## Usage

```bash
# Launch web application
uv run python main.py

# Or use streamlit directly
streamlit run app.py

# Specify port
uv run python main.py --server.port 8080
```

1. Upload sand grain image in the sidebar
2. Adjust preprocessing parameters (optional: enable auto-tune)
3. Configure detection options (edge filtering, flocculation detection)
4. Click "Run Detection"
5. View statistical summary, grain data table, and Plotly interactive charts
6. Export CSV, annotated image, or PDF report

## Morphology Parameters

| Parameter | Calculation Method | Geological Significance |
|-----------|-------------------|------------------------|
| Area (A) | Mask pixel count | Particle size basis |
| Perimeter (P) | Contour length | Abrasion degree |
| Circularity | 4pi*A/P^2 | Closer to 1 = more circular |
| Equivalent Diameter (d_eq) | sqrt(4A/pi) | Equivalent circle diameter |
| Aspect Ratio (AR) | Major/Minor axis | Flattening degree |
| Sphericity | Minor/Major axis | 3D shape inference |
| Convexity | Area/Convex hull area | Surface roughness |

## Classification System

### Zingg Shape Classification

| Class | Aspect Ratio Range | Color |
|-------|-------------------|-------|
| Spherical | < 1.5 | Green |
| Rod-like | 1.5 - 2.5 | Red |
| Discoidal | >= 2.5 | Blue |

### Flocculation Detection

Flocculation (grain clusters) are detected based on combined criteria:
- Large area (configurable thresholds)
- Low circularity
- Low convexity
- High aspect ratio

Detected flocculation grains are colored yellow and take priority over Zingg classification.

## Edge Filtering

Grains touching or too close to the image border can be automatically excluded to avoid incomplete grain measurements. Configurable border margin in pixels.

## Batch Processing

Process all images in a directory:

```python
from core.batch import process_batch

summary = process_batch(
    input_dir="/path/to/images",
    output_dir="/path/to/output",
    border_margin=5,
    generate_pdf=True,
)

print(f"Processed: {summary.total_images}")
print(f"Success rate: {summary.success_rate:.1f}%")
print(f"Total grains: {summary.total_grains}")
```

## Tech Stack

- Python >= 3.10
- OpenCV (image processing)
- Streamlit (Web UI)
- Plotly (interactive charts)
- ReportLab (PDF generation)
- numpy / scipy (numerical computing)

## Project Structure

```
sandanalyze/
├── app.py                 # Streamlit web application
├── core/
│   ├── __init__.py        # Package exports
│   ├── batch.py           # Batch processing
│   ├── classifier.py      # Zingg + flocculation classification
│   ├── detector.py        # Grain detection with flocculation/edge filtering
│   ├── exporter.py        # CSV and image export
│   ├── morphology.py      # Morphological parameter computation
│   ├── preprocessor.py    # Image preprocessing
│   ├── report.py          # PDF report generation
│   └── traditional.py     # Traditional contour-based detection
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```
