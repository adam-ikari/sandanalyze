"""SandAnalyze core processing modules.

Provides grain detection, morphology computation, classification, and export utilities.
"""
from core.batch import process_batch, process_single_image, BatchResult, BatchSummary
from core.classifier import classify_grain, classify_batch, ZinggClassifier
from core.detector import detect_grains, FlocculationConfig, DetectionResult
from core.exporter import export_annotated_image, export_csv
from core.morphology import (
    GrainMorphology,
    GrainStatistics,
    GrainContour,
    compute_morphology,
    compute_statistics,
    get_classification_color,
    get_zingg_color,
    zingg_classify,
    CLASSIFICATION_COLORS,
    ZINGG_COLORS,
)
from core.preprocessor import PreprocessConfig, preprocess, auto_tune_params, filter_edge_grains
from core.report import generate_pdf_report
from core.yolo_detector import YOLODetector, refine_with_yolo

__all__ = [
    # Preprocessing
    "PreprocessConfig",
    "preprocess",
    "auto_tune_params",
    "filter_edge_grains",
    # Detection
    "detect_grains",
    "FlocculationConfig",
    "DetectionResult",
    # Classification
    "classify_grain",
    "classify_batch",
    "ZinggClassifier",
    # Morphology
    "GrainMorphology",
    "GrainStatistics",
    "GrainContour",
    "compute_morphology",
    "compute_statistics",
    "get_classification_color",
    "get_zingg_color",
    "zingg_classify",
    "CLASSIFICATION_COLORS",
    "ZINGG_COLORS",
    # Export
    "export_csv",
    "export_annotated_image",
    # Report
    "generate_pdf_report",
    # Batch
    "process_batch",
    "process_single_image",
    "BatchResult",
    "BatchSummary",
    # YOLO
    "YOLODetector",
    "refine_with_yolo",
]
