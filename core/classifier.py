"""Classification module for sand grain shape analysis.

Implements Zingg classification with flocculation support.
"""

from dataclasses import dataclass


@dataclass
class ZinggClassifier:
    """Zingg shape classification configuration."""

    spherical_threshold: float = 1.5
    bladed_threshold: float = 2.5


def classify_grain(aspect_ratio: float, is_flocculation: bool,
                     classifier: ZinggClassifier = None) -> str:
    """Classify a grain using Zingg + Flocculation system.

    Priority: Flocculation > Zingg classification

    Args:
        aspect_ratio: Major axis / minor axis ratio.
        is_flocculation: Whether the grain is detected as flocculation.
        classifier: ZinggClassifier instance. Uses defaults if None.

    Returns:
        One of: "spherical", "rod-like", "discoidal", "flocculation"
    """
    if classifier is None:
        classifier = ZinggClassifier()

    # Priority: Flocculation first
    if is_flocculation:
        return "flocculation"

    # Zingg classification
    if aspect_ratio < classifier.spherical_threshold:
        return "spherical"
    elif aspect_ratio < classifier.bladed_threshold:
        return "rod-like"
    else:
        return "discoidal"


def classify_batch(aspect_ratios: list[float], flocculation_flags: list[bool],
                   classifier: ZinggClassifier = None) -> list[str]:
    """Classify multiple grains.

    Args:
        aspect_ratios: List of aspect ratios.
        flocculation_flags: List of flocculation flags.
        classifier: ZinggClassifier instance.

    Returns:
        List of classification strings.
    """
    if classifier is None:
        classifier = ZinggClassifier()

    return [
        classify_grain(ar, is_floc, classifier)
        for ar, is_floc in zip(aspect_ratios, flocculation_flags)
    ]
