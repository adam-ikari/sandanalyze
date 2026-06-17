"""Tests for the classification module."""

import numpy as np

from core.classifier import classify_grain, ZinggClassifier


class TestClassifyGrain:
    """Tests for grain classification."""

    def test_classify_spherical(self):
        """Low aspect ratio should classify as spherical."""
        classifier = ZinggClassifier()
        result = classify_grain(aspect_ratio=1.2, is_flocculation=False, classifier=classifier)
        assert result == "spherical"

    def test_classify_flocculation(self):
        """Flocculation should always be classified as flocculation."""
        classifier = ZinggClassifier()
        result = classify_grain(aspect_ratio=5.0, is_flocculation=True, classifier=classifier)
        assert result == "flocculation"

    def test_classify_rod(self):
        """Medium aspect ratio should classify as rod-like."""
        classifier = ZinggClassifier()
        result = classify_grain(aspect_ratio=2.0, is_flocculation=False, classifier=classifier)
        assert result == "rod-like"

    def test_classify_discoidal(self):
        """High aspect ratio should classify as discoidal."""
        classifier = ZinggClassifier()
        result = classify_grain(aspect_ratio=3.0, is_flocculation=False, classifier=classifier)
        assert result == "discoidal"

    def test_flocculation_priority(self):
        """Flocculation should take priority over Zingg classification."""
        classifier = ZinggClassifier()
        # Even with low aspect ratio, flocculation wins
        result = classify_grain(aspect_ratio=1.0, is_flocculation=True, classifier=classifier)
        assert result == "flocculation"

    def test_default_classifier(self):
        """Should use default classifier when none provided."""
        result = classify_grain(aspect_ratio=1.2, is_flocculation=False)
        assert result == "spherical"
