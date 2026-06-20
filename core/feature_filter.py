"""Multi-feature false positive filtering for grain candidates.

Provides filters that remove edge artifacts, noise-like components,
and filament-like shapes using area, circularity, aspect ratio,
and solidity features.
"""

from core.multiscale_detector import GrainCandidate


def filter_edge_false_positives(
    candidates: list[GrainCandidate], edge_margin: int = 10
) -> list[GrainCandidate]:
    """Remove small components near the image border.

    Args:
        candidates: List of GrainCandidate objects.
        edge_margin: Distance from the border considered "near edge".

    Returns:
        Candidates that are either far from the edge OR large enough
        to be kept even if near the edge.
    """
    return [
        c
        for c in        candidates
        if not (c.border_distance < edge_margin and c.area < 2000)
    ]


def filter_noise(
    candidates: list[GrainCandidate], min_area: int = 500
) -> list[GrainCandidate]:
    """Remove noise-like components (small area + low circularity).

    Args:
        candidates: List of GrainCandidate objects.
        min_area: Minimum area threshold for noise detection.

    Returns:
        Candidates that are not both small and low-circularity.
    """
    return [
        c
        for c in candidates
        if not (c.area < min_area and c.circularity < 0.2)
    ]


def filter_filaments(
    candidates: list[GrainCandidate],
    max_aspect_ratio: float = 5.0,
    min_solidity: float = 0.5,
) -> list[GrainCandidate]:
    """Remove filament-like shapes (high aspect ratio + low solidity).

    Args:
        candidates: List of GrainCandidate objects.
        max_aspect_ratio: Maximum aspect ratio before a shape is considered filament-like.
        min_solidity: Minimum solidity below which a shape is considered filament-like.

    Returns:
        Candidates that are not both high-aspect-ratio and low-solidity.
    """
    return [
        c
        for c in candidates
        if not (c.aspect_ratio > max_aspect_ratio and c.solidity < min_solidity)
    ]


def filter_strict(
    candidates: list[GrainCandidate],
    min_area: int = 800,
    min_circularity: float = 0.15,
    max_aspect_ratio: float = 4.0,
    min_solidity: float = 0.6,
) -> list[GrainCandidate]:
    """Apply strict filtering to remove false positives.

    This is a stricter version that combines multiple filters:
    - Remove small components (area < min_area)
    - Remove irregular shapes (circularity < min_circularity)
    - Remove very elongated shapes (aspect_ratio > max_aspect_ratio)
    - Remove low solidity shapes (solidity < min_solidity)

    Args:
        candidates: List of GrainCandidate objects.
        min_area: Minimum area threshold.
        min_circularity: Minimum circularity threshold.
        max_aspect_ratio: Maximum aspect ratio threshold.
        min_solidity: Minimum solidity threshold.

    Returns:
        Filtered list of candidates.
    """
    filtered = []
    for c in candidates:
        # Filter 1: Too small
        if c.area < min_area:
            continue

        # Filter 2: Too irregular (but allow larger grains to be less circular)
        if c.circularity < min_circularity and c.area < 3000:
            continue

        # Filter 3: Too elongated
        if c.aspect_ratio > max_aspect_ratio:
            continue

        # Filter 4: Too low solidity
        if c.solidity < min_solidity:
            continue

        filtered.append(c)

    return filtered
