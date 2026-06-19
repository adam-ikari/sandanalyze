import cv2
import numpy as np


def split_by_watershed(mask, min_circularity=0.3):
    """Split over-merged components in a binary mask using distance transform + watershed.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8) with potentially touching/over-merged components.
    min_circularity : float, optional
        Minimum circularity threshold (currently unused in watershed logic,
        reserved for downstream filtering).

    Returns
    -------
    np.ndarray
        Binary mask with components split where watershed found boundaries.
    """
    if mask is None or mask.size == 0:
        return mask.copy() if mask is not None else np.array([], dtype=np.uint8)

    # Ensure binary mask
    binary = (mask > 0).astype(np.uint8) * 255

    # 1. Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 2. Threshold to find sure foreground (peaks)
    # Use a high fraction of the max distance to ensure separate markers
    # for distinct components, even if they are touching.
    dist_max = dist.max()
    if dist_max > 0:
        _, sure_fg = cv2.threshold(dist, 0.75 * dist_max, 255, 0)
        sure_fg = np.uint8(sure_fg)
    else:
        return np.zeros_like(binary)

    # 3. Dilate to get sure background
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)

    # 4. Unknown = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Label markers from sure foreground
    num_markers, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all markers so background is not 0 (watershed treats 0 as unknown)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 6. Run watershed
    # Watershed needs a 3-channel image
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    # 7. Extract split components (exclude background and watershed boundaries)
    result = np.zeros_like(binary)
    # Markers == -1 are watershed boundaries, markers == 1 is background
    # Keep only markers > 1 (actual components)
    result[markers > 1] = 255

    # Remove small noise (but don't close, which would re-merge split components)
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)

    return result


def split_by_concave_points(mask, min_concave_depth=5):
    """Split a single component in a binary mask using concave points from convexity defects.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8) with a single component
    min_concave_depth : int, optional
        Minimum depth of concave points to consider for splitting.

    Returns
    -------
    np.ndarray
        Binary mask with component split where concave points were found,
        or the original mask if no split was possible.
    """
    if mask is None or mask.size == 0:
        return mask.copy() if mask is not None else np.array([], dtype=np.uint8)

    # Ensure binary mask
    binary = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return binary

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Need at least 3 points for convexity defects
    if len(contour) < 3:
        return binary

    # Compute convex hull and convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)

    # Need at least 3 hull points for convexity defects
    if len(hull) < 3:
        return binary

    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return binary

    if defects is None or len(defects) == 0:
        return binary

    # Extract concave points with sufficient depth
    concave_points = []
    for defect in defects:
        s, e, f, d = defect[0]
        if d / 256.0 >= min_concave_depth:
            concave_points.append(tuple(contour[f][0]))

    # Need at least 2 concave points to draw a split line
    if len(concave_points) < 2:
        return binary

    # Draw a line between the first two concave points to split the component
    result = binary.copy()
    pt1 = concave_points[0]
    pt2 = concave_points[1]
    cv2.line(result, pt1, pt2, 0, 2)

    return result
