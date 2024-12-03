import cv2
import numpy as np
import random
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
import streamlit as st


@st.cache_data
def dent_detection(image):
    """
    Detect dents on a single image using LBP and return the processed image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: Image with detected dents highlighted by bounding boxes.
    """

    # Default parameters
    eps = 220
    expansion = 60
    area_threshold = (350, 3000)
    aspect_ratio_range = (1.1, 3)
    extent_range = (0.2, 1.3)

    def merge_and_expand_boxes(boxes, eps, min_samples=1, expansion=60, image_shape=None):
        if len(boxes) == 0:
            return []

        # Convert boxes to center points
        centers = np.array([(x + w // 2, y + h // 2) for x, y, w, h in boxes])

        # Cluster centers using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
        labels = clustering.labels_

        merged_boxes = []
        for label in set(labels):
            if label == -1:
                continue  # Ignore noise points
            cluster_points = np.array([boxes[i] for i in range(len(boxes)) if labels[i] == label])
            x_min = np.min(cluster_points[:, 0])
            y_min = np.min(cluster_points[:, 1])
            x_max = np.max(cluster_points[:, 0] + cluster_points[:, 2])
            y_max = np.max(cluster_points[:, 1] + cluster_points[:, 3])

            # Expand the bounding box by a fixed margin
            x_min = max(0, x_min - expansion)
            y_min = max(0, y_min - expansion)
            x_max = min(image_shape[1], x_max + expansion) if image_shape else x_max + expansion
            y_max = min(image_shape[0], y_max + expansion) if image_shape else y_max + expansion

            merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged_boxes

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Local Binary Patterns (LBP)
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    # Step 3: Preprocess the LBP image
    blurred_lbp = cv2.GaussianBlur(lbp, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_lbp.astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Find contours on the thresholded LBP image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Filter contours based on area and shape properties
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_threshold[0] < area < area_threshold[1]:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            extent = area / (w * h)  # Area-to-bounding-box ratio

            # Filter by aspect ratio and extent
            if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and extent_range[0] < extent < extent_range[1]:
                boxes.append((x, y, w, h))

    # Step 6: Merge and expand bounding boxes
    merged_boxes = merge_and_expand_boxes(boxes, eps=eps, expansion=expansion, image_shape=image.shape)

    # Step 7: Draw final bounding boxes on the original image
    for (x, y, w, h) in merged_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@st.cache_data
def scratch_detection(image, is_white):
    """
    Detect scratches on a car image.

    Args:
        image (numpy.ndarray): The image of the car as a NumPy array (BGR format).
        is_white (bool): Whether the car is white (True for white, False otherwise).

    Returns:
        numpy.ndarray: Image with detected scratches highlighted and labeled.
    """

    # Convert to grayscale and HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust processing based on input
    if is_white:
        enhanced = cv2.equalizeHist(gray)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 150])
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
    else:
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

    # Laplacian + Canny edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    edges = cv2.Canny(laplacian, 50, 150)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.erode(cv2.dilate(edges, kernel, iterations=1), kernel, iterations=1)

    # Combine mask and edges
    combined = cv2.bitwise_or(morphed, mask)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    scratches = []
    min_length = 50
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        length = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * length, True)
        if length > min_length and 2.0 < aspect_ratio < 20.0 and len(approx) > 4:
            scratches.append(contour)

    # Draw scratches and add text
    output_image = image.copy()
    for contour in scratches:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), -1)  # Fill scratches with green

    # Add "Scratch Detected" text if scratches are found
    if scratches:
        cv2.putText(output_image, "Scratch Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image


@st.cache_data
def tire_detection(image):
    """
    Detects and analyzes tire contours to assess inflation status.
    
    Parameters:
        image (numpy.ndarray): Input image (BGR format).
    
    Returns:
        tuple:
            numpy.ndarray: Processed image with detected contours and annotations.
            bool: True if tire is flat, False otherwise.
    """
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_dst = dst.copy()
    for _ in range(3):
        morph_dst = cv2.morphologyEx(morph_dst, cv2.MORPH_OPEN, kernel)
        morph_dst = cv2.morphologyEx(morph_dst, cv2.MORPH_CLOSE, kernel)
    cont, _ = cv2.findContours(morph_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    closest_eccentricity = float('inf')
    closest_ellipse = None

    for c in cont:
        # Filter contours by area
        area = cv2.contourArea(c)
        if area > (image.shape[0] * image.shape[1]) / 50:  # Minimum area threshold
            if len(c) >= 5:  # At least 5 points required for ellipse fitting
                ellipse = cv2.fitEllipse(c)
                _, axes, _ = ellipse
                eccentricity = (1 - (min(axes) / max(axes)) ** 2) ** 0.5
                if eccentricity < closest_eccentricity:
                    closest_eccentricity = eccentricity
                    closest_ellipse = ellipse
    
    if closest_ellipse is not None:
        (cx, cy), (major_axis, minor_axis), angle = closest_ellipse
        angle_rad = np.deg2rad(angle)

        # Calculate horizontal and vertical projections
        horizontal_projection = abs(major_axis * np.cos(angle_rad)) + abs(minor_axis * np.sin(angle_rad))
        vertical_projection = abs(major_axis * np.sin(angle_rad)) + abs(minor_axis * np.cos(angle_rad))

        # Scaling factors
        scale_factor_w = 1.2 + closest_eccentricity * 0.3
        scale_factor_h = 1.2 + closest_eccentricity * 0.4

        # Expanded rectangle dimensions
        expanded_width = horizontal_projection * scale_factor_w
        expanded_height = vertical_projection * scale_factor_h
        x = int(cx - expanded_width / 2)
        y = int(cy - expanded_height / 2)
        width = int(expanded_width * 0.8)
        height = int(expanded_height * 0.85)

        # Adjust rectangle if eccentricity is high
        if closest_eccentricity > 0.7:
            if 0 <= angle <= 90:
                x = x - int(closest_eccentricity * 100)
                width = width - int(closest_eccentricity * 50)
            elif 90 <= angle <= 180:
                x = + int(closest_eccentricity * 100)
                width = + int(closest_eccentricity * 50)

        rect = (x, y, x + width, y + height)
    else:
        rect = None

    # GrabCut Segmentation
    if rect:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Extract foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        foreground = image * mask2[:, :, np.newaxis]

        # Additional morphological operations to refine the mask
        gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        dst = cv2.inRange(blurred, 35, 115)
        morph_dst = dst.copy()
        for _ in range(20):
            morph_dst = cv2.morphologyEx(morph_dst, cv2.MORPH_OPEN, kernel)
            morph_dst = cv2.morphologyEx(morph_dst, cv2.MORPH_CLOSE, kernel)

        # Contour analysis for deflation detection
        cont, _ = cv2.findContours(morph_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cont:
            largest_contour = max(cont, key=cv2.contourArea)

            # Create a copy of the original image to draw contours
            contoured_image = image.copy()
            cv2.drawContours(contoured_image, [largest_contour], -1, (0, 255, 0), 2)  # Draw the largest contour in green

            # Fit ellipse and compare with the contour
            if len(largest_contour) >= 5:
                min_enclosing_ellipse = cv2.fitEllipse(largest_contour)
                cv2.ellipse(contoured_image, min_enclosing_ellipse, (0, 0, 255), thickness=2)  # Draw ellipse in red

                # Create masks for intersection and union analysis
                contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                ellipse_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                cv2.ellipse(ellipse_mask, min_enclosing_ellipse, 255, thickness=cv2.FILLED)

                intersection = cv2.bitwise_and(contour_mask, ellipse_mask)
                union = cv2.bitwise_or(contour_mask, ellipse_mask)

                intersection_area = np.sum(intersection == 255)
                union_area = np.sum(union == 255)
                symmetric_difference_area = union_area - intersection_area

                # Determine tire inflation status
                if_flat = (symmetric_difference_area / union_area) > 0.05
                status_text = "Flat Tire" if if_flat else "Normal Tire"
                color = (0, 0, 255) if if_flat else (0, 255, 0)
                cv2.putText(contoured_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                print(status_text)
            else:
                contoured_image = image.copy()
                if_flat = None
        else:
            contoured_image = image.copy()
            if_flat = None
    else:
        contoured_image = image.copy()
        if_flat = None

    output_image = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB)
    return output_image, if_flat


@st.cache_data
def windshield_detection(image):
    """
    Detects potential damage on a windshield based on contours and lines in the image.

    Parameters:
        image (numpy.ndarray): Input image of the windshield in BGR format.

    Returns:
        tuple:
            numpy.ndarray: Processed image with detected contours and lines visualized.
            bool: True if damage is detected, False otherwise.
    """
    min_contour_density = 0.0001
    min_line_count = 90

    # Convert the input image to grayscale
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(original_image, 130, 255, cv2.THRESH_BINARY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 110)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    image_area = original_image.shape[0] * original_image.shape[1]
    contour_density = contour_count / image_area

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Convert edge image to BGR for visualization
    contour_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Draw contours on the image
    cv2.drawContours(contour_image, sorted_contours, -1, (0, 255, 0), 2)  # Green contours

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    line_count = len(lines) if lines is not None else 0

    # Draw Hough lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(contour_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines

    # Shattered glass detection logic
    is_damaged = contour_density > min_contour_density and line_count > min_line_count

    output_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    return contour_image, is_damaged