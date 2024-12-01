
import cv2
import numpy as np

def windshield_detection(image):
    """
    Detects potential damage on a windshield based on contours and lines in the image.

    Parameters:
        image (numpy.ndarray): Input image of the windshield in BGR format.

    Returns:
        tuple:
            contour_image (numpy.ndarray): Processed image with detected contours and lines visualized.
            is_damaged (bool): True if damage is detected, False otherwise.
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
    print(f"Contour Count: {contour_count}")
    print(f"Contour Density: {contour_density:.6f}")

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
