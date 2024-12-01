import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def tire_detection(image):
    """
    Detects and analyzes tire contours to assess inflation status.
    
    Parameters:
        image (numpy.ndarray): Input image (BGR format).
    
    Returns:
        tuple: Processed output image with annotations and a boolean indicating flat tire status.
    """
    # Step 1: Preprocessing - Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    
    # Step 2: Convert to Grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Binary Thresholding
    _, dst = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    
    # Step 4: Morphological Operations (Closing small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_dst = dst.copy()
    for _ in range(3):
        morph_dst = cv2.morphologyEx(morph_dst, cv2.MORPH_OPEN, kernel)
        morph_dst = cv2.morphologyEx(morph_dst, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Contour Detection
    cont, _ = cv2.findContours(morph_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = image.copy()
    
    # Initialize variables for finding the closest ellipse
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
    
    # Step 6: Draw the closest ellipse
    if closest_ellipse is not None:
        cv2.ellipse(img_copy, closest_ellipse, (0, 255, 0), 2)  # Green ellipse
        print(f"Closest eccentricity: {closest_eccentricity:.3f}")
    else:
        print("No valid ellipse found.")

    # Step 7: Rectangle around the ellipse
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
        width = int(expanded_width * 0.85)
        height = int(expanded_height * 0.85)

        # Adjust rectangle if eccentricity is high
        if closest_eccentricity > 0.7:
            if 0 <= angle <= 90:
                x -= int(closest_eccentricity * 100)
                width -= int(closest_eccentricity * 50)
            elif 90 <= angle <= 180:
                x += int(closest_eccentricity * 100)
                width += int(closest_eccentricity * 50)

        # Draw rectangle
        rect = (x, y, x + width, y + height)
    else:
        rect = None
        print("No valid ellipse found to calculate rectangle.")

    # Step 8: GrabCut Segmentation
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

        # Step 9: Contour analysis for deflation detection
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

    return contoured_image, if_flat

def main():
    # Read an image
    input_image = cv2.imread('img/rgb/07.jpg')  # Replace with your image path

    # Detect tire status
    processed_image, is_flat = tire_detection(input_image)

    # Display the results
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title("Tire Detection")
    plt.axis("off")
    plt.show()

    if is_flat is not None:
        print("Tire Status:", "Flat" if is_flat else "Properly Inflated")
    else:
        print("Could not determine tire status.")
        
if __name__ == "__main__":
    main()
