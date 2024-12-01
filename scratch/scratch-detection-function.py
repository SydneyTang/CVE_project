import cv2
import numpy as np

def scratch_detection(image_path, car_color):
    """
    Detect scratches on a car image.

    Args:
        image_path (str): Path to the image.
        car_color (str): Specify if the car is white ("yes"). Input is case-insensitive.

    Returns:
        output_image: Image with detected scratches highlighted and labeled.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Please provide a valid image path.")

    # Convert to grayscale and HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust processing based on input
    if car_color.lower() == "yes":
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

    return output_image


def main():
    # Get user input
    image_path = "scratch-5.jpg"  # Replace with your image path
    car_color = input("Is the car white? (yes/no): ").strip()

    # Call the scratch detection function
    try:
        result_image = scratch_detection(image_path, car_color)

        # Display results
        cv2.imshow("Original Image", cv2.imread(image_path))
        cv2.imshow("Detected Scratches", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
