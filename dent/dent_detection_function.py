import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN


def dent_detection_function(input_folder, output_folder, eps, expansion, area_threshold=(350, 3000), aspect_ratio_range=(1.1, 3), extent_range=(0.2, 1.3)):

    def merge_and_expand_boxes(boxes, eps, min_samples=1, expansion=80, image_shape=None):
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

    def detect_dents_with_lbp(image_path, save_path=None):
        # Step 1: Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load the image at {image_path}")
            return

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply Local Binary Patterns (LBP)
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

        # Step 4: Preprocess the LBP image
        blurred_lbp = cv2.GaussianBlur(lbp, (5, 5), 0)
        _, thresh = cv2.threshold(blurred_lbp.astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 5: Find contours on the thresholded LBP image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 6: Filter contours based on area and shape properties
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

        # Step 7: Merge and expand bounding boxes
        merged_boxes = merge_and_expand_boxes(boxes, eps=eps, expansion=expansion, image_shape=image.shape)

        # Step 8: Draw final bounding boxes on the original image
        for (x, y, w, h) in merged_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Step 9: Save the result
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Saved output image with detected dents at: {save_path}")

    def process_images_in_folder():
        """
        Process all images in the input folder and save results in the output folder.
        """
        if not os.path.exists(input_folder):
            print(f"Error: Folder '{input_folder}' does not exist.")
            return

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Process each image in the folder
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)

            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_path = os.path.join(output_folder, f"processed_{filename}")
                print(f"Processing: {filename}")
                detect_dents_with_lbp(input_path, output_path)
            else:
                print(f"Skipping non-image file: {filename}")

    # Run the processing function
    process_images_in_folder()
    
dent_detection_function(input_folder="Input", output_folder="./Processed", eps=220, expansion=60)
