import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the title of the web page
st.title("Multi-Function Image Processing Web App")

# Upload an image
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert the image to OpenCV format
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Define image processing functions
    def apply_grayscale(img):
        """Convert the image to grayscale."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_edge_detection(img):
        """Perform edge detection."""
        return cv2.Canny(img, 100, 200)

    def apply_blur(img):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(img, (15, 15), 0)

    def apply_threshold(img):
        """Apply binary thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    # Add buttons for different image processing options
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Grayscale"):
        processed_image = apply_grayscale(image_cv)
        st.text("This image has been converted to grayscale.")
        st.image(processed_image, caption="Grayscale Image", use_container_width=True, channels="GRAY")
        st.text('The')

    if col2.button("Edge Detection"):
        processed_image = apply_edge_detection(image_cv)
        st.text("Edges of the image have been highlighted using Canny edge detection.")
        st.image(processed_image, caption="Edge Detection", use_container_width=True, channels="GRAY")

    if col3.button("Gaussian Blur"):
        processed_image = apply_blur(image_cv)
        st.text("The image has been smoothed using Gaussian blur with a 15x15 kernel.")
        st.image(processed_image, caption="Gaussian Blur", use_container_width=True)

    if col4.button("Thresholding"):
        processed_image = apply_threshold(image_cv)
        st.text("The image has been converted to black and white using binary thresholding.")
        st.image(processed_image, caption="Thresholded Image", use_container_width=True, channels="GRAY")
