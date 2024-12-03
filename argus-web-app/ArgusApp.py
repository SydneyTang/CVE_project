import cv2
import numpy as np
from PIL import Image
import streamlit as st
# Consolidated detection logic
import damage  

# Set browser title and favicon
st.set_page_config(
    page_title="Argus",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set the title of the web page
st.title("Argus Web App 🚗👀")

# Add a subtitle using markdown
st.markdown("### Vehicle Damage Detection and Time Estimation")

# Upload an image
uploaded_file = st.file_uploader("Please upload an image file of your car damage.", type=["jpg", "jpeg", "png"])

if "processing_method" not in st.session_state:
    st.session_state.processing_method = None  # Initialize the session state

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Car Damage Image", width=600)

    # Convert the image to OpenCV format
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


    # Add buttons to the sidebar
    st.sidebar.title("Damage Detector Options")
    if st.sidebar.button("Dent"):
        st.session_state.processing_method = "Dent"

    if st.sidebar.button("Scratch"):
        st.session_state.processing_method = "Scratch"

    if st.sidebar.button("Tire"):
        st.session_state.processing_method = "Tire"

    if st.sidebar.button("Windshield"):
        st.session_state.processing_method = "Windshield"

    # Process the image based on the button clicked
    if st.session_state.processing_method == "Scratch":
        st.sidebar.subheader("Scratch Detection Options")
        is_white = st.sidebar.checkbox("If it's a white car", value=False)
        processed_image = damage.scratch_detection(image_cv, is_white=is_white)
        st.markdown("#### Scratches on the car image have been detected and highlighted!")
        st.image(processed_image, caption="Prosessed Scratch Image", width=600)
        

    elif st.session_state.processing_method == "Dent":
        processed_image = damage.dent_detection(image_cv)
        st.markdown("#### Dents on the car image have been detected and highlighted!")
        st.image(processed_image, caption="Prosessed Dent Image", width=600)
        

    elif st.session_state.processing_method == "Tire":
        processed_image, is_flat = damage.tire_detection(image_cv)
        if is_flat:
            st.markdown("#### Flat tire detected! ")
            st.markdown("#### This will take 1 hour to fix.  ")
        else:
            st.markdown("#### No flat tire detected.  ")
        st.image(processed_image, caption="Prosessed Tire Image", width=600)

    elif st.session_state.processing_method == "Windshield":
        processed_image,is_damaged = damage.windshield_detection(image_cv)
        if is_damaged:
            st.markdown("#### Shattered glass detected! ")
            st.markdown("#### This will take 1 hour to fix.  ")
        else:
            st.markdown("#### No shattered glass detected.  ")
        st.image(processed_image, caption="Prosessed Windshield Image", width=600)
