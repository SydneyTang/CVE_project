# Argus Web App 🚗👀

## Overview
**Argus Web App** is a Streamlit-based web application designed for vehicle damage detection and time estimation. Users can upload images of car damage, and the app provides insights into dents, scratches, tire issues, and windshield damage. Each detection type is powered by dedicated image processing algorithms.

---

## Features
- **Image Upload**: Users can upload images in `.jpg`, `.jpeg`, or `.png` format.
- **Damage Detection Options**:
  - **Dents**: Detect and highlight dents on the vehicle.
  - **Scratches**: Identify scratches on the car, with special options for white cars.
  - **Tires**: Detect flat tires and estimate time for repair.
  - **Windshield**: Identify shattered or damaged windshield glass.
- **Interactive Sidebar**: Choose specific detection methods through a sidebar menu.
- **Real-time Processing**: View processed images with detected damage areas highlighted.
  
---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SydneyTang/CVE_project
   ```
2. Navigate to the project directory:
   ```bash
   cd argus-web-app
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the App
1. Launch the Streamlit application:
   ```bash
   streamlit run ArgusApp.py
   ```
2. Open the provided URL (usually `http://localhost:8501`) in your web browser.

---

## Usage
1. **Upload an Image**:
   - Drag and drop an image of your vehicle's damage or use the file uploader to select an image.
2. **Select Damage Detection**:
   - Use the sidebar to choose the type of damage to detect:
     - `Dent`
     - `Scratch`
     - `Tire`
     - `Windshield`
3. **View Results**:
   - The processed image will be displayed with highlighted damage areas.
   - Additional messages provide estimated repair time or indicate no damage detected.

---

## File Structure
```plaintext
.
├── argus-web-app/         # Main project directory
│   ├── app.py             # Main Streamlit application
│   └── damage.py          # Consolidated detection logic
├── test_img/              # Example Test Images
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Future Enhancements
- Improve detection accuracy for multiple damage types in a single image.
- Support more car damage scenarios, such as bumper and paint issues.
- Add cost estimation for specific repairs.

---

## Dependencies
- **Streamlit**: For building the web interface.
- **OpenCV**: For image processing.
- **NumPy**: For numerical operations.
- **Pillow**: For image handling.
- **scikit-image**: For additional image analysis.
- **scikit-learn**: For clustering in dent detection.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
Special thanks to the creators of Streamlit and OpenCV for enabling the development of this application.
