# WebRTC Object Detection with Streamlit

This Streamlit application demonstrates real-time object detection through the web camera using a pre-trained MobileNetSSD model for video frames, and Haar Cascades for static image detection. The application allows users to switch between live video detection and static image detection modes.

## Features

- Real-time object detection using a webcam and the MobileNetSSD model.
- Static image object detection using Haar cascades for faces and cats.
- Interactive UI with Streamlit.
  
## Installation

Ensure you have Python 3.6 or newer installed on your system.

1. Clone the repository:
```bash
git clone https://github.com/shubhamshnd/Object-Detection
```
```bash
cd Object-Detection
```
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run main.py
```
## Usage

After starting the application, navigate to (Usually)http://localhost:8501 or provided URL in your web browser. You will see the Streamlit interface with the following options:

Real-time object detection (sendrecv): Use your webcam for real-time object detection. You can adjust the confidence threshold for detections.
Object Detection From Image: Upload a static image to detect faces and cats using Haar cascades.

## Note
The real-time object detection feature requires a webcam.
For the best performance, ensure you have a stable internet connection and sufficient system resources.

## Disclaimer
This demo is for educational purposes only. The accuracy of object detection may vary based on the quality of the webcam and the uploaded images.

Enjoy exploring object detection with WebRTC and Streamlit!
