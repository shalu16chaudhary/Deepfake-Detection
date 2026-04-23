MesoNet: Deepfake Detection System
Developed by: Shalu Chaudhary

MesoNet is an advanced deep learning framework implemented to verify the authenticity of digital media. This system utilizes specialized Convolutional Neural Network (CNN) architectures to detect face tampering, specifically targeting Deepfake and Face2Face manipulations in both images and video frames.

Introduction
The rapid rise of Deepfake technology in digital media has created significant risks regarding misinformation and digital forgery. This project adopts an AI-driven approach that focuses on the mesoscopic properties of an image. By analyzing these subtle details, the system can effectively distinguish between authentic and manipulated content with high precision.

Deepfake Detection Accuracy: > 98%

Face2Face Detection Accuracy: > 95%

System Architecture (Proposed System)
The system is designed as a multi-stage pipeline, following the Waterfall Development Model to ensure a structured and reliable workflow.

Preprocessing Module: Utilizing computer vision techniques (OpenCV/Haar Cascades) to isolate the face, normalize the input, and remove background noise for better analysis.

Feature Extraction Module: Leverages Meso4 and MesoInception architectures to extract subtle pixel-level artifacts and inconsistencies often left behind by AI-generation tools.

Classification Module: Employs pre-trained weights (.h5 files) to perform binary classification, providing a final decision of "Real" or "Fake."

Tech Stack & Requirements
The project is built within a modern Python-based ecosystem:

Programming Language: Python 3.11

Deep Learning Framework: TensorFlow 2.x, Keras

Web Interface: Streamlit (For a real-time web dashboard)

Image Processing: OpenCV, NumPy, Imageio

Deployment & Usage
The application is optimized for cloud deployment and can be accessed via a web browser. It provides an intuitive interface where users can upload media files and receive instant verification results with confidence scores.

Conclusion
This system provides a robust and adaptive approach to safeguarding digital media integrity. Future enhancements will focus on real-time video stream detection and integrating more complex spatio-temporal models to counter evolving forgery techniques.

Maintained by: Shalu Final Year Student | B.Tech - Computer Science & Engineering
Eshan College of Engineering

## Usage Instructions

### 1. Web Dashboard Launch Karein
To demonstrate the project with the graphical user interface, execute the following command in your terminal:

streamlit run meso_ui.py

### 2. Manual Example Prediction
For quick testing via a script without the web interface, use:

python example.py
