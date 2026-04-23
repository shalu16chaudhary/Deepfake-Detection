import streamlit as st
import numpy as np
import cv2
from classifiers import Meso4

# Load MesoNet Model
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# OpenCV Face Detector (Inbuilt & Lightweight)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_meso(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, "No Face Detected"

    (x, y, w, h) = faces[0]
    face_image = image[y:y+h, x:x+w]
    
    # Model Preprocessing (Same as before)
    img = cv2.resize(face_image, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = classifier.predict(img)
    return prediction[0][0], face_image

# UI Design
st.set_page_config(page_title="Deepfake Detection UI", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>DEEP FAKE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.write("---")

uploaded_file = st.file_uploader("Upload an image for analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image")
    
    with st.spinner('Analyzing...'):
        score, face_crop = predict_meso(image)
        
    if isinstance(face_crop, str) and face_crop == "No Face Detected":
        st.error("Could not find a face. Please upload a clearer photo.")
    else:
        with col2:
            st.image(face_crop, channels="BGR", caption="Detected Face")
            
        if score < 0.5:
            st.success(f"### RESULT: AUTHENTIC (Real)")
            st.info(f"Verification Score: {(1-score):.2f}")
        else:
            st.error(f"### RESULT: TAMPERED / LOW QUALITY (Fake)")
            st.warning(f"Detection Probability: {score:.2f}")