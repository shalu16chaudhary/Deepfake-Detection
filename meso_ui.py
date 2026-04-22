import streamlit as st
import numpy as np
import cv2
import face_recognition
from classifiers import Meso4

# Load MesoNet Model
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

def predict_meso(image):
    # STEP 1: Find faces in the image
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        return None, "No Face Detected"

    # STEP 2: Crop the first face found
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    
    # STEP 3: Preprocess the cropped face
    img = cv2.resize(face_image, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # STEP 4: Prediction
    prediction = classifier.predict(img)
    return prediction[0][0], face_image

# UI Setup
st.set_page_config(page_title="Deepfake Detection UI", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>DEEP FAKE DETECTION SYSTEM</h1>", unsafe_allow_html=True)

st.write("---")
uploaded_file = st.file_uploader("Upload an image (Only face will be analyzed)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    # Convert BGR to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, channels="BGR", caption="Original Image")
    
    with st.spinner('Detecting Face and Analyzing...'):
        score, face_crop = predict_meso(rgb_image)
        
    if isinstance(face_crop, str) and face_crop == "No Face Detected":
        st.error("Could not find a face in this image. Please upload a clearer photo.")
    else:
        with col2:
            # Display cropped face
            st.image(face_crop, caption="Detected Face (Analyzed Area)")
            
        # Result Logic
        if score < 0.5:
            st.success(f"### RESULT: REAL (Confidence: {100*(1-score):.2f}%)")
        else:
            st.error(f"### RESULT: FAKE (Confidence: {100*score:.2f}%)")
            
        st.write(f"Raw Score: {score:.4f} (Near 0 = Real, Near 1 = Fake)")

st.sidebar.markdown("""
### How it works:
1. **Face Detection**: Uses HOG algorithm to find the face.
2. **Cropping**: Removes background noise.
3. **MesoNet Analysis**: Analyzes microscopic pixel artifacts.
""")