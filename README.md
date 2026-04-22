# MesoNet

# MesoNet: Deepfake Detection System
**Developed by: Shalu**

MesoNet ek advanced deep learning framework hai jise maine digital media ki authenticity verify karne ke liye implement kiya hai. [cite_start]Yeh system videos aur images mein face tampering (Deepfake aur Face2Face) ko detect karne ke liye specialized CNN architectures ka istemal karta hai[cite: 12, 13].

## Introduction
[cite_start]Digital media mein deepfake technology ka badhta upyog disinformation ka khatra paida kar raha hai[cite: 5, 7]. [cite_start]Maine is project mein ek AI-driven approach adopt ki hai jo image ki **mesoscopic properties** par focus karti hai taaki manipulated content ko high accuracy ke saath pakda ja sake[cite: 13, 19].

* [cite_start]**Deepfake Detection Success**: > 98% [cite: 123]
* [cite_start]**Face2Face Detection Success**: > 95% [cite: 123, 407]

## System Architecture (Proposed System)
[cite_start]Maine is system ko ek multi-stage pipeline mein design kiya hai jo **Waterfall Development Model** ko follow karta hai[cite: 86, 275].

* [cite_start]**Preprocessing Module**: `face_recognition` library ka use karke chehre ko isolate kiya jata hai aur background noise hatayi jati hai[cite: 159, 195].
* [cite_start]**Feature Extraction Module**: Meso4 aur MesoInception architectures ka use karke subtle pixel artifacts aur inconsistencies extract kiye jate hain[cite: 87, 161].
* [cite_start]**Classification Module**: Pre-trained weights (.h5 files) ka use karke final "Real" ya "Fake" ka decision liya jata hai[cite: 117, 119].

## Tech Stack & Requirements
Is project ko maine modern Python environment mein setup kiya hai:
* **Languages**: Python 3.11
* [cite_start]**Deep Learning**: TensorFlow 2.x, Keras [cite: 166, 208]
* [cite_start]**Interface**: Streamlit (For Web Dashboard) [cite: 182, 374]
* [cite_start]**Image Processing**: OpenCV, face_recognition, Imageio [cite: 159, 162]

## Usage Instructions

### 1. Web Dashboard Launch Karein
Interface ke saath demo dikhane ke liye terminal mein ye command chalayein:
```bash
streamlit run meso_ui.py