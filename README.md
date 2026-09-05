# Multiple Disease Detection System

## Project Description

The Multiple Disease Detection System is an AI-based web application developed to assist in the early detection of diseases using medical images.

The system currently detects:

- Brain Tumor from MRI images
- Pneumonia from Chest X-Ray images

The application uses Deep Learning models to analyze uploaded medical images and provide prediction results with confidence scores. It also generates Grad-CAM visualizations to highlight important regions used by the model during prediction.

A downloadable medical report is generated for each prediction.

---

## Features

- User login interface
- Brain Tumor detection from MRI images
- Pneumonia detection from Chest X-Ray images
- Deep Learning-based prediction
- Confidence score display
- Grad-CAM visualization
- Medical report generation in PDF format
- User-friendly web interface

---

## Technologies Used

### Programming Language
- Python

### Framework
- Flask

### Machine Learning and Deep Learning
- TensorFlow
- Keras
- CNN

### Image Processing
- OpenCV
- NumPy

### Report Generation
- ReportLab

### Frontend
- HTML
- CSS

---

## Project Structure

```text
Multi_Disease_Detection/
│
├── app.py
├── gradcam.py
├── pneumonia_gradcam.py
├── report_generator.py
├── requirements.txt
├── README.md
│
├── datasets/
├── models/
├── reports/
├── static/
│   ├── css/
│   └── uploads/
│
└── templates/
    ├── index.html
    ├── login.html
    ├── brain_tumor.html
    ├── pneumonia.html
    └── result.html