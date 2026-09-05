# 🧠 Multiple Disease Detection System

An AI-powered web application for detecting **Brain Tumor** from MRI images and **Pneumonia** from Chest X-Ray images using Deep Learning.

## 📌 Project Overview

The **Multiple Disease Detection System** is a web-based medical image analysis application developed using **Python, Flask, TensorFlow/Keras, and OpenCV**.

The system accepts medical images uploaded by the user and uses trained Deep Learning models to predict the possible disease category. Along with the prediction, the application displays a confidence score and uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the regions that influenced the model's prediction.

The system also generates a downloadable **PDF medical report** containing the prediction results and visualization.

> **Note:** This project is intended for educational and research purposes and is not a substitute for professional medical diagnosis.

---

## ✨ Key Features

* 🔐 User login interface
* 🧠 Brain Tumor detection from MRI images
* 🫁 Pneumonia detection from Chest X-Ray images
* 🤖 Deep Learning-based image classification
* 📊 Prediction confidence score
* 🔍 Grad-CAM explainability visualization
* 📄 Automatic PDF medical report generation
* 🖼️ Medical image preprocessing using OpenCV
* 🌐 Flask-based web application
* 💻 Simple and user-friendly interface

---

## 🔬 Diseases Detected

### Brain Tumor

The system classifies brain MRI images into:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

### Pneumonia

The system classifies chest X-ray images into:

* Pneumonia
* Normal

---

## ⚙️ How the System Works

```text
User Login
     ↓
Select Disease
     ↓
Upload Medical Image
     ↓
Image Preprocessing
     ↓
Deep Learning Model
     ↓
Disease Prediction
     ↓
Confidence Score
     ↓
Grad-CAM Visualization
     ↓
Generate PDF Report
```

---

## 🧠 Deep Learning

The project uses **Convolutional Neural Network (CNN)** based Deep Learning models for medical image classification.

The models are trained to identify visual patterns present in MRI and Chest X-Ray images.

### Model Tasks

| Model             | Input       | Classification |
| ----------------- | ----------- | -------------- |
| Brain Tumor Model | MRI Image   | 4 Classes      |
| Pneumonia Model   | Chest X-Ray | 2 Classes      |

---

## 🔍 Explainable AI with Grad-CAM

The project uses **Grad-CAM** to make the model's prediction easier to understand.

Grad-CAM generates a heatmap showing the regions of the medical image that contributed most strongly to the model's prediction.

This provides an additional visual explanation instead of displaying only the predicted class.

---

## 📄 Medical Report Generation

After prediction, the application generates a downloadable PDF report containing information such as:

* Disease prediction
* Prediction confidence
* Grad-CAM visualization
* Prediction information
* Generated report details

The report is generated using **ReportLab**.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Web Framework

* Flask

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Networks (CNN)

### Image Processing

* OpenCV
* NumPy

### Explainable AI

* Grad-CAM

### Report Generation

* ReportLab

### Frontend

* HTML
* CSS

### Development Tools

* Visual Studio Code
* Google Colab

---

## 📂 Project Structure

```text
Multi_Disease_Detection/
│
├── app.py
├── gradcam.py
├── pneumonia_gradcam.py
├── report_generator.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── brain_tumor_model.h5
│   └── pneumonia_model.h5
│
├── datasets/
│
├── reports/
│
├── static/
│   └── css/
│       ├── login.css
│       └── style.css
│
└── templates/
    ├── index.html
    ├── login.html
    ├── brain_tumor.html
    ├── pneumonia.html
    └── result.html
```

---

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sanjana-nyk/Multi-Disease-Detection-System.git
```

### 2. Navigate to the project directory

```bash
cd Multi-Disease-Detection-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

### 7. Open the application

Open the local Flask address displayed in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

## 📦 Requirements

The main dependencies used in this project include:

```text
Flask
TensorFlow
NumPy
OpenCV
ReportLab
Werkzeug
```

All required packages and their versions are provided in `requirements.txt`.

---

## 📊 Project Highlights

* Developed a complete **end-to-end AI web application**
* Integrated multiple Deep Learning models into a Flask application
* Implemented medical image preprocessing and classification
* Added **Grad-CAM-based model explainability**
* Implemented automatic **PDF report generation**
* Designed separate interfaces for Brain Tumor and Pneumonia detection
* Integrated trained `.h5` models with a production-style web application structure

---

## 🔮 Future Enhancements

* Add more disease detection modules
* Improve model accuracy with larger and more diverse datasets
* Deploy the application to a cloud platform
* Add secure user authentication and database integration
* Improve medical report customization
* Add additional Explainable AI techniques
* Develop a responsive mobile-friendly interface

---

## ⚠️ Disclaimer

This application is developed for **educational and research purposes only**.

The predictions generated by this system should not be considered a medical diagnosis. Medical images and prediction results should always be reviewed by qualified healthcare professionals.

---

## 👩‍💻 Developer

**Sanjana I Naik**

BE – Computer Science and Engineering

JNN College of Engineering, Shivamogga

---

⭐ If you find this project interesting, consider giving the repository a star.
