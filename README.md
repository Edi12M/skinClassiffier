[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/u2w0l2du)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=22075798&assignment_repo_type=AssignmentRepo)

# 🔬 Skin Lesion Classification using Deep Learning

An AI-powered skin lesion classification system trained on the HAM10000 dataset, capable of classifying 7 types of skin lesions using a ResNet50 model with transfer learning.

## 📋 Project Overview

This project implements a deep learning model for dermoscopic image classification to assist in early detection of skin conditions. The model can classify lesions into the following categories:

| Code  | Name                 | Severity            |
| ----- | -------------------- | ------------------- |
| akiec | Actinic Keratoses    | Pre-cancerous       |
| bcc   | Basal Cell Carcinoma | Cancerous           |
| bkl   | Benign Keratosis     | Benign              |
| df    | Dermatofibroma       | Benign              |
| mel   | Melanoma             | Cancerous (Serious) |
| nv    | Melanocytic Nevi     | Benign              |
| vasc  | Vascular Lesions     | Benign              |

## 🛠️ Dependencies

### Core Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- NumPy >= 1.23.0
- Pillow >= 9.5.0

### Web Application

- Flask >= 2.3.0
- Flask-CORS >= 4.0.0
- Streamlit >= 1.28.0

### Install All Dependencies

```bash
# Install web app dependencies
pip install -r web_app/requirements.txt
```

Or install manually:

```bash
pip install torch torchvision flask flask-cors streamlit pillow numpy
```

## 🚀 Running the Demo

### Option 1: Flask Web Application

1. Navigate to the project root directory
2. Run the Flask server:

```bash
python web_app/app.py
```

3. Open your browser and go to: **http://localhost:5000**

### Option 2: Streamlit Application

1. Navigate to the project root directory
2. Run the Streamlit app:

```bash
streamlit run web_app/streamlit_app.py
```

3. The app will automatically open in your browser at **http://localhost:8501**

## 📁 Project Structure

```
├── skin_lesion_model.pth          # Trained PyTorch model (ResNet50)
├── AI.ipynb                       # Training notebook
├── skin_classifier.py             # Training script
├── web_app/
│   ├── app.py                     # Flask backend server
│   ├── streamlit_app.py           # Streamlit application
│   ├── requirements.txt           # Python dependencies
│   ├── templates/
│   │   └── index.html             # Flask web page
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── models/                        # Additional model files
└── reports/                       # Analysis reports
```

## 🖼️ How to Use the Web App

1. Open the web application in your browser
2. Upload a dermoscopic skin lesion image (JPG, JPEG, or PNG)
3. Click "Analyze Image"
4. View the prediction results with confidence percentages for all 7 classes

## ⚠️ Disclaimer

**This application is for educational purposes only.** It should NOT be used for medical diagnosis. Always consult a qualified dermatologist for any skin concerns or medical advice.

## 🔧 Technical Details

- **Model Architecture**: ResNet50 with Transfer Learning
- **Framework**: PyTorch
- **Dataset**: HAM10000 (10,015 dermatoscopic images)
- **Input Size**: 224x224 RGB images
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
