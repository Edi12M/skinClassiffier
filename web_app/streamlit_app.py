"""
Skin Lesion Classification Web Application
Streamlit app for serving model predictions (PyTorch version)
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

# Configuration
IMG_SIZE = 224  # ResNet50 uses 224x224

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transforms for PyTorch (matching training transforms)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names mapping
CLASS_NAMES = {
    0: 'akiec',
    1: 'bcc', 
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

# Full class descriptions
CLASS_DESCRIPTIONS = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Rough, scaly patches on the skin caused by years of sun exposure. Can potentially develop into squamous cell carcinoma.',
        'severity': 'Pre-cancerous',
        'color': '#FFA500'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer. Usually appears as a slightly transparent bump on the skin.',
        'severity': 'Cancerous',
        'color': '#FF4444'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'Non-cancerous skin growths that appear in adulthood. Includes seborrheic keratoses and solar lentigines.',
        'severity': 'Benign',
        'color': '#44AA44'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Common benign skin nodules of unknown cause. Usually found on the legs.',
        'severity': 'Benign',
        'color': '#44AA44'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'The most serious type of skin cancer. Develops in melanocytes, the cells that give skin its color.',
        'severity': 'Cancerous',
        'color': '#FF0000'
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'description': 'Common moles. Benign neoplasms of melanocytes that appear in a variety of forms.',
        'severity': 'Benign',
        'color': '#44AA44'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Skin lesions related to blood vessels. Includes angiomas, angiokeratomas, and hemorrhages.',
        'severity': 'Benign',
        'color': '#44AA44'
    }
}

@st.cache_resource
def load_model():
    """Load the trained PyTorch model (cached for performance)"""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'skin_lesion_model.pth'),
        os.path.join(os.path.dirname(__file__), 'skin_lesion_model.pth'),
        'skin_lesion_model.pth',
        '../skin_lesion_model.pth'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model = torch.load(path, map_location=device, weights_only=False)
            model.eval()
            model.to(device)
            return model
    
    st.error("Model file not found! Please ensure 'skin_lesion_model.pth' is in the correct location.")
    return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Apply PyTorch transforms
    img_tensor = val_transform(image)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(device)

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .severity-benign {
        background-color: #d4edda;
        border-left: 5px solid #44AA44;
    }
    .severity-precancerous {
        background-color: #fff3cd;
        border-left: 5px solid #FFA500;
    }
    .severity-cancerous {
        background-color: #f8d7da;
        border-left: 5px solid #FF4444;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🔬 Skin Lesion Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Dermoscopic Image Analysis using Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a deep learning model trained on the **HAM10000 dataset** 
        to classify skin lesions into 7 categories.
        """)
        
        st.header("📊 Model Info")
        st.write("""
        - **Architecture:** ResNet50
        - **Training:** Transfer Learning
        - **Framework:** PyTorch
        - **Dataset:** HAM10000 (10,015 images)
        """)
        
        st.header("🏥 Lesion Types")
        for code, info in CLASS_DESCRIPTIONS.items():
            severity_color = info['color']
            st.markdown(f"**{code}** - {info['name']}")
            st.markdown(f"<span style='color:{severity_color}'>{info['severity']}</span>", unsafe_allow_html=True)
            st.write("---")
        
        st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic skin image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a dermoscopic image of a skin lesion for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if uploaded_file is not None:
            # Load model
            with st.spinner("Loading model..."):
                model = load_model()
            
            if model is not None:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    processed_image = preprocess_image(image)
                    
                    # Predict (PyTorch)
                    with torch.no_grad():
                        outputs = model(processed_image)
                        probabilities = torch.softmax(outputs, dim=1)[0]
                        predictions = probabilities.cpu().numpy()
                    
                    predicted_class_idx = int(np.argmax(predictions))
                    confidence = float(predictions[predicted_class_idx]) * 100
                    
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    class_info = CLASS_DESCRIPTIONS[predicted_class]
                
                # Display results
                severity = class_info['severity'].lower().replace('-', '')
                st.markdown(f"""
                <div class="result-box severity-{severity}">
                    <h2 style="margin:0; color:{class_info['color']}">{class_info['name']}</h2>
                    <p style="font-size:1.2rem; margin:0.5rem 0;">Code: <strong>{predicted_class}</strong></p>
                    <p style="font-size:1.5rem; margin:0.5rem 0;">Confidence: <strong>{confidence:.1f}%</strong></p>
                    <p style="margin:0.5rem 0;"><strong>Severity:</strong> {class_info['severity']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Description:** {class_info['description']}")
                
                # Show all probabilities
                st.subheader("📊 All Predictions")
                
                # Create bar chart data
                probs_data = {
                    CLASS_DESCRIPTIONS[CLASS_NAMES[i]]['name']: float(predictions[i]) * 100
                    for i in range(len(CLASS_NAMES))
                }
                
                # Sort by probability
                sorted_probs = dict(sorted(probs_data.items(), key=lambda x: x[1], reverse=True))
                
                st.bar_chart(sorted_probs)
                
                # Detailed probabilities table
                st.subheader("📋 Detailed Probabilities")
                for name, prob in sorted_probs.items():
                    st.write(f"**{name}:** {prob:.2f}%")
        else:
            st.info("👆 Upload an image to get started!")
            
            # Show sample usage
            st.markdown("""
            ### How to use:
            1. Upload a dermoscopic skin lesion image
            2. Wait for the AI to analyze the image
            3. View the classification results and confidence scores
            
            ### Supported formats:
            - JPG/JPEG
            - PNG
            """)

if __name__ == "__main__":
    main()
