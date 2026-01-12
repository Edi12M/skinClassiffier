"""
Skin Lesion Classification Web Application
Flask backend for serving model predictions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Configuration
IMG_SIZE = 254
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_EfficientNetB0_Transfer_phase2.keras')

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
        'color': '#4CAF50'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Common benign skin nodules of unknown cause. Usually found on the legs and are harmless.',
        'severity': 'Benign',
        'color': '#4CAF50'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'The most serious type of skin cancer. Develops from the cells that give your skin its color.',
        'severity': 'Cancerous (Serious)',
        'color': '#FF0000'
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'description': 'Common moles. Usually harmless clusters of pigmented cells that appear as small, dark brown spots.',
        'severity': 'Benign',
        'color': '#4CAF50'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Skin abnormalities caused by blood vessels. Includes cherry angiomas and angiokeratomas.',
        'severity': 'Benign',
        'color': '#2196F3'
    }
}

# Load model globally
model = None

def load_model():
    """Load the trained Keras model"""
    global model
    if model is None:
        print(f"Loading model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            # Try alternative path
            alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_EfficientNetB0_Transfer.keras')
            if os.path.exists(alt_path):
                model = keras.models.load_model(alt_path)
                print(f"Model loaded from alternative path: {alt_path}")
            else:
                print(f"ERROR: Model not found at {MODEL_PATH} or {alt_path}")
                return None
    return model

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Apply EfficientNet preprocessing (scale to [-1, 1])
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    """Serve the main web page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Load model and make prediction
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        # Get prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Get all class probabilities
        all_predictions = []
        for idx, prob in enumerate(predictions[0]):
            class_code = CLASS_NAMES[idx]
            all_predictions.append({
                'class_code': class_code,
                'class_name': CLASS_DESCRIPTIONS[class_code]['name'],
                'probability': float(prob) * 100,
                'severity': CLASS_DESCRIPTIONS[class_code]['severity'],
                'color': CLASS_DESCRIPTIONS[class_code]['color']
            })
        
        # Sort by probability (highest first)
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Prepare response
        result = {
            'success': True,
            'prediction': {
                'class_code': predicted_class,
                'class_name': CLASS_DESCRIPTIONS[predicted_class]['name'],
                'description': CLASS_DESCRIPTIONS[predicted_class]['description'],
                'severity': CLASS_DESCRIPTIONS[predicted_class]['severity'],
                'confidence': round(confidence, 2)
            },
            'all_predictions': all_predictions
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    """Return information about the model"""
    return jsonify({
        'name': 'Skin Lesion Classifier',
        'architecture': 'EfficientNetB0 (Transfer Learning)',
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'num_classes': 7,
        'classes': [
            {
                'code': code,
                'name': info['name'],
                'description': info['description'],
                'severity': info['severity']
            }
            for code, info in CLASS_DESCRIPTIONS.items()
        ],
        'dataset': 'HAM10000',
        'framework': 'TensorFlow/Keras'
    })

if __name__ == '__main__':
    # Run Flask app (model loads lazily on first prediction)
    print("\n" + "="*50)
    print("Skin Lesion Classification Web App")
    print("="*50)
    print("Model path:", MODEL_PATH)
    print("Model exists:", os.path.exists(MODEL_PATH))
    print("Open your browser and go to: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
