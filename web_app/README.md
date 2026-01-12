# Skin Lesion Classification Web Application

A web-based interface for the AI-powered skin lesion classification model trained on the HAM10000 dataset.

## 🌟 Features

- **Modern UI**: Clean, responsive design with intuitive drag-and-drop image upload
- **Real-time Analysis**: Upload a skin lesion image and get instant AI predictions
- **Detailed Results**: Shows confidence percentages for all 7 skin lesion classes
- **Educational Content**: Information about each skin condition type
- **Medical Disclaimer**: Clear warnings that this is for educational purposes only

## 📁 Project Structure

```
web_app/
├── app.py                  # Flask backend server
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Main web page
└── static/
    ├── css/
    │   └── style.css      # Stylesheet
    └── js/
        └── main.js        # Frontend JavaScript
```

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then install the required packages:

```bash
cd web_app
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.12+
- Flask 2.3+
- The trained model file (`best_model_EfficientNetB0_Transfer_phase2.keras` or `best_model_EfficientNetB0_Transfer.keras`) in the parent directory

## 📊 Skin Lesion Classes

The model can classify 7 types of skin lesions:

| Code  | Name                 | Severity            |
| ----- | -------------------- | ------------------- |
| nv    | Melanocytic Nevi     | Benign              |
| bkl   | Benign Keratosis     | Benign              |
| df    | Dermatofibroma       | Benign              |
| vasc  | Vascular Lesions     | Benign              |
| akiec | Actinic Keratoses    | Pre-cancerous       |
| bcc   | Basal Cell Carcinoma | Cancerous           |
| mel   | Melanoma             | Cancerous (Serious) |

## 🖼️ How to Use

1. Open the web application in your browser
2. Scroll to the "Analyze Skin Lesion" section
3. Drag and drop an image or click to upload
4. Click "Analyze Image"
5. View the prediction results with confidence percentages

## ⚠️ Disclaimer

**This application is for educational purposes only.** It should NOT be used for medical diagnosis. Always consult a qualified dermatologist for any skin concerns or medical advice.

## 🛠️ Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Model**: EfficientNetB0 with Transfer Learning
- **Dataset**: HAM10000 (10,015 dermatoscopic images)
- **Image Processing**: TensorFlow/Keras preprocessing
- **Input Size**: 254x254 RGB images

## 📝 API Endpoints

### `GET /`

Serves the main web page.

### `POST /api/predict`

Analyzes an uploaded skin lesion image.

**Request**: `multipart/form-data` with `image` field

**Response**:

```json
{
  "success": true,
  "prediction": {
    "class_code": "nv",
    "class_name": "Melanocytic Nevi",
    "description": "Common moles...",
    "severity": "Benign",
    "confidence": 95.42
  },
  "all_predictions": [...]
}
```

### `GET /api/model-info`

Returns information about the model architecture and capabilities.

## 🎨 Screenshots

The web app features:

- Hero section with model statistics
- About section explaining the model
- Interactive image upload area
- Detailed results with probability bars
- Information cards for each skin lesion class

---

Built with ❤️ using TensorFlow/Keras and Flask
