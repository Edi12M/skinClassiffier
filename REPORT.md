# Skin Lesion Classification Using Deep Learning

## CEN352 Term Project Report

---

## Executive Summary

This report presents the development and evaluation of a deep learning model for classifying skin lesions using the HAM10000 dataset. The model utilizes **EfficientNetB0 Transfer Learning** architecture to classify dermoscopic images into 7 different skin lesion categories.

| Metric                  | Value  |
| ----------------------- | ------ |
| **Final Accuracy**      | 66.53% |
| **F1 Score (Macro)**    | 0.5383 |
| **AUC-ROC (Macro)**     | 0.9272 |
| **Average Specificity** | 94.16% |

---

## 1. Introduction

### 1.1 Project Objective

The objective of this project is to build a Convolutional Neural Network (CNN) model capable of classifying skin lesions into 7 categories with high accuracy. The model aims to assist in early detection of skin cancer through automated image analysis.

### 1.2 Classification Categories

The model classifies skin lesions into the following 7 categories:

| Class Code | Full Name            | Description                        |
| ---------- | -------------------- | ---------------------------------- |
| **akiec**  | Actinic Keratoses    | Pre-cancerous skin condition       |
| **bcc**    | Basal Cell Carcinoma | Common form of skin cancer         |
| **bkl**    | Benign Keratosis     | Non-cancerous skin growths         |
| **df**     | Dermatofibroma       | Benign skin nodules                |
| **mel**    | Melanoma             | Most dangerous form of skin cancer |
| **nv**     | Melanocytic Nevi     | Common moles                       |
| **vasc**   | Vascular Lesions     | Blood vessel-related skin lesions  |

---

## 2. Dataset Description

### 2.1 HAM10000 Dataset Overview

| Property              | Value               |
| --------------------- | ------------------- |
| **Total Images**      | 10,015              |
| **Image Resolution**  | 254 × 254 × 3 (RGB) |
| **Number of Classes** | 7                   |
| **Unique Lesions**    | 7,470               |
| **Pixel Range**       | [0, 255]            |

### 2.2 Class Distribution

The dataset exhibits significant **class imbalance**:

| Class                      | Count | Percentage |
| -------------------------- | ----- | ---------- |
| nv (Melanocytic Nevi)      | 6,705 | 66.9%      |
| mel (Melanoma)             | 1,113 | 11.1%      |
| bkl (Benign Keratosis)     | 1,099 | 11.0%      |
| bcc (Basal Cell Carcinoma) | 514   | 5.1%       |
| akiec (Actinic Keratoses)  | 327   | 3.3%       |
| vasc (Vascular Lesions)    | 142   | 1.4%       |
| df (Dermatofibroma)        | 115   | 1.1%       |

**⚠️ Note:** The dataset is highly imbalanced with the 'nv' class dominating at ~67% of all samples. This imbalance was addressed using dampened class weights during training.

### 2.3 Sample Images from Each Class

The dataset contains dermoscopic images showing various skin lesions with distinct visual characteristics for each category.

---

## 3. Technical Implementation

### 3.1 Development Environment

| Component      | Version/Details             |
| -------------- | --------------------------- |
| **TensorFlow** | 2.20.0                      |
| **GPU**        | CPU-only (No GPU available) |
| **Image Size** | 254 × 254 pixels            |
| **Batch Size** | 16                          |

### 3.2 Data Preprocessing

1. **EfficientNet Preprocessing**: Pixel values scaled to [-1, 1] range using `preprocess_input`
2. **Original Pixel Range**: [0.0, 255.0]
3. **Lesion-based Splitting**: Data split by lesion_id to prevent data leakage

### 3.3 Data Split (Grouped by Lesion ID - No Data Leakage)

| Split      | Samples | Percentage | Unique Lesions |
| ---------- | ------- | ---------- | -------------- |
| Training   | 7,002   | 69.9%      | 5,229          |
| Validation | 1,519   | 15.2%      | 1,120          |
| Test       | 1,494   | 14.9%      | 1,121          |

**✅ Lesion Overlap Check:**

- Train-Val overlap: 0 ✓
- Train-Test overlap: 0 ✓
- Val-Test overlap: 0 ✓

### 3.4 Data Augmentation

| Augmentation    | Setting     |
| --------------- | ----------- |
| Rotation        | ±20 degrees |
| Zoom            | ≤10%        |
| Horizontal Flip | Yes         |
| Vertical Flip   | Yes         |
| Fill Mode       | Nearest     |

### 3.5 Class Weights (Dampened)

To handle class imbalance, dampened class weights were applied (dampen factor = 0.5):

| Class | Dampened Weight | Original Weight |
| ----- | --------------- | --------------- |
| akiec | 1.169           | 4.832           |
| bcc   | 0.862           | 2.625           |
| bkl   | 0.618           | 1.350           |
| df    | 1.783           | 11.239          |
| mel   | 0.606           | 1.299           |
| nv    | 0.245           | 0.212           |
| vasc  | 1.717           | 10.420          |

---

## 4. Model Architecture

### 4.1 EfficientNetB0 Transfer Learning Model

| Property                 | Value                                |
| ------------------------ | ------------------------------------ |
| **Base Model**           | EfficientNetB0 (ImageNet pretrained) |
| **Total Layers**         | 246                                  |
| **Trainable Layers**     | 8 (initial), 25% later unfrozen      |
| **Non-trainable Layers** | 238                                  |

### 4.2 Custom Classification Head

The pre-trained EfficientNetB0 base was extended with:

- Global Average Pooling 2D
- Batch Normalization
- Dense Layer (512 units, ReLU)
- Dropout (0.5)
- Batch Normalization
- Dense Layer (256 units, ReLU)
- Dropout (0.3)
- Output Dense Layer (7 units, Softmax)

### 4.3 Training Configuration

| Parameter       | Phase 1                  | Phase 2                  |
| --------------- | ------------------------ | ------------------------ |
| Learning Rate   | 1e-5                     | 5e-6                     |
| Optimizer       | Adam                     | Adam                     |
| Loss Function   | Categorical Crossentropy | Categorical Crossentropy |
| Epochs          | 50 (with early stopping) | 30 (with early stopping) |
| Backbone Frozen | 75% frozen               | 70% frozen               |

### 4.4 Callbacks Used

- **EarlyStopping**: Patience = 10, monitor val_accuracy
- **ModelCheckpoint**: Save best model by val_accuracy
- **ReduceLROnPlateau**: Factor = 0.5, patience = 5, min_lr = 1e-8

---

## 5. Model Performance Results

### 5.1 Test Set Evaluation

| Metric            | Score  |
| ----------------- | ------ |
| **Test Loss**     | 0.8801 |
| **Test Accuracy** | 67.40% |

### 5.2 Overall Performance Metrics

| Metric                  | Value  |
| ----------------------- | ------ |
| **Accuracy**            | 66.53% |
| **F1 Score (Macro)**    | 0.5383 |
| **F1 Score (Micro)**    | 0.6740 |
| **F1 Score (Weighted)** | 0.6919 |
| **Precision (Macro)**   | 0.5223 |
| **Recall (Macro)**      | 0.6714 |
| **AUC-ROC (Macro)**     | 0.9272 |
| **Average Sensitivity** | 0.6677 |
| **Average Specificity** | 0.9416 |

### 5.3 Per-Class Performance Metrics

| Class | Precision | Recall | F1 Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| akiec | 0.6667    | 0.1905 | 0.2963   | 63      |
| bcc   | 0.5000    | 0.7353 | 0.5952   | 68      |
| bkl   | 0.3857    | 0.7105 | 0.5000   | 152     |
| df    | 0.2727    | 0.8571 | 0.4138   | 7       |
| mel   | 0.3605    | 0.5668 | 0.4407   | 187     |
| nv    | 0.9502    | 0.7088 | 0.8120   | 996     |
| vasc  | 0.5135    | 0.9048 | 0.6552   | 21      |

### 5.4 Sensitivity & Specificity Per Class

| Class       | Sensitivity | Specificity |
| ----------- | ----------- | ----------- |
| akiec       | 0.1905      | 0.9958      |
| bcc         | 0.7353      | 0.9649      |
| bkl         | 0.7105      | 0.8718      |
| df          | 0.8571      | 0.9892      |
| mel         | 0.5668      | 0.8562      |
| nv          | 0.7088      | 0.9257      |
| vasc        | 0.9048      | 0.9878      |
| **Average** | **0.6677**  | **0.9416**  |

### 5.5 AUC-ROC Scores Per Class

| Class            | AUC-ROC Score |
| ---------------- | ------------- |
| akiec            | 0.9258        |
| bcc              | 0.9639        |
| bkl              | 0.8976        |
| df               | 0.9796        |
| mel              | 0.8192        |
| nv               | 0.9188        |
| vasc             | 0.9766        |
| **Macro AUC**    | **0.9259**    |
| **Weighted AUC** | **0.9076**    |

---

## 6. Confusion Matrix Analysis

### 6.1 Normalized Confusion Matrix (Percentage of True Class)

The confusion matrix reveals the following insights:

| True Class | Best Predicted As   | Accuracy |
| ---------- | ------------------- | -------- |
| akiec      | bkl (misclassified) | 19.05%   |
| bcc        | bcc ✓               | 73.53%   |
| bkl        | bkl ✓               | 71.05%   |
| df         | df ✓                | 85.71%   |
| mel        | mel ✓               | 56.68%   |
| nv         | nv ✓                | 70.88%   |
| vasc       | vasc ✓              | 90.48%   |

### 6.2 Key Observations from Confusion Matrix

1. **Best Performance**: vasc (90.48%), df (85.71%), bcc (73.53%)
2. **Worst Performance**: akiec (only 19.05% correctly classified)
3. **Common Misclassifications**:
   - akiec often misclassified as bkl (44.44%)
   - mel often confused with bkl (20.86%) and nv (12.30%)
   - nv sometimes misclassified as mel (14.86%)

---

## 7. Visual Results

### 7.1 Class Distribution Charts

The dataset visualization shows the severe class imbalance with 'nv' class dominating at 6,705 images (66.9% of total).

### 7.2 Sample Images from Each Class

Representative dermoscopic images from all 7 classes demonstrate the visual diversity of skin lesions.

### 7.3 Data Augmentation Examples

Visualization of augmented images showing rotation, flipping, and zoom transformations applied during training.

### 7.4 Per-Class Performance Bar Charts

Bar charts comparing Precision, Recall, and F1 Score across all 7 classes, with the 90% target line shown as reference.

### 7.5 Confusion Matrices

- Normalized confusion matrix showing percentage distributions
- Raw count confusion matrix showing actual classification counts

### 7.6 ROC Curves

Multi-class ROC curves (One-vs-Rest) showing excellent discrimination ability across all classes, with all AUC values > 0.81.

### 7.7 Model Performance Summary Chart

Horizontal bar chart summarizing all key metrics, highlighting the high AUC-ROC (0.9272) and Specificity (0.9416).

---

## 8. Sample Prediction Demonstration

### Test Prediction Example

| Property            | Value                 |
| ------------------- | --------------------- |
| **True Class**      | mel (Melanoma)        |
| **Predicted Class** | nv (Melanocytic Nevi) |
| **Confidence**      | 73.27%                |
| **Correct**         | ❌ No                 |

This example demonstrates one of the challenging cases where melanoma was misclassified as a common mole (nv), highlighting the difficulty in distinguishing between these visually similar classes.

---

## 9. Key Findings & Discussion

### 9.1 Strengths

1. **High AUC-ROC (0.9272)**: The model shows excellent discrimination ability, meaning it can rank lesions by risk effectively.

2. **High Specificity (94.16%)**: The model is very good at correctly identifying negative cases, reducing false alarms.

3. **Best Class Performance**:
   - vasc: 90.48% recall, 0.9766 AUC
   - df: 85.71% recall, 0.9796 AUC
   - bcc: 73.53% recall, 0.9639 AUC

### 9.2 Weaknesses

1. **Lower Overall Accuracy (66.53%)**: Below the 90% target, primarily due to class imbalance.

2. **Poor akiec Detection (19.05% recall)**: Actinic keratoses are frequently misclassified as other conditions.

3. **Melanoma Sensitivity (56.68%)**: Critical for clinical applications, melanoma detection needs improvement.

### 9.3 Impact of Class Imbalance

The severe imbalance (nv = 67% vs df = 1.1%) significantly impacts:

- Model bias towards majority class (nv)
- Difficulty learning minority class features
- Lower macro-averaged metrics

---

## 10. Saved Model Artifacts

The following files were saved for deployment:

| File                                              | Purpose                       |
| ------------------------------------------------- | ----------------------------- |
| `models/skin_classifier.keras`                    | Complete model (Keras format) |
| `models/skin_classifier.h5`                       | Complete model (H5 format)    |
| `models/model_architecture.json`                  | Architecture definition       |
| `models/class_names.json`                         | Class mapping dictionary      |
| `best_model_EfficientNetB0_Transfer.keras`        | Best Phase 1 model            |
| `best_model_EfficientNetB0_Transfer_phase2.keras` | Best Phase 2 model            |

---

## 11. Conclusions

### 11.1 Summary

This project successfully developed a deep learning model for skin lesion classification using:

- **EfficientNetB0** transfer learning architecture
- **HAM10000** dataset with 10,015 dermoscopic images
- **Two-phase training** strategy with gradual backbone unfreezing

### 11.2 Key Results

| Metric              | Achieved | 
| ------------------- | -------- | 
| Accuracy            | 66.53%   | 
| AUC-ROC (Macro)     | 92.72%   | 
| Specificity         | 94.16%   | 
| F1 Score (Weighted) | 69.19%   | 

### 11.3 Clinical Relevance

While the accuracy is below the target, the **high AUC-ROC (0.9272)** and **high specificity (94.16%)** indicate that:

- The model can effectively rank lesions by risk
- Low false positive rate reduces unnecessary biopsies
- Model could serve as a screening tool with specialist confirmation


---

## 12. Web Application

A Flask-based web application was developed for model deployment, allowing users to:

- Upload skin lesion images
- Receive classification predictions with confidence percentages
- View information about each lesion type

**Location**: `web_app/` directory

---

## Appendix: Technical Specifications

### A. Libraries Used

- TensorFlow 2.20.0
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL (Pillow)

### B. Training Hardware

- CPU-only training (no GPU detected)
- Windows operating system

### C. Training Time

- Approximately 35 seconds per test batch (47 batches total)
- Multiple epochs with early stopping

---

_Report generated from skin_classifier.ipynb notebook outputs_
