"""
Skin Lesion Classification Using Deep Learning (TensorFlow/Keras)

HAM10000 Dataset - Skin Cancer Classification

Key Features:
- Split by lesion_id to prevent data leakage
- EfficientNet preprocessing
- Unfreeze top 20-30% of backbone
- LR = 1e-5 or lower
- Augmentation: Flip, Rotate +-20 degrees, Zoom <=10%
- IMG_SIZE = 384x384

Target: 90%+ Accuracy with F1 Score evaluation
"""

# =============================================================================
# 1. Import Required Libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, 
                                      Dropout, BatchNormalization, GlobalAveragePooling2D,
                                      Input, Activation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, precision_score, recall_score,
                             roc_curve, auc, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

from glob import glob

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# =============================================================================
# 2. Configuration
# =============================================================================

IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-5

# =============================================================================
# 3. Load Dataset
# =============================================================================

data_dir = 'dataset'
images_dir = os.path.join(data_dir, 'images')
extracted_images_dir = os.path.join(images_dir, 'extracted')

os.makedirs(extracted_images_dir, exist_ok=True)

zip_files = [
    os.path.join(images_dir, 'HAM10000_images_part_1 (1).zip'),
    os.path.join(images_dir, 'HAM10000_images_part_2.zip')
]

for zip_path in zip_files:
    if os.path.exists(zip_path):
        print(f"Extracting {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_images_dir)

metadata_path = os.path.join(data_dir, 'HAM10000_metadata (2).csv')
metadata_df = pd.read_csv(metadata_path)

print(f"Metadata Shape: {metadata_df.shape}")

all_image_paths = glob(os.path.join(extracted_images_dir, '*.jpg'))
print(f"Total images found: {len(all_image_paths)}")

image_path_dict = {}
for path in all_image_paths:
    image_id = os.path.splitext(os.path.basename(path))[0]
    image_path_dict[image_id] = path

dx_to_label = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
}
class_names = {v: k for k, v in dx_to_label.items()}

# =============================================================================
# 4. Load Images with Lesion IDs
# =============================================================================

def load_images_with_lesion_ids(metadata_df, image_path_dict):
    images, labels, lesion_ids = [], [], []
    
    for idx, row in metadata_df.iterrows():
        image_id = row['image_id']
        if image_id in image_path_dict:
            try:
                img = load_img(image_path_dict[image_id], target_size=(IMG_SIZE, IMG_SIZE))
                images.append(img_to_array(img))
                labels.append(dx_to_label[row['dx']])
                lesion_ids.append(row['lesion_id'])
            except:
                pass
    
    print(f"Loaded {len(images)} images, {len(set(lesion_ids))} unique lesions")
    return np.array(images), np.array(labels), np.array(lesion_ids)

X, y, lesion_ids = load_images_with_lesion_ids(metadata_df, image_path_dict)
print(f"Dataset shape: {X.shape}")

# =============================================================================
# 5. Split by Lesion ID (NO DATA LEAKAGE)
# =============================================================================

num_classes = len(np.unique(y))
X_normalized = preprocess_input(X.astype('float32'))
y_categorical = to_categorical(y, num_classes=num_classes)

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(X_normalized, y, groups=lesion_ids))

X_train, y_train = X_normalized[train_idx], y_categorical[train_idx]
X_temp, y_temp = X_normalized[temp_idx], y_categorical[temp_idx]
lesion_ids_temp = lesion_ids[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(X_temp, np.argmax(y_temp, axis=1), groups=lesion_ids_temp))

X_val, y_val = X_temp[val_idx], y_temp[val_idx]
X_test, y_test = X_temp[test_idx], y_temp[test_idx]

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# =============================================================================
# 6. Class Weights
# =============================================================================

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
dampened_weights = np.power(class_weights, 0.5)
dampened_weights = dampened_weights / np.mean(dampened_weights)
class_weight_dict = dict(enumerate(dampened_weights))

# =============================================================================
# 7. Data Augmentation (Flip, Rotate +-20, Zoom <=10%)
# =============================================================================

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# 8. Build Model
# =============================================================================

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Unfreeze top 25% of backbone
total_layers = len(base_model.layers)
freeze_until = int(total_layers * 0.75)

base_model.trainable = True
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

print(f"Trainable: {sum(1 for l in base_model.layers if l.trainable)}/{total_layers}")

# =============================================================================
# 9. Train
# =============================================================================

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8)
]

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator,
                    class_weight=class_weight_dict, callbacks=callbacks, verbose=1)

# =============================================================================
# 10. Evaluate
# =============================================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=[class_names[i] for i in range(num_classes)]))

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/skin_classifier.keras')
print("Model saved!")
