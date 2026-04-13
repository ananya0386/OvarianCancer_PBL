import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

print("Starting local training of the Histopathology model...")

DATA_DIR = "Histopathology"

filepaths = []
labels = []

# Assuming Non_Cancerous is class 0 (Non-Tumour), and the others are class 1 (Tumour)
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(folder_path, img_name))
                if folder == 'Non_Cancerous':
                    labels.append(0)
                else:
                    labels.append(1)

if len(filepaths) == 0:
    print("No images found in the Histopathology directory. Cannot train model.")
    exit(1)

print(f"Found {len(filepaths)} images.")

# Load and preprocess images
X = []
for fp in filepaths:
    img = load_img(fp, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    X.append(img_arr)

X = np.array(X)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the DenseNet121 Model (same structure as OCI.ipynb)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training model... this may take a few minutes on CPU.")

# Train for 5 epochs
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16
)

# Save the model
model.save("ovarian_cancer_densenet.keras")
print("Model successfully trained and saved to ovarian_cancer_densenet.keras!")
