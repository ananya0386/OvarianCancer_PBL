import tensorflow as tf
from PIL import Image
import numpy as np

print("Loading model...")
model = tf.keras.models.load_model("ovarian_cancer_densenet.keras")
print("Model loaded.")

print("Loading image...")
# Load an image from Histopathology/Non_Cancerous/
img_path = r"c:\Users\Ananya Kawatra\OneDrive\Documents\PBL\Histopathology\Non_Cancerous\114.JPG"
image = Image.open(img_path).convert('RGB')

# Preprocess image
img_resized = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0) # Create batch axis

print("Predicting...")
pred = model.predict(img_array)
print("Raw prediction output:", pred)
print("Scalar:", pred[0][0])
print(f"Formatted: {pred[0][0]:.1%}")

print("Done.")
