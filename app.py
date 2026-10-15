import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained CNN model
model = tf.keras.models.load_model("breast_cancer_cnn.keras")

# Class names must match encoder.classes_ from training
class_names = ["benign", "malignant", "normal"]

# App title
st.title("Breast Ultrasound Image Classifier")

st.warning(
    "Educational project only. This model is not a medical "
    "diagnostic tool. Do not use predictions for medical decisions."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded ultrasound image")

    # Preprocess image
    image = image.resize((128, 128))
    image_array = np.array(image, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))

    # Predict
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]

    # Display result
    st.subheader(f"Model prediction: {predicted_class.title()}")

    st.write("Model class scores (not calibrated medical probabilities):")

    for i, name in enumerate(class_names):
        st.write(f"{name.title()}: {probabilities[i] * 100:.2f}%")