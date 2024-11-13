import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from PIL import Image

# Load the model (this part stays the same as your existing code)
model_path = r'models\20241111-114215-full-image-set-mobilenetv2-Adam.h5'  # Replace with actual model URL or path
IMG_SIZE = 224
BATCH_SIZE = 32

# Load the breed labels (from your CSV)
labels_csv = pd.read_csv("data/labels.csv")
labels = labels_csv["breed"]
unique_breeds = np.unique(labels)

# Preprocess the image
def process_image(image, img_size=IMG_SIZE):
    """
    Takes an image file and preprocesses it for the model.
    """
    # Convert the PIL image to a numpy array
    image = np.array(image)
    
    # Resize the image
    image = tf.image.resize(image, [img_size, img_size])
    
    # Normalize the image (scale to 0-1)
    image = image / 255.0
    
    return image

def load_model(model_path):
    """
    Loads a saved Keras model from a specified path, including custom layers like KerasLayer.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model


# Load the trained model (already done in your code)
model = load_model(model_path)

# Function to predict the breed
def predict_breed(image):
    image = process_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_label = unique_breeds[np.argmax(prediction)]
    return predicted_label

# Streamlit UI improvements

# Set the page layout to dark mode and narrow mode
st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐕", layout="centered")

# Dark Mode configuration
st.markdown(
    """
    <style>
        body {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        .stFileUploader {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Styling
st.markdown("<h1 style='text-align: center; color: #1DBF73;'>Dog Breed Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of a dog, and the model will predict its breed. 🐶</p>", unsafe_allow_html=True)

# File uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded, display it and make predictions
if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)
    
    # Center the image display
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)  # Display image
    st.markdown("</div>", unsafe_allow_html=True)

    # Button to predict breed
    if st.button("Predict Breed"):
        breed = predict_breed(image)
        st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>The predicted breed is: {breed}</h3>", unsafe_allow_html=True)

# Updated footer message
st.markdown("""
    <footer style="text-align: center; font-size: 12px; padding-top: 20px; color: gray;">
        <p>Dog Breed Classifier | Powered by AI | Developed by Sugam</p>
    </footer>
""", unsafe_allow_html=True)
