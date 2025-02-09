import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from explainability_dl import integrated_gradients, preprocess_image

# Load the deep learning model
@st.cache_resource
def load_deep_learning_model():
    return tf.keras.models.load_model("dl-model.h5")

dnn_model = load_deep_learning_model()

def deep_learning_predict(image):
    processed_image = preprocess_image(image)
    prediction = dnn_model.predict(processed_image)
    return prediction[0][1]  # Assuming binary classification (0: No Cancer, 1: Cancer)

def explain_prediction(image):
    processed_image = preprocess_image(image)
    ig_map = integrated_gradients(dnn_model, processed_image, class_idx=1)
    return ig_map

def non_deep_learning_predict(image):
    # Placeholder for non-deep learning method
    # Example: Using traditional image processing techniques
    return np.random.choice([0, 1])  # Replace with actual implementation

# Streamlit App
st.title("Cancer Detection from Image")
st.write("Upload an image and select a method for cancer prediction.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
method = st.radio("Select Prediction Method", ("Deep Learning Model", "Non-Deep Learning Model"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing...")
    if method == "Deep Learning Model":
        prediction = deep_learning_predict(image)
        explanation = explain_prediction(image)
        st.write("Feature Importance Explanation Generated")
    else:
        prediction = non_deep_learning_predict(image)
    
    if prediction > 0.5:
        st.error("The model predicts this image is CANCEROUS.")
    else:
        st.success("The model predicts this image is NOT cancerous.")
