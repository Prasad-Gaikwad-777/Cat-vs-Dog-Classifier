# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:34:02 2025

@author: lenovo
"""

import streamlit as st
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Define the prediction function
def final_prediction(img_path, model_path):
    """
    Predict whether the given image is a Cat or Dog.

    Parameters:
        img_path: Path to the input image of a cat or dog to be classified.
        model_path: The trained TensorFlow/Keras model for prediction.

    Returns:
        str: Prediction result ('Cat' or 'Dog').
    """
    # Load the image
    img = cv2.imread(img_path)  # Read the image from the file
    if img is None:
        return "Error: Image not found or cannot be read."

    # Preprocess the image
    img = cv2.resize(img, (256, 256))  # Resize to 256x256
    img = img.reshape((1, 256, 256, 3))  # Add batch dimension
    

    # Load the model
    model = load_model(model_path)

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = prediction[0][0]  # Get the class index

    # Determine the result
    if 0 <= predicted_class < 0.80:
        return "Cat"
    elif predicted_class >= 0.81:
        return "Dog"
    else:
        return "Oops! Sorry, I can't decide this..."

# Streamlit Application
def main():
    
    st.sidebar.title("Upload Image for Analysis")

    html_temp = """
    <div style="background-color:tomato;padding:15px">
    <h2 style="color:white;text-align:center;">CAT vs DOG CLASSIFIER</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="fileUploader")
    model_path = r"C:\Users\lenovo\OneDrive\Desktop\Personal Projects\Python Projects Personal\Dog and Cat Detection Deep learning campusx\final models\model_1.h5"  # Replace with your model path

    if st.sidebar.button("Check"):
        if uploaded_file is not None:
            temp_path = "temp_image.jpg"
            image = Image.open(uploaded_file)
            image.save(temp_path)   

            try:
                prediction = final_prediction(temp_path, model_path)

                # Display the output in the main area
                st.markdown(f"""<div style='text-align: center; font-size: 36px; font-weight: bold;'>Classified as: {prediction}</div>""", unsafe_allow_html=True)

                resized_image = image.resize((640, 480))  # Resize image to 640x480
                st.image(resized_image, caption="Uploaded Image", use_container_width=True)
                

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

            # Remove temporary file
            os.remove(temp_path)
        else:
            st.sidebar.warning("Please upload an image first.")

if __name__ == '__main__':
    main()
