import warnings
import logging
import absl.logging
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st

# Suppress warnings and logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")

# Load trained model
model = tf.keras.models.load_model('efficientnetb0_transfer_learning_with_batch_processing_finetuned.h5')

# Index-to-label and label-to-food name mappings
index_to_class = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '3', '4', '5', '6', '7', '8', '9']
with open('label_to_item.json', 'r') as f:
    label_to_item = json.load(f)

# Configure Streamlit app
st.set_page_config(page_title="Food Item Classifier", page_icon="🍴", layout="wide")
st.markdown("<h1 style='text-align: center;'>Food Item Classifier 🍽️</h1>", unsafe_allow_html=True)

# Divide the page into three columns
col1, col2, col3 = st.columns([2, 1.5, 2])

# File upload in the left column
with col1:
    st.markdown("<h3>Upload Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop or select a food image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display image preview in the middle column
    with col2:
        st.markdown("<h3>Image Preview</h3>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Selected Image", use_container_width=True)

    # Preprocess image and make predictions
    img_array = img_to_array(image.resize((224, 224)))
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = index_to_class[predicted_class_index]
    predicted_food_item = label_to_item[predicted_label]
    confidence = predictions[0][predicted_class_index] * 100

    # Display prediction and confidence in the right column
    with col3:
        st.markdown("<h3>Prediction</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size: 20px; color: #2b2b2b; padding: 10px; background-color: #f9f9f9; border-radius: 10px;'>"
            f"<b>Hey!👋 I am Saubhagya's CNN Model and... <b/></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='font-size: 20px; color: #2b2b2b; padding: 10px; background-color: #f9f9f9; border-radius: 10px;'>"
            f"I am <strong>{confidence:.1f}%</strong> sure that there is <strong>{predicted_food_item}</strong> in this Image👀...Looks Yummy😋...HiHi😁</div>",
            unsafe_allow_html=True
        )