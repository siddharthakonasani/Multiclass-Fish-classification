import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Fish Species Classifier - MobileNetV2")

st.title("🐟 Multiclass Fish Image Classification - MobileNetV2")

MODEL_PATH = 'MobileNetV2_fish_model.h5'  # Ensure model file has this exact name and is in same dir

@st.cache(allow_output_mutation=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please upload the model file in the app directory.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# List of class names matching the training label order
class_names = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food house_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def plot_probabilities(probs, classes):
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, probs, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # highest at top
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Probabilities for All Classes')
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i + 0.1, f"{v*100:.2f}%", color='blue')
    st.pyplot(fig)

if model is not None:
    uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded Fish Image', use_column_width=True)

            processed_img = preprocess_image(img)
            preds = model.predict(processed_img)[0]
            pred_class_idx = np.argmax(preds)
            pred_class = class_names[pred_class_idx]
            pred_prob = preds[pred_class_idx]

            st.success(f"Prediction: **{pred_class}**")
            st.write(f"Confidence: **{pred_prob*100:.2f}%**")

            plot_probabilities(preds, class_names)
        except Exception as e:
            st.error(f"Error processing image or prediction: {e}")
    else:
        st.info("Please upload a fish image to classify.")
else:
    st.warning("Model is not loaded. Please fix the model loading issue and restart the app.")
