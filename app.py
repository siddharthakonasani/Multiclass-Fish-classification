import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set page title
st.title("🐟 Multiclass Fish Image Classification - MobileNetV2")

# Load the trained MobileNetV2 model (ensure the .h5 file is in the same directory)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('MobileNetV2_fish_model.h5')
    return model


model = load_model()

# List of class names in the same order as the training's class_indices
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

# File uploader
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img: Image.Image):
    """Resize, convert to array, and normalize image."""
    img = img.resize((224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def plot_probabilities(probs, classes):
    """Plot a horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, probs, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # highest probability at top
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Probabilities for All Classes')
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i + 0.1, f"{v*100:.2f}%", color='blue')
    st.pyplot(fig)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Fish Image', use_column_width=True)

    # Preprocess
    processed_img = preprocess_image(img)

    # Prediction
    preds = model.predict(processed_img)[0]
    pred_class_idx = np.argmax(preds)
    pred_class = class_names[pred_class_idx]
    pred_prob = preds[pred_class_idx]

    st.success(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{pred_prob*100:.2f}%**")

    # Show full class probabilities bar chart
    plot_probabilities(preds, class_names)
else:
    st.info("Please upload a fish image to classify.")
