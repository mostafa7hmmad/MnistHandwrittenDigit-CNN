import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    layout="centered"
)

import numpy as np
import tensorflow as tf
from PIL import Image

# =====================
# Load Model
# =====================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_Adam.h5")

model = load_model()

# =====================
# App UI
# =====================
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9) and the model will predict it.")

# =====================
# Image Upload
# =====================
uploaded_file = st.file_uploader(
    "Upload an image (28x28 or any size)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(preds)
    confidence = np.max(preds)

    # Results
    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted Digit:** {predicted_digit}")
    st.write(f"**Confidence:** {confidence:.2%}")

    st.bar_chart(preds[0])
