import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import tempfile
import os

# ---------- Page Config ----------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9)")

# ---------- Load Model ----------
MODEL_PATH = "model/digit_cnn.keras"  # or .keras if you changed
model = load_model(MODEL_PATH)

# ---------- Preprocessing ----------
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = img[y:y+h, x:x+w]

    digit = cv2.resize(digit, (20, 20))

    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    padded[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    padded = padded / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded

# ---------- File Upload ----------
uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=200)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    processed = preprocess_image(temp_path)

    if processed is None:
        st.error("No digit detected in image")
    else:
        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.subheader("✅ Prediction Result")
        st.metric("Predicted Digit", digit)
        st.metric("Confidence", f"{confidence * 100:.2f} %")

    os.remove(temp_path)
