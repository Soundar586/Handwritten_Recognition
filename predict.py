import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- Load Model ----------
MODEL_PATH = "model/digit_cnn.keras"  # or .keras
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    import cv2
    import numpy as np

    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Invert image (MNIST style)
    img = cv2.bitwise_not(img)

    # Threshold (binarize)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find contours (digit outline)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No digit found in image")

    # Get bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = img[y:y+h, x:x+w]

    # Resize digit to fit into 20x20 box (MNIST standard)
    digit = cv2.resize(digit, (20, 20))

    # Create blank 28x28 image
    padded = np.zeros((28, 28), dtype=np.uint8)

    # Center the digit
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    padded[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    # Normalize
    padded = padded / 255.0

    # Reshape for CNN
    padded = padded.reshape(1, 28, 28, 1)

    return padded

def predict_digit(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_digit, confidence


# ---------- Test Prediction ----------
if __name__ == "__main__":
    test_image_path = "data/digit.png"
    print("Image loaded, starting prediction...")

    digit, confidence = predict_digit(test_image_path)
    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence * 100:.2f}%")
