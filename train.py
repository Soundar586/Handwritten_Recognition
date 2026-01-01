"""
train.py
---------
Builds, trains, evaluates, and saves a CNN model
for handwritten digit recognition.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # optional: suppress warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from preprocess import load_and_preprocess_data

# ---------- Load Data ----------
x_train, y_train, x_test, y_test = load_and_preprocess_data()

# ---------- Build CNN Model ----------
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu",
           input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# ---------- Compile Model ----------
model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------- Model Summary ----------
model.summary()

# ---------- Train Model ----------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# ---------- Evaluate Model ----------
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# ---------- Save Model ----------
os.makedirs("model", exist_ok=True)
model.save("model/digit_cnn.keras")

print("Model saved as model/digit_cnn.keras")
