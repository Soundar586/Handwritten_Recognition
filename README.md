# ✍️ Handwritten Digit Recognition System using Deep Learning

## Overview
This project implements a deep learning–based handwritten digit recognition system using a Convolutional Neural Network (CNN).  
The system accurately recognizes handwritten digits (0–9) from images by learning visual patterns from the MNIST dataset and applying robust preprocessing techniques to handle real-world handwritten inputs.

An interactive web interface is provided using Streamlit for easy image upload and prediction.

---

## Objectives
- Recognize handwritten digits using a CNN
- Achieve high classification accuracy on unseen data
- Handle real-world handwritten images through proper preprocessing
- Provide a user-friendly web interface for prediction
- Demonstrate practical application of deep learning concepts

---

## 🛠️ Tools & Technologies
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Image Processing:** OpenCV, Pillow  
- **UI Framework:** Streamlit  
- **Dataset:** MNIST  

---

## System Architecture

Input Image

↓ 

Image Preprocessing
(grayscale, inversion, thresholding,
cropping, centering, resizing)

↓

Convolutional Neural Network (CNN)

↓

Softmax Classification
(0–9)

↓

Predicted Digit + Confidence

---

## Key Features
-  CNN-based handwritten digit classification  
-  MNIST-trained model with ~99% test accuracy  
-  Robust preprocessing to handle real handwritten images  
-  Image centering and normalization (MNIST-style)  
-  Confidence score for each prediction  
-  Interactive Streamlit web application  

---

##  Model Performance
- **Test Accuracy:** ~99% on MNIST test dataset
- The model generalizes well after aligning inference preprocessing with training data distribution.

---

##  Image Preprocessing Strategy
To ensure correct predictions for real-world images, the following preprocessing steps are applied:

1. Convert image to grayscale  
2. Invert colors to match MNIST format  
3. Apply thresholding to remove noise  
4. Detect and crop digit using contours  
5. Resize digit to 20×20  
6. Center digit in a 28×28 frame  
7. Normalize pixel values to range [0,1]  

This alignment significantly improves prediction accuracy on user-uploaded images.

---

---

##  How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Train the Model (Optional – Already Trained)
```bash
python train.py
```
### Run the Web Application
```bash
streamlit run app.py
```
---

### Future Enhancements

Support for handwritten letters (EMNIST)

Digit drawing canvas in UI

Model explainability (feature visualization)

Mobile-friendly interface

Cloud deployment

---
### Conclusion

The Handwritten Digit Recognition System successfully applies deep learning techniques to recognize handwritten digits with high accuracy.
By carefully aligning preprocessing steps with the training data distribution, the system effectively handles real-world handwritten images and provides reliable predictions through an intuitive user interface.
