import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the configuration for Streamlit secrets
api_key = st.secrets["G_Key"]
genai.configure(api_key=api_key)

# Function to preprocess and load image dataset
def load_images(image_paths):
    data, labels = [], []
    for label, imagePath in image_paths:
        image = Image.open(imagePath)
        image = img_to_array(image)
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)

# Function to build and train model
def train_model(image_paths):
    data, labels = load_images(image_paths)
    
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # Split data into training and testing sets
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
    
    # Normalize the data
    trainX, testX = trainX / 255.0, testX / 255.0
    
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)
    
    # Evaluate the model
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY, predictions, target_names=lb.classes_))
    
    return model

# Function to classify image using the trained model
def classify_image(model, image_file):
    image = Image.open(image_file)
    image = image.resize((64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    return "Class 1" if prediction > 0.5 else "Class 2"

st.title("AI Image Classifier")

# User inputs
item1 = st.text_input("Enter the first item to classify (e.g., motorcycle):")
item2 = st.text_input("Enter the second item to classify (e.g., car):")

# Temporary directories for image storage
if item1 and item2:
    with tempfile.TemporaryDirectory() as tempdir:
        st.write(f"Temporary directory created at {tempdir}")
        
        # Upload images for the first item
        st.write(f"Upload images for {item1}")
        item1_files = st.file_uploader(f"Choose images for {item1}...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        
        # Upload images for the second item
        st.write(f"Upload images for {item2}")
        item2_files = st.file_uploader(f"Choose images for {item2}...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        
        if item1_files and item2_files:
            image_paths = []
            
            # Save item1 images to temporary directory
            for file in item1_files:
                file_path = os.path.join(tempdir, f"{item1}_{file.name}")
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                image_paths.append((item1, file_path))
            
            # Save item2 images to temporary directory
            for file in item2_files:
                file_path = os.path.join(tempdir, f"{item2}_{file.name}")
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                image_paths.append((item2, file_path))
            
            # Train the model
            model = train_model(image_paths)
            st.write(f"Model trained to classify {item1} vs {item2}")

# Upload an image for classification
uploaded_file = st.file_uploader("Choose an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and item1 and item2:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classify_image(model, uploaded_file)
    st.write(f'This image is classified as: {label}')
