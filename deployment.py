import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("C:\\Users\\yehia\\Desktop\\Cellula Internship\\Week2 Data\\TeethClassification_finetuned.h5")

# Define class names
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Streamlit app title
st.title('Teeth Classification App')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        # Convert the file to an image
        img = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)
        predicted_class_name = class_names[predicted_class[0]]

        # Display the prediction
        st.write(f'Predicted Class: {predicted_class_name}')
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
