# Fine-Tuning-Deploying-VGG16-Teeth-Classification

This project aims to classify various dental conditions using a fine-tuned Convolutional Neural Network (CNN). The application is built using Streamlit and TensorFlow, allowing users to upload images of teeth and receive predictions about the type of dental condition present.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Finetuning](#model-finetuning)
- [Deployment](#deployment)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This repository contains code for training and deploying a machine learning model to classify dental conditions based on images of teeth. The project includes:
- A Jupyter Notebook for finetuning a pre-trained CNN model on a dataset of teeth images.
- A Python script for deploying the fine-tuned model as a web application using Streamlit.

## Model Finetuning
The model is fine-tuned using a pre-trained CNN on a dataset of dental images. The training process involves:
1. Loading a pre-trained model.
2. Fine-tuning the model on a custom dataset.
3. Saving the fine-tuned model for deployment.

### Finetuning Process
The finetuning process is implemented in the Jupyter Notebook (`Fine-Tune-VGG16-Teeth_Diseases.ipynb`). The key steps are:
- Data Preprocessing: Resize images, normalize pixel values, and create data generators for training and validation.
- Model Architecture: Load a pre-trained CNN (VGG-16) and add custom layers for classification.
- Training: Compile the model and train it on the dataset, monitoring validation accuracy and loss.
- Saving the Model: The trained model is saved as `TeethClassification_finetuned.h5`.

## Deployment
The deployment is handled using a Streamlit web application, allowing users to upload images and get predictions from the trained model.

### Deployment Script
The deployment script (`deployment.py`) contains the following key components:
- **Model Loading:** The trained model is loaded using TensorFlow.
- **Image Uploading:** Users can upload an image of teeth via a file uploader.
- **Image Preprocessing:** Uploaded images are resized and normalized before being fed into the model.
- **Prediction:** The model predicts the class of the dental condition, which is then displayed to the user.

## Requirements
- Python 3.7 or higher
- TensorFlow 2.5 or higher
- Streamlit 1.3 or higher
- Pillow (PIL)

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yahiaahmed4/Fine-Tuning-deploying-VGG16-Teeth-Classification.git
   cd Fine-Tuning-deploying-VGG16-Teeth-Classification
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model:**
   - Place the `TeethClassification_finetuned.h5` model in the same directory as the `deployment.py` script.

## Usage
### Running the Web Application
To run the Streamlit web application, use the following command:
```bash
streamlit run deployment.py
```
After running the command, a web browser will open, allowing you to upload an image and receive a prediction.

### Example
1. Open the app.
2. Upload an image of teeth (JPG format).
3. View the predicted dental condition displayed on the screen.

## Troubleshooting
- **403 Error on Image Upload:** If you encounter an AxiosError with a 403 status code when uploading images, ensure that CORS and XSRF protection are disabled in your Streamlit configuration. You can do this by adding the following to your `.streamlit/config.toml` file:
   ```toml
   [server]
   enableXsrfProtection = false
   enableCORS = false
   ```
