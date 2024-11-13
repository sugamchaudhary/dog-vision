# 🐕 **Dog Vision: Classifying Dog Breeds with Deep Learning**

## Overview

**Dog Vision** is a project that aims to classify dog breeds based on images using deep learning techniques. The project leverages TensorFlow 2.0 and TensorFlow Hub for creating a deep neural network model that can accurately predict the breed of a dog from an image. 

The model uses a pre-trained MobileNetV2 model as a feature extractor and is fine-tuned with a custom dataset of dog images. The solution involves processing and resizing images, training the model on a labeled dog breed dataset, and making predictions for unseen dog images.

The application is built using Streamlit, providing an easy-to-use web interface where users can upload an image of a dog and get the predicted breed in real-time.


## Dataset Description

The dataset used for this project consists of images of dogs from a variety of breeds. It includes both a training set, where the breed labels are provided, and a test set, where the goal is to predict the breed based on the images.

The dataset contains **120 unique dog breeds** and is designed to challenge the model in recognizing different breeds based on visual features. Each image in the dataset is uniquely identified by its filename, which serves as its ID.

The objective of this project is to develop a classifier capable of identifying a dog's breed from an image with high accuracy. The dataset provides a rich variety of dog breeds, allowing the model to learn discriminative features that can generalize well to new images.


## File Descriptions

The following files and folders are included in the project:

- **train.zip**: This file contains the training set images of dogs, where each image has an associated breed label. These images will be used to train the model.

- **test.zip**: This file contains the test set images of dogs, which are unlabeled. The goal is to predict the breed of each dog in these images.

- **sample_submission.csv**: A sample submission file that shows the correct format for submitting predictions. It contains placeholders for the image IDs and the corresponding predicted breed probabilities.

- **labels.csv**: This file contains the breed labels for each image in the training set. It maps each image ID to its corresponding dog breed, which will be used to train the model.

- **custom_dog_images/**: A folder where users can upload custom dog images for prediction. The uploaded images are processed and passed through the model to predict the breed.

- **logs/**: A folder that stores logs, including training logs and TensorBoard outputs, to help monitor and analyze the training process.

- **models/**: A folder containing saved models. This includes the final trained model (`.h5` file) that can be used to make predictions on new images.

- **app.py**: The main Python script that runs the Streamlit web application. This script allows users to upload dog images and get breed predictions.

- **dog-vision.ipynb**: A Jupyter notebook that contains the full code for model training, data preprocessing, and other relevant tasks. This notebook can be used to retrain the model or experiment with different approaches.

- **requirements.txt**: A text file listing the Python packages and dependencies required to run the project. This can be used to set up the environment and install the necessary libraries.


## Step-by-Step Installation Guide

1. **Install Python 3.8.20**:  
   Ensure that you have Python version 3.8.20 installed on your system. If you don’t have it, download and install it from the official [Python website](https://www.python.org/downloads/release/python-3820/).

2. **Clone the Repository**:  
   Start by cloning the repository from GitHub to your local machine. This will allow you to access the project files and begin working with them.

   ```bash
   git clone https://github.com/sugamchaudhary/dog-vision.git
   cd dog-vision

3. **Create a Virtual Environment**:  
   It is recommended to create a virtual environment to manage the dependencies for this project. You can use Conda to create a new environment specifically for this project. This will help to isolate your project’s dependencies from your global Python installation.

   ```bash
   Create a virtual environment: python -m venv venv
   Activate the virtual environment
   On Windows: venv\Scripts\activate
   On macOS/Linux: source venv/bin/activate

4. **Install Required Packages**:  
   Once the virtual environment is set up and activated, you can install all the necessary libraries for the project by using the `requirements.txt` file. This file contains a list of all the dependencies required to run the project. You can install them by running the appropriate command within your virtual environment.

   ```bash
   pip install -r requirements.txt

## Prepare the Dataset

Before running the app, you need to prepare the dataset. The dataset consists of training and test images, which are compressed in `.zip` format. You will need to extract these `.zip` files and place the images in their respective folders. The dataset should be structured correctly, with images in the `train` folder and test images in the `test` folder. The breed labels are stored in the `labels.csv` file, which should be kept in the project directory for easy access by the app.

Ensure that the dataset is properly organized and the paths to the images are correctly referenced in your code to enable smooth operation of the classifier.

## Run the Streamlit App

To run the Streamlit app, make sure that all dependencies are installed, and the dataset is prepared as described above. Once you have set up everything, you can launch the app by running the following command: 

```bash
streamlit run app.py
```

The app will open in your default web browser where you can interact with the dog breed classifier. It allows you to upload images of dogs and predict their breed using the trained model.

### Using the App

1. **Upload an Image**: You can upload an image of a dog by using the file uploader provided in the app. The app supports various image formats like JPG, JPEG, and PNG.
   
2. **Predict the Breed**: Once the image is uploaded, simply click on the "Predict" button, and the app will display the predicted breed of the dog based on the image.

3. **View Results**: The app will show you the predicted breed, which is derived from the model’s output after processing the image.

## Model Details

The model used in this project is a convolutional neural network (CNN) built using TensorFlow and TensorFlow Hub. It is based on the MobileNetV2 architecture, which is a lightweight model suitable for mobile and web applications. The model was pre-trained on a large image dataset and fine-tuned on the dog breed dataset to classify 120 different dog breeds.

### Modify the Model Architecture or Retrain the Model

If you want to modify the model architecture or retrain the model with different parameters, you can do so by editing the code in the `dog-vision.ipynb` file. The notebook allows you to adjust hyperparameters, change the model architecture, and retrain the model on your dataset.

You can also use different pre-trained models from TensorFlow Hub if you wish to experiment with other architectures. Make sure to modify the code accordingly to match the input size and requirements of the new model.
