# Gesture Prediction Model

This repository contains a gesture prediction model implemented using TensorFlow and TensorFlow Lite (TFLite). The model is designed to recognize three types of gestures: circle, cross, and pad, based on accelerometer data collected at 50Hz over 2.5-second windows.

## Features
- Loads gesture data from CSV files.
- Preprocesses data using sliding window segmentation.
- Trains a fully connected neural network with ReLU activations and dropout.
- Converts the trained model to TFLite format for deployment on embedded systems.
- Provides inference using the converted TFLite model.

## Project Structure
```
 gesture_model.tflite      # Converted TFLite model
 model_keras.h5            # Trained Keras model
 circle.csv, cross.csv, pad.csv  # Raw gesture data
 circle-test.csv           # Sample test data
 train_model.py            # Script for training and converting the model
 infer_tflite.py           # Script for testing inference with TFLite
 GestureModel.cpp          # Testing TFLite model on Raspberry Pi
 README.md                 # Project documentation
```

## Installation
Ensure you have the required dependencies installed:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Training the Model
Run the following script to train the model and convert it to TFLite format:
```bash
python train_model.py
```

## Running Inference with TFLite
To test gesture recognition on new data using the TFLite model, run:
```bash
python infer_tflite.py
```

## Data Processing
- The dataset is loaded from CSV files.
- Each sample is segmented into 125-frame windows (2.5s at 50Hz).
- The flattened window is used as input to the neural network.

## Model Architecture
The model consists of:
- Dense layers with ReLU activation
- Dropout layers to prevent overfitting
- Softmax output layer for classification

## License
This project is open-source and can be used freely for research and development.


