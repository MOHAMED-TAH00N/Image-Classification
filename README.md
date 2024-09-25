# Intel Image Classification using Convolutional Neural Networks

## Introduction
This project focuses on classifying images into six categories: buildings, forest, glacier, mountain, sea, and street. The goal is to develop a Convolutional Neural Network (CNN) model that accurately classifies images based on these predefined classes. This task is a common example of supervised machine learning in computer vision.

## Dataset Overview
The dataset is structured into training and testing sets:
- **Training Set**: Located in `seg_train`, containing subfolders for each class.
- **Testing Set**: Located in `seg_test`, structured similarly to the training set.
- **Prediction Set**: Contains images to be classified without labels.

## Data Exploration
The code includes various steps to explore and visualize the dataset:
- **Image Count**: The number of images in each folder for both training and testing datasets is printed.
- **Image Size Analysis**: The dimensions of images in the training, testing, and prediction sets are examined, helping to identify discrepancies in image sizes.

## Data Preparation
Images are resized to a standard size of 100x100 pixels to ensure uniformity. The labels for training and testing datasets are converted to categorical format.

## Model Architecture
The CNN model consists of several layers:
- **Convolutional Layers**: Multiple layers of convolution with Leaky ReLU activations.
- **Batch Normalization**: Applied after convolutional layers to stabilize learning.
- **Max Pooling**: Reduces dimensionality, helping to extract dominant features.
- **Dropout Layers**: Prevents overfitting by randomly setting a fraction of input units to 0.
- **Global Average Pooling**: Reduces the spatial dimensions, providing a compact representation.
- **Dense Layers**: Fully connected layers leading to the output layer, which uses the softmax activation function for multi-class classification.

## Model Compilation
The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function, with accuracy as the evaluation metric.

## Data Augmentation
To enhance model robustness, data augmentation techniques are applied, including:
- Rotation
- Zoom
- Shifts in both dimensions

## Training
The model is trained using the augmented data for 2 epochs with a batch size of 256. Early stopping and learning rate reduction callbacks are employed to optimize training.

## Model Evaluation
The model is evaluated on the test dataset, providing the test loss and accuracy metrics.

## Visualization of Results
Training history is visualized with plots showing accuracy and loss over epochs. This helps in understanding the model's performance and identifying potential overfitting.

## Predictions
The model is used to predict the classes of new images from the prediction set. A function displays images alongside their predicted classes.

### Example Predictions
Several images from the prediction set are shown with their predicted classes, demonstrating the model's ability to classify unseen data effectively.

## Conclusion
The CNN model demonstrates reasonable performance on the Intel Image Classification dataset. Future improvements could include:
- Increasing the number of epochs for training.
- Utilizing more advanced data augmentation techniques.
- Experimenting with different architectures or transfer learning from pre-trained models.

## Getting Started
To run this project, make sure to install the required libraries and follow the setup instructions in the code files.

