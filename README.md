# multi_class_image_classification


# Introduction

This documentation provides an overview of the multiclass image classification project in my portfolio. The project aims to classify images into multiple classes using a deep learning model. The implementation is based on the PyTorch framework and utilizes the ResNet-50 architecture pretrained on ImageNet.

## Project Structure

The project consists of the following key components:

PotholeClassifier Class: The PotholeClassifier class serves as the main component for training and testing the multiclass image classification model. It handles data loading, model initialization, training loop, and evaluation.

Data Preparation: The project assumes that the dataset is organized in a specific directory structure. The directories for each class are specified using class_dirs. Within each class directory, separate directories for train, validation, and test datasets are expected.

Model Architecture: The model architecture used for the classification task is ResNet-50, a popular deep convolutional neural network. The architecture is initialized in the PotholeClassifier class, with the last fully connected layer replaced to match the number of output classes.

Training: The training process involves iterating over the training dataset for a specified number of epochs. During each epoch, the model is trained using batches of images and labels. The loss is calculated using the BCEWithLogitsLoss criterion, and the Adam optimizer is used for updating the model parameters.

Validation: After each epoch, the model's performance is evaluated on the validation dataset. The validation loss, accuracy per class, and confusion matrix are calculated and recorded. The best model based on the validation loss is saved.

Testing: Once training is complete, the saved best model is loaded, and the final evaluation is performed on the test dataset. The accuracy and confusion matrix are computed.

Visualization: The project includes visualizations to aid in understanding the training progress. The validation loss over epochs is plotted using Matplotlib.

