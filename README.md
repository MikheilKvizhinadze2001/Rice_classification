# Rice Image Classification using Convolutional Neural Networks

This repository contains the code for a machine learning project that classifies images of rice into five distinct classes. Each class represents a different type of rice. The dataset consists of 75,000 images, with 15,000 images for each rice type.

## Project Overview

The goal of this project is to build a Convolutional Neural Network (CNN) that can accurately classify images of rice into their respective types. The model is trained on a dataset of 60,000 images and validated on a separate set of 7,500 images. The model’s performance is then evaluated on a test set of 7,500 unseen images.

## Model Architecture

The model architecture consists of three convolutional layers, each followed by a max-pooling layer to reduce dimensionality. After flattening the output of the convolutional layers, we use a fully connected layer, apply dropout for regularization, and finally use a softmax activation function in the output layer to obtain class probabilities.

## Training

The model is trained using the Adam optimizer and the sparse categorical cross-entropy loss function, suitable for multi-class classification problems. We also employ data augmentation techniques such as shearing, zooming, and horizontal flipping on the training set to increase the diversity of our training data and improve the model’s ability to generalize.

## Evaluation

The model’s performance is evaluated based on its accuracy on the test set. Additionally, we provide visualizations of the model’s training and validation loss and accuracy to gain insights into the model’s performance.

Link to the dataset: 
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
