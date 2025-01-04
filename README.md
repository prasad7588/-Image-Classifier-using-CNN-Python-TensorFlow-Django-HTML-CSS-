# CIFAR-10 Image Classifier Using CNN

This project implements an image classifier using a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10** dataset into 10 categories (e.g., airplanes, cars, cats, dogs). The model is trained using TensorFlow and Keras and includes techniques like data augmentation, batch normalization, and learning rate scheduling for improved performance.  

---

## Features

- **Dataset**: CIFAR-10, a popular dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.  
- **Deep Learning Techniques**:  
  - Convolutional Neural Network (CNN) for feature extraction.  
  - Batch Normalization and Dropout for improved generalization.  
  - Data Augmentation to expand the dataset diversity.  
- **Optimization**:  
  - Learning rate scheduling with `ReduceLROnPlateau`.  
  - Early stopping to prevent overfitting and improve training efficiency.  
- **Interactive Web Integration**: Model saved as `model.h5` for easy integration with a Django-based web interface.

---

## Tech Stack

- **Programming Language**: Python  
- **Libraries**:  
  - TensorFlow and Keras for building and training the CNN.  
  - NumPy for numerical operations.  
  - ImageDataGenerator for data augmentation.  
  - SGD optimizer with momentum for efficient training.  

---

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/cifar10-image-classifier.git
   cd cifar10-image-classifier
