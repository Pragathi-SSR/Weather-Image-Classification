# Weather-Image-Classification
This project employs deep learning techniques and transfer learning to classify weather images into categories such as cloudy, rainy, sunny, and sunrise. It aims to enhance accuracy in weather condition detection for practical applications in meteorology and related fields.

**Project Description :** This project involves classifying weather images using deep learning techniques, CNN, Transfer Learning. Images were resized to 224x224 pixels, converted to numerical arrays, and processed with CNNs (VGG, ResNet, MobileNet).Evaluation metrics include accuracy, precision, recall, and F1-score, highlighting performance and insights gained from classification efforts.

![image](https://github.com/user-attachments/assets/3c5d84bc-c2f8-4f65-8a58-400a9088c0ac)


**Project Objective :** To develop an accurate and efficient image classification model that can automatically identify and categorize different weather conditions (cloudy, rainy, sunny, and sunrise) using Convolutional Neural Networks (CNNs) and transfer learning techniques. This project aims to leverage advanced deep learning models to enhance the accuracy of weather condition detection, thereby providing a robust tool for applications in meteorology and related fields.

**DOMAIN :** Machine Learning and Deep Learning.


### Project Features
**Image Resizing:** All images resized to 224x224 pixels for uniformity and compatibility with CNN models.

**Normalization:** Pixel values normalized to a range of 0 to 1 for faster convergence during training.

**Label Encoding:** Weather conditions encoded into numerical labels to facilitate model training.**{'Cloudy': 0, 'Rainy': 1, 'Shine': 2, 'Sunrise': 3}**

**Dataset Splitting :** Dataset divided into training, validation, and test sets to train, tune, and evaluate the model.

**Transfer Learning:** Utilized pre-trained models like VGG16 for feature extraction and fine-tuning on the weather dataset.

**Convolutional Neural Network (CNN):** Implemented a CNN architecture with multiple convolutional and pooling layers to extract hierarchical features from images.

**Performance Metrics:** Evaluated model performance using accuracy, precision, recall, and F1-score to ensure robust classification.

## Dataset Overview

The dataset consists of images categorized into different weather conditions, including cloudy, rainy, sunny, and sunrise. Each category contains a variety of images representing the specific weather condition, contributing to a diverse and comprehensive dataset.The images were collected from https://data.mendeley.com/datasets/4drtyfjtfy/1.
 - **Cloudy Images :**
   ![image](https://github.com/user-attachments/assets/f2744f25-e65a-4d53-a825-9280954a727d)
 - **Rainy Images :**
  ![image](https://github.com/user-attachments/assets/004db807-8dae-4ed5-9d2c-5a9d97460ce4)

 - **Shine Images :**
  ![image](https://github.com/user-attachments/assets/45c7466d-d32d-4632-8a81-66425be8a30d)

 - **Sunrise Images :**
  ![image](https://github.com/user-attachments/assets/1cd84450-caa9-4460-9151-439dd3030959)

## Tools and Techniques

- Python
- Numpy
- Pandas
- Matplotlib
- CNN
- VGG, DenseNet, ResNet, MobileNet, InceptionV3
- Transfer Learning.
- PIL
- Tensorflow, Keras

### Analysis

- There are four types of claases in the dataset (Cloudy, Rainy, Shine, Sunrise).
- Shape of image arrays
    - Cloudy : (298, 224, 224, 3)
    - Rainy : (214, 224, 224, 3)
    - Shine : (251, 224, 224, 3)
    - Sunrise : (356, 224, 224, 3)

- Shape of the combined data : (1119, 224, 224, 3)

### Model Performence
The VGG16 model with a few additional layers achieves 95% accuracy in both testing and validation sets.

### Model Scores on Testing data.

- **Accuracy:** 95%	

- **Precision:** 94.5%

- **Recall:** 95% 
 
- **F1-Score:** 95.4%


**Outcome :** The model implementation assists in accurately classifying weather conditions from images, demonstrating the utility of CNNs and transfer learning in image classification tasks.

### Final model predictions and graph.

![image](https://github.com/user-attachments/assets/6372c785-7091-447c-96a9-639b117e5edd)

![image](https://github.com/user-attachments/assets/743f7241-c7ed-485d-9961-74c4c37deb2f)
