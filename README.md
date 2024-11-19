
---

# **Meat Classification with Custom Machine Learning Model and Augmentation**

## **Project Overview**
This project implements a machine learning model to classify meat images into three categories: **pork**, **horse**, and **beef**. The approach utilizes a custom convolutional neural network with an innovative **Push-Pull Convolutional Unit**. It also evaluates the model on both original and augmented datasets to test its robustness under realistic conditions.

---

## **Features**
- **Custom Dataset Handling**: Dynamically processes and loads images into train, validation, and test datasets.
- **Push-Pull Convolution**: Integrates the "PushPullConv2DUnit" for effective feature extraction.
- **Performance Metrics**: Provides accuracy, precision, recall, and F1-score evaluations.
- **Augmentation Testing**: Tests the model on augmented datasets to simulate real-world conditions.
- **GPU Support**: Automatically utilizes CUDA if available.

---

## **Project Structure**
```
root/
├── dataset/               # Contains the original images
│   ├── pork/
│   ├── horse/
│   └── beef/
├── augmented_dataset/     # Contains augmented images for testing
│   ├── pork/
│   ├── horse/
│   └── beef/
├── pushpull-conv/         # PushPull convolution module
├── MeatClassifier.py      # Main model and training script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## **Setup Instructions**
### **1. Clone Repository**
```bash
git clone <https://github.com/Craingit/MeatTypeClassification>
```

### **2. Install Dependencies**
Ensure you have Python 3.8+ and pip installed. Then run:
```bash
pip install -r requirements.txt
```

### **3. Dataset Setup**
Organize your dataset as follows:
```
dataset/
├── pork/
├── horse/
└── beef/
```
Each subfolder should contain images of the respective meat type.

### **4. Augmented Dataset**
Use augmentation.py for your dataset to augmentate the images in your dataset.

### **5. Run Training**
To train the model, execute:
```bash
python modeltrain.py
```

---

## **Key Components**
### **Custom Dataset Class**
The `MeatDataset` class dynamically loads and preprocesses the dataset using:
- **Image Resizing**: Scales all images to 224×224 pixels.
- **Normalization**: Standardizes pixel values using ImageNet statistics.

### **Model Architecture**
The `MeatClassifier` is a custom neural network featuring:
- **PushPullConv2DUnit**: A specialized convolutional layer for enhanced feature learning.
- **Batch Normalization & ReLU Activation**: To stabilize and accelerate training.
- **Fully Connected Layers**: For classifying the images into three categories.

### **Training & Validation**
The model trains for a configurable number of epochs, minimizing **Cross-Entropy Loss** using the **Adam Optimizer**. During validation, metrics such as **accuracy, precision, recall,** and **F1-score** are computed.

### **Testing with Augmented Images**
To evaluate robustness, the model is tested on an augmented dataset containing corrupted or distorted images.

---

## **Performance Metrics**
The model outputs key metrics after training and testing:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Percentage of relevant predictions among all positive predictions.
- **Recall**: Ability to identify all relevant cases.
- **F1-Score**: Harmonic mean of precision and recall.

Example results:
```
Validation Metrics - Accuracy: 0.92, Precision: 0.91, Recall: 0.92, F1-Score: 0.91
Augmented Testing Metrics - Accuracy: 0.89, Precision: 0.88, Recall: 0.89, F1-Score: 0.88
```

---

## **Acknowledgements**
- **Push-Pull Convolution Unit**: Adapted from [PushPullConv GitHub Repository](https://github.com/bgswaroop/pushpull-conv).
- **Dataset**: Public meat dataset obtained from [Kaggle](https://www.kaggle.com/datasets/iqbalagistany/pork-meat-and-horse-meat-dataset).

---

## **Future Improvements**
- Implement advanced data augmentation techniques during training.
- Integrate additional meat categories for broader classification.
- Explore transfer learning models such as ResNet or MobileNet.

---
