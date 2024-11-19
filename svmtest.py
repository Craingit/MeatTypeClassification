import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from glob import glob

# Dataset class
class MeatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Paths
data_dir = r'C:\Users\Crain\Desktop\dataset'
augmented_dir = r'C:\Users\Crain\Desktop\augmented_dataset'

# Load dataset
classes = ['pork', 'horse', 'beef']
images, labels = [], []
for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_path in glob(os.path.join(class_dir, '*.jpg')):
        images.append(img_path)
        labels.append(label)

# Split dataset
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels
)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=42, stratify=test_labels
)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = models.resnet18(pretrained=True)
feature_extractor.fc = torch.nn.Identity()  # Remove the classification layer
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

def extract_features(loader, model):
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.extend(batch_labels)
    return np.vstack(features), np.array(labels)

# Datasets and DataLoaders
train_dataset = MeatDataset(train_images, train_labels, transform=transform)
val_dataset = MeatDataset(val_images, val_labels, transform=transform)
test_dataset = MeatDataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Extract features
train_features, train_labels = extract_features(train_loader, feature_extractor)
val_features, val_labels = extract_features(val_loader, feature_extractor)
test_features, test_labels = extract_features(test_loader, feature_extractor)

# Train SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(train_features, train_labels)

# Evaluate SVM
def evaluate_model(features, labels, model):
    preds = model.predict(features)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1

# Validation
val_metrics = evaluate_model(val_features, val_labels, svm)
print(f"Validation Metrics - Accuracy: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}, F1-Score: {val_metrics[3]:.4f}")

# Testing with augmented images
augmented_images, augmented_labels = [], []
for label, class_name in enumerate(classes):
    class_dir = os.path.join(augmented_dir, class_name)
    for img_path in glob(os.path.join(class_dir, '*.jpg')):
        augmented_images.append(img_path)
        augmented_labels.append(label)

augmented_dataset = MeatDataset(augmented_images, augmented_labels, transform=transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=16, shuffle=False)
augmented_features, augmented_labels = extract_features(augmented_loader, feature_extractor)

# Evaluate on augmented images
aug_metrics = evaluate_model(augmented_features, augmented_labels, svm)
print(f"Augmented Testing Metrics - Accuracy: {aug_metrics[0]:.4f}, Precision: {aug_metrics[1]:.4f}, Recall: {aug_metrics[2]:.4f}, F1-Score: {aug_metrics[3]:.4f}")
