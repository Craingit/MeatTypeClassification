import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


data_dir = r'C:\Users\Crain\Desktop\dataset'
classes = ['pork', 'horse', 'beef'] 


images = []
labels = []
for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_path in glob(os.path.join(class_dir, '*.jpg')):  
        images.append(img_path)
        labels.append(label)

print(f"Loaded {len(images)} images.")


plt.figure(figsize=(10, 5))
for i, img_path in enumerate(images[:5]):
    img = Image.open(img_path)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
plt.title(classes[labels[i]])
plt.show()


df = pd.DataFrame({'Image_Path': images, 'Label': labels})
sns.countplot(data=df, x='Label')
plt.title("Class Distribution")
plt.show()


sizes = [Image.open(img).size for img in images[:10]]
print("Sample Image Sizes:", sizes)


target_size = (224, 224)


train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=42, stratify=test_labels)

print(f"Training Set: {len(train_images)} images")
print(f"Validation Set: {len(val_images)} images")
print(f"Testing Set: {len(test_images)} images")


class MeatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = MeatDataset(train_images, train_labels, transform=transform)
val_dataset = MeatDataset(val_images, val_labels, transform=transform)
test_dataset = MeatDataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


import sys
sys.path.append(r"C:\Users\Crain\Desktop\pushpull-conv\project\models\utils")
from push_pull_unit import PushPullConv2DUnit

class MeatClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(MeatClassifier, self).__init__()
        self.conv1 = PushPullConv2DUnit(
            in_channels=3, out_channels=64, kernel_size=(3, 3), 
            avg_kernel_size=(3, 3), pull_inhibition_strength=1, 
            trainable_pull_inhibition=False, stride=(1, 1), padding='same'
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 112 * 112, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MeatClassifier(num_classes=3)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss/len(train_loader)}")


def evaluate_model(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), torch.tensor(labels).to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, precision, recall, f1

val_metrics = evaluate_model(model, val_loader)
print(f"Validation Metrics - Accuracy: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}, F1-Score: {val_metrics[3]:.4f}")

torch.save(model.state_dict(), "trained_meat_classifier.pth")


augmented_dir = r'C:\Users\Crain\Desktop\augmented_dataset'
augmented_images, augmented_labels = [], []
for label, class_name in enumerate(classes):
    class_dir = os.path.join(augmented_dir, class_name)
    for img_path in glob(os.path.join(class_dir, '*.jpg')):
        augmented_images.append(img_path)
        augmented_labels.append(label)

augmented_dataset = MeatDataset(augmented_images, augmented_labels, transform=transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=16, shuffle=False)


all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in augmented_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Augmented Testing Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
