import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import cv2

global IMAGE_SIZE
IMAGE_SIZE = 128

# Dataset object to organize image samples
class CustomImageDataset(Dataset):
    def __init__(self, labels, images, transform=None, target_transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, num_filters=16, kernel_size=3, padding=2):
        """
        Initializes the CNN.
        Parameters:
            num_classes (int): Number of output classes.
            num_filters (int): Number of filters in the first convolutional layer.
                The second layer will have 2x this number.
            kernel_size (int): Size of the convolving kernel.
            padding (int): Zero-padding added to both sides of the input.
        """
        super(SimpleCNN, self).__init__()
        self.c1 = nn.Conv2d(1, num_filters, tuple([kernel_size, kernel_size]), padding=padding)
        self.c2 = nn.Conv2d(num_filters, num_filters * 2, tuple([kernel_size, kernel_size]), padding=padding)
        self.pooling = nn.MaxPool2d(2, 2)
        dimensions = int((((IMAGE_SIZE + 2 * padding - kernel_size + 1) // 2) - kernel_size + 1 + 2 * padding) // 2)
        print(num_filters * 2 * dimensions * dimensions)
        self.f1 = nn.Linear(num_filters * 2 * dimensions * dimensions, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        Performs forward pass of the input.
        Parameters:
            x (Tensor): Input tensor.
        Returns:
            out (Tensor): Output tensor.
        """
        x = self.pooling(F.relu(self.c1(x)))
        x = self.pooling(F.relu(self.c2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x




# Display the image
#cv2.imshow('Image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# extract image data
print("Extracting Benign Data\n")
benignPath = pathlib.Path("./exploration/benign")
fileNames = list(benignPath.iterdir())
data = []
labels = []
for i in range(len(fileNames)):
    if i%2 == 1:
        img = np.float32(np.loadtxt(f"{fileNames[i]}"))
        res = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        data.append(res)
        labels.append(0)


print("Extracting Malware Data\n")
malwarePath = pathlib.Path("./exploration/malware")
fileNames = list(malwarePath.iterdir())
for i in range(len(fileNames)):
    if i%2 == 1:
        img = np.float32(np.loadtxt(f"{fileNames[i]}"))
        res = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        data.append(res)
        labels.append(1)


# create custom dataset
image_labels_df = pd.DataFrame({'Images': data, 'Labels': labels})
fullDataset = CustomImageDataset(image_labels_df['Labels'], image_labels_df['Images'], transform=transforms.ToTensor())

# split data
train_size = int(0.75 * len(data))
test_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(fullDataset, [train_size, test_size])

# create data loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# load device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def train(model, train_loader, lr=0.01, momentum=0.9):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validate(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print('Validation accuracy: {:.2f}%'.format((accuracy) * 100))
    return accuracy


def visulization(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.title('Validation accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def start():
    print('Starting Training -----------------------')
    params = [[32,3,2]]
    for num_filters, kernel_size, padding in params:
        model = SimpleCNN(num_filters=num_filters, kernel_size=kernel_size, padding=padding)
        model = model.to(device)
        accuracies = []
        for epoch in range(15):  # Loop over the dataset multiple times
            print('Epoch {}/{}'.format(epoch + 1, 15))
            train(model, train_loader)
            accuracy = validate(model, val_loader)
            accuracies.append(accuracy)
        visulization(accuracies)  # visualize trends



if __name__ == "__main__":
    start()


