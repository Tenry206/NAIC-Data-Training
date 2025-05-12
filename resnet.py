import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

FOLDER_PATH = "/home/littlecrabby/NAIC-Data-Training/dataset/"
epoch_count = 10
pretrained_weights = True
batch_size = 32
learning_rate = 0.001

# Data transformation and loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])

train_dataset = datasets.ImageFolder(FOLDER_PATH + "train/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load ResNet model without pretrained weights
model = models.resnet18(weights=pretrained_weights)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjusting output layer for your number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda") #if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epoch_count):  # Number of epochs
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'resnet18_trained.pth')
