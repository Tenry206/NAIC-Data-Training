import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

FOLDER_PATH = "/home/littlecrabby/NAIC-Data-Training/dataset/"
epoch_count = 15
pretrained_weights = True
batch_size = 32
learning_rate = 0.0005
dimension = 224
filename = ""    

# Data transformation and loading
transform = transforms.Compose([
    transforms.Resize((dimension, dimension)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])

train_dataset = datasets.ImageFolder(FOLDER_PATH + "train/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_type = input("resnet18[1] mobilenet_v2[2] squeezenet1_0[3]: ")

if model_type == "1":
    model = models.resnet18(weights=pretrained_weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    filename = "resnet18"
elif model_type == "2":
    model = models.mobilenet_v2(weights=pretrained_weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
    filename = "mobilenet_v2"
else:
    print("Defaulting to squeezenet1_0")
    model = models.squeezenet1_0(weights=pretrained_weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
    filename = "squeezenet1_0"

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda") #if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epoch_count):
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

    print(f"Epoch [{epoch+1}/{epoch_count}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), f'./models/{filename}.pth')
