import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from torch.utils.data import DataLoader

file_info = tuple(os.walk("./models/"))
batch_size = 32
dimension = 224
FOLDER_PATH = "/home/littlecrabby/NAIC-Data-Training/dataset/"
transform = transforms.Compose([
    transforms.Resize((dimension, dimension)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])


def main():
    selection()
    test()

def selection():
    global filename
    for index, filename in enumerate(file_info[0][2]):
        print(f"{index} - {filename}")
    filename = file_info[0][2][int(input("Select file: "))]

def test():
    if "resnet" in filename:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 8)
    elif "mobilenet" in filename:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 8)
    elif "squeezenet" in filename:
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 8)
    else:
        raise Exception
    model.load_state_dict(torch.load(f"./models/{filename}", weights_only=True, map_location='cpu'))
    model.eval()
    device = torch.device("cpu")
    test_dataset = datasets.ImageFolder(FOLDER_PATH + "test/", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = datasets.ImageFolder(FOLDER_PATH + "valid/", transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    correct = 0
    total = 0
    loss_function = nn.CrossEntropyLoss()  # Adjust this based on your task (e.g., for classification)
    total_loss = 0

    # Evaluate on the test set
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print accuracy and average loss
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
main()