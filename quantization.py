import torch
import torch.nn as nn
from torchvision import models
import os

file_info = tuple(os.walk("./models/"))
filename = ""

def main():
    selection()
    quantize()

def selection():
    global filename
    for index, filename in enumerate(file_info[0][2]):
        print(f"{index} - {filename}")
    filename = file_info[0][2][int(input("Select file: "))]

def quantize():
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
    #print(model)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    #print(quantized_model)
    new_filename = f"{filename.split('.')[0]}_quantized.pth"
    torch.save(quantized_model, f"./quantized_models/{new_filename}")
    print(f"Successfully quantized {new_filename}")

main()