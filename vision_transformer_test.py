import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os

# ----------------------- Patch Embedding -----------------------
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.linear_project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x)         # (B, d_model, H/P, W/P)
        x = x.flatten(2)                   # (B, d_model, P)
        x = x.transpose(1, 2)              # (B, P, d_model)
        return x

# ---------------------- Positional Encoding ---------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2*i)/d_model)))
                if i + 1 < d_model:
                    pe[pos, i+1] = np.cos(pos / (10000 ** ((2*i)/d_model)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pe[:, :x.size(1), :]
        return x

# ----------------------- Attention Modules -----------------------
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_model // n_heads) for _ in range(n_heads)])
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.projection(x)

# ----------------------- Transformer Block -----------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ------------------------ Vision Transformer ------------------------
class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
        self.max_seq_len = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, self.max_seq_len)
        self.encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0])

# ------------------------ Training Setup ------------------------
# Paths
dataset_path = r"C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\DataSet"
train_path = os.path.join(dataset_path, "Train")
test_path = os.path.join(dataset_path, "Test")

# Parameters
img_size = (64, 64)
patch_size = (16, 16)
n_channels = 3
d_model = 128
n_heads = 8
n_layers = 6
n_classes = 2
batch_size = 16
epochs = 10
lr = 0.001

# Transform with augmentation
transform = T.Compose([
    T.Resize(img_size),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor()
])

# Load dataset
train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=T.Compose([
    T.Resize(img_size),
    T.ToTensor()
]))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model, loss, optimizer
model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ------------------------ Training Loop ------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.3f}")

# ------------------------ Evaluation ------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}%")
