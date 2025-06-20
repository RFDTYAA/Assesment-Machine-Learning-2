import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MNISTDataset
from models import MnistClassifier
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# 👉 Definisikan argumen dulu sebelum dipakai
class Args:
    epochs = 10
    batch_size = 64
    lr = 0.001
    num_classes = 10
    checkpoint = 'mnist_model.pth'

args = Args()

# 👉 Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
train_dataset = MNISTDataset(
    images_path='data/mnist/train-images.idx3-ubyte',
    labels_path='data/mnist/train-labels.idx1-ubyte'
)

test_dataset = MNISTDataset(
    images_path='data/mnist/t10k-images.idx3-ubyte',
    labels_path='data/mnist/t10k-labels.idx1-ubyte'
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load Model
model = MnistClassifier(num_classes=args.num_classes)
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

train_losses = []
test_losses = []

# Training loop
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for data, labels in tqdm(train_loader):
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': args.epochs,
    'optimizer_state_dict': optimizer.state_dict()
}, args.checkpoint)

# Plot Loss
plt.plot(train_losses, label='Train Loss')
plt.title('Train Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
