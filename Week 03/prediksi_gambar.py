import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import MnistClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistClassifier(num_classes=10)
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model = model.to(device)
model.eval()

image_path = 'data/my_number.jpg'  # GANTI dengan path gambarmu
image = Image.open(image_path).convert('L')  # 'L' = grayscale

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

print(f'Predicted Class: {predicted.item()}')

plt.imshow(image.squeeze().cpu(), cmap='gray')
plt.title(f'Prediksi: {predicted.item()}')
plt.axis('off')
plt.show()
