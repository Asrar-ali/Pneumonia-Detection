import torch
from model import PneumoniaCNN
from preprocess import test_loader

model = PneumoniaCNN()
model.load_state_dict(torch.load('pneumonia_cnn.pth'))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")