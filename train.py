import torch
import torch.nn as nn
import torch.optim as optim
from model import PneumoniaCNN
from preprocess import train_loader, test_loader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
train_losses, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'pneumonia_cnn.pth')