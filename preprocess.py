import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os


class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.transform = transform
        self.images = []

        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (150, 150))  # Resize for consistency

        if self.transform:
            image = self.transform(image)
        return image, label


# Define transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = ChestXRayDataset('chest_xray/train', transform=train_transform)
test_dataset = ChestXRayDataset('chest_xray/test', transform=test_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)