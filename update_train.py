import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from PIL import Image, ExifTags
from datasets import load_dataset
from vig import vig_ti_224_gelu  # your ViG/DeepGCN model

class HuggingFaceImageNet100(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        label = item['label']
        # Convert all images to RGB to ensure 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Optional: handle EXIF orientation safely (can be extended if needed)
        orientation_key = None
        for key, value in ExifTags.TAGS.items():
            if value == 'Orientation':
                orientation_key = key
                break
        if hasattr(img, '_getexif') and img._getexif() is not None and orientation_key is not None:
            exif = img._getexif()
            orientation = exif.get(orientation_key, None)
            # Could add rotation correction here if desired
        if self.transform:
            img = self.transform(img)
        return img, label

def load_pretrained_model(weights_path, device):
    model = vig_ti_224_gelu(pretrained=False, num_classes=100)
    state_dict = torch.load(weights_path, map_location=device)
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    missing = set(model_dict.keys()) - set(filtered_state_dict.keys())
    unexpected = set(state_dict.keys()) - set(filtered_state_dict.keys())
    print("Missing keys (not loaded):", missing)
    print("Unexpected keys (skipped):", unexpected)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    return model

def get_train_transforms():
    return Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += images.size(0)
    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = 'vig_ti_74.5.pth'
    dataset = load_dataset("clane9/imagenet-100")
    train_dataset = HuggingFaceImageNet100(dataset['train'], transform=get_train_transforms())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    model = load_pretrained_model(weights_path, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    num_epochs = 5
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Accuracy: {acc:.2f}%")
        torch.save(model.state_dict(), f"vig_ti_finetuned_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()

