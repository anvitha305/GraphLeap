import torch
import urllib.request
import os
from vig import vig_ti_224_gelu  # your modified vig.py module
from timm.data import resolve_data_config, create_transform
from torchvision.datasets import ImageFolder
from data.myloader import create_loader

def download_weights(url, dst_path='vig_ti_74.5.pth'):
    if not os.path.exists(dst_path):
        print(f'Downloading weights from {url}...')
        urllib.request.urlretrieve(url, dst_path)
        print('Download complete.')
    else:
        print('Weights file already exists.')

def load_model(weights_path, device):
    model = vig_ti_224_gelu(pretrained=False, num_classes=1000)
    model.to(device)
    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_url = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/vig/vig_ti_74.5.pth'
    weights_path = 'vig_ti_74.5.pth'

    # Step 1: Download weights if not exist locally
    download_weights(weights_url, weights_path)

    # Step 2: Load model and weights
    model = load_model(weights_path, device)
    print("Model loaded and set to evaluation mode.")

    # Step 3: Prepare validation dataloader (fix: explicitly create dataset + transform)
    val_data_path = 'val'  # or your real ImageNet val path
    data_config = resolve_data_config({}, model=model)
    data_transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config.get('crop_pct', 0.875),
        interpolation=data_config['interpolation']
    )
    dataset_eval = ImageFolder(val_data_path, transform=data_transform)
    val_loader = create_loader(
        dataset_eval,  # Pass the dataset object, NOT the path string!
        input_size=data_config['input_size'],
        batch_size=32,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=4,
        distributed=False,
        crop_pct=data_config.get('crop_pct', 0.875),
        pin_memory=True,
    )

    # Step 4: Run simple evaluation loop
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Loss: {loss_sum / total:.4f}, Accuracy: {100.0 * correct / total:.2f}%')

if __name__ == "__main__":
    main()

