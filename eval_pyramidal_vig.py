import torch
import os
import tarfile
from pyramid_vig import pvig_ti_224_gelu  # import the pyramidal ViG model
from timm.data import resolve_data_config, create_transform
from torchvision.datasets import ImageFolder
from data.myloader import create_loader

def load_model(weights_path, device):
    model = pvig_ti_224_gelu(pretrained=False, num_classes=1000)
    model.to(device)
    state_dict = torch.load(weights_path, map_location=device)
    # The state dict may be under 'model' or 'state_dict' key if tarball contains dict
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights_path = 'pvig_ti_78.5.pth'  # weight file in current directory

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file '{weights_path}' not found in the current directory. Please place it there before running.")

    # Load model
    model = load_model(weights_path, device)
    print("Pyramidal ViG model loaded and set to evaluation mode.")

    val_data_path = 'val'  # change to your ImageNet val path

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
        dataset_eval,
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

