import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_dataset
from huggingface_hub import login

from graphleap_wrapper import (
    vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu,
    pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu
)
# ==========================================
# 1. Environment & Auth
# ==========================================
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

# ==========================================
# 2. Model Mapping (Direct FP32 Builders)
# ==========================================
MODEL_MAP = {
    'vig_ti_74.5.pth':   lambda: vig_ti_224_gelu(num_classes=1000),
    'vig_s_80.6.pth':    lambda: vig_s_224_gelu(num_classes=1000),
    'vig_b_82.6.pth':    lambda: vig_b_224_gelu(num_classes=1000),
    'pvig_ti_78.5.pth':  lambda: pvig_ti_224_gelu(num_classes=1000),
    'pvig_s_82.1.pth':   lambda: pvig_s_224_gelu(num_classes=1000),
    'pvig_m_83.1.pth':   lambda: pvig_m_224_gelu(num_classes=1000),
    'pvig_b_83.66.pth':  lambda: pvig_b_224_gelu(num_classes=1000),
}

# ==========================================
# 3. Data & Accuracy Handling
# ==========================================
class ImageNetSubset(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.dataset = hf_ds
        self.transform = transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'].convert('RGB')
        if self.transform: img = self.transform(img)
        return img, item['label']

def get_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total

# ==========================================
# 4. Main Validation Loop
# ==========================================
def main():
    # Use CUDA if available for standard FP32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_dir = 'pretrained_weights'
    
    print(f"Using device: {device}")
    print("Fetching 5,000-image validation subset...")
    
    ds = load_dataset("imagenet-1k", split='validation', trust_remote_code=True, token=token)
    val_ds = ds.shuffle(seed=42).select(range(5000))

    transform = Compose([
        Resize(256), CenterCrop(224), ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader = DataLoader(ImageNetSubset(val_ds, transform), batch_size=64, shuffle=False, num_workers=4)

    summary = []

    for fname, builder in MODEL_MAP.items():
        w_path = os.path.join(weights_dir, fname)
        if not os.path.exists(w_path):
            continue

        print(f"\n>>> Validating {fname}...")
        model = builder().to(device)

        # Load weights
        ckpt = torch.load(w_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

        # Calculate Accuracy
        acc = get_acc(model, loader, device)
        summary.append(f"{fname:<20} | FP32 Accuracy: {acc:>5.2f}%")
        print(summary[-1])

    print("\n--- Final Results (FP32) ---\n" + "\n".join(summary))

if __name__ == "__main__":
    main()

