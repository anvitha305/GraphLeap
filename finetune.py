import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, CenterCrop, ToTensor, Normalize
from datasets import load_dataset
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np
import json
from graphleap_wrapper import (
    vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu,
    pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu
)

# ==========================================
# Memory Configuration (ADD AT START)
# ==========================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# ==========================================
# Configuration
# ==========================================
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

BASELINE_ACCURACIES = {
    'vig_ti_74.5.pth': 74.5,
    'vig_s_80.6.pth': 80.6,
    'vig_b_82.6.pth': 82.6,
    'pvig_ti_78.5.pth': 78.5,
    'pvig_s_82.1.pth': 82.1,
    'pvig_m_83.1.pth': 83.1,
    'pvig_b_83.66.pth': 83.66,
}

ISOTROPIC_MODELS = {
    'vig_ti_74.5.pth': lambda: vig_ti_224_gelu(num_classes=1000),
    'vig_s_80.6.pth': lambda: vig_s_224_gelu(num_classes=1000),
    'vig_b_82.6.pth': lambda: vig_b_224_gelu(num_classes=1000),
}

PYRAMIDAL_MODELS = {
    'pvig_ti_78.5.pth': lambda: pvig_ti_224_gelu(num_classes=1000),
    'pvig_s_82.1.pth': lambda: pvig_s_224_gelu(num_classes=1000),
    'pvig_m_83.1.pth': lambda: pvig_m_224_gelu(num_classes=1000),
    'pvig_b_83.66.pth': lambda: pvig_b_224_gelu(num_classes=1000),
}

# ==========================================
# Data Loading
# ==========================================
class ImageNetSubset(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.dataset = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'].convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['label']

def get_loaders(batch_size=32, num_workers=2):
    """Load training and validation datasets with reduced defaults"""
    print("Loading datasets...")

    # Training data
    train_ds = load_dataset("imagenet-1k", split='train', trust_remote_code=True, token=token)
    train_ds = train_ds.select(range(50000))

    train_transform = Compose([
        RandomResizedCrop(224, scale=(0.08, 1.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation data
    val_ds = load_dataset("imagenet-1k", split='validation', trust_remote_code=True, token=token)
    val_ds = val_ds.shuffle(seed=42).select(range(5000))

    val_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        ImageNetSubset(train_ds, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        ImageNetSubset(val_ds, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory=True, drop_last=False
    )

    return train_loader, val_loader

# ==========================================
# Training & Validation
# ==========================================
def validate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Clear intermediate results
            del imgs, labels, outputs

    return 100 * correct / total

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with memory optimization"""
    model.train()
    total_loss = 0.0
    batch_count = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Forward pass with memory efficiency
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        batch_count += 1
        
        # Clear intermediate results
        del imgs, labels, outputs, loss
        
        # Periodically clear cache
        if batch_count % 10 == 0:
            torch.cuda.empty_cache()

    return total_loss / batch_count

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, model_name, metrics):
    """Save training checkpoint for resuming"""
    os.makedirs('checkpoints', exist_ok=True)

    checkpoint_path = f'checkpoints/{model_name}_checkpoint.pt'
    metrics_path = f'checkpoints/{model_name}_metrics.json'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
    }, checkpoint_path)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    print(f"  ✓ Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, model_name, device):
    """Load training checkpoint to resume"""
    checkpoint_path = f'checkpoints/{model_name}_checkpoint.pt'
    metrics_path = f'checkpoints/{model_name}_metrics.json'

    if not os.path.exists(checkpoint_path):
        print(f"  No checkpoint found for {model_name}, starting from scratch")
        return 0, {}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    start_epoch = checkpoint['epoch'] + 1
    print(f"  ✓ Resumed from epoch {start_epoch} (best val acc: {checkpoint['val_acc']:.2f}%)")

    return start_epoch, metrics

def finetune_model(model_name, model_builder, train_loader, val_loader, device,
                   num_epochs=30, lr=1e-5, freeze_ratio=0.5):
    """Fine-tune a model with checkpoint support and memory optimization"""

    print(f"\nFine-tuning {model_name} for {num_epochs} epochs...")

    # Load model with memory cleanup
    torch.cuda.empty_cache()
    model = model_builder().to(device)
    
    # Load pretrained weights
    weights_path = os.path.join('pretrained_weights', model_name)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    del checkpoint, state_dict
    torch.cuda.empty_cache()

    # Freeze early layers
    if freeze_ratio > 0 and hasattr(model.model, 'backbone'):
        num_layers = len(model.model.backbone)
        freeze_until = int(num_layers * freeze_ratio)

        for i, layer in enumerate(model.model.backbone):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        print(f"  Froze first {freeze_until}/{num_layers} layers")

    # Setup optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.05
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    # Try to load checkpoint
    start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, model_name, device)

    train_losses = metrics.get('train_losses', [])
    val_accuracies = metrics.get('val_accuracies', [])
    epochs_list = metrics.get('epochs_list', [])
    best_acc = metrics.get('best_acc', 0.0)

    # Training loop - NO EARLY STOPPING
    for epoch in range(start_epoch, num_epochs):
        try:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            torch.cuda.empty_cache()
            
            val_acc = validate(model, val_loader, device)
            torch.cuda.empty_cache()

            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            epochs_list.append(epoch + 1)

            scheduler.step()

            print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss {train_loss:.4f}, Val Acc {val_acc:.2f}%")

            # Update best accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs('finetuned_models', exist_ok=True)
                torch.save(model.state_dict(), f'finetuned_models/{model_name}_best.pth')

            # Save checkpoint every epoch
            checkpoint_metrics = {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'epochs_list': epochs_list,
                'best_acc': best_acc
            }
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, model_name, checkpoint_metrics)
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  ⚠ OOM at epoch {epoch+1}. Try reducing batch_size further.")
                torch.cuda.empty_cache()
                raise
            else:
                raise

    print(f"  ✓ Training complete! Best Val Acc: {best_acc:.2f}%")

    return {
        'model_name': model_name,
        'epochs': epochs_list,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_acc': best_acc,
        'baseline_acc': BASELINE_ACCURACIES[model_name]
    }

# ==========================================
# Plotting
# ==========================================
def plot_results(iso_results, pyr_results):
    """Create comparison charts for isotropic and pyramidal models"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vision GNN Fine-tuning Results (GraphLeap Accuracy Recovery)', fontsize=16, fontweight='bold')

    # ========== Isotropic Models ==========
    ax1, ax2 = axes[0]

    for result in iso_results:
        model_short = result['model_name'].split('_')[1].upper()
        ax1.plot(result['epochs'], result['val_accuracies'],
                marker='o', label=model_short, linewidth=2)
        ax1.axhline(y=result['baseline_acc'], linestyle='--', alpha=0.5)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Isotropic ViG Models - Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([60, 85])

    models_iso = [r['model_name'].split('_')[1].upper() for r in iso_results]
    baseline_accs = [r['baseline_acc'] for r in iso_results]
    best_accs = [r['best_acc'] for r in iso_results]
    recovery = [best - baseline for best, baseline in zip(best_accs, baseline_accs)]

    x_pos = np.arange(len(models_iso))
    ax2.bar(x_pos - 0.2, baseline_accs, 0.4, label='Baseline', alpha=0.8)
    ax2.bar(x_pos + 0.2, best_accs, 0.4, label='After Fine-tuning', alpha=0.8)

    for i, (model, rec) in enumerate(zip(models_iso, recovery)):
        ax2.text(i + 0.2, best_accs[i] + 0.5, f'+{rec:.2f}%', ha='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Isotropic ViG - Accuracy Recovery', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_iso)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([60, 85])

    # ========== Pyramidal Models ==========
    ax3, ax4 = axes[1]

    for result in pyr_results:
        model_short = result['model_name'].split('_')[1].upper()
        ax3.plot(result['epochs'], result['val_accuracies'],
                marker='s', label=model_short, linewidth=2)
        ax3.axhline(y=result['baseline_acc'], linestyle='--', alpha=0.5)

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax3.set_title('Pyramidal ViG Models - Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([75, 85])

    models_pyr = [r['model_name'].split('_')[1].upper() for r in pyr_results]
    baseline_accs_pyr = [r['baseline_acc'] for r in pyr_results]
    best_accs_pyr = [r['best_acc'] for r in pyr_results]
    recovery_pyr = [best - baseline for best, baseline in zip(best_accs_pyr, baseline_accs_pyr)]

    x_pos_pyr = np.arange(len(models_pyr))
    ax4.bar(x_pos_pyr - 0.2, baseline_accs_pyr, 0.4, label='Baseline', alpha=0.8)
    ax4.bar(x_pos_pyr + 0.2, best_accs_pyr, 0.4, label='After Fine-tuning', alpha=0.8)

    for i, (model, rec) in enumerate(zip(models_pyr, recovery_pyr)):
        ax4.text(i + 0.2, best_accs_pyr[i] + 0.2, f'+{rec:.2f}%', ha='center', fontsize=10, fontweight='bold')

    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Pyramidal ViG - Accuracy Recovery', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos_pyr)
    ax4.set_xticklabels(models_pyr)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([75, 85])

    plt.tight_layout()
    plt.savefig('fine_tuning_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved chart to 'fine_tuning_results.png'")
    plt.show()

# ==========================================
# Main
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Reduced batch size to 32 (was 64)
    # Adjust down further if still OOM
    train_loader, val_loader = get_loaders(batch_size=32, num_workers=2)

    # Fine-tune isotropic models
    print("\n" + "="*60)
    print("FINE-TUNING ISOTROPIC ViG MODELS")
    print("="*60)
    iso_results = []
    for model_name, builder in ISOTROPIC_MODELS.items():
        result = finetune_model(
            model_name, builder, train_loader, val_loader, device,
            num_epochs=30, lr=1e-5, freeze_ratio=0.5
        )
        iso_results.append(result)
        torch.cuda.empty_cache()

    # Fine-tune pyramidal models
    print("\n" + "="*60)
    print("FINE-TUNING PYRAMIDAL ViG MODELS")
    print("="*60)
    pyr_results = []
    for model_name, builder in PYRAMIDAL_MODELS.items():
        result = finetune_model(
            model_name, builder, train_loader, val_loader, device,
            num_epochs=30, lr=1e-5, freeze_ratio=0.5
        )
        pyr_results.append(result)
        torch.cuda.empty_cache()

    # Plot results
    print("\n" + "="*60)
    print("GENERATING CHARTS")
    print("="*60)
    plot_results(iso_results, pyr_results)

    # Print summary
    print("\n" + "="*60)
    print("FINE-TUNING SUMMARY")
    print("="*60)

    print("\nIsotropic Models:")
    for r in iso_results:
        recovery = r['best_acc'] - r['baseline_acc']
        print(f"  {r['model_name']:<20} | Baseline: {r['baseline_acc']:.2f}% → "
              f"Best: {r['best_acc']:.2f}% | Recovery: {recovery:+.2f}%")

    print("\nPyramidal Models:")
    for r in pyr_results:
        recovery = r['best_acc'] - r['baseline_acc']
        print(f"  {r['model_name']:<20} | Baseline: {r['baseline_acc']:.2f}% → "
              f"Best: {r['best_acc']:.2f}% | Recovery: {recovery:+.2f}%")

if __name__ == "__main__":
    main()
