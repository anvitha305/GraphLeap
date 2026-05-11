"""
Script to verify if Vision GNN weights are correct and compatible
"""

import torch
import os
import hashlib
from pathlib import Path

# Expected weights info: (filename, expected_size_MB, description)
EXPECTED_WEIGHTS = {
    'vig_ti': {
        'filename': 'vig_ti_74.5.pth',
        'size_range': (40, 60),
        'accuracy': 74.5,
        'source': 'https://github.com/huawei-noah/Efficient-AI-Backbones'
    },
    'vig_s': {
        'filename': 'vig_s_80.6.pth',
        'size_range': (80, 120),
        'accuracy': 80.6,
        'source': 'https://github.com/huawei-noah/Efficient-AI-Backbones'
    },
    'vig_b': {
        'filename': 'vig_b_82.6.pth',
        'size_range': (200, 250),
        'accuracy': 82.6,
        'source': 'https://github.com/huawei-noah/Efficient-AI-Backbones'
    },
    'pvig_ti': {
        'filename': 'pvig_ti_78.5.pth',
        'size_range': (40, 60),
        'accuracy': 78.5,
        'source': 'BaiduDisk'
    },
    'pvig_s': {
        'filename': 'pvig_s_82.1.pth',
        'size_range': (80, 120),
        'accuracy': 82.1,
        'source': 'BaiduDisk'
    },
    'pvig_m': {
        'filename': 'pvig_m_83.1.pth',
        'size_range': (150, 200),
        'accuracy': 83.1,
        'source': 'BaiduDisk'
    },
    'pvig_b': {
        'filename': 'pvig_b_83.66.pth',
        'size_range': (200, 280),
        'accuracy': 83.66,
        'source': 'BaiduDisk'
    },
}

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def verify_weight_structure(model_name, weights_path):
    """Verify the internal structure of the weights"""
    try:
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle nested dict
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # Check if it's a valid state dict
        if not isinstance(state_dict, dict):
            return False, "Not a valid state dict (not a dictionary)"
        
        if len(state_dict) == 0:
            return False, "State dict is empty"
        
        # Get some info about the weights
        num_params = len(state_dict)
        total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
        
        return True, {
            'num_layers': num_params,
            'total_parameters': total_params,
            'sample_keys': list(state_dict.keys())[:5]
        }
    
    except Exception as e:
        return False, f"Error loading weights: {str(e)}"

def verify_weights(weights_dir='./pretrained_weights'):
    """Verify all weights in a directory"""
    
    print("\n" + "="*90)
    print("VISION GNN WEIGHTS VERIFICATION")
    print("="*90 + "\n")
    
    if not os.path.exists(weights_dir):
        print(f"⚠️  Weights directory not found: {weights_dir}")
        print(f"Creating directory: {weights_dir}\n")
        os.makedirs(weights_dir, exist_ok=True)
    
    print(f"Checking weights in: {weights_dir}\n")
    
    results = {}
    
    for model_name, info in EXPECTED_WEIGHTS.items():
        filename = info['filename']
        weights_path = os.path.join(weights_dir, filename)
        
        print(f"Model: {model_name}")
        print(f"  Expected accuracy: {info['accuracy']}%")
        print(f"  Expected size: {info['size_range'][0]}-{info['size_range'][1]} MB")
        print(f"  Source: {info['source']}")
        
        # Check if file exists
        if not os.path.exists(weights_path):
            print(f"  ❌ FILE NOT FOUND: {weights_path}")
            results[model_name] = 'MISSING'
            print()
            continue
        
        # Check file size
        file_size_mb = get_file_size_mb(weights_path)
        size_ok = info['size_range'][0] <= file_size_mb <= info['size_range'][1]
        
        print(f"  ✓ File found: {weights_path}")
        print(f"  File size: {file_size_mb:.1f} MB", end="")
        if size_ok:
            print(" ✓ (within expected range)")
        else:
            print(f" ⚠️  (expected {info['size_range'][0]}-{info['size_range'][1]} MB)")
        
        # Verify structure
        is_valid, details = verify_weight_structure(model_name, weights_path)
        
        if is_valid:
            print(f"  ✓ Valid PyTorch state dict")
            print(f"    - Layers: {details['num_layers']}")
            print(f"    - Total parameters: {details['total_parameters']:,}")
            print(f"    - Sample layers: {details['sample_keys']}")
            results[model_name] = 'VALID'
        else:
            print(f"  ❌ Invalid state dict: {details}")
            results[model_name] = 'INVALID'
        
        print()
    
    # Summary
    print("="*90)
    print("SUMMARY")
    print("="*90)
    
    valid_count = sum(1 for v in results.values() if v == 'VALID')
    missing_count = sum(1 for v in results.values() if v == 'MISSING')
    invalid_count = sum(1 for v in results.values() if v == 'INVALID')
    
    print(f"Valid:   {valid_count}")
    print(f"Missing: {missing_count}")
    print(f"Invalid: {invalid_count}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 90)
    
    if missing_count > 0:
        print("\n❌ Missing weights for:")
        for model_name, status in results.items():
            if status == 'MISSING':
                info = EXPECTED_WEIGHTS[model_name]
                print(f"   - {model_name}: {info['source']}")
                if 'BaiduDisk' in info['source']:
                    print(f"     Download manually or use only isotropic models")
    
    if invalid_count > 0:
        print("\n❌ Invalid weights for:")
        for model_name, status in results.items():
            if status == 'INVALID':
                print(f"   - {model_name}: Re-download weights (file may be corrupted)")
    
    if valid_count > 0:
        print(f"\n✓ Ready to train with:")
        valid_models = [m for m, s in results.items() if s == 'VALID']
        print(f"  Models: {', '.join(valid_models)}")
    
    print("\n" + "="*90)
    
    return results

def recommend_training_config(results):
    """Recommend which models to train based on available weights"""
    valid_models = [m for m, s in results.items() if s == 'VALID']
    
    print("\nRECOMMENDED TRAINING CONFIG:")
    print("-" * 90)
    
    isotropic = [m for m in valid_models if m.startswith('vig_')]
    pyramidal = [m for m in valid_models if m.startswith('pvig_')]
    
    if isotropic:
        print(f"\nIsotropic models available: {', '.join(isotropic)}")
        print(f"model_names = {isotropic}")
    
    if pyramidal:
        print(f"\nPyramidal models available: {', '.join(pyramidal)}")
        print(f"model_names = {pyramidal}")
    
    if not valid_models:
        print("\nNo valid weights found!")
        print("Use only isotropic models (most reliable):")
        print("model_names = ['vig_ti', 'vig_s', 'vig_b']")
    
    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    results = verify_weights()
    recommend_training_config(results)
