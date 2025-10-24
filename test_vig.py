import torch
from vig import vig_ti_224_gelu  # or vig_ti_224_gelu, vig_b_224_gelu
torch.set_num_threads(16)
def run_vig_on_various_sizes(model, device, sizes):
    model = model.to(device)
    model.eval()
    for img_size in sizes:
        x = torch.randn(1, 3, img_size, img_size).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"Input size: {img_size}x{img_size} -> Output shape: {out.shape}")

if __name__ == "__main__":
    # Choose your ViG model
    model = vig_ti_224_gelu(pretrained=False)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
   # torch.set_num_threads(16)
    # List of input sizes to test (must be divisible by 16 for ViG's 4x stride-2 stem)
    input_sizes = [256, 512, 1024, 2048]
    run_vig_on_various_sizes(model, device, input_sizes)

