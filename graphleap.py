"""
GraphLeap Usage Example
Demonstrates how to use the batched CPU-GPU GraphLeap implementation
"""

import torch
from vig import vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu

def example_standard_inference():
    """Standard inference without GraphLeap"""
    print("=" * 50)
    print("Standard Inference (GPU only)")
    print("=" * 50)
    
    # Create model
    model = vig_ti_224_gelu(num_classes=1000, use_graphleap=False)
    model.eval()
    model = model.cuda()
    
    # Create batch of images
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # Standard inference
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")
    print("Standard inference completed successfully!")
    print()


def example_graphleap_inference():
    """GraphLeap inference with CPU-GPU overlap"""
    print("=" * 50)
    print("GraphLeap Inference (CPU-GPU Overlap)")
    print("=" * 50)
    
    # Create model with GraphLeap enabled
    model = vig_ti_224_gelu(num_classes=1000, use_graphleap=True)
    model.eval()
    model = model.cuda()
    
    # Create batch of images
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # GraphLeap inference
    # Graph construction happens on CPU in parallel with GPU feature updates
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")
    print("GraphLeap inference completed successfully!")
    print()
    
    # Clean up
    model.cpu_graph_constructor.stop()


def example_dynamic_switching():
    """Example showing how to switch between modes"""
    print("=" * 50)
    print("Dynamic Mode Switching")
    print("=" * 50)
    
    # Create model
    model = vig_s_224_gelu(num_classes=1000, use_graphleap=False)
    model.eval()
    model = model.cuda()
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # Standard mode
    print("Running in standard mode...")
    with torch.no_grad():
        outputs1 = model(images)
    print(f"Standard output shape: {outputs1.shape}")
    
    # Enable GraphLeap
    print("\nEnabling GraphLeap mode...")
    model.enable_graphleap()
    
    with torch.no_grad():
        outputs2 = model(images)
    print(f"GraphLeap output shape: {outputs2.shape}")
    
    # Disable GraphLeap
    print("\nDisabling GraphLeap mode...")
    model.disable_graphleap()
    
    with torch.no_grad():
        outputs3 = model(images)
    print(f"Back to standard output shape: {outputs3.shape}")
    print()


def example_training():
    """Training example (always uses standard mode)"""
    print("=" * 50)
    print("Training Mode")
    print("=" * 50)
    
    # Create model (GraphLeap is automatically disabled during training)
    model = vig_ti_224_gelu(num_classes=1000, use_graphleap=True)
    model.train()  # Training mode
    model = model.cuda()

#Standard output shape: torch.Size([2, 1000])    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    labels = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training step (uses standard mode automatically)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print("Note: GraphLeap is disabled during training")
    print()


def example_batched_inference_comparison():
    """Compare standard vs GraphLeap for batched inference"""
    print("=" * 50)
    print("Batched Inference Comparison")
    print("=" * 50)
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Standard model
        model_std = vig_ti_224_gelu(num_classes=1000, use_graphleap=False)
        model_std.eval()
        model_std = model_std.cuda()
        
        images = torch.randn(batch_size, 3, 224, 224).cuda()
        
        # Warmup
        with torch.no_grad():
            _ = model_std(images)
        
        # Time standard inference
        torch.cuda.synchronize()
        import time
        start = time.time()
        with torch.no_grad():
            _ = model_std(images)
        torch.cuda.synchronize()
        std_time = time.time() - start
        
        # GraphLeap model
        model_gl = vig_ti_224_gelu(num_classes=1000, use_graphleap=True)
        model_gl.eval()
        model_gl = model_gl.cuda()
        
        # Warmup
        with torch.no_grad():
            _ = model_gl(images)
        
        # Time GraphLeap inference
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model_gl(images)
        torch.cuda.synchronize()
        gl_time = time.time() - start
        
        speedup = std_time / gl_time
        print(f"  Standard time: {std_time*1000:.2f} ms")
        print(f"  GraphLeap time: {gl_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Clean up
        model_gl.cpu_graph_constructor.stop()
        del model_std, model_gl


def example_large_images():
    """Example with larger image sizes where GraphLeap shows more benefit"""
    print("=" * 50)
    print("Large Image Inference")
    print("=" * 50)
    
    image_sizes = [224, 384, 512]
    batch_size = 1
    
    for img_size in image_sizes:
        print(f"\nImage size: {img_size}x{img_size}")
        
        # GraphLeap model
        model = vig_ti_224_gelu(num_classes=1000, use_graphleap=True)
        model.eval()
        model = model.cuda()
        
        images = torch.randn(batch_size, 3, img_size, img_size).cuda()
        
        try:
            with torch.no_grad():
                outputs = model(images)
            print(f"  Output shape: {outputs.shape}")
            print(f"  GraphLeap successfully processed {img_size}x{img_size} images")
        except Exception as e:
            print(f"  Error: {e}")
        finally:
            model.cpu_graph_constructor.stop()
            del model


if __name__ == "__main__":
    print("\n" + "="*50)
    print("GraphLeap Implementation Examples")
    print("="*50 + "\n")
    
    # Run examples
    try:
        images = torch.randn(16, 3, 224, 224).cuda()
        # GraphLeap model
        model_gl = vig_ti_224_gelu(num_classes=1000, use_graphleap=True)
        model_gl.eval()
        model_gl = model_gl.cuda()
        import time
        # Warmup
        with torch.no_grad():
            _ = model_gl(images)

        # Time GraphLeap inference
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model_gl(images)
        torch.cuda.synchronize()
        gl_time = time.time() - start
        print("{:.4f} ms".format(gl_time*1000))
    except Exception as e:
        print(f"Error in GraphLeap inference: {e}\n")
    

