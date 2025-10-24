import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vig import vig_ti_224_gelu

image_sizes = [256, 512, 768, 1024]
batch_sizes = [1, 2, 4]

results = {
    'image_size': [],
    'batch_size': [],
    'graph_time': [],
    'feature_time': [],
    'h2d_time': [],
    'd2h_time': []
}

for size in image_sizes:
    for batch in batch_sizes:
        model = vig_ti_224_gelu(num_classes=1000, use_graphleap=True)
        model.eval()
        model.cuda()

        images = torch.randn(batch, 3, size, size)
        torch.cuda.synchronize()
        start_h2d = time.time()
        images_gpu = images.cuda(non_blocking=True)
        torch.cuda.synchronize()
        h2d_time = time.time() - start_h2d

        model.profile = {k: 0.0 for k in model.profile}
        with torch.no_grad():
            torch.cuda.synchronize()
            t_start = time.time()
            outputs = model(images_gpu)
            torch.cuda.synchronize()
        graph_time = model.profile['graph_construction_total']
        feature_time = model.profile['feature_update_total']

        torch.cuda.synchronize()
        start_d2h = time.time()
        _ = outputs.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start_d2h

        print(f"Size {size} Batch {batch}: Graph {graph_time*1000:.2f}ms, "
              f"Feature {feature_time*1000:.2f}ms, H2D {h2d_time*1000:.2f}ms, D2H {d2h_time*1000:.2f}ms")

        results['image_size'].append(size)
        results['batch_size'].append(batch)
        results['graph_time'].append(graph_time*1000)
        results['feature_time'].append(feature_time*1000)
        results['h2d_time'].append(h2d_time*1000)
        results['d2h_time'].append(d2h_time*1000)

fig, axs = plt.subplots(1, 4, figsize=(22, 5))
ylabels = ['Graph Construction (ms)', 'Feature Update (ms)', 'Host→Device (ms)', 'Device→Host (ms)']
keys = ['graph_time', 'feature_time', 'h2d_time', 'd2h_time']

for i, key in enumerate(keys):
    for batch in batch_sizes:
        xs = [sz for sz, b in zip(results['image_size'], results['batch_size']) if b == batch]
        ys = [v for v, b in zip(results[key], results['batch_size']) if b == batch]
        axs[i].plot(xs, ys, marker='o', label=f'Batch {batch}')
    axs[i].set_xlabel('Image Size')
    axs[i].set_ylabel(ylabels[i])
    axs[i].legend()
    axs[i].set_title(ylabels[i])
plt.tight_layout()
plt.savefig('vig_ti_224_gelu_profile.png')
plt.show()

