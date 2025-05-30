import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Paths to VGGT prediction cache
vggt_cache_dir = "unet_predictions/step_010000/refined"
output_dir = "vggt_visualizations/refined"
os.makedirs(output_dir, exist_ok=True)

# Collect all depth files
depth_files = sorted(f for f in os.listdir(vggt_cache_dir) if f.endswith("_rgb.npy"))

# Sample 30
samples = random.sample(depth_files, min(30, len(depth_files)))

for i, depth_file in enumerate(samples):
    base = depth_file.replace("_depth.npy", "")

    # Load depth and confidence
    depth = np.load(os.path.join(vggt_cache_dir, depth_file))

    # Plot
    plt.plot(1, 2, 1)
    plt.imshow(depth, cmap='plasma')
    plt.colorbar()
    plt.title("VGGT Depth")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}.png"))
    plt.close()

print(f"✅ Saved {len(samples)} VGGT visualizations to: {output_dir}")
