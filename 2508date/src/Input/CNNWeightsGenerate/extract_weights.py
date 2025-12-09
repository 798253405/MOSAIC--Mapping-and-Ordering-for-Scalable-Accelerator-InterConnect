#!/usr/bin/env python3
"""
Extract pretrained weights from PyTorch models and save in weight.txt format.
- VGG16: from torchvision
- DarkNet-19: from YOLO official weights

Output format: each line contains weights for ONE OUTPUT CHANNEL, space-separated floats
  - Conv layer: in_ch * kernel_h * kernel_w weights + 1 bias per line
  - Dense layer: in_features weights + 1 bias per line
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os


def extract_vgg16_weights(output_path):
    """Extract VGG16 pretrained weights - one output channel per line"""
    print("Loading VGG16 pretrained model...")
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.eval()

    print("Extracting VGG16 weights...")
    lines = []
    total_lines = 0

    # Extract Conv2D weights (features)
    conv_idx = 0
    for name, module in model.features.named_modules():
        if isinstance(module, nn.Conv2d):
            # weight shape: (out_channels, in_channels, H, W)
            # bias shape: (out_channels,)
            w = module.weight.data.cpu().numpy()  # (out_ch, in_ch, H, W)
            b = module.bias.data.cpu().numpy() if module.bias is not None else np.zeros(w.shape[0])

            out_ch, in_ch, kh, kw = w.shape
            weights_per_line = in_ch * kh * kw + 1  # +1 for bias

            for oc in range(out_ch):
                # Flatten weights for this output channel: (in_ch, H, W) -> flat
                oc_weights = w[oc].flatten()  # in_ch * kh * kw weights
                line_data = np.concatenate([oc_weights, [b[oc]]])
                lines.append(line_data)

            print(f"  Conv2D layer {conv_idx}: {w.shape} -> {out_ch} lines, {weights_per_line} values/line")
            total_lines += out_ch
            conv_idx += 1

    # Extract Dense weights (classifier)
    dense_idx = 0
    for name, module in model.classifier.named_modules():
        if isinstance(module, nn.Linear):
            # weight shape: (out_features, in_features)
            # bias shape: (out_features,)
            w = module.weight.data.cpu().numpy()  # (out_features, in_features)
            b = module.bias.data.cpu().numpy() if module.bias is not None else np.zeros(w.shape[0])

            out_features, in_features = w.shape
            weights_per_line = in_features + 1  # +1 for bias

            for of in range(out_features):
                line_data = np.concatenate([w[of], [b[of]]])
                lines.append(line_data)

            print(f"  Dense layer {dense_idx}: {w.shape} -> {out_features} lines, {weights_per_line} values/line")
            total_lines += out_features
            dense_idx += 1

    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for line_data in lines:
            line = ' '.join(f'{w:.8g}' for w in line_data)
            f.write(line + '\n')

    total_params = sum(len(l) for l in lines)
    print(f"VGG16: {total_lines} lines, {total_params:,} total values")
    return lines


def download_darknet19_weights():
    """Download darknet19 weights from YOLO official"""
    import urllib.request
    url = "https://pjreddie.com/media/files/darknet19.weights"
    weights_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "darknet19.weights")

    if not os.path.exists(weights_file):
        print(f"Downloading darknet19.weights from {url}...")
        urllib.request.urlretrieve(url, weights_file)
        print(f"Downloaded to {weights_file}")
    else:
        print(f"Using cached {weights_file}")

    return weights_file


def extract_darknet19_weights(output_path):
    """Extract DarkNet-19 pretrained weights - one output channel per line"""
    print("\nExtracting DarkNet-19 weights...")

    # Download weights
    weights_file = download_darknet19_weights()

    # DarkNet-19 layer configuration from darknetSimplify.txt
    # Format: (type, in_ch, out_ch, kernel_h, kernel_w)
    layer_configs = [
        ('conv', 3, 32, 3, 3),      # Conv2D 3 3 3 32
        ('conv', 32, 64, 3, 3),     # Conv2D 32 3 3 64
        ('conv', 64, 128, 3, 3),    # Conv2D 64 3 3 128
        ('conv', 128, 64, 1, 1),    # Conv2D 128 1 1 64
        ('conv', 64, 128, 3, 3),    # Conv2D 64 3 3 128
        ('conv', 128, 256, 3, 3),   # Conv2D 128 3 3 256
        ('conv', 256, 128, 1, 1),   # Conv2D 256 1 1 128
        ('conv', 128, 256, 3, 3),   # Conv2D 128 3 3 256
        ('conv', 256, 512, 3, 3),   # Conv2D 256 3 3 512
        ('conv', 512, 256, 1, 1),   # Conv2D 512 1 1 256
        ('conv', 256, 512, 3, 3),   # Conv2D 256 3 3 512
        ('conv', 512, 256, 1, 1),   # Conv2D 512 1 1 256
        ('conv', 256, 512, 3, 3),   # Conv2D 256 3 3 512
        ('conv', 512, 1024, 3, 3),  # Conv2D 512 3 3 1024
        ('conv', 1024, 512, 1, 1),  # Conv2D 1024 1 1 512
        ('conv', 512, 1024, 3, 3),  # Conv2D 512 3 3 1024
        ('conv', 1024, 512, 1, 1),  # Conv2D 1024 1 1 512
        ('conv', 512, 1024, 3, 3),  # Conv2D 512 3 3 1024
        ('conv_nobias', 1024, 1000, 1, 1),  # Conv2D 1024 1 1 1000 (classifier)
    ]

    print(f"\nParsing {weights_file}...")
    lines = []
    total_lines = 0

    with open(weights_file, 'rb') as f:
        # Header: 4 int32 values (major, minor, revision, seen)
        header = np.fromfile(f, dtype=np.int32, count=4)
        print(f"  Header: {header}")

        for idx, config in enumerate(layer_configs):
            layer_type, in_ch, out_ch, kernel_h, kernel_w = config
            num_weights = out_ch * in_ch * kernel_h * kernel_w
            weights_per_line = in_ch * kernel_h * kernel_w + 1  # +1 for bias

            if layer_type == 'conv':
                # Read BN params first: beta, gamma, mean, var (each out_ch floats)
                bn_beta = np.fromfile(f, dtype=np.float32, count=out_ch)
                bn_gamma = np.fromfile(f, dtype=np.float32, count=out_ch)
                bn_mean = np.fromfile(f, dtype=np.float32, count=out_ch)
                bn_var = np.fromfile(f, dtype=np.float32, count=out_ch)

                # Read conv weights
                conv_weights = np.fromfile(f, dtype=np.float32, count=num_weights)
                conv_weights = conv_weights.reshape(out_ch, in_ch, kernel_h, kernel_w)

                # Use BN beta as bias (approximation)
                for oc in range(out_ch):
                    oc_weights = conv_weights[oc].flatten()
                    line_data = np.concatenate([oc_weights, [bn_beta[oc]]])
                    lines.append(line_data)

                print(f"  Layer {idx}: Conv({in_ch}->{out_ch}, {kernel_h}x{kernel_w}) -> {out_ch} lines, {weights_per_line} values/line")
                total_lines += out_ch

            elif layer_type == 'conv_nobias':
                # Last conv layer without BN, has bias
                bias = np.fromfile(f, dtype=np.float32, count=out_ch)
                conv_weights = np.fromfile(f, dtype=np.float32, count=num_weights)
                conv_weights = conv_weights.reshape(out_ch, in_ch, kernel_h, kernel_w)

                for oc in range(out_ch):
                    oc_weights = conv_weights[oc].flatten()
                    line_data = np.concatenate([oc_weights, [bias[oc]]])
                    lines.append(line_data)

                print(f"  Layer {idx}: Conv({in_ch}->{out_ch}, {kernel_h}x{kernel_w}) with bias -> {out_ch} lines, {weights_per_line} values/line")
                total_lines += out_ch

    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for line_data in lines:
            line = ' '.join(f'{w:.8g}' for w in line_data)
            f.write(line + '\n')

    total_params = sum(len(l) for l in lines)
    print(f"DarkNet-19: {total_lines} lines, {total_params:,} total values")
    return lines


def generate_random_input(height, width, channels, output_path):
    """Generate random input image in the same format as input2.txt"""
    print(f"\nGenerating random input ({height}x{width}x{channels})...")

    # Generate random values (similar to normalized ImageNet range)
    np.random.seed(42)
    data = np.random.randn(channels, height, width) * 0.5  # Similar to normalized values

    # Save to file: each line is one row of one channel
    with open(output_path, 'w') as f:
        for c in range(channels):
            for h in range(height):
                line = ' '.join(f'{v:.8g}' for v in data[c, h, :])
                f.write(line + '\n')

    print(f"Saved to {output_path}")


def main():
    # Use relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.dirname(script_dir)  # ../Input

    # VGG16
    vgg_weight = os.path.join(input_dir, 'vgg16_weight.txt')
    extract_vgg16_weights(vgg_weight)

    # DarkNet-19
    darknet_weight = os.path.join(input_dir, 'darknet_weight.txt')
    extract_darknet19_weights(darknet_weight)

    # Generate input files
    vgg_input = os.path.join(input_dir, 'vgg16_input.txt')
    generate_random_input(224, 224, 3, vgg_input)

    darknet_input = os.path.join(input_dir, 'darknet_input.txt')
    generate_random_input(64, 64, 3, darknet_input)

    print("\n" + "="*50)
    print("Done! Generated files in src/Input/:")
    print("  - vgg16_weight.txt")
    print("  - vgg16_input.txt")
    print("  - darknet_weight.txt")
    print("  - darknet_input.txt")


if __name__ == "__main__":
    main()
