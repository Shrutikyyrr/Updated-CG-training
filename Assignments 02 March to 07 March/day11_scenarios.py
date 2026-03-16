"""
Day 11 - Deep Learning with CNN (PyTorch)
==========================================
Scenarios:
  1. CNN Handwritten Digit Recognition (Conv2D layer + Full CNN)
  2. CNN Blood Cell Analysis (Conv2D with ReLU + Full CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# SCENARIO 1: CNN Handwritten Digit Recognition
# ─────────────────────────────────────────────────────────────
# A machine learning engineer is designing a CNN to process
# grayscale images of handwritten digits (28x28 pixels).
# She starts with a single convolutional layer:
#   - Input: 1 channel (grayscale), 28x28 pixels
#   - 32 filters, each 3x3
#   - Stride = 1, Padding = 1 (same padding, output size stays 28x28)
# After understanding the single layer, she builds a full CNN
# with 2 conv blocks + fully connected layers for 10-class output.
# This is the standard approach for MNIST digit classification.
print("=" * 60)
print("SCENARIO 1: CNN - Handwritten Digit Recognition (PyTorch)")
print("Input: 28x28 grayscale | 32 filters | 3x3 kernel")
print("=" * 60)

# Single Conv2D layer
conv = nn.Conv2d(
    in_channels=1,    # grayscale input
    out_channels=32,  # 32 filters -> 32 feature maps
    kernel_size=3,    # 3x3 filter
    stride=1,
    padding=1         # same padding: output stays 28x28
)

x   = torch.randn(1, 1, 28, 28)  # (batch, channel, height, width)
out = conv(x)
print(f"Output Shape : {out.shape}")
print(f"Weights      : {conv.weight.shape}")
print(f"Bias         : {conv.bias.shape}")
print(f"Total Params : {32 * 1 * 3 * 3 + 32}")

# Full CNN Architecture
print("\n--- Full CNN Architecture (2 Conv Blocks + FC Layers) ---")

class DigitCNN(nn.Module):
    """
    CNN for handwritten digit recognition (MNIST style)
    Input  : 1 x 28 x 28 (grayscale)
    Output : 10 class scores (digits 0-9)

    Block 1: Conv(1->32) -> BatchNorm -> ReLU -> MaxPool  => 32 x 14 x 14
    Block 2: Conv(32->64) -> BatchNorm -> ReLU -> MaxPool => 64 x 7 x 7
    FC: 3136 -> 128 -> Dropout -> 10
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1  = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2, 2)   # 32 x 14 x 14

        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2, 2)   # 64 x 7 x 7

        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = DigitCNN()
print(model)

dummy_input = torch.randn(1, 1, 28, 28)
output      = model(dummy_input)
print(f"\nOutput shape for 1 image : {output.shape}")
print(f"(10 scores, one per digit 0-9)")

# ─────────────────────────────────────────────────────────────
# SCENARIO 2: CNN Blood Cell Analysis
# ─────────────────────────────────────────────────────────────
# A biomedical researcher is designing a CNN to analyze
# microscopic grayscale images of blood cells (64x64 pixels).
# She starts with a single convolutional layer:
#   - Input: 1 channel (grayscale), 64x64 pixels
#   - 16 filters, each 5x5
#   - Stride = 1, Padding = 2 (same padding for 5x5 kernel)
#   - ReLU activation applied after convolution
# Goal: extract local structural features like cell boundaries
# and texture to classify cells as healthy or abnormal.
# A full CNN is then built for binary classification.
print("\n" + "=" * 60)
print("SCENARIO 2: CNN - Blood Cell Analysis (PyTorch)")
print("Input: 64x64 grayscale | 16 filters | 5x5 kernel | ReLU")
print("=" * 60)

# Single Conv2D layer with ReLU
conv = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=5,
    stride=1,
    padding=2    # same padding for 5x5: padding = (5-1)/2 = 2
)
relu = nn.ReLU()

x   = torch.randn(1, 1, 64, 64)
out = relu(conv(x))
print(f"Output Shape : {out.shape}")
print(f"Weights      : {conv.weight.shape}")
print(f"Bias         : {conv.bias.shape}")
print(f"Total Params : {16 * 1 * 5 * 5 + 16}")

# Full CNN for Blood Cell Classification
print("\n--- Full CNN Architecture (Blood Cell Classifier) ---")

class BloodCellCNN(nn.Module):
    """
    CNN for blood cell classification
    Input  : 1 x 64 x 64 (grayscale microscope image)
    Output : 2 class scores (0=Healthy, 1=Abnormal)

    Block 1: Conv(1->16, 5x5) -> ReLU -> MaxPool  => 16 x 32 x 32
    Block 2: Conv(16->32, 3x3) -> ReLU -> MaxPool => 32 x 16 x 16
    FC: 8192 -> 128 -> 2
    """
    def __init__(self):
        super(BloodCellCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu  = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)   # 16 x 32 x 32

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # 32 x 16 x 16

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = BloodCellCNN()
print(model)

dummy_input = torch.randn(1, 1, 64, 64)
output      = model(dummy_input)
print(f"\nOutput shape for 1 image : {output.shape}")
print(f"(2 scores: Healthy / Abnormal)")

print("\n" + "=" * 60)
print("DAY 11 - ALL SCENARIOS COMPLETE")
print("=" * 60)
