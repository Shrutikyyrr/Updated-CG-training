"""
Day 13 - Unsupervised Learning (Clustering) & Transfer Learning
================================================================
Scenarios:
  1. Plant Disease Classification (ResNet-50 Transfer Learning)
  2. Music Genre Classification (ResNet-50 Transfer Learning)
  3. Overfitting in Fine-tuned Model + K-Means Clustering

Note: ResNet-50 pretrained model download is not required.
      Feature extraction output is simulated using numpy arrays.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# SCENARIO 1: Plant Disease Classification (Transfer Learning)
# ─────────────────────────────────────────────────────────────
# A farmer's cooperative wants to detect plant diseases from
# leaf photos. Training a CNN from scratch needs thousands of
# labeled images and days of compute time.
# Instead, they use Transfer Learning with ResNet-50:
#   - ResNet-50 is pretrained on ImageNet (1.2M images, 1000 classes)
#   - Its convolutional layers already know how to detect edges,
#     textures, and shapes — useful for leaf analysis too
#   - We freeze the ResNet-50 backbone and only train a new
#     classification head on top (5 disease classes)
# This approach needs far less data and trains much faster.
# Here, ResNet-50 feature output (2048-dim) is simulated.
print("=" * 60)
print("SCENARIO 1: Plant Disease Classification (Transfer Learning)")
print("Backbone: ResNet-50 (pretrained) | Head: Custom 5-class classifier")
print("=" * 60)

np.random.seed(42)
num_samples = 100
num_classes = 5
class_names = ['Healthy', 'Rust', 'Blight', 'Mildew', 'Mosaic']

# Simulated ResNet-50 feature output (2048-dim per image)
features = np.random.randn(num_samples, 2048)
labels   = np.random.randint(0, num_classes, num_samples)

print(f"Feature shape (ResNet-50 output) : {features.shape}")
print(f"Number of disease classes        : {num_classes}")
print(f"Classes                          : {class_names}")

class PlantDiseaseClassifier(nn.Module):
    """
    Classification head on top of frozen ResNet-50 features.
    Input  : 2048-dim feature vector
    Output : 5 class scores (disease types)
    """
    def __init__(self, input_dim=2048, num_classes=5):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = PlantDiseaseClassifier()
print(f"\nClassifier head:")
print(model)

sample_features = torch.FloatTensor(features[:4])
output          = model(sample_features)
predicted       = torch.argmax(output, dim=1)
print(f"\nSample predictions (4 leaf images):")
for i, pred in enumerate(predicted):
    print(f"  Image {i + 1}: {class_names[pred.item()]}")

print("\nSimulated Training Progress:")
print("Epoch 1/5  - Loss: 1.6234  Accuracy: 22.00%")
print("Epoch 2/5  - Loss: 1.4521  Accuracy: 38.00%")
print("Epoch 3/5  - Loss: 1.2103  Accuracy: 55.00%")
print("Epoch 4/5  - Loss: 0.9876  Accuracy: 68.00%")
print("Epoch 5/5  - Loss: 0.7654  Accuracy: 78.00%")
print("\nTest Accuracy: 76.00%")
print("Transfer Learning saved significant training time vs training from scratch.")

# ─────────────────────────────────────────────────────────────
# SCENARIO 2: Music Genre Classification (Transfer Learning)
# ─────────────────────────────────────────────────────────────
# A music streaming platform wants to auto-tag songs by genre
# (Rock, Pop, Jazz, Classical, Hip-Hop, Electronic).
# Audio is converted to mel-spectrogram images (visual representation
# of sound frequencies over time). These images are then fed into
# a ResNet-50 model pretrained on ImageNet.
# Even though ResNet-50 was trained on photos, its low-level feature
# detectors (edges, patterns) transfer well to spectrogram images.
# Only the final classification head is trained from scratch.
print("\n" + "=" * 60)
print("SCENARIO 2: Music Genre Classification (Transfer Learning)")
print("Input: Mel-spectrogram images | Backbone: ResNet-50 | 6 genres")
print("=" * 60)

np.random.seed(7)
num_samples  = 80
num_genres   = 6
genre_names  = ['Rock', 'Pop', 'Jazz', 'Classical', 'Hip-Hop', 'Electronic']

# Simulated ResNet-50 features from mel-spectrogram images
spectrogram_features = np.random.randn(num_samples, 2048)
genre_labels         = np.random.randint(0, num_genres, num_samples)

print(f"Spectrogram feature shape : {spectrogram_features.shape}")
print(f"Genres                    : {genre_names}")

class MusicGenreClassifier(nn.Module):
    """
    Classification head on top of ResNet-50 features.
    Input  : 2048-dim feature vector
    Output : 6 genre class scores
    """
    def __init__(self, input_dim=2048, num_classes=6):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, 512)
        self.bn1     = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(512, 128)
        self.fc3     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = MusicGenreClassifier()
print(f"\nClassifier head:")
print(model)

sample      = torch.FloatTensor(spectrogram_features[:3])
output      = model(sample)
predicted   = torch.argmax(output, dim=1)
print(f"\nSample genre predictions (3 songs):")
for i, pred in enumerate(predicted):
    print(f"  Song {i + 1}: {genre_names[pred.item()]}")

print("\nSimulated Training Progress:")
print("Epoch  1/10 - Loss: 1.7891  Val Accuracy: 18.75%")
print("Epoch  3/10 - Loss: 1.3245  Val Accuracy: 43.75%")
print("Epoch  5/10 - Loss: 0.9876  Val Accuracy: 62.50%")
print("Epoch  8/10 - Loss: 0.6543  Val Accuracy: 75.00%")
print("Epoch 10/10 - Loss: 0.4321  Val Accuracy: 81.25%")
print("\nFinal Test Accuracy: 79.17%")

# ─────────────────────────────────────────────────────────────
# SCENARIO 3: Overfitting in Fine-tuned Model + K-Means Clustering
# ─────────────────────────────────────────────────────────────
# Part A: Overfitting
# When fine-tuning a pretrained model on a small dataset, the model
# can memorize the training data instead of learning general patterns.
# This is called overfitting — training accuracy keeps rising but
# validation accuracy stops improving or starts dropping.
# Signs: large gap between train and val accuracy after epoch 8-9.
# Solutions: Early stopping, Dropout, Data Augmentation, L2 regularization.
#
# Part B: K-Means Clustering
# After extracting features from images using ResNet-50, we can
# group similar images together without any labels using K-Means.
# This is useful for organizing large unlabeled image datasets,
# finding similar products, or discovering hidden patterns.
print("\n" + "=" * 60)
print("SCENARIO 3: Overfitting Detection + K-Means Clustering")
print("=" * 60)

# --- Part A: Overfitting ---
print("\n--- Part A: Overfitting in Fine-tuned Model ---")
print("Tracking training vs validation accuracy over 20 epochs:\n")

epochs    = list(range(1, 21))
train_acc = [0.45, 0.55, 0.63, 0.70, 0.76, 0.81, 0.85, 0.88, 0.91, 0.93,
             0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.988, 0.990, 0.992, 0.994]
val_acc   = [0.43, 0.53, 0.61, 0.67, 0.72, 0.75, 0.77, 0.78, 0.78, 0.77,
             0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70, 0.69, 0.68, 0.67]

print(f"{'Epoch':<8} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10}")
print("-" * 44)
for e, tr, va in zip(epochs, train_acc, val_acc):
    gap  = tr - va
    flag = "  <-- OVERFITTING" if gap > 0.15 else ""
    print(f"{e:<8} {tr:<12.3f} {va:<12.3f} {gap:<10.3f}{flag}")

print("\nOverfitting starts around epoch 8-9 (val accuracy plateaus then drops).")
print("Solutions: Early Stopping | Dropout | Data Augmentation | L2 Regularization")

# --- Part B: K-Means Clustering ---
print("\n--- Part B: K-Means Clustering on Image Features ---")
print("Grouping 150 images into 3 clusters based on visual similarity:\n")

np.random.seed(42)
cluster1 = np.random.randn(50, 2) + [2, 2]
cluster2 = np.random.randn(50, 2) + [-2, -2]
cluster3 = np.random.randn(50, 2) + [2, -2]
data     = np.vstack([cluster1, cluster2, cluster3])

scaler      = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data_scaled)
labels = kmeans.labels_

print(f"Total images     : {data.shape[0]}")
print(f"Feature dims     : {data.shape[1]}")
print(f"Number of clusters : 3")
print(f"Inertia (within-cluster sum of squares) : {kmeans.inertia_:.4f}")

unique, counts = np.unique(labels, return_counts=True)
for cid, cnt in zip(unique, counts):
    print(f"  Cluster {cid}: {cnt} images")

print("\nK-Means successfully grouped similar images without any labels!")

print("\n" + "=" * 60)
print("DAY 13 - ALL SCENARIOS COMPLETE")
print("=" * 60)
