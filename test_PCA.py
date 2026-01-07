# best_eigenfaces_full_verbose_fixed.py
# Fully fixed with import for accuracy_score and correct verification accuracy

import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score  # Added accuracy_score here
from collections import Counter
from PIL import Image
import os
import time

print("=== Optimized Eigenfaces for Face Recognition ===\n")
start_total = time.time()

# ====================== Simple Dataset ======================
class SimpleDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('L')  # Grayscale
        img = img.resize((112, 112), Image.BILINEAR)
        x = np.array(img).flatten().astype(np.float32)
        x = (x - 127.5) / 128.0  # Normalize to [-1, 1]
        return x, self.labels[i]

# ====================== Load Dataset ======================
print("1. Loading VGGFace2 dataset...")
dataset_root = 'VGGFace2'  # Change if your path is different
full_dataset = datasets.ImageFolder(dataset_root)
targets = np.array(full_dataset.targets)

print(f"   Total images: {len(full_dataset)}")
print(f"   Total classes: {len(full_dataset.classes)}\n")

# Select first 10 classes
print("2. Selecting first 10 classes...")
num_classes = 10
mask = targets < num_classes
selected_paths = [full_dataset.samples[i][0] for i in range(len(full_dataset)) if mask[i]]
selected_labels = targets[mask]

print(f"   Found {len(selected_paths)} images across {num_classes} classes\n")

# Show class names and counts
class_names = sorted(full_dataset.classes)[:num_classes]
print("   === Selected Classes ===")
for i, name in enumerate(class_names):
    count = np.sum(selected_labels == i)
    print(f"   Class {i:2d}: {name:25} → {count} images")
print()

# ====================== Train/Test Split ======================
print("3. Creating stratified train/test split (80/20)...")
train_paths, test_paths, y_train, y_test = train_test_split(
    selected_paths, selected_labels, test_size=0.2, random_state=42, stratify=selected_labels
)

print(f"   Train images: {len(train_paths)}")
print(f"   Test images : {len(test_paths)}\n")

# ====================== Load Data ======================
train_ds = SimpleDataset(train_paths, y_train)
test_ds = SimpleDataset(test_paths, y_test)

print("4. Loading and preprocessing training images...")
X_train = np.stack([item[0] for item in train_ds])
y_train = np.array([item[1] for item in train_ds])
print(f"   → Train data shape: {X_train.shape}")

print("5. Loading and preprocessing test images...")
X_test = np.stack([item[0] for item in test_ds])
y_test = np.array([item[1] for item in test_ds])
print(f"   → Test data shape: {X_test.shape}\n")

# ====================== PCA ======================
print("6. Computing mean face and centering...")
mean_face = X_train.mean(axis=0)
X_centered = X_train - mean_face
print("   → Mean face computed\n")

print("7. Computing covariance matrix and eigenvectors (this may take 10-30 seconds)...")
cov_start = time.time()
cov = np.dot(X_centered.T, X_centered) / len(X_train)
eigvals, eigvecs = np.linalg.eigh(cov)
cov_time = time.time() - cov_start
print(f"   → Eigen decomposition completed in {cov_time:.2f} seconds\n")

# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print(f"   Top 5 eigenvalues: {eigvals[:5]}")
print(f"   Total variance explained by all: {eigvals.sum():.2f}\n")

# Select components for 99% variance
cumvar = np.cumsum(eigvals) / eigvals.sum()
K = np.searchsorted(cumvar, 0.99) + 1
K = min(K, 250)
print(f"   Using {K} components (captures {cumvar[K-1]:.4f} of variance)\n")

W = eigvecs[:, :K]  # Projection matrix

# Project features
print("8. Projecting images into eigenface space...")
train_feats = X_centered @ W
test_feats = (X_test - mean_face) @ W
print(f"   → Train features: {train_feats.shape}")
print(f"   → Test features : {test_feats.shape}\n")

# Save model
print("9. Saving model...")
np.savez('best_eigenfaces.npz', mean=mean_face, W=W, classes=class_names)
print("   → Saved as 'best_eigenfaces.npz'\n")

# ====================== Evaluation ======================
print("10. Starting evaluation...\n")

def l2_distance(a, b):
    return np.linalg.norm(a - b)

# Verification pairs
print("   Generating verification pairs...")
same_dists = []
diff_dists = []
n_pairs = 3000
same_count = diff_count = 0

np.random.seed(42)
for _ in range(n_pairs):
    # Same person
    lbl = np.random.choice(np.unique(y_test))
    idxs = np.where(y_test == lbl)[0]
    if len(idxs) >= 2:
        i, j = np.random.choice(idxs, 2, replace=False)
        same_dists.append(l2_distance(test_feats[i], test_feats[j]))
        same_count += 1
    
    # Different person
    l1, l2 = np.random.choice(np.unique(y_test), 2, replace=False)
    i = np.random.choice(np.where(y_test == l1)[0])
    j = np.random.choice(np.where(y_test == l2)[0])
    diff_dists.append(l2_distance(test_feats[i], test_feats[j]))
    diff_count += 1

print(f"   → {same_count} same-person pairs, {diff_count} different-person pairs\n")

all_dists = np.concatenate([same_dists, diff_dists])
pair_labels = np.concatenate([np.ones(len(same_dists)), np.zeros(len(diff_dists))])  # 1 = same, 0 = different

# AUC (higher score = more similar)
scores = -all_dists
auc = roc_auc_score(pair_labels, scores)
print(f"   Verification AUC: {auc:.4f}")

# Best threshold using Youden's J
fpr, tpr, thresholds = roc_curve(pair_labels, scores)
J = tpr - fpr
best_idx = np.argmax(J)
best_thresh_score = thresholds[best_idx]
best_thresh_dist = -best_thresh_score  # Convert back to distance (smaller dist = same)
preds = (all_dists < best_thresh_dist).astype(int)  # Predict same if distance below threshold
acc = accuracy_score(pair_labels, preds)
print(f"   Best distance threshold: {best_thresh_dist:.2f}")
print(f"   Verification Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

# Rank-1 (Vectorized - Fast!)
print("   Computing Rank-1 accuracy (vectorized)...")
dists_to_train = np.linalg.norm(test_feats[:, np.newaxis, :] - train_feats[np.newaxis, :, :], axis=2)
pred_labels = y_train[np.argmin(dists_to_train, axis=1)]
correct = np.sum(pred_labels == y_test)
rank1_acc = correct / len(y_test)
print(f"   Rank-1 Identification Accuracy: {rank1_acc:.4f} ({rank1_acc*100:.2f}%)\n")

# ====================== Final Summary ======================
total_time = time.time() - start_total
print("=" * 60)
print("                  FINAL RESULTS")
print("=" * 60)
print(f"Verification AUC           : {auc:.4f}")
print(f"Verification Accuracy      : {acc:.4f} ({acc*100:.2f}%)")
print(f"Rank-1 Accuracy            : {rank1_acc:.4f} ({rank1_acc*100:.2f}%)")
print(f"Total runtime              : {total_time:.1f} seconds")
print(f"Number of eigenfaces used  : {K}")
print("=" * 60)
print("Eigenfaces training and evaluation completed successfully!")