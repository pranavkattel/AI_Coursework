import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
import os

# ======================
# Model Definition (same as training)
# ======================

class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        expanded_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv1 = torch.nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(expanded_channels)
        self.prelu1 = torch.nn.PReLU(expanded_channels)

        self.dwconv = torch.nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(expanded_channels)
        self.prelu2 = torch.nn.PReLU(expanded_channels)

        self.conv3 = torch.nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv3(out))
        if self.use_residual:
            out = out + x
        return out

class MobileFaceNet(torch.nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.prelu1 = torch.nn.PReLU(64)

        self.dwconv1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.bn_dw1 = torch.nn.BatchNorm2d(64)
        self.prelu_dw1 = torch.nn.PReLU(64)

        self.bottlenecks = torch.nn.Sequential(
            Bottleneck(64, 64, expansion=2, stride=2),
            *[Bottleneck(64, 64, expansion=2, stride=1) for _ in range(4)],
            Bottleneck(64, 128, expansion=4, stride=2),
            *[Bottleneck(128, 128, expansion=2, stride=1) for _ in range(6)],
            Bottleneck(128, 128, expansion=4, stride=2),
            *[Bottleneck(128, 128, expansion=2, stride=1) for _ in range(2)],
        )

        self.conv_last = torch.nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.bn_last = torch.nn.BatchNorm2d(512)
        self.prelu_last = torch.nn.PReLU(512)

        self.gdconv = torch.nn.Conv2d(512, 512, kernel_size=7, groups=512, bias=False)
        self.bn_gd = torch.nn.BatchNorm2d(512)

        self.linear = torch.nn.Linear(512, embedding_size, bias=False)
        self.bn_embed = torch.nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu_dw1(self.bn_dw1(self.dwconv1(out)))
        out = self.bottlenecks(out)
        out = self.prelu_last(self.bn_last(self.conv_last(out)))
        out = self.bn_gd(self.gdconv(out))
        out = out.view(out.size(0), -1)
        out = self.bn_embed(self.linear(out))
        return out

# ======================
# Dataset Wrapper
# ======================

class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

# ======================
# Main Evaluation Script
# ======================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MobileFaceNet(embedding_size=512).to(device)
    model_path = 'mobilefacenet_arcface.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found! Train the model first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Transforms (same as validation)
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load dataset and split exactly like training
    dataset_root = 'VGGFace2'  # CHANGE IF YOUR PATH IS DIFFERENT
    full_dataset = datasets.ImageFolder(dataset_root, transform=None)
    
    num_classes = 10
    all_labels = np.array(full_dataset.targets)
    selected_indices = np.where(all_labels < num_classes)[0]
    selected_dataset = Subset(full_dataset, selected_indices)
    labels = all_labels[selected_indices]

    # Reproduce the same 80/20 stratified split
    val_indices = []
    for class_id in range(num_classes):
        class_idx = np.where(labels == class_id)[0]
        if len(class_idx) > 0:
            _, val_idx = train_test_split(class_idx, test_size=0.2, random_state=42)
            val_indices.extend(val_idx)

    val_subset = Subset(selected_dataset, val_indices)
    val_dataset = TransformDataset(val_subset, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(val_dataset)} validation images from {num_classes} classes.")

    # Extract embeddings
    embeddings = []
    emb_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu().numpy())
            emb_labels.extend(labels.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)  # Shape: [N, 512]
    emb_labels = np.array(emb_labels)

    # Normalize embeddings (important for cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Generate pairs: 5000 same + 5000 different (adjust number if dataset is small)
    n_pairs = 5000
    same_sims = []
    diff_sims = []

    np.random.seed(42)
    for _ in range(n_pairs):
        # Same identity
        label = np.random.choice(np.unique(emb_labels))
        idxs = np.where(emb_labels == label)[0]
        if len(idxs) >= 2:
            i, j = np.random.choice(idxs, 2, replace=False)
            sim = np.dot(embeddings[i], embeddings[j])
            same_sims.append(sim)
            

        # Different identity
        l1, l2 = np.random.choice(np.unique(emb_labels), 2, replace=False)
        i = np.random.choice(np.where(emb_labels == l1)[0])
        j = np.random.choice(np.where(emb_labels == l2)[0])
        sim = np.dot(embeddings[i], embeddings[j])
        diff_sims.append(sim)

    # Combine
    similarities = np.array(same_sims + diff_sims)
    true_labels = np.array([1] * len(same_sims) + [0] * len(diff_sims))

    # Metrics
    auc = roc_auc_score(true_labels, similarities)
    print(f"\nVerification AUC: {auc:.4f}")

    # Find best threshold
    fpr, tpr, thresholds = roc_curve(true_labels, similarities)
    # Youden's J statistic
    J = tpr - fpr
    best_thresh_idx = np.argmax(J)
    best_threshold = thresholds[best_thresh_idx]
    best_acc = accuracy_score(true_labels, (similarities > best_threshold).astype(int))

    print(f"Best threshold (Youden): {best_threshold:.3f}")
    print(f"Verification Accuracy at best threshold: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # Common thresholds
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        acc = accuracy_score(true_labels, (similarities > thresh).astype(int))
        print(f"Accuracy @ threshold {thresh:.1f}: {acc:.4f}")

    # Optional: Rank-1 Identification Accuracy
    print("\nComputing Rank-1 Identification Accuracy...")
    correct = 0
    for i in range(len(embeddings)):
        query_emb = embeddings[i]
        query_label = emb_labels[i]
        # Cosine similarity to all others (excluding self)
        sims = np.dot(embeddings, query_emb)
        sims[i] = -1  # exclude self
        pred_idx = np.argmax(sims)
        if emb_labels[pred_idx] == query_label:
            correct += 1
    rank1_acc = correct / len(embeddings)
    print(f"Rank-1 Identification Accuracy: {rank1_acc:.4f} ({rank1_acc*100:.2f}%)")

    print("\nEvaluation complete!")