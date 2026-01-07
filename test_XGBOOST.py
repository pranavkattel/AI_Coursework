import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import os
import joblib

# ====================== CNN Feature Extractor ======================
class FaceCNN(nn.Module):
    def __init__(self, feature_dim=512):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * 7 * 7, feature_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ====================== Fast Dataset using PIL ======================
class FastFaceDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        
        self.base_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # To [-1, 1]
        ])
        
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L')  # Grayscale
        if self.aug_transform:
            img = self.aug_transform(img)
        img = self.base_transform(img)
        return img, self.labels[idx]


# ====================== Main Script ======================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load dataset
    dataset_root = 'VGGFace2'
    if not os.path.exists(dataset_root):
        print(f"ERROR: Folder '{dataset_root}' not found!")
        exit()

    full_dataset = datasets.ImageFolder(dataset_root)
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes.\n")

    max_classes = min(20, len(class_names))  # Increase if you want more
    print(f"Using {max_classes} classes for training.\n")

    paths = np.array([s[0] for s in full_dataset.samples])
    labels = np.array(full_dataset.targets)
    mask = labels < max_classes
    paths, labels = paths[mask], labels[mask]
    class_names = class_names[:max_classes]

    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}\n")

    # Fast data loaders
    batch_size = 64  # Increase to 128 if no OOM
    train_dataset = FastFaceDataset(train_paths, y_train, augment=True)
    test_dataset = FastFaceDataset(test_paths, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Model
    model = FaceCNN(feature_dim=512).to(device)
    temp_head = nn.Linear(512, max_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training with early stopping
    num_epochs = 2
    patience = 10
    best_acc = 0.0
    patience_counter = 0

    print("Starting fast training...\n")
    for epoch in range(num_epochs):
        model.train()
        temp_head.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            with autocast():
                features = model(images)
                logits = temp_head(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_face_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered!")
                break

    # Load best model and remove head
    model.load_state_dict(torch.load('best_face_cnn.pth'))
    model.fc = nn.Identity()
    print("\nBest model loaded. Extracting features...\n")

    # Feature extraction
    def extract_features(loader):
        model.eval()
        features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                feats = model(images)
                feats = F.normalize(feats, p=2, dim=1)  # L2 normalize
                features.append(feats.cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.vstack(features), np.array(all_labels)

    train_features, _ = extract_features(train_loader)
    test_features, y_test_true = extract_features(test_loader)
    print(f"Features extracted: {train_features.shape[0]} train, {test_features.shape[0]} test, dim={train_features.shape[1]}\n")

    # ==================== KNN Classifier (Instant & High Accuracy) ====================
    print("Training K-Nearest Neighbors classifier (cosine similarity)...")
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric='cosine',   # Best for normalized face embeddings
        n_jobs=-1
    )
    knn.fit(train_features, y_train)  # Extremely fast

    # Evaluation
    test_preds = knn.predict(test_features)
    test_probs = knn.predict_proba(test_features)

    acc = accuracy_score(y_test_true, test_preds)
    try:
        auc = roc_auc_score(y_test_true, test_probs, multi_class='ovr')
    except:
        auc = "N/A"

    # Rank-1 Accuracy (pure nearest neighbor)
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(test_features, train_features)
    pred_indices = np.argmax(sim_matrix, axis=1)
    rank1 = np.sum(y_train[pred_indices] == y_test_true) / len(y_test_true)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"KNN Classification Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Multi-class AUC             : {auc}")
    print(f"Rank-1 Accuracy (Key metric): {rank1:.4f} ({rank1*100:.2f}%)")
    print("="*70)

    # Save models
    torch.save(model.state_dict(), 'face_cnn_extractor.pth')
    joblib.dump(knn, 'knn_face_classifier.pkl')
    print("\nModels saved: 'face_cnn_extractor.pth' and 'knn_face_classifier.pkl'")

    # Optional: Single image prediction
    def predict_image(image_path):
        if not os.path.exists(image_path):
            print("Image not found!")
            return
        img = Image.open(image_path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            feat = model(img)
            feat = F.normalize(feat, p=2, dim=1).cpu().numpy()

        prob = knn.predict_proba(feat)[0]
        pred = np.argmax(prob)
        print(f"\nPrediction: {class_names[pred]} (confidence: {prob[pred]:.4f})")

    # Uncomment to test:
    # predict_image("test_image.jpg")