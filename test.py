import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error tracing
torch.autograd.set_detect_anomaly(True)  # Detect NaN/Inf

# Set multiprocessing start method for Windows
multiprocessing.set_start_method('spawn', force=True)

# Bottleneck block (depthwise separable with expansion)
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        expanded_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.prelu1 = nn.PReLU(expanded_channels)

        self.dwconv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.prelu2 = nn.PReLU(expanded_channels)

        self.conv3 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv3(out))
        if self.use_residual:
            out = out + x
        return out

# MobileFaceNet architecture (based on the paper)
class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)

        # Depthwise conv after initial (stride=1)
        self.dwconv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.bn_dw1 = nn.BatchNorm2d(64)
        self.prelu_dw1 = nn.PReLU(64)

        # Bottleneck layers
        self.bottlenecks = nn.Sequential(
            Bottleneck(64, 64, expansion=2, stride=2),
            *[Bottleneck(64, 64, expansion=2, stride=1) for _ in range(4)],
            Bottleneck(64, 128, expansion=4, stride=2),
            *[Bottleneck(128, 128, expansion=2, stride=1) for _ in range(6)],
            Bottleneck(128, 128, expansion=4, stride=2),
            *[Bottleneck(128, 128, expansion=2, stride=1) for _ in range(2)],
        )

        # Final 1x1 conv to 512
        self.conv_last = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(512)
        self.prelu_last = nn.PReLU(512)

        # Global Depthwise Conv (GDConv 7x7)
        self.gdconv = nn.Conv2d(512, 512, kernel_size=7, groups=512, bias=False)
        self.bn_gd = nn.BatchNorm2d(512)

        # Final linear to embedding
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn_embed = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu_dw1(self.bn_dw1(self.dwconv1(out)))
        out = self.bottlenecks(out)
        out = self.prelu_last(self.bn_last(self.conv_last(out)))
        out = self.bn_gd(self.gdconv(out))
        out = out.view(out.size(0), -1)  # Flatten
        out = self.bn_embed(self.linear(out))
        return out

# Wrapper for applying transforms to subsets
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

if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if running as frozen executable (e.g., PyInstaller)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load full dataset without transform
    dataset_root = 'VGGFace2'  # Change this to your actual dataset path if different
    full_dataset = datasets.ImageFolder(dataset_root, transform=None)

    # Subset to only the first 10 classes
    num_classes = 10  # Hardcoded to 10 as requested
    all_labels = np.array(full_dataset.targets)
    selected_indices = np.where(all_labels < num_classes)[0]
    selected_dataset = Subset(full_dataset, selected_indices)
    labels = all_labels[selected_indices]  # Updated labels for selected classes
    print(f"Using {num_classes} classes, total samples: {len(selected_dataset)}")  # Debug

    # Stratified split per class (80/20)
    train_indices = []
    val_indices = []

    for class_id in range(num_classes):
        class_idx = np.where(labels == class_id)[0]
        if len(class_idx) > 0:
            train_idx, val_idx = train_test_split(class_idx, test_size=0.2, random_state=42)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

    # Create subsets and apply transforms
    train_subset = Subset(selected_dataset, train_indices)
    val_subset = Subset(selected_dataset, val_indices)

    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, val_transform)

    # DataLoaders - Set num_workers=0 to avoid shared memory issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileFaceNet(embedding_size=512).to(device)
    arcface_loss = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=512, margin=0.5, scale=64.0).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    num_epochs = 50  # Adjust based on monitoring
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape {images.shape}, Labels min/max {labels.min()}/{labels.max()}")  # Debug labels
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            loss = arcface_loss(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)
                val_loss += arcface_loss(embeddings, labels).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'mobilefacenet_arcface.pth')
    print("Training complete. Model saved as 'mobilefacenet_arcface.pth'.")