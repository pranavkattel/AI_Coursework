import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import datasets
from PIL import Image
import os

if not hasattr(cv2, 'face'):
    print("Install opencv-contrib-python!")
    exit()

# ====================== Main ======================
if __name__ == '__main__':
    print(f"OpenCV version: {cv2.__version__}\n")

    # Load data
    dataset_root = 'VGGFace2'
    full_dataset = datasets.ImageFolder(dataset_root)
    targets = np.array(full_dataset.targets)
    mask = targets < 10
    selected_paths = [full_dataset.samples[i][0] for i in range(len(full_dataset)) if mask[i]]
    selected_labels = targets[mask]
    
    class_names = sorted(full_dataset.classes)[:10]
    print("=== Selected Classes ===")
    for i, name in enumerate(class_names):
        count = np.sum(selected_labels == i)
        print(f"Class {i}: {name} ({count} images)")
    print()
    
    train_paths, test_paths, y_train, y_test = train_test_split(
        selected_paths, selected_labels, test_size=0.2, random_state=42, stratify=selected_labels
    )
    
    # Load + preprocess (smaller size + equalization)
    IMG_SIZE = 80  # Reduced from 112 to avoid memory error
    def load_images(paths):
        images = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img)  # Better contrast
            images.append(img)
        return images

    print("Loading train images...")
    train_images = load_images(train_paths)
    print(f"Loaded {len(train_images)} train images")
    
    print("Loading test images...")
    test_images = load_images(test_paths)
    print(f"Loaded {len(test_images)} test images\n")

    # LBPH with memory-friendly params
    lbph = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=5,   # Reduced grid for less memory
        grid_y=5,
        threshold=100.0
    )
    
    print("Training LBPH (this should now work)...")
    lbph.train(train_images, np.array(y_train))
    lbph.save('lbph_fixed.yml')
    print("Training complete!\n")
    
    # Evaluation
    print("Evaluating...")
    preds = []
    confidences = []
    for img in test_images:
        label, conf = lbph.predict(img)
        preds.append(label)
        confidences.append(conf)
    
    # Rank-1 (raw)
    rank1 = accuracy_score(y_test, preds)
    print(f"Rank-1 Accuracy: {rank1:.4f} ({rank1*100:.2f}%)")
    
    # With threshold
    threshold = 90
    valid_preds = [p if c < threshold else -1 for p, c in zip(preds, confidences)]
    acc_thresh = sum(1 for i, p in enumerate(valid_preds) if p == y_test[i] and p != -1) / len(y_test)
    print(f"Accuracy with threshold {threshold}: {acc_thresh:.4f}")
    
    # Verification AUC (proxy)
    n_pairs = 1000
    same = []
    diff = []
    np.random.seed(42)
    for _ in range(n_pairs):
        lbl = np.random.choice(np.unique(y_test))
        idxs = np.where(y_test == lbl)[0]
        if len(idxs) >= 2:
            i, j = np.random.choice(idxs, 2, replace=False)
            same.append(abs(confidences[i] - confidences[j]))
        l1, l2 = np.random.choice(np.unique(y_test), 2, replace=False)
        i = np.random.choice(np.where(y_test == l1)[0])
        j = np.random.choice(np.where(y_test == l2)[0])
        diff.append(abs(confidences[i] - confidences[j]))
    
    dists = np.array(same + diff)
    labels = np.array([1]*len(same) + [0]*len(diff))
    auc = roc_auc_score(labels, -dists)
    print(f"Verification AUC: {auc:.4f}")

    # Custom test
    def test_custom(image_path, threshold=90):
        print(f"\nTesting: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Not found!")
            return
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        label, conf = lbph.predict(img)
        if conf < threshold:
            print(f"Match! Class {label} - {class_names[label]} (distance: {conf:.2f})")
        else:
            print(f"No match (distance: {conf:.2f})")

    test_custom('test_image.jpg', threshold=90)