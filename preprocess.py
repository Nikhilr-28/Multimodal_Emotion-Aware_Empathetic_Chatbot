"""
preprocess.py
Unified dataset loading for FER-2013, RAF-DB, and AffectNet.
4-class schema: 0=happiness, 1=sadness, 2=anger, 3=fear
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import defaultdict
from torch.utils.data import Dataset, ConcatDataset, Subset
from PIL import Image
import torch


NUM_CLASSES = 4

EMOTION_LABELS = {
    0: "happiness",
    1: "sadness",
    2: "anger",
    3: "fear"
}

EMOTION_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}


# None = drop that class
FER_LABEL_MAP = {
    "happy":    0,
    "sad":      1,
    "angry":    2,
    "fear":     3,
    "disgust":  None,
    "neutral":  None,
    "surprise": None,
}

# RAF-DB folders are numbered: 1=Surprise, 2=Fear, 3=Disgust,
# 4=Happy, 5=Sad, 6=Anger, 7=Neutral
RAFDB_LABEL_MAP = {
    1: None,  # surprise - drop
    2: 3,     # fear
    3: None,  # disgust  - drop
    4: 0,     # happiness
    5: 1,     # sadness
    6: 2,     # anger
    7: None,  # neutral  - drop
}

# AffectNet folder names are lowercased before lookup
# because Test/ uses mixed case (Anger vs anger)
AFFECTNET_LABEL_MAP = {
    "happy":    0,
    "sad":      1,
    "anger":    2,
    "fear":     3,
    "contempt": None,
    "disgust":  None,
    "neutral":  None,
    "surprise": None,
}


class FER2013Dataset(Dataset):
    # root expects: root/train/<emotion>/ and root/test/<emotion>/
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ("train", "test")
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(root_dir, split)
        for folder_name, label in FER_LABEL_MAP.items():
            if label is None:
                continue
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, img_name), label))

        print(f"FER-2013   [{split:5s}]: {len(self.samples):6d} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # FER is grayscale natively
        if self.transform:
            image = self.transform(image)
        return image, label


class RAFDBDataset(Dataset):
    # root expects: root/DATASET/train/<1-7>/ and root/DATASET/test/<1-7>/
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ("train", "test")
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(root_dir, "DATASET", split)
        for folder_num, label in RAFDB_LABEL_MAP.items():
            if label is None:
                continue
            folder_path = os.path.join(split_dir, str(folder_num))
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
                    self.samples.append((os.path.join(folder_path, img_name), label))

        print(f"RAF-DB     [{split:5s}]: {len(self.samples):6d} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class AffectNetDataset(Dataset):
    # root expects: root/Train/<emotion>/ and root/Test/<emotion>/
    # split must be "Train" or "Test" (capital T to match folder names)
    def __init__(self, root_dir, split="Train", transform=None):
        assert split in ("Train", "Test")
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(root_dir, split)
        for folder_name in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = AFFECTNET_LABEL_MAP.get(folder_name.lower())
            if label is None:
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, img_name), label))

        print(f"AffectNet  [{split:5s}]: {len(self.samples):6d} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _balance_indices(datasets, max_per_class, seed):
    """
    Given a list of datasets with .samples, collect all flat indices
    grouped by class and cap each class at max_per_class.
    Returns a list of balanced flat indices into the concatenated dataset.
    """
    class_indices = defaultdict(list)
    offset = 0
    for ds in datasets:
        for i, (_, label) in enumerate(ds.samples):
            class_indices[label].append(offset + i)
        offset += len(ds.samples)

    rng = torch.Generator().manual_seed(seed)
    balanced = []
    for label in sorted(class_indices.keys()):
        idxs = torch.tensor(class_indices[label])
        perm = torch.randperm(len(idxs), generator=rng)
        selected = idxs[perm[:max_per_class]].tolist()
        balanced.extend(selected)

    return balanced


def get_datasets(fer_root, rafdb_root, affectnet_root,
                 train_transform=None, test_transform=None,
                 val_fraction=0.2, seed=42,
                 balance=True):
    """
    Returns train_dataset, val_dataset, test_dataset, class_counts.

    Val is carved from training indices but uses test_transform.
    class_counts is a [4] LongTensor for FocalLoss weighting.

    If balance=True, training data is capped per class at the size of
    the smallest class to prevent majority-class overfitting.
    Test set is never balanced — evaluated on natural distribution.
    """
    print("\n" + "-" * 40)

    fer_train       = FER2013Dataset(fer_root,         split="train", transform=train_transform)
    rafdb_train     = RAFDBDataset(rafdb_root,         split="train", transform=train_transform)
    affectnet_train = AffectNetDataset(affectnet_root,  split="Train", transform=train_transform)

    fer_val         = FER2013Dataset(fer_root,         split="train", transform=test_transform)
    rafdb_val       = RAFDBDataset(rafdb_root,         split="train", transform=test_transform)
    affectnet_val   = AffectNetDataset(affectnet_root,  split="Train", transform=test_transform)

    raw_datasets    = [fer_train, rafdb_train, affectnet_train]
    combined_train  = ConcatDataset(raw_datasets)
    combined_val    = ConcatDataset([fer_val, rafdb_val, affectnet_val])

    # Class counts before balancing
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for ds in raw_datasets:
        for _, label in ds.samples:
            class_counts[label] += 1

    print("Class counts (before balancing):")
    for idx, name in EMOTION_LABELS.items():
        print(f"  {idx} {name:10s}: {class_counts[idx]}")

    if balance:
        max_per_class = class_counts.min().item()
        print(f"Balancing: capping each class at {max_per_class} samples")
        balanced_indices = _balance_indices(raw_datasets, max_per_class, seed)

        # Apply same indices to both transform versions
        combined_train = Subset(ConcatDataset(raw_datasets), balanced_indices)
        combined_val   = Subset(ConcatDataset([fer_val, rafdb_val, affectnet_val]),
                                balanced_indices)

        # Recompute class counts after balancing
        class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        for label in range(NUM_CLASSES):
            class_counts[label] = max_per_class
    else:
        combined_train = ConcatDataset(raw_datasets)
        combined_val   = ConcatDataset([fer_val, rafdb_val, affectnet_val])

    # Val split from balanced training indices
    n          = len(combined_train)
    val_size   = int(n * val_fraction)
    train_size = n - val_size
    indices    = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    train_dataset = Subset(combined_train, indices[:train_size])
    val_dataset   = Subset(combined_val,   indices[train_size:])

    # Test set — never balanced
    fer_test       = FER2013Dataset(fer_root,         split="test", transform=test_transform)
    rafdb_test     = RAFDBDataset(rafdb_root,         split="test", transform=test_transform)
    affectnet_test = AffectNetDataset(affectnet_root,  split="Test", transform=test_transform)
    test_dataset   = ConcatDataset([fer_test, rafdb_test, affectnet_test])

    print(f"\nTrain: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")
    print("Class counts (after balancing):" if balance else "Class counts:")
    for idx, name in EMOTION_LABELS.items():
        print(f"  {idx} {name:10s}: {class_counts[idx]}")
    print("-" * 40 + "\n")

    return train_dataset, val_dataset, test_dataset, class_counts


def get_unbalanced_datasets(fer_root, rafdb_root, affectnet_root,
                            train_transform=None, test_transform=None,
                            val_fraction=0.2, seed=42):
    """
    Same as get_datasets but with balance=False.
    Returns train_dataset, val_dataset, test_dataset, class_counts
    on the natural class distribution.
    """
    return get_datasets(
        fer_root=fer_root,
        rafdb_root=rafdb_root,
        affectnet_root=affectnet_root,
        train_transform=train_transform,
        test_transform=test_transform,
        val_fraction=val_fraction,
        seed=seed,
        balance=False,
    )


def _label_from_concat(dataset, flat_idx):
    # Handle Subset wrapping a ConcatDataset
    if isinstance(dataset, Subset):
        actual_idx = dataset.indices[flat_idx]
        return _label_from_concat(dataset.dataset, actual_idx)
    # Handle ConcatDataset
    for ds in dataset.datasets:
        if flat_idx < len(ds):
            return ds.samples[flat_idx][1]
        flat_idx -= len(ds)


def print_split_distribution(train_ds, val_ds, test_ds):
    for split_name, subset in [("train", train_ds), ("val", val_ds)]:
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        for i in subset.indices:
            label = _label_from_concat(subset.dataset, i)
            counts[label] += 1
        print(f"\n{split_name}:")
        for idx, name in EMOTION_LABELS.items():
            print(f"  {name:10s}: {counts[idx]}")

    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for ds in test_ds.datasets:
        for _, label in ds.samples:
            counts[label] += 1
    print("\ntest:")
    for idx, name in EMOTION_LABELS.items():
        print(f"  {name:10s}: {counts[idx]}")


if __name__ == "__main__":
    BASE           = r"A:\MHC_535\project\Multimodal_Emotion_Aware_Empathetic_Chatbot"
    FER_ROOT       = os.path.join(BASE, "FER-2013")
    RAFDB_ROOT     = os.path.join(BASE, "RAF-DB Dataset")
    AFFECTNET_ROOT = os.path.join(BASE, "affectnet")

    train_ds, val_ds, test_ds, class_counts = get_datasets(
        fer_root=FER_ROOT,
        rafdb_root=RAFDB_ROOT,
        affectnet_root=AFFECTNET_ROOT,
        balance=True,
    )

    print_split_distribution(train_ds, val_ds, test_ds)