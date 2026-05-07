"""
plot_results.py
Generates normalized confusion matrix and per-class F1 bar chart
for ConvNeXt-Base only. Saves both to results/ folder.

Usage:
    python plot_results.py
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from preprocess import get_datasets, EMOTION_LABELS, NUM_CLASSES

BASE           = r"A:/MHC_535/project/Multimodal_Emotion_Aware_Empathetic_Chatbot"
FER_ROOT       = os.path.join(BASE, "FER-2013")
RAFDB_ROOT     = os.path.join(BASE, "RAF-DB Dataset")
AFFECTNET_ROOT = os.path.join(BASE, "affectnet")
CKPT_PATH      = os.path.join(BASE, "scripts/checkpoints/convnext_base_best.pth")
OUTPUT_DIR     = "results"
IMAGE_SIZE     = 224
BATCH_SIZE     = 64
NUM_WORKERS    = 6
CLASS_NAMES    = list(EMOTION_LABELS.values())


def get_test_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="  inference", leave=False):
        images = images.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_title("ConvNeXt-Base — Confusion Matrix (normalized)", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "convnext_confusion_matrix_norm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_class_f1(y_true, y_pred):
    f1s    = f1_score(y_true, y_pred, average=None,
                      labels=list(range(NUM_CLASSES))) * 100
    colors = ["#4CAF50", "#2196F3", "#F44336", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_NAMES, f1s, color=colors, edgecolor="white", width=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 Score (%)", fontsize=11)
    ax.set_title("ConvNeXt-Base — Per-Class F1 Score", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, f1s):  # type: ignore[arg-type]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "convnext_per_class_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_tf = get_test_transform()
    _, _, test_ds, _ = get_datasets(
        fer_root=FER_ROOT,
        rafdb_root=RAFDB_ROOT,
        affectnet_root=AFFECTNET_ROOT,
        test_transform=test_tf,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)
    print(f"Test set: {len(test_ds)} images")

    model = timm.create_model("convnext_base.fb_in22k_ft_in1k",
                               pretrained=False, num_classes=NUM_CLASSES,
                               drop_rate=0.3)
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    print(f"Loaded checkpoint — val_acc={checkpoint['val_acc']:.2f}% "
          f"at epoch {checkpoint['epoch']}")

    y_true, y_pred = run_inference(model, test_loader, device)

    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)}")

    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_f1(y_true, y_pred)
    print("\nDone.")


if __name__ == "__main__":
    main()