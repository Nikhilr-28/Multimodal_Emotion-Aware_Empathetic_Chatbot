"""
train.py
Train ConvNeXt-Base or EfficientNetV2-M on FER-2013, RAF-DB, AffectNet.
Usage:
    python train.py --model convnext_base
    python train.py --model efficientnetv2_m
    python train.py --model convnext_base --epochs 5  # sanity check
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
import timm
from tqdm import tqdm

from preprocess import get_datasets, EMOTION_LABELS, NUM_CLASSES


CONFIGS = {
    "convnext_base": {
        "timm_name":          "convnext_base.fb_in22k_ft_in1k",
        "batch_h100":         256,
        "batch_local":        48,
        "num_workers_h100":   16,
        "num_workers_local":  6,
        "lr":                 3e-4,
        "mixup_alpha":        0.4,
        "pct_start":          0.05,
    },
    "efficientnetv2_m": {
        "timm_name":          "tf_efficientnetv2_m.in21k_ft_in1k",
        "batch_h100":         320,
        "batch_local":        48,
        "num_workers_h100":   16,
        "num_workers_local":  6,
        "lr":                 4e-4,
        "mixup_alpha":        0.3,
        "pct_start":          0.05,
    },
}

TRAIN_CONFIG = {
    "backbone_lr_scale":   0.1,
    "weight_decay":        0.1,
    "epochs":              120,
    "early_stop_patience": 20,
    "image_size":          224,
    "label_smoothing":     0.1,
    "dropout":             0.3,
}

BASE           = r"A:\MHC_535\project\Multimodal_Emotion_Aware_Empathetic_Chatbot"
FER_ROOT       = os.path.join(BASE, "FER-2013")
RAFDB_ROOT     = os.path.join(BASE, "RAF-DB Dataset")
AFFECTNET_ROOT = os.path.join(BASE, "affectnet")


def get_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.alpha,
                               reduction="none", label_smoothing=self.label_smoothing)
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def mixup_batch(images, labels, alpha, device):
    lam      = torch.distributions.Beta(alpha, alpha).sample().item()
    idx      = torch.randperm(images.size(0)).to(device)
    mixed    = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, mixup_alpha):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha, device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss    = lam * criterion(outputs, labels_a) + \
                      (1 - lam) * criterion(outputs, labels_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels_a.size(0)
        # accuracy against the dominant label for logging only
        correct    += outputs.argmax(1).eq(labels_a).sum().item()
        total      += labels_a.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=CONFIGS.keys())
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--h100",   action="store_true")
    args = parser.parse_args()

    cfg     = CONFIGS[args.model]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_h100 = args.h100 and torch.cuda.is_available()

    batch_size  = cfg["batch_h100"]   if is_h100 else cfg["batch_local"]
    num_workers = cfg["num_workers_h100"] if is_h100 else cfg["num_workers_local"]
    epochs      = args.epochs if args.epochs else TRAIN_CONFIG["epochs"]

    print(f"\nModel      : {args.model}")
    print(f"Device     : {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {epochs}\n")

    train_tf, test_tf = get_transforms(TRAIN_CONFIG["image_size"])
    train_ds, val_ds, test_ds, class_counts = get_datasets(
        fer_root=FER_ROOT,
        rafdb_root=RAFDB_ROOT,
        affectnet_root=AFFECTNET_ROOT,
        train_transform=train_tf,
        test_transform=test_tf,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # dropout applied to classifier head via drop_rate
    model = timm.create_model(cfg["timm_name"], pretrained=True,
                               num_classes=NUM_CLASSES,
                               drop_rate=TRAIN_CONFIG["dropout"])
    model = model.to(device)

    class_weights = (1.0 / class_counts.float())
    class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0,
                          label_smoothing=TRAIN_CONFIG["label_smoothing"])

    # differential LR: backbone gets 10x smaller LR than head
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params     = [p for n, p in model.named_parameters() if "head" in n]
    optimizer = AdamW([
    {"params": backbone_params, "lr": cfg["lr"] * TRAIN_CONFIG["backbone_lr_scale"]},
    {"params": head_params,     "lr": cfg["lr"]},
    ], weight_decay=TRAIN_CONFIG["weight_decay"])

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[cfg["lr"] * TRAIN_CONFIG["backbone_lr_scale"], cfg["lr"]],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=cfg["pct_start"],
        anneal_strategy="cos"
    )

    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir  = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")

    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                    optimizer, scheduler, device, scaler,
                                    cfg["mixup_alpha"])
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)

        print(f"  train loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  val   loss: {val_loss:.4f}  acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch":      epoch,
                "model":      args.model,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "val_loss":   val_loss,
            }, ckpt_path)
            print(f"  checkpoint saved (val_acc={val_acc:.2f}%)")
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN_CONFIG["early_stop_patience"]:
                print(f"\nEarly stopping — no improvement for {patience_ctr} epochs.")
                break

    print("\nLoading best checkpoint for test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss : {test_loss:.4f}  Test acc : {test_acc:.2f}%")
    print(f"Best val  : {checkpoint['val_acc']:.2f}% at epoch {checkpoint['epoch']}")


if __name__ == "__main__":
    main()