import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm
import torch.nn.functional as F


# =========================
# Preprocess for Inception
# =========================
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Load and extract features for FID
# =========================
def extract_features(folder, inception, device):
    feats = []
    files = sorted(os.listdir(folder))
    for fname in tqdm(files, desc=f"Extracting features from {folder}"):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB")
        except:
            continue
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = inception(img).squeeze()  # 2048-dim feature
        feats.append(feat.cpu().numpy())
    feats = np.stack(feats, axis=0)
    return feats


# =========================
# Compute FID
# =========================
def compute_fid(fake_folder, real_folder, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    inception.fc = nn.Identity()  # output 2048-dim features
    inception.to(device)
    inception.eval()

    real_feats = extract_features(real_folder, inception, device)
    fake_feats = extract_features(fake_folder, inception, device)

    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)
    print(f"[FID] = {fid:.4f}")
    return float(fid)


# =========================
# Compute Inception Score
# =========================
def compute_inception_score(folder, device=None, splits=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    inception.to(device)
    inception.eval()

    files = sorted(os.listdir(folder))
    preds = []

    for fname in tqdm(files, desc="Computing Inception Score"):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB")
        except:
            continue
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = inception(img)
            p = F.softmax(logits, dim=1)
            preds.append(p.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    N = preds.shape[0]

    eps = 1e-16
    split_scores = []
    for k in range(splits):
        start = k * N // splits
        end = (k + 1) * N // splits
        part = preds[start:end]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        kl = np.mean(np.sum(kl, axis=1))
        split_scores.append(np.exp(kl))

    mean_is = float(np.mean(split_scores))
    std_is  = float(np.std(split_scores))
    print(f"[Inception Score] = {mean_is:.4f} ± {std_is:.4f}")
    return mean_is, std_is


# =========================
# Main
# =========================
if __name__ == "__main__":
    fake_folder = "logs/KAN/images/fake"    # <<< folder ảnh sinh ra
    real_folder = "data/cifar10/test"   # <<< folder ảnh thật

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute FID
    fid_score = compute_fid(fake_folder, real_folder, device=device)

    # Compute Inception Score
    mean_is, std_is = compute_inception_score(fake_folder, device=device)
