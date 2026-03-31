import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# ─── Fixed paths ───
SPECS_DIR = "/kaggle/working/specs/"
METADATA  = "/kaggle/input/datasets/imsparsh/fma-free-music-archive-small-medium/fma_metadata/"
WEIGHTS   = "/kaggle/working/weights/encoder.pth"
OUTPUT    = "/kaggle/working/"

# ─── Force both GPUs ───
torch.cuda.set_device(0)
device = torch.device('cuda')

# ─── Encoder ───
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x):
        return self.backbone(x)

encoder = AudioEncoder()
encoder.load_state_dict(torch.load(WEIGHTS, map_location=device))
encoder = nn.DataParallel(encoder, device_ids=[0, 1])
encoder = encoder.to(device)
encoder.eval()

# ─── Load metadata ───
tracks = pd.read_csv(os.path.join(METADATA, "tracks.csv"), index_col=0, header=[0, 1])
small = tracks[tracks[("set", "subset")] == "small"]

train_meta = small[small[("set", "split")] == "training"]
val_meta   = small[small[("set", "split")] == "validation"]
test_meta  = small[small[("set", "split")] == "test"]

def filter_existing(meta, specs_dir):
    valid = [tid for tid in meta.index if os.path.exists(os.path.join(specs_dir, f"{tid}.npy"))]
    return meta.loc[valid]

train_meta = filter_existing(train_meta, SPECS_DIR)
val_meta   = filter_existing(val_meta, SPECS_DIR)
test_meta  = filter_existing(test_meta, SPECS_DIR)

train_ids = train_meta.index.tolist()
val_ids   = val_meta.index.tolist()
test_ids  = test_meta.index.tolist()

train_labels = train_meta[("track", "genre_top")].values.astype(str)
val_labels   = val_meta[("track", "genre_top")].values.astype(str)
test_labels  = test_meta[("track", "genre_top")].values.astype(str)

# ═══════════════════════════════════════════════════════════
# Phase 4A — Extract all embeddings
# ═══════════════════════════════════════════════════════════
def extract_embeddings(track_ids, specs_dir, enc, dev, batch_size=128):
    all_embs = []
    batch_tensors = []
    for i, tid in enumerate(track_ids):
        spec = np.load(os.path.join(specs_dir, f"{tid}.npy")).astype(np.float32)
        crop = spec[:, :, 538:754].copy()  # (1, 128, 216)
        batch_tensors.append(torch.from_numpy(crop).unsqueeze(0))  # (1, 1, 128, 216)

        if len(batch_tensors) == batch_size or i == len(track_ids) - 1:
            batch = torch.cat(batch_tensors, dim=0).to(dev)  # (B, 1, 128, 216)
            with torch.no_grad():
                h = enc(batch)  # (B, 512)
            h = h.cpu().numpy()
            norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
            h = h / norms
            all_embs.append(h)
            batch_tensors = []

    return np.concatenate(all_embs, axis=0)

print("Extracting embeddings...")
train_embeddings = extract_embeddings(train_ids, SPECS_DIR, encoder, device)
val_embeddings   = extract_embeddings(val_ids, SPECS_DIR, encoder, device)
test_embeddings  = extract_embeddings(test_ids, SPECS_DIR, encoder, device)

print(f"Embeddings extracted. Train: {train_embeddings.shape[0]}, Val: {val_embeddings.shape[0]}, Test: {test_embeddings.shape[0]}")

# ═══════════════════════════════════════════════════════════
# Phase 4B — Linear Probe (10% of train)
# ═══════════════════════════════════════════════════════════
print("\n─── Linear Probe ───")
np.random.seed(42)
n_probe = max(1, int(0.10 * len(train_embeddings)))
probe_idx = np.random.choice(len(train_embeddings), size=n_probe, replace=False)

X_probe = train_embeddings[probe_idx]
y_probe = train_labels[probe_idx]

clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X_probe, y_probe)

val_preds = clf.predict(val_embeddings)

print(classification_report(val_labels, val_preds))

macro_f1 = f1_score(val_labels, val_preds, average='macro')
print(f"Linear Probe Macro F1: {macro_f1:.4f}")

# ═══════════════════════════════════════════════════════════
# Phase 4C — t-SNE Visualization
# ═══════════════════════════════════════════════════════════
print("\n─── t-SNE ───")
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
coords = tsne.fit_transform(val_embeddings)

genres = sorted(list(set(val_labels)))
cmap = plt.cm.tab10

fig, ax = plt.subplots(figsize=(14, 10))

for i, genre in enumerate(genres):
    mask = val_labels == genre
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=[cmap(i)], label=genre, s=25, alpha=0.75)

ax.set_title("EchoFind — Latent Space (t-SNE) | SimCLR on FMA-Small (No Labels Used)",
             fontsize=14, fontweight='bold')
ax.text(0.5, 1.02, f"Linear Probe Macro F1: {macro_f1:.4f} | Clean Retrieval Accuracy: 99.62%",
        transform=ax.transAxes, ha='center', va='bottom', fontsize=11, style='italic')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "tsne_embeddings.png"), dpi=300, bbox_inches='tight')
plt.show()
print("t-SNE saved to /kaggle/working/tsne_embeddings.png")

# ═══════════════════════════════════════════════════════════
# Extension B — OOD Detection (Mahalanobis)
# ═══════════════════════════════════════════════════════════
print("\n─── OOD Detection ───")

# Per-class mean vectors
unique_genres = sorted(list(set(train_labels)))
genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
class_means = np.zeros((len(unique_genres), 512), dtype=np.float64)

for i, genre in enumerate(unique_genres):
    mask = train_labels == genre
    class_means[i] = train_embeddings[mask].mean(axis=0)

# Shared covariance + regularization
cov = np.cov(train_embeddings.T)
cov += 1e-5 * np.eye(512)
cov_inv = np.linalg.inv(cov)

def mahalanobis_score(embedding):
    """Compute minimum Mahalanobis distance to any class mean."""
    dists = []
    for cm in class_means:
        d = mahalanobis(embedding.astype(np.float64), cm, cov_inv)
        dists.append(d)
    return min(dists)

# Compute threshold from train set
print("Computing Mahalanobis scores on train set...")
train_scores = np.array([mahalanobis_score(e) for e in train_embeddings])
threshold = np.percentile(train_scores, 95)
print(f"OOD threshold (95th pct): {threshold:.4f}")

def detect_ood(embedding):
    score = mahalanobis_score(embedding)
    if score >= threshold:
        return "Unknown/Anomaly", score
    # Predict genre as nearest class mean
    dists = []
    for cm in class_means:
        d = mahalanobis(embedding.astype(np.float64), cm, cov_inv)
        dists.append(d)
    nearest_idx = np.argmin(dists)
    return unique_genres[nearest_idx], score

# ─── Generate synthetic signals & convert to log-mel spec ───
SR = 22050
DURATION = 30  # seconds to match expected spec length
N_SAMPLES = SR * DURATION

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_mels=128, hop_length=512, n_fft=2048
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def signal_to_embedding(signal_np):
    """Convert raw 1D signal to L2-normed 512-d embedding."""
    waveform = torch.from_numpy(signal_np).float().unsqueeze(0)  # (1, N)
    mel = mel_transform(waveform)
    log_mel = amp_to_db(mel)  # (1, 128, T)
    T = log_mel.shape[2]
    # Pad or crop to 1292
    if T < 1292:
        log_mel = torch.nn.functional.pad(log_mel, (0, 1292 - T), mode="constant", value=log_mel.min().item())
    else:
        log_mel = log_mel[:, :, :1292]
    # Center crop
    crop = log_mel[:, :, 538:754]  # (1, 128, 216)
    tensor = crop.unsqueeze(0).to(device)  # (1, 1, 128, 216)
    with torch.no_grad():
        h = encoder(tensor)
    h = h.cpu().numpy().flatten()
    h = h / (np.linalg.norm(h) + 1e-8)
    return h

# White noise
white_noise = np.random.randn(N_SAMPLES).astype(np.float32) * 0.5
emb_noise = signal_to_embedding(white_noise)
result_noise, score_noise = detect_ood(emb_noise)
print(f"\n[White Noise]:   detect_ood → \"{result_noise}\" (score: {score_noise:.4f})")

# 440Hz sine wave
t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
sine_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
emb_sine = signal_to_embedding(sine_440)
result_sine, score_sine = detect_ood(emb_sine)
print(f"[440Hz Sine]:    detect_ood → \"{result_sine}\" (score: {score_sine:.4f})")

# Linear chirp (20Hz → 8000Hz)
chirp_signal = np.sin(2 * np.pi * (20 + (8000 - 20) / (2 * DURATION) * t) * t).astype(np.float32) * 0.5
emb_chirp = signal_to_embedding(chirp_signal)
result_chirp, score_chirp = detect_ood(emb_chirp)
print(f"[Linear Chirp]:  detect_ood → \"{result_chirp}\" (score: {score_chirp:.4f})")
