# !pip install faiss-cpu -q

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import faiss
from scipy.signal import fftconvolve

# ─── Fixed paths ───
SPECS_DIR = "/kaggle/working/specs/"
METADATA  = "/kaggle/input/datasets/imsparsh/fma-free-music-archive-small-medium/fma_metadata/"
WEIGHTS   = "/kaggle/working/weights/encoder.pth"
OUTPUT    = "/kaggle/working/"

# ─── Force both GPUs ───
torch.cuda.set_device(0)
device = torch.device('cuda')

# ─── Encoder (must match training architecture exactly) ───
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x):
        return self.backbone(x)

# ─── Load encoder with DataParallel on both GPUs ───
encoder = AudioEncoder()
encoder.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
encoder = nn.DataParallel(encoder, device_ids=[0, 1])
encoder = encoder.to(device)
encoder.eval()

# ─── Load metadata & get test split ───
tracks = pd.read_csv(os.path.join(METADATA, "tracks.csv"), index_col=0, header=[0, 1])
small = tracks[tracks[("set", "subset")] == "small"]
test_ids = small[small[("set", "split")] == "test"].index.tolist()
test_ids = [tid for tid in test_ids if os.path.exists(os.path.join(SPECS_DIR, f"{tid}.npy"))]
print(f"Valid test IDs: {len(test_ids)}")

# ─── Helper: extract embedding from spec array ───
def extract_embedding(spec_np, encoder_model, dev):
    """spec_np: (1, 128, 1292) numpy array. Returns L2-normed 512-d numpy vector."""
    crop = spec_np[:, :, 538:754].copy()  # center 5s crop → (1, 128, 216)
    tensor = torch.from_numpy(crop).float().unsqueeze(0).to(dev)  # (1, 1, 128, 216)
    with torch.no_grad():
        h = encoder_model(tensor)  # (1, 512)
    h = h.cpu().numpy().flatten()  # (512,)
    h = h / (np.linalg.norm(h) + 1e-8)
    return h

# ═══════════════════════════════════════════════════════════
# Step 1 — Build FAISS index from test split
# ═══════════════════════════════════════════════════════════
print("Building FAISS index...")
dim = 512
index = faiss.IndexFlatIP(dim)
id_list = []

for i, tid in enumerate(test_ids):
    spec = np.load(os.path.join(SPECS_DIR, f"{tid}.npy")).astype(np.float32)
    emb = extract_embedding(spec, encoder, device)
    index.add(emb.reshape(1, -1).astype(np.float32))
    id_list.append(tid)

faiss.write_index(index, os.path.join(OUTPUT, "faiss_index.bin"))
np.save(os.path.join(OUTPUT, "id_map.npy"), np.array(id_list))
print(f"Index built. {index.ntotal} tracks indexed.")

# ═══════════════════════════════════════════════════════════
# Step 2 — query() function
# ═══════════════════════════════════════════════════════════
def query(spec_npy_path, add_noise=True):
    spec = np.load(spec_npy_path).astype(np.float32)  # (1, 128, 1292)
    crop = spec[:, :, 538:754].copy()  # (1, 128, 216)

    if add_noise:
        signal_std = np.std(crop)
        noise_std = signal_std / (10 ** (5 / 20))
        noise = np.random.randn(*crop.shape).astype(np.float32) * noise_std
        crop = crop + noise

        # Mild reverb via exponential decay IR
        ir = np.exp(-np.linspace(0, 5, 2000)).astype(np.float32)
        ir = ir / np.sum(ir)  # normalize
        # Apply convolution along time axis for each channel and mel band
        reverbed = np.zeros_like(crop)
        for c in range(crop.shape[0]):
            for m in range(crop.shape[1]):
                conv_result = fftconvolve(crop[c, m, :], ir, mode='full')
                reverbed[c, m, :] = conv_result[:crop.shape[2]]
        crop = reverbed

    tensor = torch.from_numpy(crop).float().unsqueeze(0).to(device)  # (1, 1, 128, 216)
    with torch.no_grad():
        h = encoder(tensor)
    h = h.cpu().numpy().flatten()
    h = h / (np.linalg.norm(h) + 1e-8)

    D, I = index.search(h.reshape(1, -1).astype(np.float32), k=1)
    predicted_tid = id_list[I[0, 0]]
    similarity = D[0, 0]
    return predicted_tid, similarity

# ═══════════════════════════════════════════════════════════
# Step 3 — Evaluate on all test tracks
# ═══════════════════════════════════════════════════════════
print("\n─── Evaluation with noise ───")
correct_noisy = 0
results_noisy = []

for tid in test_ids:
    spec_path = os.path.join(SPECS_DIR, f"{tid}.npy")
    pred_tid, sim = query(spec_path, add_noise=True)
    is_correct = (pred_tid == tid)
    if is_correct:
        correct_noisy += 1
    results_noisy.append((tid, pred_tid, sim, is_correct))

acc_noisy = 100.0 * correct_noisy / len(test_ids)
print(f"Retrieval Accuracy (with noise): {acc_noisy:.2f}%")

print("\n─── Evaluation without noise ───")
correct_clean = 0
results_clean = []

for tid in test_ids:
    spec_path = os.path.join(SPECS_DIR, f"{tid}.npy")
    pred_tid, sim = query(spec_path, add_noise=False)
    is_correct = (pred_tid == tid)
    if is_correct:
        correct_clean += 1
    results_clean.append((tid, pred_tid, sim, is_correct))

acc_clean = 100.0 * correct_clean / len(test_ids)
print(f"Retrieval Accuracy (clean): {acc_clean:.2f}%")

# ─── Print 5 example rows (noisy) ───
print("\n─── Sample results (with noise) ───")
for true_id, pred_id, sim, correct in results_noisy[:5]:
    mark = "✓" if correct else "✗"
    print(f"True: {true_id:06d} | Predicted: {pred_id:06d} | Sim: {sim:.3f} | Correct: {mark}")
