import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast

# ─── Fixed paths ───
SPECS_DIR = "/kaggle/working/specs/"
METADATA  = "/kaggle/input/datasets/imsparsh/fma-free-music-archive-small-medium/fma_metadata/"
WEIGHTS   = "/kaggle/working/weights/"
os.makedirs(WEIGHTS, exist_ok=True)

# ─── Force GPU 0 as primary ───
torch.cuda.set_device(0)
device = torch.device('cuda')

# ─── Load metadata & build split lists ───
tracks = pd.read_csv(os.path.join(METADATA, "tracks.csv"), index_col=0, header=[0, 1])
small = tracks[tracks[("set", "subset")] == "small"]

train_ids = small[small[("set", "split")] == "training"].index.tolist()
val_ids   = small[small[("set", "split")] == "validation"].index.tolist()
test_ids  = small[small[("set", "split")] == "test"].index.tolist()

# ─── Safety check: keep only IDs with existing .npy ───
def filter_existing(ids, specs_dir):
    return [tid for tid in ids if os.path.exists(os.path.join(specs_dir, f"{tid}.npy"))]

train_ids = filter_existing(train_ids, SPECS_DIR)
val_ids   = filter_existing(val_ids, SPECS_DIR)
test_ids  = filter_existing(test_ids, SPECS_DIR)

print(f"Valid train IDs: {len(train_ids)}")
print(f"Valid val IDs:   {len(val_ids)}")
print(f"Valid test IDs:  {len(test_ids)}")

# ─── Augmentation constants ───
CROP_FRAMES = 216  # ~5 seconds at hop_length=512, sr=22050
TOTAL_FRAMES = 1292
FREQ_MASK_MAX = 30
TIME_MASK_MAX = 50
NOISE_STD = 0.01

# ─── Dataset ───
class SimCLRAudioDataset(Dataset):
    def __init__(self, track_ids, specs_dir):
        self.track_ids = track_ids
        self.specs_dir = specs_dir

    def __len__(self):
        return len(self.track_ids)

    def _random_crop(self, spec):
        max_start = TOTAL_FRAMES - CROP_FRAMES
        start = np.random.randint(0, max_start + 1)
        return spec[:, :, start:start + CROP_FRAMES].copy()

    def _augment(self, tensor):
        # Frequency masking
        f_start = torch.randint(0, 128 - FREQ_MASK_MAX, (1,)).item()
        f_width = torch.randint(0, FREQ_MASK_MAX + 1, (1,)).item()
        tensor[:, f_start:f_start + f_width, :] = 0.0

        # Time masking
        t_start = torch.randint(0, CROP_FRAMES - TIME_MASK_MAX, (1,)).item()
        t_width = torch.randint(0, TIME_MASK_MAX + 1, (1,)).item()
        tensor[:, :, t_start:t_start + t_width] = 0.0

        # Gaussian noise
        tensor = tensor + torch.randn_like(tensor) * NOISE_STD

        return tensor

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        spec = np.load(os.path.join(self.specs_dir, f"{tid}.npy"))  # (1, 128, 1292)
        spec = spec.astype(np.float32)

        crop1 = self._random_crop(spec)
        crop2 = self._random_crop(spec)

        view1 = self._augment(torch.from_numpy(crop1))
        view2 = self._augment(torch.from_numpy(crop2))

        return view1, view2  # both (1, 128, 216)

# ─── DataLoaders ───
train_dataset = SimCLRAudioDataset(train_ids, SPECS_DIR)
val_dataset   = SimCLRAudioDataset(val_ids, SPECS_DIR)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True, drop_last=False)

# ─── Encoder ───
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.backbone = resnet  # output dim = 512

    def forward(self, x):
        return self.backbone(x)

# ─── Projection Head ───
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ─── Full SimCLR Model ───
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

# ─── NT-Xent Loss ───
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        # z: (2N, 128)
        N2 = z.shape[0]
        N = N2 // 2

        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarities
        mask = torch.eye(N2, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N, N2), torch.arange(0, N)]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss

# ─── Initialize model with DataParallel on both GPUs ───
model = SimCLRModel()
model = nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)

criterion = NTXentLoss(temperature=0.1).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler = GradScaler()

# ─── Training loop ───
best_val_loss = float('inf')
first_batch_logged = False

for epoch in range(1, 51):
    # ── Train ──
    model.train()
    train_losses = []

    for batch_idx, (view1, view2) in enumerate(train_loader):
        view1 = view1.to(device, non_blocking=True)  # (B, 1, 128, 216)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            z1 = model(view1)  # (B, 128)
            z2 = model(view2)  # (B, 128)
            z = torch.cat([z1, z2], dim=0)  # (2B, 128)
            loss = criterion(z)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

        if not first_batch_logged:
            print(f"GPU 0 memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
            print(f"GPU 1 memory allocated: {torch.cuda.memory_allocated(1) / 1e6:.2f} MB")
            first_batch_logged = True

    avg_train_loss = np.mean(train_losses)

    # ── Validation ──
    model.eval()
    val_losses = []

    with torch.no_grad():
        for view1, view2 in val_loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            with autocast():
                z1 = model(view1)
                z2 = model(view2)
                z = torch.cat([z1, z2], dim=0)
                loss = criterion(z)

            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch:02d}/50 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

    # ── Save best encoder only ──
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.module.encoder.state_dict(), os.path.join(WEIGHTS, "encoder.pth"))

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
