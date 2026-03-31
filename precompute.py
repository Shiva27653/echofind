
# !pip install torchaudio -q

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from multiprocessing import Pool

# ─── Fixed paths ───
AUDIO_BASE = "/kaggle/input/datasets/imsparsh/fma-free-music-archive-small-medium/fma_small/fma_small/"
METADATA   = "/kaggle/input/datasets/imsparsh/fma-free-music-archive-small-medium/fma_metadata/"
SPECS_OUT  = "/kaggle/working/specs/"
WEIGHTS    = "/kaggle/working/weights/"

os.makedirs(SPECS_OUT, exist_ok=True)
os.makedirs(WEIGHTS, exist_ok=True)

# ─── Load corrupt IDs ───
with open(os.path.join(METADATA, "not_found.pickle"), "rb") as f:
    not_found = pickle.load(f)
corrupt_ids = set(int(x) for x in not_found["audio"])

# ─── Load metadata & build valid track list ───
tracks = pd.read_csv(os.path.join(METADATA, "tracks.csv"), index_col=0, header=[0, 1])
small = tracks[tracks[("set", "subset")] == "small"]
all_track_ids = small.index.tolist()
valid_ids = [tid for tid in all_track_ids if tid not in corrupt_ids]
print(f"Total small tracks: {len(all_track_ids)}, corrupt: {len(corrupt_ids)}, valid: {len(valid_ids)}")

# ─── Target spec length (30 s @ 22050 Hz, hop 512) ───
TARGET_LENGTH = 1292

# ─── Worker function ───
def process_track(track_id):
    try:
        subdir = f"{track_id:06d}"[:3]
        fname  = f"{track_id:06d}.mp3"
        path   = os.path.join(AUDIO_BASE, subdir, fname)

        if not os.path.exists(path):
            return None

        waveform, sr = torchaudio.load(path)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
            waveform = resampler(waveform)

        # mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_mels=128, hop_length=512, n_fft=2048
        )
        amp_to_db = torchaudio.transforms.AmplitudeToDB()

        mel_spec = mel_transform(waveform)
        log_mel  = amp_to_db(mel_spec)  # (1, 128, T)

        # pad or crop to TARGET_LENGTH
        T = log_mel.shape[2]
        if T < TARGET_LENGTH:
            pad = TARGET_LENGTH - T
            log_mel = torch.nn.functional.pad(log_mel, (0, pad), mode="constant", value=log_mel.min().item())
        else:
            log_mel = log_mel[:, :, :TARGET_LENGTH]

        # save
        out_path = os.path.join(SPECS_OUT, f"{track_id}.npy")
        np.save(out_path, log_mel.numpy())
        return track_id
    except Exception as e:
        return None

# ─── Run with multiprocessing ───
if __name__ == "__main__":
    count = 0
    results = []

    with Pool(4) as pool:
        for i, result in enumerate(pool.imap_unordered(process_track, valid_ids), 1):
            if result is not None:
                count += 1
            if i % 500 == 0:
                print(f"Processed {i}/{len(valid_ids)} tracks, saved so far: {count}")

    print(f"\nTotal spectrograms saved: {count}")
    print(f"Phase 1 complete. {count} spectrograms saved.")
