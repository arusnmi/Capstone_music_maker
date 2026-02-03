"""
train_and_generate_mel.py
GPU-ONLY — Zero-CLI Mel Spectrogram Training & Generation
"""

import os
import math
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================
# USER CONFIG (EDIT HERE)
# =========================
NPZ_DIR = "mel_npz"
EPOCHS = 20
BATCH_SIZE = 16
SEQ_LEN = 16
N_MELS = 128
USE_AMP = True

SAVE_MODEL = "checkpoints/best_lstm_retrain.pth"
OUTPUT_AUDIO = "outputs/retrain_smoke.wav"
GEN_DURATION_SEC = 30

SAMPLE_RATE = 22050
HOP_LENGTH = 512
# =========================


# =========================
# GPU ONLY ENFORCEMENT
# =========================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU REQUIRED")

torch.backends.cudnn.benchmark = True
DEVICE = torch.device(0)


# =========================
# Dataset
# =========================
class MelNPZSequenceDataset(Dataset):
    def __init__(self, npz_dir, seq_len, n_mels):
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.paths = list(Path(npz_dir).rglob("*.npz"))

        if not self.paths:
            raise RuntimeError("No .npz files found")

        self.items = []
        frames = []

        print("🔍 Loading NPZ files...")
        for p in tqdm(self.paths):
            with np.load(p) as z:
                mel = z["mel_spectrogram"].astype(np.float32).T  # (T, 128)

            if mel.shape[1] != n_mels or mel.shape[0] <= seq_len:
                continue

            for i in range(mel.shape[0] - seq_len):
                self.items.append((p, i))

            frames.append(mel)

        full = np.concatenate(frames, axis=0)
        self.mean = full.mean(axis=0, keepdims=True)
        self.std = full.std(axis=0, keepdims=True) + 1e-9

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, i = self.items[idx]
        with np.load(p) as z:
            mel = z["mel_spectrogram"].astype(np.float32).T

        seq = mel[i : i + self.seq_len]
        tgt = mel[i + self.seq_len]

        seq = (seq - self.mean) / self.std
        tgt = (tgt - self.mean.squeeze()) / self.std.squeeze()

        return torch.from_numpy(seq), torch.from_numpy(tgt)


# =========================
# Model
# =========================
class LSTMNextFrame(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size=512,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Linear(512, input_size)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1])


# =========================
# Training
# =========================
def train_model(dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    model = LSTMNextFrame(N_MELS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_loss = float("inf")

    for ep in range(1, EPOCHS + 1):
        model.train()
        mse_sum = 0.0
        count = 0

        for seq, tgt in tqdm(loader, desc=f"Epoch {ep}/{EPOCHS}"):
            seq = seq.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred = model(seq)
                loss = loss_fn(pred, tgt)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            mse_sum += loss.item() * seq.size(0)
            count += seq.size(0)

        mse = mse_sum / count
        rmse = math.sqrt(mse)
        print(f"[EPOCH {ep}] MSE={mse:.6f}  RMSE={rmse:.6f}")

        if mse < best_loss:
            best_loss = mse
            os.makedirs(Path(SAVE_MODEL).parent, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": dataset.mean,
                    "std": dataset.std,
                },
                SAVE_MODEL,
            )
            print("💾 Best model saved")

    return model


# =========================
# Generation
# =========================
def generate_audio(model, dataset):
    ckpt = torch.load(
    SAVE_MODEL,
    map_location="cpu",
    weights_only=False
)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    mean = ckpt["mean"]
    std = ckpt["std"]

    seed_path = random.choice(dataset.paths)
    with np.load(seed_path) as z:
        seed = z["mel_spectrogram"].astype(np.float32).T[:SEQ_LEN]

    seed = (seed - mean) / std
    seed_t = torch.from_numpy(seed).unsqueeze(0).to(DEVICE)

    fps = SAMPLE_RATE / HOP_LENGTH
    n_frames = int(GEN_DURATION_SEC * fps)

    generated = []

    with torch.no_grad():
        for _ in range(n_frames):
            pred = model(seed_t)
            generated.append(pred.squeeze(0).cpu().numpy())
            seed_t = torch.cat([seed_t[:, 1:], pred.unsqueeze(1)], dim=1)

    mel = np.concatenate([seed, np.array(generated)], axis=0)
    mel = mel * std + mean
    mel = mel.T

    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_iter=64,
    )

    out = Path(OUTPUT_AUDIO)
    out = out.with_name(out.stem + "_mel.wav")
    os.makedirs(out.parent, exist_ok=True)

    sf.write(out, audio, SAMPLE_RATE)
    print(f"🎧 Audio written → {out}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    dataset = MelNPZSequenceDataset(NPZ_DIR, SEQ_LEN, N_MELS)
    model = train_model(dataset)
    generate_audio(model, dataset)
