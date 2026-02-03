"""
generate_mel_spectrogram.py
Script to load the trained LSTM model and generate a mel spectrogram.
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import librosa.display

# =========================
# CONFIG (same as training)
# =========================
NPZ_DIR = "mel_npz"
SEQ_LEN = 16
N_MELS = 128
SAMPLE_RATE = 22050
HOP_LENGTH = 512
CHECKPOINT_PATH = "checkpoints/best_lstm_retrain.pth"
OUTPUT_MEL = "generated_mel_spectrogram.npz"
GEN_DURATION_SEC = 30  # Duration of generated mel

# =========================
# GPU ENFORCEMENT (if available)
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model Definition (same as training)
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
# Generation Function
# =========================
def generate_mel_spectrogram():
    # Load checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    mean = ckpt["mean"]
    std = ckpt["std"]

    # Initialize model
    model = LSTMNextFrame(N_MELS)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    # Get seed from existing data
    paths = list(Path(NPZ_DIR).rglob("*.npz"))
    if not paths:
        raise RuntimeError("No .npz files found in mel_npz directory")

    seed_path = random.choice(paths)
    with np.load(seed_path) as z:
        seed = z["mel_spectrogram"].astype(np.float32).T[:SEQ_LEN]

    # Normalize seed
    seed = (seed - mean) / std
    seed_t = torch.from_numpy(seed).unsqueeze(0).to(DEVICE)

    # Calculate number of frames to generate
    fps = SAMPLE_RATE / HOP_LENGTH
    n_frames = int(GEN_DURATION_SEC * fps)

    generated = []

    with torch.no_grad():
        for _ in range(n_frames):
            pred = model(seed_t)
            generated.append(pred.squeeze(0).cpu().numpy())
            # Update seed with new prediction
            seed_t = torch.cat([seed_t[:, 1:], pred.unsqueeze(1)], dim=1)

    # Combine seed and generated
    mel = np.concatenate([seed, np.array(generated)], axis=0)
    # Denormalize
    mel = mel * std + mean
    # Transpose back to (n_mels, time)
    mel = mel.T

    # Save as npz
    np.savez(OUTPUT_MEL, mel_spectrogram=mel)
    print(f"🎵 Mel spectrogram generated and saved to {OUTPUT_MEL}")
    print(f"Shape: {mel.shape}")

    # Visualize the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('generated_mel_spectrogram.png')
    plt.show()  # This will display the plot if running in an environment that supports it
    print("📊 Mel spectrogram visualization saved as 'generated_mel_spectrogram.png'")

if __name__ == "__main__":
    generate_mel_spectrogram()