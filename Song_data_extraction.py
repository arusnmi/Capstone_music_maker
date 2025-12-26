import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from glob import glob
from pathlib import Path
import seaborn as sns
import librosa
import librosa.display



# Base folder containing genre subfolders with mp3 files
base_mp3_dir = Path(r"C:\Users\warty\OneDrive\Desktop\Python_projects\Capstone_music_maker\Scenario 2_ AI Music Composer & Listener Insight platform\MP3-Example")

# Collect up to the first 15 .mp3 files from each immediate subfolder
audio_files = []
for subfolder in sorted([p for p in base_mp3_dir.iterdir() if p.is_dir()]):
    mp3s = sorted(subfolder.glob('*.mp3'))
    # If you want to include nested folders use subfolder.rglob('*.mp3') instead
    audio_files.extend([str(p) for p in mp3s[:1]])



for file in audio_files:
    Y, sr = librosa.load(file)
    print(f"File: {file}, Sample Rate: {sr}, Audio Shape: {Y.shape}")

    # Prepare output directories
    plots_base = Path('plots')
    plots_base.mkdir(parents=True, exist_ok=True)
    file_plots_dir = plots_base / Path(file).stem
    file_plots_dir.mkdir(parents=True, exist_ok=True)

    # Waveform
    fig = plt.figure(figsize=(12, 4))
    librosa.display.waveshow(Y, sr=sr)
    plt.title(f"Waveform of {Path(file).name}")
    waveform_path = file_plots_dir / f"{Path(file).stem}_waveform.png"
    fig.savefig(waveform_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Inline audio preview
    ipd.display(ipd.Audio(Y, rate=sr))

    # Extract and save MFCCs
    mfccs = librosa.feature.mfcc(y=Y, sr=sr, n_mfcc=13)
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCCs of {Path(file).name}')
    plt.tight_layout()
    mfcc_path = file_plots_dir / f"{Path(file).stem}_mfcc.png"
    fig.savefig(mfcc_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Extract and save Spectrogram
    S = librosa.feature.melspectrogram(y=Y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram of {Path(file).name}')
    plt.tight_layout()
    spec_path = file_plots_dir / f"{Path(file).stem}_melspectrogram.png"
    fig.savefig(spec_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Extract and save Chroma feature
    chroma = librosa.feature.chroma_stft(y=Y, sr=sr)
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title(f'Chroma Feature of {Path(file).name}')
    plt.tight_layout()
    chroma_path = file_plots_dir / f"{Path(file).stem}_chroma.png"
    fig.savefig(chroma_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Extract and save Tonnetz feature
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(Y), sr=sr)
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
    plt.colorbar()
    plt.title(f'Tonnetz Feature of {Path(file).name}')
    plt.tight_layout()
    tonnetz_path = file_plots_dir / f"{Path(file).stem}_tonnetz.png"
    fig.savefig(tonnetz_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plots to: {file_plots_dir}")
