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
    # Collect only the first file from each genre folder
    audio_files.extend([str(p) for p in mp3s[:1]])



features_list = []

for file in audio_files:
    try:
        # Load audio (preserve native sampling rate)
        Y, sr = librosa.load(file, sr=None)
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

        # Extract MFCCs (keep for plotting but also compute stats)
        mfccs = librosa.feature.mfcc(y=Y, sr=sr, n_mfcc=13)
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCCs of {Path(file).name}')
        plt.tight_layout()
        mfcc_path = file_plots_dir / f"{Path(file).stem}_mfcc.png"
        fig.savefig(mfcc_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Extract Spectrogram
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

        # Extract Chroma feature
        chroma = librosa.feature.chroma_stft(y=Y, sr=sr)
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(f'Chroma Feature of {Path(file).name}')
        plt.tight_layout()
        chroma_path = file_plots_dir / f"{Path(file).stem}_chroma.png"
        fig.savefig(chroma_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Extract Tonnetz feature
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(Y), sr=sr)
        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
        plt.colorbar()
        plt.title(f'Tonnetz Feature of {Path(file).name}')
        plt.tight_layout()
        tonnetz_path = file_plots_dir / f"{Path(file).stem}_tonnetz.png"
        fig.savefig(tonnetz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # --- New feature extractions (numerical summaries) ---
        # Tempo (beats per minute)
        tempo, beat_frames = librosa.beat.beat_track(y=Y, sr=sr)

        # MFCC stats (mean and std for each coefficient)
        mfcc_means = mfccs.mean(axis=1)
        mfcc_stds = mfccs.std(axis=1)

        # Chroma mean strength for 12 pitch classes
        chroma_means = chroma.mean(axis=1)

        # Spectral centroid (brightness)
        spec_centroid = librosa.feature.spectral_centroid(y=Y, sr=sr)
        spec_centroid_mean = float(spec_centroid.mean())
        spec_centroid_std = float(spec_centroid.std())

        # Energy (use RMS)
        rms = librosa.feature.rms(y=Y)
        rms_mean = float(rms.mean())
        rms_std = float(rms.std())

        # Build a flat dictionary of features for the file
        feat = {
            'file_path': str(file),
            'genre': Path(file).parent.name,
            'file_name': Path(file).name,
            'tempo_bpm': float(tempo),
            'spectral_centroid_mean': spec_centroid_mean,
            'spectral_centroid_std': spec_centroid_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
        }

        # Add MFCC summary fields
        for i, (m_mean, m_std) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
            feat[f'mfcc_{i}_mean'] = float(m_mean)
            feat[f'mfcc_{i}_std'] = float(m_std)

        # Add chroma mean fields (12 bins)
        for i, c in enumerate(chroma_means, start=1):
            feat[f'chroma_{i}_mean'] = float(c)

        features_list.append(feat)

        print(f"Saved plots to: {file_plots_dir}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

# After processing all files, save features to CSV
if features_list:
    features_df = pd.DataFrame(features_list)
    features_df.to_csv('audio_features.csv', index=False)
    print(f"Saved audio features for {len(features_df)} files to audio_features.csv")
else:
    print("No features were extracted.")
