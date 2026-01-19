import os
import librosa
import numpy as np

# -------- CONFIG --------
INPUT_DIR = r"Scenario 2_ AI Music Composer & Listener Insight platform\MP3-Example"
OUTPUT_DIR = r"mel_npz"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# ------------------------

def extract_mel_spectrogram(mp3_path):
    """Load MP3 and extract log-scaled Mel Spectrogram"""
    y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".mp3"):
                mp3_path = os.path.join(root, file)

                # Preserve folder structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                output_path = os.path.join(
                    output_subdir,
                    file.replace(".mp3", ".npz")
                )

                try:
                    mel_spec = extract_mel_spectrogram(mp3_path)

                    np.savez_compressed(
                        output_path,
                        mel_spectrogram=mel_spec
                    )

                    print(f"Saved: {output_path}")

                except Exception as e:
                    print(f"❌ Error processing {mp3_path}: {e}")


if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)
