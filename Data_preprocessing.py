import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import librosa
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.switch_backend('Agg')

SCENARIO_DIR = Path(r"C:/Users/warty/OneDrive/Desktop/Python_projects/Capstone_music_maker/Scenario 2_ AI Music Composer & Listener Insight platform")
MUSIC_INFO = SCENARIO_DIR / "Music Info.csv"


def extract_metadata_for_track(track_id: str, scenario_dir: Path = SCENARIO_DIR, df: pd.DataFrame = None):
    """Convenience function to return the requested metadata dict for a track_id.
    Matches keys:
    track_id, name, tags, genre, mode, time_signature, danceability, energy, loudness,
    speechiness, acousticness, instrumentalness, liveness, valence

    This function will attempt to coerce numeric fields to floats (and mode to int) when possible,
    and will return empty strings for missing values.
    """
    if df is None:
        df = pd.read_csv(scenario_dir / "Music Info.csv")

    row = df.loc[df['track_id'].astype(str) == str(track_id)]
    if row.empty:
        metadata_dict = {
            "track_id": str(track_id),
            "name": "",
            "tags": "",
            "genre": "",
            "mode": "",
            "time_signature": "",
            "danceability": "",
            "energy": "",
            "loudness": "",
            "speechiness": "",
            "acousticness": "",
            "instrumentalness": "",
            "liveness": "",
            "valence": ""
        }
        return metadata_dict

    meta = row.iloc[0].to_dict()

    metadata_dict = {
        "track_id": str(track_id),
        "name": meta.get("name", ""),
        "tags": meta.get("tags", ""),
        "genre": meta.get("genre", ""),
        "mode": meta.get("mode", ""),
        "time_signature": meta.get("time_signature", ""),
        "danceability": meta.get("danceability", ""),
        "energy": meta.get("energy", ""),
        "loudness": meta.get("loudness", ""),
        "speechiness": meta.get("speechiness", ""),
        "acousticness": meta.get("acousticness", ""),
        "instrumentalness": meta.get("instrumentalness", ""),
        "liveness": meta.get("liveness", ""),
        "valence": meta.get("valence", "")
    }

    # Coerce numeric-like fields to floats where possible
    for k in ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "time_signature"]:
        try:
            v = metadata_dict.get(k, "")
            if v is not None and v != "" and not pd.isna(v):
                metadata_dict[k] = float(v)
            else:
                metadata_dict[k] = ""
        except Exception:
            metadata_dict[k] = ""

    # coerce mode to int when possible
    try:
        if metadata_dict.get("mode", "") != "" and not pd.isna(metadata_dict.get("mode")):
            metadata_dict["mode"] = int(float(metadata_dict["mode"]))
        else:
            metadata_dict["mode"] = ""
    except Exception:
        metadata_dict["mode"] = ""

    return metadata_dict


def filter_genre_and_copy_scenario(scenario_dir: Path,
                                   copy_suffix: str = "-genre_filtered",
                                   perform_deletions: bool = False):
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario folder not found: {scenario_dir}")

    df = pd.read_csv(scenario_dir / "Music Info.csv")

    # Keep all metadata columns; do not drop any columns automatically.
    # (Removed previous filtering such as columns containing 'spotify' to preserve metadata.)

    # Keep all metadata columns (e.g., danceability, time_signature) so they can be
    # included in each song's .npz file and used later for analysis.


    AUDIO_DIR = scenario_dir / 'MP3-Example'

    extracted_features = []

    FEATURE_DIR = scenario_dir / 'npz_features'
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    npz_sorted_dir = scenario_dir / 'npz_by_genre'
    npz_sorted_dir.mkdir(exist_ok=True)
    not_sorted_tracks = []

    token_genre_map = {}
    if AUDIO_DIR.exists():
        for sub in AUDIO_DIR.iterdir():
            if not sub.is_dir():
                continue
            for mp3 in sub.glob('*.mp3'):
                if '-' in mp3.name:
                    token = mp3.name.split('-', 1)[1].rsplit('.', 1)[0]
                    token_genre_map[token] = sub.name

    for _, row in df.iterrows():
        track_id = str(row['track_id'])
        audio_path = None

        for root, _, files in os.walk(AUDIO_DIR):
            for file in files:
                if file.endswith('.mp3') and track_id in file:
                    audio_path = Path(root) / file
                    break
            if audio_path:
                break

        if audio_path is None:
            continue

        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Audio-derived features (as requested: tempo, energy, MFCC, chroma, centroids)
        energy = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        loudness = float(np.mean(librosa.amplitude_to_db(np.abs(y))))

        # Simple summary scalars for CSV merging
        mfcc_mean = float(np.mean(np.abs(mfcc)))
        chroma_mean = float(np.mean(np.abs(chroma)))
        centroids_mean = float(np.mean(np.abs(centroids)))

        # get metadata for this track (from CSV) using typed extractor
        metadata = extract_metadata_for_npz(track_id, scenario_dir, df)

        npz_path = FEATURE_DIR / f"{track_id}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Save full arrays (mfcc, chroma, centroids), requested scalar features (tempo, energy),
            # and metadata fields (both as a dict and as individual prefixed entries).
            meta_kwargs = {}
            for k, v in metadata.items():
                meta_kwargs[f"meta_{k}"] = v

            np.savez_compressed(
                npz_path,
                tempo=float(tempo),
                energy=float(energy),
                mfcc=mfcc,
                chroma=chroma,
                centroids=centroids,
                mfcc_mean=mfcc_mean,
                chroma_mean=chroma_mean,
                centroids_mean=centroids_mean,
                metadata_json=json.dumps(metadata),
                **meta_kwargs
            )
        except Exception as e:
            print(f"[ERROR] Failed to save .npz for {track_id}: {e}")
            not_sorted_tracks.append(track_id)
            continue

        extracted_features.append({
            'track_id': track_id,
            'tempo': float(tempo),
            'energy': float(energy),
            'loudness': loudness,
            'mfcc_mean': mfcc_mean,
            'chroma_mean': chroma_mean,
            'centroids_mean': centroids_mean
        })
        print(f"[INFO] Extracted features for track {track_id} -> {npz_path}")
        print(f"[INFO] Extracted features for track {track_id} -> {npz_path}")

        genre_to_use = None
        if 'genre' in row and pd.notna(row['genre']) and str(row['genre']).strip() != '':
            genre_to_use = str(row['genre']).strip()
        else:
            if audio_path is not None and '-' in audio_path.name:
                token = audio_path.name.split('-', 1)[1].rsplit('.', 1)[0]
                genre_to_use = token_genre_map.get(token)

        if genre_to_use:
            dest_dir = npz_sorted_dir / genre_to_use
            dest_dir.mkdir(exist_ok=True)
            dest_path = dest_dir / f"{track_id}.npz"
            try:
                if not dest_path.exists():
                    shutil.copy2(npz_path, dest_path)
                    print(f"[INFO] Sorted npz for track {track_id} into genre folder: {dest_path}")
                else:
                    print(f"[INFO] NPZ already sorted: {dest_path}")
            except Exception as e:
                print(f"[WARN] Failed to sort npz for {track_id}: {e}")
                not_sorted_tracks.append(track_id)
        else:
            print(f"[WARN] Could not determine genre for {track_id} (file: {audio_path.name})")
            not_sorted_tracks.append(track_id)

    feature_df = pd.DataFrame(extracted_features)
    df = df.merge(feature_df, on='track_id', how='left')

    if 'loudness' in df.columns:
        min_loud, max_loud = df['loudness'].min(), df['loudness'].max()
        if max_loud != min_loud:
            df['loudness'] = (df['loudness'] - min_loud) / (max_loud - min_loud)

    if 'mode' in df.columns:
        df['mode'] = df['mode'].astype(int)
        if 'loudness' in df.columns:
            min_loud, max_loud = df['loudness'].min(), df['loudness'].max()
            if max_loud != min_loud:
                df['loudness'] = (df['loudness'] - min_loud) / (max_loud - min_loud)

        if 'mode' in df.columns:
            df['mode'] = df['mode'].apply(lambda x: 1 if x == 1 else 0)

    mask_present = df['genre'].notna() & df['genre'].astype(str).str.strip().ne('')
    missing_idx = df.loc[~mask_present].index.tolist()
    missing_track_ids = df.loc[~mask_present, 'track_id'].astype(str).tolist()

    mp3_root = scenario_dir / 'MP3-Example'
    inferred = {}
    not_found = []
    token_genre_map = {}

    if mp3_root.exists():
        for sub in mp3_root.iterdir():
            if not sub.is_dir():
                continue
            for mp3 in sub.glob('*.mp3'):
                if '-' in mp3.name:
                    token = mp3.name.split('-', 1)[1].rsplit('.', 1)[0]
                    token_genre_map[token] = sub.name

        df['inferred_genre'] = df['track_id'].astype(str).map(token_genre_map).fillna('')

        for tid, idx in zip(missing_track_ids, missing_idx):
            if tid in token_genre_map:
                df.at[idx, 'genre'] = token_genre_map[tid]
                inferred[tid] = token_genre_map[tid]
            else:
                not_found.append(tid)
    else:
        df['inferred_genre'] = ''
        not_found = missing_track_ids.copy()

    df.to_csv(scenario_dir / 'Music Info_genre_filled.csv', index=False)

    mask_present_after = df['genre'].notna() & df['genre'].astype(str).str.strip().ne('')
    df_present = df[mask_present_after].copy()

    mp3_root = scenario_dir / 'MP3-Example'
    copied = 0
    missing_npzs = []

    if mp3_root.exists():
        for _, row in df_present.iterrows():
            genre = str(row['genre']).strip()
            track_id = str(row['track_id'])
            dest_dir = npz_sorted_dir / genre
            dest_dir.mkdir(exist_ok=True)

            src_npz = FEATURE_DIR / f"{track_id}.npz"
            dest_npz = dest_dir / f"{track_id}.npz"
            if src_npz.exists():
                if not dest_npz.exists():
                    try:
                        shutil.copy2(src_npz, dest_npz)
                        copied += 1
                        print(f"[INFO] Copied NPZ {src_npz} -> {dest_npz}")
                    except Exception as e:
                        print(f"[WARN] Failed to copy NPZ {src_npz} -> {dest_npz}: {e}")
                else:
                    print(f"[INFO] NPZ already exists: {dest_npz}")
            else:
                missing_npzs.append(track_id)
    else:
        print(f"[WARN] MP3 root not found: {mp3_root} (skipping NPZ organization verification)")

    if missing_npzs:
        print(f"[WARN] Could not find .npz files for track_ids: {', '.join(missing_npzs)}")

    print(f"[INFO] NPZ sorting complete: {copied} files copied to {npz_sorted_dir}")

    df_present.to_csv(scenario_dir / 'Music Info_genre_present.csv', index=False)

    summary = {
        'original_rows': len(df),
        'rows_kept': len(df_present),
        'rows_removed': len(df) - len(df_present),
        'filled_count': len(inferred),
        'remaining_missing': len(not_found),
        'timestamp': datetime.utcnow().isoformat()
    }
    return summary


def get_metadata_for_track(scenario_dir: Path, track_id: str, df: pd.DataFrame = None):
    """Return a metadata dictionary for a given track_id from the Music Info CSV or provided DataFrame."""
    if df is None:
        df = pd.read_csv(scenario_dir / "Music Info.csv")

    row = df.loc[df['track_id'].astype(str) == str(track_id)]
    if row.empty:
        return {
            "track_id": str(track_id),
            "name": "",
            "tags": "",
            "genre": "",
            "mode": "",
            "time_signature": "",
            "danceability": "",
            "energy": "",
            "loudness": "",
            "speechiness": "",
            "acousticness": "",
            "instrumentalness": "",
            "liveness": "",
            "valence": ""
        }

    meta = row.iloc[0].to_dict()
    metadata_dict = {
        "track_id": str(track_id),
        "name": meta.get("name", ""),
        "tags": meta.get("tags", ""),
        "genre": meta.get("genre", ""),
        "mode": meta.get("mode", ""),
        "time_signature": meta.get("time_signature", ""),
        "danceability": meta.get("danceability", ""),
        "energy": meta.get("energy", ""),
        "loudness": meta.get("loudness", ""),
        "speechiness": meta.get("speechiness", ""),
        "acousticness": meta.get("acousticness", ""),
        "instrumentalness": meta.get("instrumentalness", ""),
        "liveness": meta.get("liveness", ""),
        "valence": meta.get("valence", "")
    }
    return metadata_dict


def extract_metadata_for_track(track_id: str, scenario_dir: Path = SCENARIO_DIR, df: pd.DataFrame = None):
    """
    Return a metadata dict for a given track_id with typed numeric fields when possible.
    Matches the requested metadata structure for inclusion in .npz files.
    """
    meta = get_metadata_for_track(scenario_dir, track_id, df)

    # Try to coerce numeric fields to float where appropriate; keep empty strings if missing.
    numeric_fields = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "time_signature"
    ]
    for k in numeric_fields:
        v = meta.get(k, "")
        try:
            if v is not None and v != "" and not pd.isna(v):
                meta[k] = float(v)
            else:
                meta[k] = ""
        except Exception:
            meta[k] = ""

    # Mode typically 0/1; coerce to int when possible.
    try:
        if meta.get("mode", "") != "" and not pd.isna(meta.get("mode")):
            meta["mode"] = int(float(meta["mode"]))
        else:
            meta["mode"] = ""
    except Exception:
        meta["mode"] = ""

    return meta

def extract_metadata_for_npz(track_id: str, scenario_dir: Path = SCENARIO_DIR, df: pd.DataFrame = None):
    """Return metadata dict for inclusion in .npz files (typed where possible)."""
    if df is None:
        df = pd.read_csv(scenario_dir / "Music Info.csv")

    row = df.loc[df['track_id'].astype(str) == str(track_id)]
    if row.empty:
        return {
            "track_id": str(track_id),
            "name": "",
            "tags": "",
            "genre": "",
            "mode": "",
            "time_signature": "",
            "danceability": "",
            "energy": "",
            "loudness": "",
            "speechiness": "",
            "acousticness": "",
            "instrumentalness": "",
            "liveness": "",
            "valence": ""
        }

    meta = row.iloc[0].to_dict()
    metadata_dict = {
        "track_id": str(track_id),
        "name": meta.get("name", ""),
        "tags": meta.get("tags", ""),
        "genre": meta.get("genre", ""),
        "mode": meta.get("mode", ""),
        "time_signature": meta.get("time_signature", ""),
        "danceability": meta.get("danceability", ""),
        "energy": meta.get("energy", ""),
        "loudness": meta.get("loudness", ""),
        "speechiness": meta.get("speechiness", ""),
        "acousticness": meta.get("acousticness", ""),
        "instrumentalness": meta.get("instrumentalness", ""),
        "liveness": meta.get("liveness", ""),
        "valence": meta.get("valence", "")
    }

    for k in ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "time_signature"]:
        try:
            v = metadata_dict.get(k, "")
            if v is not None and v != "" and not pd.isna(v):
                metadata_dict[k] = float(v)
            else:
                metadata_dict[k] = ""
        except Exception:
            metadata_dict[k] = ""

    try:
        if metadata_dict.get("mode", "") != "" and not pd.isna(metadata_dict.get("mode")):
            metadata_dict["mode"] = int(float(metadata_dict["mode"]))
        else:
            metadata_dict["mode"] = ""
    except Exception:
        metadata_dict["mode"] = ""

    return metadata_dict

GENRE_TO_LABEL = {
    'Blues': 1, 'Country': 2, 'Electronic': 3, 'Folk': 4, 'Jazz': 5,
    'Latin': 6, 'Metal': 7, 'New Age': 8, 'Pop': 9, 'Punk': 10,
    'Rap': 11, 'Reggae': 12, 'RnB': 13, 'Rock': 14, 'World': 15,
}

def apply_genre_labeling(scenario_dir: Path,
                         source_csv_name: str = 'Music Info_genre_present.csv',
                         out_csv_name: str = 'Music Info_genre_numeric.csv',
                         genre_col: str = 'genre',
                         label_col: str = 'genre_label',
                         mapping: dict = None,
                         cap_labels_to: int = 15):
    df = pd.read_csv(scenario_dir / source_csv_name)

    if mapping is None:
        genres = df[genre_col].dropna().astype(str).str.strip()
        genres = genres[genres != '']
        top_genres = genres.value_counts().index.tolist()[:cap_labels_to]
        mapping = {g: i + 1 for i, g in enumerate(top_genres)}

    df[label_col] = df[genre_col].map(mapping).fillna(0).astype(int)
    out_path = scenario_dir / out_csv_name
    df.to_csv(out_path, index=False)
    return str(out_path), mapping

def apply_artist_labeling(scenario_dir: Path,
                          source_csv_name: str = 'Music Info_genre_numeric.csv',
                          out_csv_name: str = 'Music Info_labeled.csv',
                          artist_col: str = 'artist',
                          label_col: str = 'artist_label'):
    df = pd.read_csv(scenario_dir / source_csv_name)
    artists = df[artist_col].dropna().astype(str).str.strip()
    mapping = {a: i + 1 for i, a in enumerate(artists.value_counts().index.tolist())}
    df[label_col] = df[artist_col].map(mapping).fillna(0).astype(int)

    out_path = scenario_dir / out_csv_name
    df.to_csv(out_path, index=False)
    return str(out_path), mapping

def generate_visualizations(scenario_dir: Path,
                            source_csv_name: str = 'Music Info_labeled.csv'):
    df = pd.read_csv(scenario_dir / source_csv_name)
    out_dir = scenario_dir / 'plots'
    out_dir.mkdir(exist_ok=True)

    if {'energy', 'valence'}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='energy', y='valence', alpha=0.6)
        plt.title('Energy vs Valence')
        plt.savefig(out_dir / 'energy_vs_valence.png')
        plt.close()

if __name__ == '__main__':
    summary = filter_genre_and_copy_scenario(SCENARIO_DIR)
    apply_genre_labeling(SCENARIO_DIR)
    apply_artist_labeling(SCENARIO_DIR)
    generate_visualizations(SCENARIO_DIR)
