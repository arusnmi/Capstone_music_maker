import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# use a non-interactive backend when running in headless environments
plt.switch_backend('Agg')

# Paths
SCENARIO_DIR = Path(r"C:/Users/warty/OneDrive/Desktop/Python_projects/Capstone_music_maker/Scenario 2_ AI Music Composer & Listener Insight platform")
MUSIC_INFO = SCENARIO_DIR / "Music Info.csv"


def filter_genre_and_copy_scenario(scenario_dir: Path, copy_suffix: str = "-genre_filtered", perform_deletions: bool = False):
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario folder not found: {scenario_dir}")

    music_info_path = scenario_dir / "Music Info.csv"
    df = pd.read_csv(music_info_path)

    # Identify rows missing genre
    mask_present = df['genre'].notna() & df['genre'].astype(str).str.strip().ne('')
    missing_idx = df.loc[~mask_present].index.tolist()
    missing_track_ids = df.loc[~mask_present, 'track_id'].astype(str).tolist()

    # Search MP3 files to infer genre
    mp3_root = scenario_dir / 'MP3-Example'
    inferred = {}
    not_found = []

    # mapping token -> folder (genre)
    token_genre_map = {}

    if mp3_root.exists():
        # Build a quick lookup by scanning all mp3 filenames once
        files = []
        for sub in mp3_root.iterdir():
            if not sub.is_dir():
                continue
            for mp3 in sub.glob('*.mp3'):
                files.append((mp3, sub.name))  # (path, folder name as candidate genre)
                if '-' in mp3.name:
                    token = mp3.name.split('-', 1)[1].rsplit('.', 1)[0]
                    token_genre_map[token] = sub.name

        # Add an `inferred_genre` column for any rows that have a matching mp3 token
        df['inferred_genre'] = df['track_id'].astype(str).map(token_genre_map).fillna('')

        # Now try to fill missing genres using the inferred_genre
        for tid, idx in zip(missing_track_ids, missing_idx):
            if tid in token_genre_map:
                genre_candidate = token_genre_map[tid]
                inferred[tid] = genre_candidate
                df.at[idx, 'genre'] = genre_candidate
            else:
                not_found.append(tid)
    else:
        df['inferred_genre'] = ''
        not_found = missing_track_ids.copy()

    filled_count = len(inferred)
    remaining_missing = len(not_found)

    # Save the filled CSV
    filled_csv_path = scenario_dir / 'Music Info_genre_filled.csv'
    df.to_csv(filled_csv_path, index=False)

    # Save filtered CSV with only rows that now have genre
    mask_present_after = df['genre'].notna() & df['genre'].astype(str).str.strip().ne('')
    df_present = df[mask_present_after].copy()
    filtered_csv_name = "Music Info_genre_present.csv"
    filtered_csv_path = scenario_dir / filtered_csv_name
    df_present.to_csv(filtered_csv_path, index=False)

    # NOTE: No folder copy or deletions are performed anymore.
    files_deleted = 0
    deleted_files = []

    original_rows = len(pd.read_csv(music_info_path))
    rows_kept = len(df_present)
    rows_removed = original_rows - rows_kept

    summary = {
        'original_rows': original_rows,
        'rows_kept': rows_kept,
        'rows_removed': rows_removed,
        'filled_count': filled_count,
        'remaining_missing': remaining_missing,
        'filled_csv_path': str(filled_csv_path),
        'filtered_csv_path': str(filtered_csv_path),
        'scenario_copy_path': None,
        'files_deleted': files_deleted,
        'deleted_files_sample': deleted_files[:20],
        'timestamp': datetime.utcnow().isoformat(),
        'inferred_sample': dict(list(inferred.items())[:20]),
        'not_found_sample': not_found[:20]
    }

    return summary


# Mapping of textual genres to integer labels (1..15)
GENRE_TO_LABEL = {
    'Blues': 1,
    'Country': 2,
    'Electronic': 3,
    'Folk': 4,
    'Jazz': 5,
    'Latin': 6,
    'Metal': 7,
    'New Age': 8,
    'Pop': 9,
    'Punk': 10,
    'Rap': 11,
    'Reggae': 12,
    'RnB': 13,
    'Rock': 14,
    'World': 15,
}


def apply_genre_labeling(scenario_dir: Path,
                         source_csv_name: str = 'Music Info_genre_present.csv',
                         out_csv_name: str = 'Music Info_genre_numeric.csv',
                         genre_col: str = 'genre',
                         label_col: str = 'genre_label',
                         mapping: dict = None,
                         cap_labels_to: int = 15):
    """Map textual genres to integer labels derived from the source CSV and save a new CSV.

    If `mapping` is None, the function builds a mapping by counting genre frequency
    in `source_csv_name` and assigns integer labels 1.. based on descending frequency.

    Labels are capped to `cap_labels_to` (default 15): only the top `cap_labels_to`
    genres receive labels 1..cap_labels_to; other genres will receive 0 (unknown).

    The function expects the source CSV to contain only rows with a valid genre
    (e.g., the output of `filter_genre_and_copy_scenario`). If not, missing/blank
    genres are ignored when constructing the mapping so that NaN is not treated
    as a valid genre label.

    If the source CSV doesn't exist, the function falls back to 'Music Info.csv'.

    Returns a tuple (out_csv_path_str, mapping_used).
    """
    source = scenario_dir / source_csv_name
    if not source.exists():
        source = scenario_dir / 'Music Info.csv'
        if not source.exists():
            raise FileNotFoundError(f"No source CSV found at {scenario_dir}")

    df = pd.read_csv(source)
    if genre_col not in df.columns:
        raise KeyError(f"'{genre_col}' column not found in {source}")

    # Build mapping from dataset if not provided
    if mapping is None:
        # Drop missing and blank genres to avoid treating NaNs as a valid genre label
        genres = df[genre_col].dropna().astype(str).str.strip()
        genres = genres[genres != '']
        if genres.empty:
            # Fallback to hardcoded mapping if no genres present in the dataset
            mapping = GENRE_TO_LABEL.copy()
        else:
            # Rank genres by frequency and assign labels 1..cap_labels_to
            top_genres = genres.value_counts().index.tolist()
            capped = top_genres[:cap_labels_to]
            mapping = {g: i + 1 for i, g in enumerate(capped)}

    # Map genres to labels. Unmapped genres will become 0 to indicate unknown.
    df[label_col] = df[genre_col].map(mapping).fillna(0).astype(int)

    out_path = scenario_dir / out_csv_name
    df.to_csv(out_path, index=False)
    return str(out_path), mapping


def apply_artist_labeling(scenario_dir: Path,
                          source_csv_name: str = 'Music Info_genre_numeric.csv',
                          out_csv_name: str = 'Music Info_labeled.csv',
                          artist_col: str = 'artist',
                          label_col: str = 'artist_label',
                          mapping: dict = None):
    """Map textual artists to integer labels derived from the source CSV and save a combined CSV.

    If `mapping` is None, the function builds a mapping by counting artist frequency
    in `source_csv_name` and assigns integer labels 1.. based on descending frequency.

    The function tries the following sources in order: `source_csv_name`,
    `Music Info_genre_present.csv`, then `Music Info.csv`.

    Rows with missing or blank artist are ignored when creating the mapping; those
    rows will receive label 0 in the output.

    Returns a tuple (out_csv_path_str, mapping_used).
    """
    source = scenario_dir / source_csv_name
    if not source.exists():
        # fallback to genre-present, then original
        source = scenario_dir / 'Music Info_genre_present.csv'
        if not source.exists():
            source = scenario_dir / 'Music Info.csv'
            if not source.exists():
                raise FileNotFoundError(f"No source CSV found at {scenario_dir}")

    df = pd.read_csv(source)
    if artist_col not in df.columns:
        raise KeyError(f"'{artist_col}' column not found in {source}")

    # Build mapping from dataset if not provided
    if mapping is None:
        artists = df[artist_col].dropna().astype(str).str.strip()
        artists = artists[artists != '']
        if artists.empty:
            # No artists present to map
            mapping = {}
        else:
            artist_list = artists.value_counts().index.tolist()
            mapping = {a: i + 1 for i, a in enumerate(artist_list)}

    # Map artists to labels. Unmapped artists will become 0 to indicate unknown.
    df[label_col] = df[artist_col].map(mapping).fillna(0).astype(int)

    out_path = scenario_dir / out_csv_name
    df.to_csv(out_path, index=False)
    return str(out_path), mapping


def generate_visualizations(scenario_dir: Path,
                            source_csv_name: str = 'Music Info_labeled.csv',
                            out_dir_name: str = 'plots',
                            top_n_genres: int = 10):
    """Create and save charts (Energy vs Valence scatter, Tempo distribution, Loudness by valence/genre).

    - Scatter: energy vs valence colored by genre (top N)
    - Tempo distribution: boxplot of tempo by genre (top N)
    - Loudness by valence: boxplot of loudness by valence bins and/or by genre

    The function writes PNG files into `scenario_dir/out_dir_name` and returns a list of paths.
    """
    out_dir = scenario_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    source = scenario_dir / source_csv_name
    if not source.exists():
        raise FileNotFoundError(f"Visualization source CSV not found: {source}")

    df = pd.read_csv(source)

    saved = []

    # Prepare top genres
    if 'genre' in df.columns:
        genre_counts = df['genre'].dropna().astype(str).str.strip().value_counts()
        top_genres = genre_counts.index.tolist()[:top_n_genres]
        df['genre_for_plot'] = df['genre'].astype(str).where(df['genre'].isin(top_genres), other='Other')
    else:
        df['genre_for_plot'] = 'Unknown'

    # 1) Scatter: energy vs valence
    if {'energy', 'valence'}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='energy', y='valence', hue='genre_for_plot', alpha=0.6, s=40)
        plt.title('Energy vs Valence (colored by genre)')
        plt.xlabel('Energy')
        plt.ylabel('Valence')
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        scatter_path = out_dir / 'energy_vs_valence_scatter.png'
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        saved.append(str(scatter_path))

    # 2) Tempo distribution by genre (boxplot)
    if 'tempo' in df.columns or 'tempo_bpm' in df.columns:
        tempo_col = 'tempo' if 'tempo' in df.columns else 'tempo_bpm'
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df['genre_for_plot'].isin(top_genres + ['Other'])], x='genre_for_plot', y=tempo_col)
        plt.title(f'Tempo distribution by genre (top {top_n_genres})')
        plt.xlabel('Genre')
        plt.ylabel('Tempo (BPM)')
        plt.xticks(rotation=45, ha='right')
        tempo_path = out_dir / 'tempo_by_genre_boxplot.png'
        plt.tight_layout()
        plt.savefig(tempo_path)
        plt.close()
        saved.append(str(tempo_path))

    # 3) Loudness comparison across valence bins and by genre
    if 'loudness' in df.columns and 'valence' in df.columns:
        # create valence bins: low, medium, high
        df['valence_bin'] = pd.cut(df['valence'], bins=[-0.01, 0.33, 0.66, 1.0], labels=['low', 'medium', 'high'])
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='valence_bin', y='loudness')
        plt.title('Loudness by Valence bin')
        plt.xlabel('Valence (mood)')
        plt.ylabel('Loudness (dB)')
        loudness_valence_path = out_dir / 'loudness_by_valence_bin.png'
        plt.tight_layout()
        plt.savefig(loudness_valence_path)
        plt.close()
        saved.append(str(loudness_valence_path))

        # Loudness by genre (top)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df['genre_for_plot'].isin(top_genres + ['Other'])], x='genre_for_plot', y='loudness')
        plt.title(f'Loudness by Genre (top {top_n_genres})')
        plt.xlabel('Genre')
        plt.ylabel('Loudness (dB)')
        plt.xticks(rotation=45, ha='right')
        loudness_genre_path = out_dir / 'loudness_by_genre_boxplot.png'
        plt.tight_layout()
        plt.savefig(loudness_genre_path)
        plt.close()
        saved.append(str(loudness_genre_path))

    # Return saved file paths
    return saved


if __name__ == '__main__':
    print("Filling missing genres from MP3 files (no folder copies or deletions will be performed)...")
    summary = filter_genre_and_copy_scenario(SCENARIO_DIR, perform_deletions=False)
    print("Done.")
    print(f"Rows before: {summary['original_rows']}")
    print(f"Genres inferred/found: {summary['filled_count']}")
    print(f"Still missing genres: {summary['remaining_missing']}")
    print(f"Filled CSV saved to: {summary['filled_csv_path']}")
    print(f"Filtered CSV saved to: {summary['filtered_csv_path']}")
    if summary['files_deleted']:
        print(f"Files deleted in copy: {summary['files_deleted']}")
    if summary['inferred_sample']:
        print('Sample inferred genres:', summary['inferred_sample'])
    if summary['not_found_sample']:
        print('Sample track_ids not found:', summary['not_found_sample'])

    # Also generate a numeric-labeled CSV for modeling (genres -> integers 1..15)
    try:
        # Use the filtered CSV that only contains rows with a genre
        numeric_path, mapping_used = apply_genre_labeling(SCENARIO_DIR, source_csv_name='Music Info_genre_present.csv')
        print(f"Numeric-labeled CSV saved to: {numeric_path}")
        print(f"Genre to label mapping used: {mapping_used}")
        print(f"Rows kept (with genre): {summary.get('rows_kept')}, rows removed: {summary.get('rows_removed')}")
    except Exception as e:
        print(f"Could not create numeric-labeled CSV: {e}")

    # Now label artists and produce a combined CSV containing both genre and artist labels
    try:
        labeled_path, artist_mapping = apply_artist_labeling(SCENARIO_DIR,
                                                            source_csv_name='Music Info_genre_numeric.csv',
                                                            out_csv_name='Music Info_labeled.csv')
        print(f"Artist-labeled CSV saved to: {labeled_path}")
        print(f"Number of artists labeled: {len(artist_mapping)}")
        print(f"Sample artist mapping: {dict(list(artist_mapping.items())[:20])}")
    except Exception as e:
        print(f"Could not create artist-labeled CSV: {e}")

    # Generate visualizations and save charts to the scenario 'plots' folder
    try:
        saved_paths = generate_visualizations(SCENARIO_DIR, source_csv_name='Music Info_labeled.csv', out_dir_name='plots', top_n_genres=12)
        print('Saved charts:')
        for p in saved_paths:
            print(' -', p)
    except Exception as e:
        print(f"Could not create visualizations: {e}")



