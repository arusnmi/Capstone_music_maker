import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# Paths
SCENARIO_DIR = Path(r"C:/Users/warty/OneDrive/Desktop/Python_projects/Capstone_music_maker/Scenario 2_ AI Music Composer & Listener Insight platform")
MUSIC_INFO = SCENARIO_DIR / "Music Info.csv"


def filter_genre_and_copy_scenario(scenario_dir: Path, copy_suffix: str = "-genre_filtered", perform_deletions: bool = False):
    """Fill missing genres by searching MP3 files, save filled CSV and filtered CSV.

    Behavior:
    - Search `MP3-Example` for files whose name contains a track_id with missing genre.
    - If found, infer genre from the MP3's subfolder name (preferred) or file name prefix before the first '-'.
    - Update the DataFrame with inferred genres, save `Music Info_genre_filled.csv`.
    - Save `Music Info_genre_present.csv` which contains only rows where genre is present.
    - NOTE: This function no longer creates copies of the scenario folder and will not delete any files.

    Returns summary dict with counts and paths.
    """
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

    summary = {
        'original_rows': len(pd.read_csv(music_info_path)),
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



