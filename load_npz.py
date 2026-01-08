"""Small utility to inspect .npz feature files created by the project.

Usage examples:
  python load_npz.py path/to/TRAAXPX128F42AA93D.npz
  python load_npz.py path/to/TRAAXPX128F42AA93D.npz --plot-mfcc --plot-chroma --out-dir=plots/inspect

Outputs:
 - Prints list of keys, types and shapes
 - Prints parsed metadata if present
 - Optionally saves MFCC and chroma plots (PNG) to --out-dir
"""

from pathlib import Path
import argparse
import numpy as np
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inspect_npz(npz_path: Path, plot_mfcc: bool = False, plot_chroma: bool = False, out_dir: Path = None):
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"Loaded NPZ: {npz_path}")
        print(f"Keys ({len(keys)}): {', '.join(keys)}\n")

        for k in keys:
            v = data[k]
            # For arrays, show dtype and shape
            if hasattr(v, 'shape'):
                try:
                    vshape = v.shape
                except Exception:
                    vshape = None
                print(f"- {k}: array, dtype={getattr(v, 'dtype', None)}, shape={vshape}")
                # print summary for numeric arrays
                try:
                    if np.issubdtype(getattr(v, 'dtype', None), np.number):
                        print(f"    min={np.nanmin(v):.4g}, max={np.nanmax(v):.4g}, mean={np.nanmean(v):.4g}")
                except Exception:
                    pass
            else:
                # other python objects
                print(f"- {k}: {type(v).__name__} -> {v}")

        # Print parsed metadata_json if exists
        if 'metadata_json' in keys:
            try:
                meta = json.loads(str(data['metadata_json']))
                print('\nMetadata parsed from metadata_json:')
                for mk, mv in meta.items():
                    print(f"  {mk}: {mv}")
            except Exception as e:
                print(f"Could not parse metadata_json: {e}")

        # Check for explicit meta_ prefixed keys
        meta_keys = [k for k in keys if k.startswith('meta_')]
        if meta_keys:
            print('\nAdditional metadata (meta_* keys):')
            for mk in meta_keys:
                try:
                    print(f"  {mk}: {data[mk]}")
                except Exception:
                    print(f"  {mk}: <unreadable>")

        # Optional plotting
        if (plot_mfcc or plot_chroma) and out_dir is None:
            out_dir = npz_path.parent / 'inspect_plots'
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        if plot_mfcc and 'mfcc' in keys:
            mfcc = data['mfcc']
            plt.figure(figsize=(10, 4))
            plt.imshow(mfcc, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'MFCC - {npz_path.name}')
            plt.xlabel('Frame')
            plt.ylabel('MFCC Coefficient')
            if out_dir:
                out_file = out_dir / f"{npz_path.stem}_mfcc.png"
                plt.tight_layout()
                plt.savefig(out_file)
                print(f"Saved MFCC plot -> {out_file}")
            plt.close()
        elif plot_mfcc:
            print("MFCC not found in NPZ; skipping MFCC plot.")

        if plot_chroma and 'chroma' in keys:
            chroma = data['chroma']
            plt.figure(figsize=(10, 3))
            plt.imshow(chroma, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar()
            plt.title(f'Chroma - {npz_path.name}')
            plt.xlabel('Frame')
            plt.ylabel('Chroma Bin')
            if out_dir:
                out_file = out_dir / f"{npz_path.stem}_chroma.png"
                plt.tight_layout()
                plt.savefig(out_file)
                print(f"Saved chroma plot -> {out_file}")
            plt.close()
        elif plot_chroma:
            print("Chroma not found in NPZ; skipping chroma plot.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Inspect .npz feature files produced by the project")
    p.add_argument('npz', type=str, help='Path to the .npz file')
    p.add_argument('--plot-mfcc', action='store_true', help='Save MFCC plot (if present)')
    p.add_argument('--plot-chroma', action='store_true', help='Save chroma plot (if present)')
    p.add_argument('--out-dir', type=str, default=None, help='Output directory for plots (defaults to sibling folder)')
    args = p.parse_args()

    try:
        inspect_npz(Path(args.npz), plot_mfcc=args.plot_mfcc, plot_chroma=args.plot_chroma, out_dir=Path(args.out_dir) if args.out_dir else None)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
