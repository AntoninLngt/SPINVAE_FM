"""
Dataset generation script for FM synthesizer presets and corresponding audio.

Generates N (parameters, audio) pairs and saves them under data/ with the
following structure::

    data/
        audio/
            sample_00001.wav
            ...
        params/
            sample_00001.json
            ...
        splits/
            train.npy
            val.npy
            test.npy
        metadata.csv

Usage::

    python scripts/generate_dataset.py [--n-samples N] [--seed SEED]
                                       [--output-dir DIR] [--skip-existing]

MIDI note 56 and velocity 75 are fixed for all samples so that timbre
variation comes solely from the FM synthesis parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from the repository root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from synth.fm_synth import (  # noqa: E402
    SAMPLE_RATE,
    generate_fm,
    sample_random_params,
    save_wav,
)

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    # Minimal fallback so the script works without tqdm installed.
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_MIDI_NOTE = 56
FIXED_VELOCITY  = 75

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.1


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def log_uniform(rng, low, high):
    return np.exp(rng.uniform(np.log(low), np.log(high)))

def _sample_params(rng):
    return {
        "midi_note": FIXED_MIDI_NOTE,
        "velocity": FIXED_VELOCITY,

        "mod_ratio": rng.uniform(0.5, 4.0),
        "mod_index": rng.uniform(0.0, 10.0),

        "attack": log_uniform(rng, 0.01, 0.5),
        "decay": log_uniform(rng, 0.05, 0.5),
        "sustain": rng.uniform(0.3, 1.0),
        "release": log_uniform(rng, 0.05, 0.5),
    }

def _normalize(waveform: np.ndarray) -> np.ndarray:
    """Normalize a waveform to the range [-1, 1].

    Returns the waveform unchanged if it is entirely silent.
    """
    peak = np.max(np.abs(waveform))
    if peak > 0.0:
        waveform = waveform / peak
    return waveform.astype(np.float32)


def _save_splits(splits_dir: str, indices: np.ndarray) -> None:
    """Randomly partition *indices* 80/10/10 and save .npy files."""
    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    os.makedirs(splits_dir, exist_ok=True)
    np.save(os.path.join(splits_dir, "train.npy"), train_idx)
    np.save(os.path.join(splits_dir, "val.npy"),   val_idx)
    np.save(os.path.join(splits_dir, "test.npy"),  test_idx)

    print(
        f"Splits saved → train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int,
    output_dir: str = "data",
    seed: int | None = None,
    skip_existing: bool = False,
) -> None:
    """Generate a dataset of FM synthesizer (parameters, audio) pairs.

    Parameters
    ----------
    n             : Number of samples to generate.
    output_dir    : Root directory for the dataset (default: ``data``).
    seed          : Optional integer seed for reproducibility.
    skip_existing : When *True*, skip samples whose files already exist.
    """
    audio_dir  = os.path.join(output_dir, "audio")
    params_dir = os.path.join(output_dir, "params")
    splits_dir = os.path.join(output_dir, "splits")
    meta_path  = os.path.join(output_dir, "metadata.csv")

    os.makedirs(audio_dir,  exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    meta_rows: list[dict] = []

    # Read existing metadata so we can append without duplicates.
    existing_ids: set[str] = set()
    if skip_existing and os.path.isfile(meta_path):
        with open(meta_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["sample_id"])
                meta_rows.append(row)

    generated_indices: list[int] = []

    for i in tqdm(range(1, n + 1), desc="Generating samples", unit="sample"):
        sample_id   = f"sample_{i:05d}"
        audio_file  = os.path.join(audio_dir,  f"{sample_id}.wav")
        params_file = os.path.join(params_dir, f"{sample_id}.json")

        # Skip if both files already exist and skip_existing is requested.
        if skip_existing and sample_id in existing_ids:
            generated_indices.append(i)
            continue

        params   = _sample_params(rng)
        waveform = generate_fm(params)
        waveform = _normalize(waveform)

        save_wav(waveform, audio_file, SAMPLE_RATE)

        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

        row = {
            "sample_id":   sample_id,
            "audio_file":  os.path.join("audio",  f"{sample_id}.wav"),
            "params_file": os.path.join("params", f"{sample_id}.json"),
        }
        row.update({k: str(v) for k, v in params.items()})
        meta_rows.append(row)
        generated_indices.append(i)

    # Write metadata CSV.
    if meta_rows:
        fieldnames = list(meta_rows[0].keys())
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(meta_rows)
        print(f"Metadata saved → {meta_path}")

    # Save split indices (1-based sample numbers, shuffled).
    all_indices = np.array(generated_indices, dtype=np.int32)
    rng.shuffle(all_indices)
    _save_splits(splits_dir, all_indices)

    print(f"Dataset generation complete: {len(generated_indices)} samples in '{output_dir}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an FM synthesizer dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Root output directory for the dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples whose audio and parameter files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        n=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )
