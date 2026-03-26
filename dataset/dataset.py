"""
PyTorch Dataset for the FM synthesizer preset collection.

Each sample exposes the 6 continuous FM parameters as a normalised float32
tensor.  Audio features (RMS, spectral centroid, attack, inharmonicity) can
optionally be computed on-the-fly and returned alongside the preset vector for
use with the AR-loss during timbre regularisation.

The 6 preset parameters and their normalisation ranges
-------------------------------------------------------
==========  =========  =======
Parameter   Range      Scale
==========  =========  =======
mod_ratio   [0.5, 4.0]   linear
mod_index   [0.0, 10.0]  linear
attack      [0.01, 0.5]  linear
decay       [0.05, 0.5]  linear
sustain     [0.3,  1.0]  linear
release     [0.05, 0.5]  linear
==========  =========  =======
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Parameter names and normalisation constants
# ---------------------------------------------------------------------------

PARAM_NAMES: list[str] = [
    "mod_ratio",
    "mod_index",
    "attack",
    "decay",
    "sustain",
    "release",
]

# Min / max for each of the 6 continuous parameters (linear scale)
PARAM_MIN = np.array([0.5,  0.0,  0.01, 0.05, 0.3,  0.05], dtype=np.float32)
PARAM_MAX = np.array([4.0,  10.0, 0.5,  0.5,  1.0,  0.5 ], dtype=np.float32)


def normalize_params(raw: np.ndarray) -> np.ndarray:
    """Map raw parameter values to [0, 1]."""
    return (raw - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)


def denormalize_params(norm: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize_params`."""
    return norm * (PARAM_MAX - PARAM_MIN) + PARAM_MIN


# ---------------------------------------------------------------------------
# Optional audio feature extraction
# ---------------------------------------------------------------------------

def _compute_audio_features(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract 4 perceptual audio features from a mono waveform.

    Features
    --------
    0 – RMS energy (overall loudness)
    1 – Spectral centroid (brightness), normalised to [0, 1] over sr/2 Hz
    2 – Attack time: fraction of total duration until peak amplitude
    3 – Inharmonicity proxy: spectral flux (frame-to-frame spectral change)

    Returns a float32 array of shape (4,), each value in [0, 1] (clipped).
    """
    n = len(waveform)

    # --- RMS ---
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    # --- Spectral centroid ---
    spectrum = np.abs(np.fft.rfft(waveform))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
    denom    = spectrum.sum()
    centroid = float((freqs * spectrum).sum() / denom) if denom > 0.0 else 0.0
    centroid_norm = centroid / (sr / 2.0)

    # --- Attack time ---
    peak_idx    = int(np.argmax(np.abs(waveform)))
    attack_frac = peak_idx / max(n - 1, 1)

    # --- Inharmonicity proxy: spectral flux ---
    frame_size = 512
    hop        = 256
    frames     = [
        np.abs(np.fft.rfft(waveform[i: i + frame_size]))
        for i in range(0, n - frame_size + 1, hop)
    ]
    if len(frames) >= 2:
        flux = float(
            np.mean([
                np.mean(np.abs(frames[k + 1] - frames[k]))
                for k in range(len(frames) - 1)
            ])
        )
    else:
        flux = 0.0
    # Normalise flux heuristically to [0, 1]
    flux_norm = min(flux / (np.max(np.abs(waveform)) + 1e-8), 1.0)

    features = np.array([rms, centroid_norm, attack_frac, flux_norm], dtype=np.float32)
    return np.clip(features, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FMPresetDataset(Dataset):
    """PyTorch Dataset that loads FM synthesizer presets from a metadata CSV.

    Parameters
    ----------
    metadata_csv    : path to the ``metadata.csv`` file produced by
                      ``scripts/generate_dataset.py``.
    split_file      : optional ``.npy`` file containing 1-based sample indices
                      for a particular split (train / val / test).
    audio_dir       : root directory for WAV files; required only when
                      *compute_audio_features* is ``True``.
    compute_audio_features : if ``True`` each ``__getitem__`` call also
                             returns a 4-d audio-feature tensor (for AR-loss).
    """

    def __init__(
        self,
        metadata_csv: str,
        split_file:   Optional[str] = None,
        audio_dir:    Optional[str] = None,
        compute_audio_features: bool = False,
    ) -> None:
        super().__init__()
        self.compute_audio_features = compute_audio_features
        self.audio_dir              = audio_dir

        # ---- Load metadata ------------------------------------------------
        rows: list[dict] = []
        with open(metadata_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # ---- Filter by split if requested ---------------------------------
        if split_file is not None:
            indices   = np.load(split_file).tolist()   # 1-based sample ids
            index_set = set(int(i) for i in indices)
            rows = [r for r in rows if self._sample_number(r["sample_id"]) in index_set]

        self._rows = rows

        # ---- Pre-extract normalised parameter tensors ---------------------
        raw = np.array(
            [[float(r[p]) for p in PARAM_NAMES] for r in rows],
            dtype=np.float32,
        )
        self._params: np.ndarray = normalize_params(raw)  # (N, 6)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_number(sample_id: str) -> int:
        """Convert 'sample_00042' → 42."""
        return int(sample_id.split("_")[-1])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        """Return (u,) or (u, a) tensors for sample *idx*.

        Returns
        -------
        u : float32 tensor of shape (6,) – normalised preset parameters.
        a : float32 tensor of shape (4,) – audio features (only when
            *compute_audio_features* is ``True``).
        """
        u = torch.from_numpy(self._params[idx])

        if not self.compute_audio_features:
            return u

        # Load the waveform and compute features
        row       = self._rows[idx]
        audio_rel = row["audio_file"].replace("\\", os.sep).replace("/", os.sep)
        if self.audio_dir is not None:
            # audio_rel is e.g. "audio/sample_00001.wav"
            audio_path = os.path.join(self.audio_dir, os.path.basename(audio_rel))
        else:
            audio_path = audio_rel

        try:
            from scipy.io import wavfile as _wf  # type: ignore
            sr, wav = _wf.read(audio_path)
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32) / 32768.0
        except Exception:
            wav = np.zeros(16000, dtype=np.float32)
            sr  = 16000

        a = torch.from_numpy(_compute_audio_features(wav, sr=sr))
        return u, a
