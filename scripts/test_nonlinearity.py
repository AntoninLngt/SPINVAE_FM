"""
Test the non-linearity of the FM synthesizer by interpolating between two presets.

This script:
  1. Loads two presets from the dataset (first/last or maximally different).
  2. Linearly interpolates all numeric FM parameters between the two presets
     using **T = 9 steps** (matching the SpinVAE paper convention)::

         interpolated_param[t] = preset_a + (t-1)/(T-1) * (preset_b - preset_a)
         for t in 1 … T

  3. Synthesizes audio for every interpolated step via ``render_audio`` /
     ``generate_fm``.
  4. Reports per-step parameter values and audio-difference metrics
     (RMS, log-spectral distance).
  5. Optionally saves each step's audio as a WAV file in a preview folder.
  6. Optionally plots waveforms and spectrograms for every step.
  7. Optionally plays each waveform in sequence.

Usage::

    # Use first and last presets in the dataset (default)
    python scripts/test_nonlinearity.py

    # Choose the maximally different pair (slower for large datasets)
    python scripts/test_nonlinearity.py --selection-mode max_distance

    # Specify exact preset indices (1-based)
    python scripts/test_nonlinearity.py --preset-a 1 --preset-b 5000

    # Save audio and show plots
    python scripts/test_nonlinearity.py --save-audio --plot

    # Control number of interpolation steps (default T = 9)
    python scripts/test_nonlinearity.py --n-steps 20

    # Play audio (requires sounddevice)
    python scripts/test_nonlinearity.py --play
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from the repository root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from synth.fm_synth import SAMPLE_RATE, generate_fm, save_wav  # noqa: E402

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt  # type: ignore
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

try:
    import sounddevice as sd  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

try:
    import librosa  # type: ignore
    import librosa.display  # type: ignore
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Numeric parameter keys that will be interpolated.
# "midi_note" and "velocity" are fixed in the dataset and kept constant.
# ---------------------------------------------------------------------------
_INTERPOLATE_KEYS = [
    "mod_ratio",
    "mod_index",
    "attack",
    "decay",
    "sustain",
    "release",
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _list_param_files(data_dir: str) -> list[str]:
    """Return a sorted list of absolute paths to all JSON parameter files."""
    params_dir = os.path.join(data_dir, "params")
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(
            f"Parameter directory not found: {params_dir}\n"
            "Run  scripts/generate_dataset.py  first."
        )
    files = sorted(
        os.path.join(params_dir, f)
        for f in os.listdir(params_dir)
        if f.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(
            f"No JSON parameter files found in {params_dir}.\n"
            "Run  scripts/generate_dataset.py  first."
        )
    return files


def _load_params(path: str) -> dict:
    """Load a parameter dictionary from a JSON file."""
    with open(path) as fh:
        return json.load(fh)


def _params_to_vector(params: dict) -> np.ndarray:
    """Convert the interpolatable keys of *params* to a 1-D numpy array."""
    return np.array([float(params[k]) for k in _INTERPOLATE_KEYS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Preset selection
# ---------------------------------------------------------------------------

def _select_first_last(param_files: list[str]) -> tuple[dict, dict, int, int]:
    """Select the first and last preset in the dataset."""
    idx_a, idx_b = 0, len(param_files) - 1
    preset_a = _load_params(param_files[idx_a])
    preset_b = _load_params(param_files[idx_b])
    return preset_a, preset_b, idx_a + 1, idx_b + 1  # 1-based indices for display


def _select_max_distance(
    param_files: list[str],
    rng: np.random.Generator,
    candidate_pool: int = 200,
) -> tuple[dict, dict, int, int]:
    """Find the pair of presets with maximum Euclidean distance in parameter space.

    To keep the search tractable for large datasets we first draw
    *candidate_pool* random indices and search within that subset.
    """
    n = len(param_files)
    pool_size = min(candidate_pool, n)
    candidate_indices = rng.choice(n, size=pool_size, replace=False)

    # Load vectors for all candidates.
    vectors = []
    for ci in candidate_indices:
        p = _load_params(param_files[ci])
        vectors.append(_params_to_vector(p))
    vectors = np.stack(vectors)  # (pool_size, n_params)

    # Normalise each dimension to [0, 1] across the candidate pool.
    v_min = vectors.min(axis=0)
    v_max = vectors.max(axis=0)
    v_range = v_max - v_min
    v_range[v_range == 0.0] = 1.0  # avoid division by zero
    vectors_norm = (vectors - v_min) / v_range

    # Brute-force search for the pair with maximum Euclidean distance.
    best_dist = -1.0
    best_i, best_j = 0, 1
    for i in range(pool_size):
        for j in range(i + 1, pool_size):
            dist = float(np.linalg.norm(vectors_norm[i] - vectors_norm[j]))
            if dist > best_dist:
                best_dist = dist
                best_i, best_j = i, j

    idx_a = int(candidate_indices[best_i])
    idx_b = int(candidate_indices[best_j])
    preset_a = _load_params(param_files[idx_a])
    preset_b = _load_params(param_files[idx_b])
    print(f"Max-distance pair: index {idx_a + 1} ↔ {idx_b + 1}  (dist={best_dist:.4f})")
    return preset_a, preset_b, idx_a + 1, idx_b + 1


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_params(preset_a: dict, preset_b: dict, T: int) -> list[dict]:
    """Return a list of *T* linearly interpolated parameter dicts.

    Step index *t* runs from 1 to T (inclusive).  The interpolation factor
    for step *t* is ``(t-1) / (T-1)``, so:

    - step 1  → preset_a  (factor = 0)
    - step T  → preset_b  (factor = 1)

    Formally, for each interpolatable key::

        interpolated[key][t] = preset_a[key] + (t-1)/(T-1) * (preset_b[key] - preset_a[key])

    All other keys (``midi_note``, ``velocity``, …) are copied unchanged from
    *preset_a*.
    """
    result = []
    for t in range(1, T + 1):
        alpha = (t - 1) / (T - 1)  # 0.0 at t=1, 1.0 at t=T
        step_params = dict(preset_a)  # copy fixed keys (midi_note, velocity, …)
        for key in _INTERPOLATE_KEYS:
            step_params[key] = float(
                preset_a[key] + alpha * (preset_b[key] - preset_a[key])
            )
        result.append(step_params)
    return result


# Keep the old name as an alias for backwards compatibility.
interpolate_presets = interpolate_params


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def render_audio(params: dict) -> np.ndarray:
    """Synthesize a waveform from an FM parameter dictionary.

    A thin wrapper around ``generate_fm`` that makes the pipeline explicit::

        waveform[t] = render_audio(interpolated_param[t])
    """
    return generate_fm(params)


def play_audio(waveform: np.ndarray, sample_rate: int) -> None:
    """Play a waveform through the system's default audio output.

    Requires the optional *sounddevice* library.  Prints a warning and returns
    immediately when the library is not available.
    """
    if not _SOUNDDEVICE_AVAILABLE:
        print("[play_audio] sounddevice not installed – skipping playback.")
        return
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()


# ---------------------------------------------------------------------------
# Audio-difference metrics
# ---------------------------------------------------------------------------

def rms(waveform: np.ndarray) -> float:
    """Root-mean-square energy of a waveform."""
    return float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))


def log_spectral_distance(wave_a: np.ndarray, wave_b: np.ndarray) -> float:
    """Log-spectral distance (LSD) between two same-length waveforms.

    LSD = sqrt( mean( (log10(|X_a(f)|²) - log10(|X_b(f)|²))² ) )

    A small epsilon is added before taking the log to avoid −∞.
    """
    n = min(len(wave_a), len(wave_b))
    eps = 1e-10
    spec_a = np.abs(np.fft.rfft(wave_a[:n].astype(np.float64))) ** 2
    spec_b = np.abs(np.fft.rfft(wave_b[:n].astype(np.float64))) ** 2
    log_diff = np.log10(spec_a + eps) - np.log10(spec_b + eps)
    return float(np.sqrt(np.mean(log_diff ** 2)))


# ---------------------------------------------------------------------------
# STFT-based helpers (demo-notebook style)
# ---------------------------------------------------------------------------

_STFT_N_FFT = 2048
_STFT_HOP = 512


def compute_stft_db(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a log-magnitude (dB) STFT spectrogram.

    Uses librosa when available; falls back to ``numpy.fft`` otherwise.

    Returns a 2-D array of shape *(n_freqs, n_frames)* in dB.
    """
    wave = waveform.astype(np.float32)
    if _LIBROSA_AVAILABLE:
        stft = librosa.stft(wave, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP)
        return librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    # Fallback: use numpy with a simple frame loop.
    n = len(wave)
    n_fft = _STFT_N_FFT
    hop = _STFT_HOP
    frames = range(0, n - n_fft + 1, hop)
    window = np.hanning(n_fft)
    specs = []
    for start in frames:
        frame = wave[start: start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame))
        specs.append(spec)
    S = np.array(specs).T  # (n_freqs, n_frames)
    eps = 1e-10
    return 20.0 * np.log10(S + eps) - 20.0 * np.log10(np.max(S) + eps)


def spectral_distance(wave_a: np.ndarray, wave_b: np.ndarray) -> float:
    """STFT-based spectral distance between two waveforms.

    Computes the RMS difference of the dB spectrograms, which is
    consistent with the log-spectral distance used in the demo notebook.
    """
    db_a = compute_stft_db(wave_a)
    db_b = compute_stft_db(wave_b)
    # Match the shorter time axis.
    n_frames = min(db_a.shape[1], db_b.shape[1])
    diff = db_a[:, :n_frames] - db_b[:, :n_frames]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Trajectory metrics (demo-notebook philosophy)
# ---------------------------------------------------------------------------

def compute_metrics(waveforms: list[np.ndarray]) -> dict:
    """Compute trajectory metrics for a list of interpolated waveforms.

    Returns a dictionary with:
    - ``consecutive_distances``: spectral distance between each pair of
      adjacent steps  (length T-1)
    - ``distances_to_target``: spectral distance from each step to the
      final waveform  (length T)
    - ``cumulative_distances``: running sum of consecutive distances
      (length T, starts at 0)
    - ``straight_distance``: spectral distance from first to last step
    - ``linearity``: straight_distance / cumulative_distance
    - ``smoothness``: std of consecutive_distances (lower = smoother)
    """
    n = len(waveforms)
    # (a) Distance between consecutive steps.
    consec = [
        spectral_distance(waveforms[i], waveforms[i + 1])
        for i in range(n - 1)
    ]
    # (b) Distance from each step to target (last waveform).
    to_target = [spectral_distance(w, waveforms[-1]) for w in waveforms]
    # (c) Cumulative trajectory distance.
    cumulative = [0.0]
    for d in consec:
        cumulative.append(cumulative[-1] + d)
    # (d) Straight-line distance (first → last).
    straight = spectral_distance(waveforms[0], waveforms[-1])
    # (e) Linearity metric.
    total_path = cumulative[-1]
    linearity = straight / total_path if total_path > 0 else 1.0
    # (f) Smoothness (lower std = more uniform spacing).
    smoothness = float(np.std(consec)) if consec else 0.0

    return {
        "consecutive_distances": consec,
        "distances_to_target": to_target,
        "cumulative_distances": cumulative,
        "straight_distance": straight,
        "linearity": linearity,
        "smoothness": smoothness,
    }


# ---------------------------------------------------------------------------
# Feature extraction (demo-notebook spirit)
# ---------------------------------------------------------------------------

def compute_features(waveforms: list[np.ndarray], sample_rate: int = SAMPLE_RATE) -> dict:
    """Compute per-step audio descriptors.

    Returns a dictionary with arrays of length T:
    - ``spectral_centroid``: mean spectral centroid over time
    - ``spectral_bandwidth``: mean spectral bandwidth over time
    - ``mfcc_mean``: mean of MFCC coefficients (list of arrays, or None)
    """
    centroids: list[float] = []
    bandwidths: list[float] = []
    mfccs: list[np.ndarray | None] = []

    for wave in waveforms:
        w = wave.astype(np.float32)
        if _LIBROSA_AVAILABLE:
            sc = librosa.feature.spectral_centroid(y=w, sr=sample_rate)
            centroids.append(float(np.mean(sc)))
            sb = librosa.feature.spectral_bandwidth(y=w, sr=sample_rate)
            bandwidths.append(float(np.mean(sb)))
            mfcc = librosa.feature.mfcc(y=w, sr=sample_rate, n_mfcc=13)
            mfccs.append(np.mean(mfcc, axis=1))
        else:
            # Numpy fallback: estimate centroid from mean power spectrum
            # using the same windowed frames as compute_stft_db for consistency.
            n = len(w)
            n_fft = _STFT_N_FFT
            hop = _STFT_HOP
            window = np.hanning(n_fft)
            frame_powers = []
            for start in range(0, n - n_fft + 1, hop):
                frame = w[start: start + n_fft] * window
                frame_powers.append(np.abs(np.fft.rfft(frame)) ** 2)
            power = np.mean(frame_powers, axis=0) if frame_powers else np.zeros(n_fft // 2 + 1)
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            total_power = power.sum() + 1e-10
            sc = float(np.sum(freqs * power) / total_power)
            variance = float(np.sum(((freqs - sc) ** 2) * power) / total_power)
            centroids.append(sc)
            bandwidths.append(float(np.sqrt(max(variance, 0.0))))
            mfccs.append(None)

    return {
        "spectral_centroid": centroids,
        "spectral_bandwidth": bandwidths,
        "mfcc_mean": mfccs,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Shared mel-spectrogram parameters used by plot_spectrograms.
_MEL_N_FFT = 2048
_MEL_HOP = 512
_MEL_N_MELS = 128

# Waveform amplitude limits for plot_waveforms.
_WAVEFORM_YLIM_MIN = -1
_WAVEFORM_YLIM_MAX = 1


def plot_waveforms(
    waveforms: list[np.ndarray],
    alphas: np.ndarray,
    sample_rate: int,
) -> None:
    """Plot one waveform per interpolation step in a single figure.

    Layout: 1 row × N columns.

    Parameters
    ----------
    waveforms   : List of mono waveforms (one per step).
    alphas      : Interpolation factors corresponding to each step.
    sample_rate : Sample rate in Hz.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[plot] matplotlib not installed – skipping waveform plot.")
        return

    n_steps = len(waveforms)
    fig, axes = plt.subplots(
        1, n_steps,
        figsize=(3 * n_steps, 3),
        squeeze=False,
    )
    fig.suptitle("Interpolation: waveforms", fontsize=14)

    for i, (wave, alpha) in enumerate(zip(waveforms, alphas)):
        duration = len(wave) / sample_rate
        t = np.linspace(0.0, duration, len(wave), endpoint=False)
        ax = axes[0][i]
        ax.plot(t, wave, linewidth=0.5)
        ax.set_ylim(_WAVEFORM_YLIM_MIN, _WAVEFORM_YLIM_MAX)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Step {i + 1}\nα={alpha:.2f}")
        if i == 0:
            ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def plot_spectrograms(
    waveforms: list[np.ndarray],
    sample_rate: int,
    alphas: np.ndarray | None = None,
) -> None:
    """Plot one mel spectrogram per interpolation step in a single figure.

    Uses ``librosa.feature.melspectrogram`` + ``librosa.power_to_db`` +
    ``librosa.display.specshow`` with a shared colour scale across all steps.

    Layout: 1 row × N columns.

    Parameters
    ----------
    waveforms   : List of waveforms (mono or multi-channel; converted to mono
                  internally).
    sample_rate : Sample rate in Hz.
    alphas      : Optional interpolation factors used for subplot titles.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[plot] matplotlib not installed – skipping spectrogram plot.")
        return
    if not _LIBROSA_AVAILABLE:
        print("[plot] librosa not installed – skipping spectrogram plot.")
        return

    n_steps = len(waveforms)

    # Pre-compute all mel spectrograms (dB) with consistent parameters.
    db_specs = []
    for wave in waveforms:
        wave_sq = wave.squeeze()
        mono = librosa.to_mono(wave_sq.T) if wave_sq.ndim > 1 else wave_sq.astype(np.float32)
        S = librosa.feature.melspectrogram(
            y=mono,
            sr=sample_rate,
            n_fft=_MEL_N_FFT,
            hop_length=_MEL_HOP,
            n_mels=_MEL_N_MELS,
        )
        db_specs.append(librosa.power_to_db(S, ref=np.max))

    # Shared colour scale for visual consistency across steps.
    global_vmin = min(s.min() for s in db_specs)
    global_vmax = max(s.max() for s in db_specs)

    fig, axes = plt.subplots(
        1, n_steps,
        figsize=(3 * n_steps, 3),
        squeeze=False,
    )
    fig.suptitle("Interpolation: mel spectrograms", fontsize=14)

    for i, S_db in enumerate(db_specs):
        ax = axes[0][i]
        librosa.display.specshow(
            S_db,
            sr=sample_rate,
            hop_length=_MEL_HOP,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            vmin=global_vmin,
            vmax=global_vmax,
            cmap="viridis",
        )
        if alphas is not None:
            ax.set_title(f"Step {i + 1}\nα={alphas[i]:.2f}")
        else:
            ax.set_title(f"Step {i + 1}")

    plt.tight_layout()
    plt.show()


def _plot_steps(
    waveforms: list[np.ndarray],
    alphas: np.ndarray,
    sample_rate: int,
) -> None:
    """Plot waveforms and mel spectrograms for every interpolation step.

    Delegates to :func:`plot_waveforms` and :func:`plot_spectrograms`, which
    each produce their own figure so the two views remain independent.
    """
    plot_waveforms(waveforms, alphas, sample_rate)
    plot_spectrograms(waveforms, sample_rate, alphas=alphas)


def _plot_metrics(
    alphas: np.ndarray,
    rms_values: list[float],
    lsd_values: list[float],
    metrics: dict | None = None,
    features: dict | None = None,
) -> None:
    """Multi-panel metrics plot matching the demo-notebook layout.

    Layout (2 rows × 3 columns):
    - [0,0] Distance between consecutive steps
    - [0,1] Distance to target
    - [0,2] Cumulative distance trajectory
    - [1,0] Spectral centroid vs α
    - [1,1] Spectral bandwidth vs α
    - [1,2] RMS energy vs α
    """
    if not _MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Interpolation trajectory metrics", fontsize=14)

    # Convenience references.
    # Note: when `metrics` is provided, consecutive distances use STFT-based
    # spectral distance (dB RMS).  The `lsd_values` fallback uses the legacy
    # log-spectral distance (different units); prefer passing metrics explicitly.
    consec = metrics["consecutive_distances"] if metrics else lsd_values
    to_target = metrics["distances_to_target"] if metrics else None
    cumulative = metrics["cumulative_distances"] if metrics else None
    centroids = features["spectral_centroid"] if features else None
    bandwidths = features["spectral_bandwidth"] if features else None

    # --- [0,0] Distance between consecutive steps ---
    ax = axes[0][0]
    ax.plot(alphas[1:], consec, marker="o", color="orange")
    ax.set_xlabel("α (interpolation factor)")
    ax.set_ylabel("Spectral distance (dB RMS)")
    ax.set_title("Distance between consecutive steps")
    ax.grid(True)

    # --- [0,1] Distance to target ---
    ax = axes[0][1]
    if to_target is not None:
        ax.plot(alphas, to_target, marker="o", color="steelblue")
        ax.set_xlabel("α (interpolation factor)")
        ax.set_ylabel("Spectral distance to target (dB RMS)")
        ax.set_title("Distance to target (last step)")
        ax.grid(True)
    else:
        ax.set_visible(False)

    # --- [0,2] Cumulative distance trajectory ---
    ax = axes[0][2]
    if cumulative is not None:
        ax.plot(alphas, cumulative, marker="o", color="green")
        if metrics is not None:
            straight = metrics["straight_distance"]
            ax.axhline(straight, linestyle="--", color="red", label=f"Straight-line ({straight:.2f})")
            ax.legend(fontsize=8)
        ax.set_xlabel("α (interpolation factor)")
        ax.set_ylabel("Cumulative spectral distance")
        ax.set_title("Cumulative trajectory distance")
        ax.grid(True)
    else:
        ax.set_visible(False)

    # --- [1,0] Spectral centroid ---
    ax = axes[1][0]
    if centroids is not None:
        ax.plot(alphas, centroids, marker="o", color="purple")
        ax.set_xlabel("α (interpolation factor)")
        ax.set_ylabel("Spectral centroid (Hz)")
        ax.set_title("Spectral centroid vs α")
        ax.grid(True)
    else:
        ax.set_visible(False)

    # --- [1,1] Spectral bandwidth ---
    ax = axes[1][1]
    if bandwidths is not None:
        ax.plot(alphas, bandwidths, marker="o", color="brown")
        ax.set_xlabel("α (interpolation factor)")
        ax.set_ylabel("Spectral bandwidth (Hz)")
        ax.set_title("Spectral bandwidth vs α")
        ax.grid(True)
    else:
        ax.set_visible(False)

    # --- [1,2] RMS energy ---
    ax = axes[1][2]
    ax.plot(alphas, rms_values, marker="o")
    ax.set_xlabel("α (interpolation factor)")
    ax.set_ylabel("RMS energy")
    ax.set_title("RMS energy per step")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_test(
    data_dir: str,
    n_steps: int,
    selection_mode: str,
    preset_a_idx: int | None,
    preset_b_idx: int | None,
    save_audio: bool,
    preview_dir: str,
    plot: bool,
    play: bool,
    seed: int | None,
) -> None:
    """Run the non-linearity test end-to-end."""

    # ------------------------------------------------------------------
    # 1. Locate parameter files
    # ------------------------------------------------------------------
    param_files = _list_param_files(data_dir)
    print(f"Found {len(param_files)} parameter files in '{data_dir}/params'.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 2. Select the two endpoint presets
    # ------------------------------------------------------------------
    if preset_a_idx is not None and preset_b_idx is not None:
        # User specified explicit indices (1-based).
        idx_a = preset_a_idx - 1
        idx_b = preset_b_idx - 1
        if not (0 <= idx_a < len(param_files)):
            raise ValueError(f"--preset-a {preset_a_idx} out of range [1, {len(param_files)}].")
        if not (0 <= idx_b < len(param_files)):
            raise ValueError(f"--preset-b {preset_b_idx} out of range [1, {len(param_files)}].")
        preset_a = _load_params(param_files[idx_a])
        preset_b = _load_params(param_files[idx_b])
        display_a, display_b = preset_a_idx, preset_b_idx
    elif selection_mode == "max_distance":
        preset_a, preset_b, display_a, display_b = _select_max_distance(param_files, rng)
    else:  # "first_last"
        preset_a, preset_b, display_a, display_b = _select_first_last(param_files)

    print(f"\nPreset A (index {display_a}):")
    for k in _INTERPOLATE_KEYS:
        print(f"    {k:12s} = {preset_a[k]:.6g}")

    print(f"\nPreset B (index {display_b}):")
    for k in _INTERPOLATE_KEYS:
        print(f"    {k:12s} = {preset_b[k]:.6g}")

    # ------------------------------------------------------------------
    # 3. Linearly interpolate between the two presets
    # ------------------------------------------------------------------
    print(f"\nInterpolating with {n_steps} steps (T = {n_steps}) …")
    step_params = interpolate_params(preset_a, preset_b, n_steps)
    alphas = np.array([(t - 1) / (n_steps - 1) for t in range(1, n_steps + 1)])

    # ------------------------------------------------------------------
    # 4. Synthesize audio for each step and collect metrics
    # ------------------------------------------------------------------
    waveforms: list[np.ndarray] = []
    rms_values: list[float] = []
    lsd_values: list[float] = []  # differences between consecutive steps

    # Print table header.
    header_parts = [f"{'Step':>4}", f"{'α':>6}"] + [f"{k:>12}" for k in _INTERPOLATE_KEYS] + [f"{'RMS':>8}"]
    header_line = "  ".join(header_parts)
    print("\n" + header_line)
    print("-" * len(header_line))

    for step_i, (alpha, params) in enumerate(zip(alphas, step_params)):
        wave = render_audio(params)
        waveforms.append(wave)

        r = rms(wave)
        rms_values.append(r)

        # Log-spectral distance vs previous step.
        if step_i > 0:
            lsd = log_spectral_distance(waveforms[step_i - 1], wave)
            lsd_values.append(lsd)

        row_parts = [f"{step_i + 1:4d}", f"{alpha:6.3f}"]
        row_parts += [f"{params[k]:12.6g}" for k in _INTERPOLATE_KEYS]
        row_parts.append(f"{r:8.5f}")
        print("  ".join(row_parts))

    # ------------------------------------------------------------------
    # 5. Highlight sudden audio changes
    # ------------------------------------------------------------------
    if lsd_values:
        lsd_arr = np.array(lsd_values)
        mean_lsd = lsd_arr.mean()
        std_lsd  = lsd_arr.std()
        threshold = mean_lsd + 2.0 * std_lsd

        print(f"\nLog-spectral distance statistics:")
        print(f"    mean = {mean_lsd:.5f}")
        print(f"    std  = {std_lsd:.5f}")
        print(f"    threshold (mean + 2·std) = {threshold:.5f}")

        jumps = [i + 1 for i, lsd in enumerate(lsd_values) if lsd > threshold]
        if jumps:
            print(f"\n⚠️  Sudden audio changes detected between steps: {jumps}")
            print("   This indicates non-linear behaviour in the FM synthesizer.")
        else:
            print("\n✅  No sudden audio changes detected above threshold.")
            print("   The interpolation appears smooth across these steps.")

    # ------------------------------------------------------------------
    # 5b. Trajectory metrics & features (demo-notebook style)
    # ------------------------------------------------------------------
    print("\nComputing trajectory metrics …")
    metrics = compute_metrics(waveforms)
    features = compute_features(waveforms, int(step_params[0].get("sample_rate", SAMPLE_RATE)))

    print("\n--- Interpolation trajectory summary ---")
    print(f"    Total path distance  : {metrics['cumulative_distances'][-1]:.4f} dB RMS")
    print(f"    Straight-line dist.  : {metrics['straight_distance']:.4f} dB RMS")
    print(f"    Linearity score      : {metrics['linearity']:.4f}  (1.0 = perfectly straight)")
    print(f"    Smoothness (std)     : {metrics['smoothness']:.4f}  (0.0 = perfectly uniform)")
    print("----------------------------------------")

    # ------------------------------------------------------------------
    # 6. Save audio files (optional)
    # ------------------------------------------------------------------
    if save_audio:
        os.makedirs(preview_dir, exist_ok=True)
        sr = int(step_params[0].get("sample_rate", SAMPLE_RATE))
        for step_i, wave in enumerate(waveforms):
            alpha = alphas[step_i]
            filename = os.path.join(preview_dir, f"step_{step_i + 1:03d}_alpha_{alpha:.3f}.wav")
            save_wav(wave, filename, sr)
        print(f"\nSaved {len(waveforms)} audio files to '{preview_dir}'.")

    # ------------------------------------------------------------------
    # 7. Play audio sequentially (optional)
    # ------------------------------------------------------------------
    if play:
        sr = int(step_params[0].get("sample_rate", SAMPLE_RATE))
        print(f"\nPlaying {n_steps} steps …  (press Ctrl-C to stop)")
        try:
            for step_i, wave in enumerate(waveforms):
                alpha = alphas[step_i]
                print(f"  Playing step {step_i + 1}/{n_steps}  α={alpha:.3f}", end="\r", flush=True)
                play_audio(wave, sr)
                time.sleep(0.1)  # brief pause between steps
            print()
        except KeyboardInterrupt:
            if _SOUNDDEVICE_AVAILABLE:
                sd.stop()
            print("\nPlayback interrupted.")

    # ------------------------------------------------------------------
    # 8. Visualisation (optional)
    # ------------------------------------------------------------------
    if plot:
        _plot_steps(waveforms, alphas, int(step_params[0].get("sample_rate", SAMPLE_RATE)))
        _plot_metrics(alphas, rms_values, lsd_values, metrics=metrics, features=features)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the non-linearity of the FM synthesizer by interpolating between two presets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset location
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root directory of the FM dataset (must contain a 'params/' sub-folder).",
    )

    # Preset selection
    sel_group = parser.add_mutually_exclusive_group()
    sel_group.add_argument(
        "--selection-mode",
        choices=["first_last", "max_distance"],
        default="first_last",
        help=(
            "How to choose the two endpoint presets. "
            "'first_last' uses the first and last files in the dataset; "
            "'max_distance' searches for the maximally different pair."
        ),
    )
    parser.add_argument(
        "--preset-a", type=int, default=None,
        metavar="INDEX",
        help="1-based index of preset A (overrides --selection-mode).",
    )
    parser.add_argument(
        "--preset-b", type=int, default=None,
        metavar="INDEX",
        help="1-based index of preset B (overrides --selection-mode).",
    )

    # Interpolation
    parser.add_argument(
        "--n-steps", type=int, default=9,
        help="Number of interpolation steps T (including endpoints).  Default is 9 as per the SpinVAE paper.",
    )

    # Output
    parser.add_argument(
        "--save-audio", action="store_true",
        help="Save each interpolated step as a WAV file.",
    )
    parser.add_argument(
        "--preview-dir", type=str, default="data/preview",
        help="Directory where interpolated WAV files are saved (with --save-audio).",
    )

    # Extras
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot waveforms and spectrograms for every step (requires matplotlib).",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play each interpolated step sequentially (requires sounddevice).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (used by --selection-mode max_distance).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate: if one of --preset-a / --preset-b is given, both must be given.
    if (args.preset_a is None) != (args.preset_b is None):
        print("Error: --preset-a and --preset-b must both be provided, or neither.")
        sys.exit(1)

    run_test(
        data_dir=args.data_dir,
        n_steps=args.n_steps,
        selection_mode=args.selection_mode,
        preset_a_idx=args.preset_a,
        preset_b_idx=args.preset_b,
        save_audio=args.save_audio,
        preview_dir=args.preview_dir,
        plot=args.plot,
        play=args.play,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
