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
# Visualisation
# ---------------------------------------------------------------------------

def _plot_steps(
    waveforms: list[np.ndarray],
    alphas: np.ndarray,
    sample_rate: int,
    title_prefix: str = "Interpolation",
) -> None:
    """Plot waveform and spectrogram for every interpolation step."""
    if not _MATPLOTLIB_AVAILABLE:
        print("[plot] matplotlib not installed – skipping plots.")
        return

    n_steps = len(waveforms)
    fig, axes = plt.subplots(
        n_steps, 2,
        figsize=(12, 2.5 * n_steps),
        squeeze=False,
    )
    fig.suptitle(f"{title_prefix}: waveforms and spectrograms", fontsize=14)

    for i, (wave, alpha) in enumerate(zip(waveforms, alphas)):
        duration = len(wave) / sample_rate
        t = np.linspace(0.0, duration, len(wave), endpoint=False)

        # --- Waveform ---
        ax_wave = axes[i][0]
        ax_wave.plot(t, wave, linewidth=0.5)
        ax_wave.set_ylim(-1.1, 1.1)
        ax_wave.set_ylabel(f"α={alpha:.2f}")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_title(f"Step {i + 1} waveform")

        # --- Spectrogram ---
        ax_spec = axes[i][1]
        ax_spec.specgram(
            wave.astype(np.float64),
            Fs=sample_rate,
            NFFT=256,
            noverlap=128,
            cmap="inferno",
        )
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title(f"Step {i + 1} spectrogram")

    plt.tight_layout()
    plt.show()


def _plot_metrics(
    alphas: np.ndarray,
    rms_values: list[float],
    lsd_values: list[float],
) -> None:
    """Plot RMS energy and log-spectral distance across interpolation steps."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Audio metrics across interpolation steps", fontsize=14)

    ax1.plot(alphas, rms_values, marker="o")
    ax1.set_xlabel("α (interpolation factor)")
    ax1.set_ylabel("RMS energy")
    ax1.set_title("RMS energy per step")
    ax1.grid(True)

    ax2.plot(alphas[1:], lsd_values, marker="o", color="orange")
    ax2.set_xlabel("α (interpolation factor)")
    ax2.set_ylabel("Log-spectral distance (LSD)")
    ax2.set_title("LSD between consecutive steps")
    ax2.grid(True)

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
        _plot_metrics(alphas, rms_values, lsd_values)

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
