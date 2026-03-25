"""Interactive FM synthesizer preset interpolation explorer.

Lets the user pick two FM synthesizer presets from the dataset by index
and listen to a 9-step linear interpolation between them.

Usage::

    # Interactive mode (prompts for preset indices)
    python scripts/interpolation_explorer.py

    # Non-interactive mode with explicit indices
    python scripts/interpolation_explorer.py --preset-a 1 --preset-b 5000

    # Show waveform / spectrogram plots
    python scripts/interpolation_explorer.py --plot

    # Custom number of interpolation steps
    python scripts/interpolation_explorer.py --n-steps 5

    # Change pause duration between playback steps (seconds)
    python scripts/interpolation_explorer.py --pause 0.3

    # Change dataset directory
    python scripts/interpolation_explorer.py --data-dir /path/to/data
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

from synth.fm_synth import SAMPLE_RATE, generate_fm  # noqa: E402

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    _SOUNDDEVICE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Interpolatable FM parameter keys.
# "midi_note" and "velocity" are fixed across the dataset and not interpolated.
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
# Core functions (public API)
# ---------------------------------------------------------------------------

def load_preset(index: int, data_dir: str = "data") -> dict:
    """Load an FM synthesizer preset by its 1-based index.

    Parameters
    ----------
    index    : 1-based preset index (1 … number of presets in the dataset).
    data_dir : Root directory of the dataset (must contain a ``params/``
               sub-folder with JSON files named ``sample_NNNNN.json``).

    Returns
    -------
    params : dict suitable for passing to ``generate_fm``.

    Raises
    ------
    FileNotFoundError : if the parameter directory or the requested file does
                        not exist.
    ValueError        : if *index* is out of range.
    """
    params_dir = os.path.join(data_dir, "params")
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(
            f"Parameter directory not found: {params_dir}\n"
            "Run  scripts/generate_dataset.py  first."
        )

    param_files = sorted(
        f for f in os.listdir(params_dir) if f.endswith(".json")
    )
    if not param_files:
        raise FileNotFoundError(
            f"No JSON parameter files found in {params_dir}.\n"
            "Run  scripts/generate_dataset.py  first."
        )

    n = len(param_files)
    if not (1 <= index <= n):
        raise ValueError(f"Index {index} out of range [1, {n}].")

    path = os.path.join(params_dir, param_files[index - 1])
    with open(path) as fh:
        return json.load(fh)


def interpolate_presets(
    preset_a: dict,
    preset_b: dict,
    T: int = 9,
) -> list[dict]:
    """Linearly interpolate between two FM synthesizer presets.

    Returns a list of *T* parameter dictionaries.  Step *t* (1-based) has
    interpolation factor ``alpha = (t-1) / (T-1)``, so:

    - step 1 → ``preset_a``  (alpha = 0)
    - step T → ``preset_b``  (alpha = 1)

    For each interpolatable key::

        interpolated_param[t] = preset_a[key] + (t-1)/(T-1) * (preset_b[key] - preset_a[key])

    All other keys (``midi_note``, ``velocity``, …) are copied unchanged
    from *preset_a*.

    Parameters
    ----------
    preset_a : FM parameter dict for the first endpoint.
    preset_b : FM parameter dict for the second endpoint.
    T        : Number of interpolation steps (default 9).

    Returns
    -------
    steps : list of *T* parameter dicts, from preset_a to preset_b.
    """
    if T < 2:
        raise ValueError(f"T must be at least 2, got {T}.")

    steps = []
    for t in range(1, T + 1):
        alpha = (t - 1) / (T - 1)
        step = dict(preset_a)  # copy fixed keys (midi_note, velocity, …)
        for key in _INTERPOLATE_KEYS:
            step[key] = float(
                preset_a[key] + alpha * (preset_b[key] - preset_a[key])
            )
        steps.append(step)
    return steps


def play_sound(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play a numpy waveform through the system's default audio output.

    Requires the optional *sounddevice* library.  Prints a warning and
    returns immediately when the library is not available.

    Parameters
    ----------
    waveform    : 1-D float32 array (values in [-1, 1]).
    sample_rate : Playback sample rate in Hz (default: ``SAMPLE_RATE``).
    """
    if not _SOUNDDEVICE_AVAILABLE:
        print("[play_sound] sounddevice not installed – skipping playback.")
        return
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()


def play_interpolation(
    interpolated_presets: list[dict],
    pause: float = 0.2,
) -> None:
    """Generate and play audio for each step in *interpolated_presets*.

    For each step the function:
    1. Synthesizes a waveform with ``generate_fm``.
    2. Prints the step number, alpha (interpolation factor), and parameter
       values.
    3. Plays the waveform via ``play_sound``.
    4. Waits *pause* seconds before the next step.

    Parameters
    ----------
    interpolated_presets : list of parameter dicts as returned by
                           ``interpolate_presets``.
    pause                : Seconds of silence inserted between consecutive
                           playback steps (default 0.2).
    """
    T = len(interpolated_presets)
    sample_rate = int(interpolated_presets[0].get("sample_rate", SAMPLE_RATE))

    # Print table header.
    header_parts = [f"{'Step':>4}", f"{'alpha':>6}"] + [
        f"{k:>12}" for k in _INTERPOLATE_KEYS
    ]
    header = "  ".join(header_parts)
    print("\n" + header)
    print("-" * len(header))

    try:
        for step_i, params in enumerate(interpolated_presets):
            alpha = step_i / (T - 1)
            row = [f"{step_i + 1:4d}", f"{alpha:6.3f}"] + [
                f"{params[k]:12.6g}" for k in _INTERPOLATE_KEYS
            ]
            print("  ".join(row))

            waveform = generate_fm(params)
            play_sound(waveform, sample_rate)

            if pause > 0 and step_i < T - 1:
                time.sleep(pause)

    except KeyboardInterrupt:
        if _SOUNDDEVICE_AVAILABLE:
            sd.stop()
        print("\nPlayback interrupted.")


# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------

def _plot_interpolation(
    interpolated_presets: list[dict],
) -> None:
    """Plot waveform and spectrogram for every interpolation step.

    Requires *matplotlib*; prints a warning and returns immediately if it is
    not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[plot] matplotlib not installed – skipping plots.")
        return

    T = len(interpolated_presets)
    sample_rate = int(interpolated_presets[0].get("sample_rate", SAMPLE_RATE))
    fig, axes = plt.subplots(T, 2, figsize=(12, 2.5 * T), squeeze=False)
    fig.suptitle("FM interpolation: waveforms and spectrograms", fontsize=14)

    for step_i, params in enumerate(interpolated_presets):
        alpha = step_i / (T - 1)
        wave = generate_fm(params)
        duration = len(wave) / sample_rate
        t_axis = np.linspace(0.0, duration, len(wave), endpoint=False)

        ax_wave = axes[step_i][0]
        ax_wave.plot(t_axis, wave, linewidth=0.5)
        ax_wave.set_ylim(-1.1, 1.1)
        ax_wave.set_ylabel(f"α={alpha:.2f}")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_title(f"Step {step_i + 1} waveform")

        ax_spec = axes[step_i][1]
        ax_spec.specgram(
            wave.astype(np.float64),
            Fs=sample_rate,
            NFFT=256,
            noverlap=128,
            cmap="inferno",
        )
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title(f"Step {step_i + 1} spectrogram")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interactive main loop
# ---------------------------------------------------------------------------

def _prompt_index(prompt: str, n_presets: int) -> int:
    """Prompt the user for a preset index in the range [1, n_presets]."""
    while True:
        try:
            raw = input(prompt).strip()
            value = int(raw)
            if 1 <= value <= n_presets:
                return value
            print(f"  Please enter a number between 1 and {n_presets}.")
        except ValueError:
            print("  Invalid input – please enter an integer.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)


def _count_presets(data_dir: str) -> int:
    """Return the number of JSON parameter files in *data_dir*/params/."""
    params_dir = os.path.join(data_dir, "params")
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(
            f"Parameter directory not found: {params_dir}\n"
            "Run  scripts/generate_dataset.py  first."
        )
    count = sum(1 for f in os.listdir(params_dir) if f.endswith(".json"))
    if count == 0:
        raise FileNotFoundError(
            f"No JSON parameter files found in {params_dir}.\n"
            "Run  scripts/generate_dataset.py  first."
        )
    return count


def main(
    data_dir: str = "data",
    T: int = 9,
    pause: float = 0.2,
    plot: bool = False,
    preset_a_idx: int | None = None,
    preset_b_idx: int | None = None,
) -> None:
    """Interactive main loop for the preset interpolation explorer.

    Parameters
    ----------
    data_dir     : Root directory of the FM dataset.
    T            : Number of interpolation steps (default 9).
    pause        : Seconds between consecutive playback steps.
    plot         : When *True*, show waveform / spectrogram plots.
    preset_a_idx : If provided together with *preset_b_idx*, skip the
                   interactive prompt and use these 1-based indices directly.
    preset_b_idx : See *preset_a_idx*.
    """
    n_presets = _count_presets(data_dir)
    print(f"Dataset loaded: {n_presets} presets in '{data_dir}/params'.")
    print(f"Interpolation steps T = {T}.\n")

    while True:
        # ----------------------------------------------------------------
        # 1. Choose two presets
        # ----------------------------------------------------------------
        if preset_a_idx is not None and preset_b_idx is not None:
            idx_a, idx_b = preset_a_idx, preset_b_idx
        else:
            print(f"Enter two preset indices (1–{n_presets}).")
            idx_a = _prompt_index(f"  Preset A index [1–{n_presets}]: ", n_presets)
            idx_b = _prompt_index(f"  Preset B index [1–{n_presets}]: ", n_presets)

        # ----------------------------------------------------------------
        # 2. Load presets
        # ----------------------------------------------------------------
        preset_a = load_preset(idx_a, data_dir)
        preset_b = load_preset(idx_b, data_dir)

        print(f"\nPreset A (index {idx_a}):")
        for k in _INTERPOLATE_KEYS:
            print(f"    {k:12s} = {preset_a[k]:.6g}")

        print(f"\nPreset B (index {idx_b}):")
        for k in _INTERPOLATE_KEYS:
            print(f"    {k:12s} = {preset_b[k]:.6g}")

        # ----------------------------------------------------------------
        # 3. Interpolate
        # ----------------------------------------------------------------
        print(f"\nInterpolating {T} steps between preset {idx_a} and preset {idx_b} …")
        interpolated = interpolate_presets(preset_a, preset_b, T)

        # ----------------------------------------------------------------
        # 4. Play all steps
        # ----------------------------------------------------------------
        play_interpolation(interpolated, pause=pause)

        # ----------------------------------------------------------------
        # 5. Optional visualisation
        # ----------------------------------------------------------------
        if plot:
            _plot_interpolation(interpolated)

        # ----------------------------------------------------------------
        # 6. Continue or quit (only when running interactively)
        # ----------------------------------------------------------------
        if preset_a_idx is not None and preset_b_idx is not None:
            break  # non-interactive mode: run once and exit

        try:
            again = input("\nTry another pair? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if again not in ("y", "yes"):
            break

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive FM synthesizer preset interpolation explorer.\n"
            "Choose two presets and listen to a 9-step linear interpolation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root directory of the FM dataset (must contain a 'params/' sub-folder).",
    )
    parser.add_argument(
        "--preset-a", type=int, default=None, metavar="INDEX",
        help="1-based index of preset A (skips interactive prompt).",
    )
    parser.add_argument(
        "--preset-b", type=int, default=None, metavar="INDEX",
        help="1-based index of preset B (skips interactive prompt).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=9,
        help="Number of interpolation steps T (including endpoints).",
    )
    parser.add_argument(
        "--pause", type=float, default=0.2,
        help="Seconds of silence between consecutive playback steps.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot waveforms and spectrograms for every step (requires matplotlib).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if (args.preset_a is None) != (args.preset_b is None):
        print("Error: --preset-a and --preset-b must both be provided, or neither.")
        sys.exit(1)

    main(
        data_dir=args.data_dir,
        T=args.n_steps,
        pause=args.pause,
        plot=args.plot,
        preset_a_idx=args.preset_a,
        preset_b_idx=args.preset_b,
    )
