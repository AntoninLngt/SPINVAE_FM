"""
Test the Phase 2 VAE by interpolating between two FM presets in latent space.

This script:
  1. Loads a checkpoint (default: ``checkpoints/phase2_final.pt``).
  2. Loads two preset JSON files (A and B) from ``data/params/``.
  3. Normalizes the presets and encodes them to the latent space via the
     VAE encoder (deterministic: returns the posterior mean μ).
  4. Linearly interpolates **T = 9 steps** in the latent space::

         z_t = z_a + (t-1)/(T-1) * (z_b - z_a)   for t = 1 … T

  5. Decodes each latent vector back to the (normalized) parameter space and
     then denormalizes to recover physical FM parameters.
  6. Optionally synthesises audio for every decoded preset via the FM synth.
  7. Computes per-step metrics: RMS energy and log-spectral distance between
     consecutive steps.
  8. Prints decoded presets and metrics to stdout.
  9. Produces two visualisation figures: waveforms and spectrograms.
  10. Optionally saves each step's audio as a WAV file.
  11. Optionally plays back each step in sequence.

Usage::

    python scripts/test_vae.py \\
        --preset_a sample_00001.json \\
        --preset_b sample_00002.json \\
        --steps 9

    # Save audio and show plots
    python scripts/test_vae.py --preset_a sample_00001.json \\
        --preset_b sample_00002.json --save_audio --plot

    # Use a different checkpoint
    python scripts/test_vae.py --preset_a sample_00001.json \\
        --preset_b sample_00002.json \\
        --checkpoint checkpoints/phase3_final.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Allow running from the repository root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore[assignment]

from dataset.dataset import (  # noqa: E402
    PARAM_MAX,
    PARAM_MIN,
    PARAM_NAMES,
    denormalize_params,
    normalize_params,
)
from models.vae import PresetVAE  # noqa: E402
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
# STFT / spectrogram parameters (consistent with test_nonlinearity.py)
# ---------------------------------------------------------------------------
_STFT_N_FFT = 2048
_STFT_HOP = 512
_MEL_N_FFT = 2048
_MEL_HOP = 512
_MEL_N_MELS = 128


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a YAML config file, falling back to hard-coded defaults."""
    if yaml is not None and os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {
        "data": {
            "metadata_csv": "data/metadata.csv",
            "splits_dir":   "data/splits",
            "audio_dir":    "data/audio",
        },
        "model":    {"input_dim": 6, "hidden_dim": 32, "latent_dim": 16},
        "training": {"seed": 42},
        "output":   {"checkpoint_dir": "checkpoints"},
    }


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

def _load_preset(json_path: str) -> dict:
    """Load a preset parameter dictionary from a JSON file."""
    with open(json_path) as fh:
        return json.load(fh)


def _resolve_preset_path(preset_arg: str, data_dir: str) -> str:
    """Resolve a preset argument to an absolute file path.

    Accepts:
    - An absolute path.
    - A path relative to the current working directory.
    - A filename (e.g. ``sample_00001.json``) resolved under
      ``<data_dir>/params/``.
    """
    if os.path.isabs(preset_arg) or os.path.isfile(preset_arg):
        return os.path.abspath(preset_arg)
    candidate = os.path.join(data_dir, "params", preset_arg)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Cannot find preset '{preset_arg}'.  "
        f"Tried '{preset_arg}' and '{candidate}'."
    )


def _preset_to_tensor(params: dict, device: torch.device) -> torch.Tensor:
    """Normalize a preset dict and return a (1, 6) float32 tensor."""
    raw = np.array([float(params[k]) for k in PARAM_NAMES], dtype=np.float32)
    norm = normalize_params(raw)
    return torch.tensor(norm, dtype=torch.float32, device=device).unsqueeze(0)


def _tensor_to_preset(
    tensor: torch.Tensor,
    base_params: dict,
) -> dict:
    """Denormalize a (1, 6) or (6,) tensor and merge with fixed fields.

    Fixed fields (``midi_note``, ``velocity``, ``duration``, ``sample_rate``)
    are copied from *base_params* so that the decoded preset can be passed
    directly to ``generate_fm``.
    """
    norm_array = tensor.squeeze().cpu().numpy().astype(np.float32)
    raw = denormalize_params(norm_array)
    preset = {name: float(raw[i]) for i, name in enumerate(PARAM_NAMES)}
    # Copy fixed / metadata fields from the source preset.
    for key in ("midi_note", "velocity", "duration", "sample_rate"):
        if key in base_params:
            preset[key] = base_params[key]
    return preset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(
    checkpoint_path: str,
    model_cfg: dict,
    device: torch.device,
) -> PresetVAE:
    """Instantiate a PresetVAE and load weights from a checkpoint."""
    model = PresetVAE(
        input_dim  = int(model_cfg.get("input_dim",  6)),
        hidden_dim = int(model_cfg.get("hidden_dim", 32)),
        latent_dim = int(model_cfg.get("latent_dim", 16)),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Latent-space interpolation
# ---------------------------------------------------------------------------

def interpolate_latent(
    model:    PresetVAE,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    T:        int,
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Encode A and B, interpolate T steps in latent space, decode each.

    Parameters
    ----------
    model    : Trained PresetVAE in eval mode.
    tensor_a : (1, input_dim) normalized preset tensor for preset A.
    tensor_b : (1, input_dim) normalized preset tensor for preset B.
    T        : Number of interpolation steps (must be ≥ 2).

    Returns
    -------
    decoded  : List of T decoded (1, input_dim) tensors (normalized outputs).
    alphas   : 1-D numpy array of T interpolation factors in [0, 1].
    """
    with torch.no_grad():
        mu_a, _ = model.encode(tensor_a)   # (1, latent_dim)
        mu_b, _ = model.encode(tensor_b)   # (1, latent_dim)

    alphas = np.linspace(0.0, 1.0, T, dtype=np.float64)

    decoded: list[torch.Tensor] = []
    with torch.no_grad():
        for alpha in alphas:
            z_t = (1.0 - alpha) * mu_a + alpha * mu_b
            u_hat = model.decode(z_t)
            decoded.append(u_hat)

    return decoded, alphas


# ---------------------------------------------------------------------------
# Audio rendering
# ---------------------------------------------------------------------------

def render_audio(params: dict) -> np.ndarray:
    """Synthesise a waveform from an FM parameter dictionary."""
    return generate_fm(params)


def play_audio(waveform: np.ndarray, sample_rate: int) -> None:
    """Play a waveform through the default audio output (requires sounddevice)."""
    if not _SOUNDDEVICE_AVAILABLE:
        print("[play_audio] sounddevice not installed – skipping playback.")
        return
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rms(waveform: np.ndarray) -> float:
    """Root-mean-square energy of a waveform."""
    return float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))


def log_spectral_distance(wave_a: np.ndarray, wave_b: np.ndarray) -> float:
    """Log-spectral distance (LSD) between two same-length waveforms.

    LSD = sqrt( mean( (log10(|X_a(f)|²) - log10(|X_b(f)|²))² ) )
    """
    n = min(len(wave_a), len(wave_b))
    eps = 1e-10
    spec_a = np.abs(np.fft.rfft(wave_a[:n].astype(np.float64))) ** 2
    spec_b = np.abs(np.fft.rfft(wave_b[:n].astype(np.float64))) ** 2
    log_diff = np.log10(spec_a + eps) - np.log10(spec_b + eps)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def compute_stft_db(waveform: np.ndarray) -> np.ndarray:
    """Log-magnitude (dB) STFT spectrogram.

    Uses librosa when available, falls back to numpy otherwise.
    Returns a 2-D array *(n_freqs, n_frames)* in dB.
    """
    wave = waveform.astype(np.float32)
    if _LIBROSA_AVAILABLE:
        stft = librosa.stft(wave, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP)
        return librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    # Numpy fallback.
    n = len(wave)
    window = np.hanning(_STFT_N_FFT)
    specs = []
    for start in range(0, n - _STFT_N_FFT + 1, _STFT_HOP):
        frame = wave[start: start + _STFT_N_FFT] * window
        specs.append(np.abs(np.fft.rfft(frame)))
    S = np.array(specs).T if specs else np.zeros((_STFT_N_FFT // 2 + 1, 1))
    eps = 1e-10
    return 20.0 * np.log10(S + eps) - 20.0 * np.log10(np.max(S) + eps)


def spectral_distance(wave_a: np.ndarray, wave_b: np.ndarray) -> float:
    """STFT-based spectral distance (RMS of dB spectrogram difference)."""
    db_a = compute_stft_db(wave_a)
    db_b = compute_stft_db(wave_b)
    n_frames = min(db_a.shape[1], db_b.shape[1])
    diff = db_a[:, :n_frames] - db_b[:, :n_frames]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_trajectory_metrics(waveforms: list[np.ndarray]) -> dict:
    """Compute trajectory metrics for a list of interpolated waveforms.

    Returns
    -------
    dict with keys:
    - ``consecutive_distances`` – spectral distance between adjacent steps (T-1)
    - ``distances_to_target`` – spectral distance to the last step (T)
    - ``cumulative_distances`` – running sum of consecutive distances (T)
    - ``straight_distance`` – spectral distance from first to last step
    - ``linearity`` – straight_distance / total_path (1.0 = perfectly linear)
    - ``smoothness`` – std of consecutive_distances (lower = smoother)
    """
    n = len(waveforms)
    consec = [
        spectral_distance(waveforms[i], waveforms[i + 1])
        for i in range(n - 1)
    ]
    to_target = [spectral_distance(w, waveforms[-1]) for w in waveforms]
    cumulative = [0.0]
    for d in consec:
        cumulative.append(cumulative[-1] + d)
    straight = spectral_distance(waveforms[0], waveforms[-1])
    total_path = cumulative[-1]
    linearity = straight / total_path if total_path > 0.0 else 1.0
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
# Visualisation
# ---------------------------------------------------------------------------

def plot_waveforms(
    waveforms: list[np.ndarray],
    alphas: np.ndarray,
    sample_rate: int,
    save_path: str | None = None,
) -> None:
    """Plot one waveform per interpolation step in a single figure.

    Parameters
    ----------
    waveforms   : List of mono waveforms (one per step).
    alphas      : Interpolation factors corresponding to each step.
    sample_rate : Sample rate in Hz.
    save_path   : If provided, save the figure to this path instead of
                  displaying it interactively.
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
    fig.suptitle("VAE latent interpolation: waveforms", fontsize=14)

    for i, (wave, alpha) in enumerate(zip(waveforms, alphas)):
        duration = len(wave) / sample_rate
        t = np.linspace(0.0, duration, len(wave), endpoint=False)
        ax = axes[0][i]
        ax.plot(t, wave, linewidth=0.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Step {i + 1}\nα={alpha:.2f}")
        if i == 0:
            ax.set_ylabel("Amplitude")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Waveform plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_spectrograms(
    waveforms: list[np.ndarray],
    sample_rate: int,
    alphas: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Plot one mel spectrogram per interpolation step in a single figure.

    Uses librosa when available; falls back to a plain STFT spectrogram
    (via numpy) when librosa is not installed.

    Parameters
    ----------
    waveforms   : List of waveforms (one per step).
    sample_rate : Sample rate in Hz.
    alphas      : Optional interpolation factors for subplot titles.
    save_path   : If provided, save the figure to this path.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[plot] matplotlib not installed – skipping spectrogram plot.")
        return

    n_steps = len(waveforms)
    fig, axes = plt.subplots(
        1, n_steps,
        figsize=(3 * n_steps, 3),
        squeeze=False,
    )
    fig.suptitle("VAE latent interpolation: spectrograms", fontsize=14)

    if _LIBROSA_AVAILABLE:
        # Pre-compute all mel spectrograms (dB).
        db_specs = []
        for wave in waveforms:
            mono = wave.astype(np.float32).squeeze()
            if mono.ndim > 1:
                mono = librosa.to_mono(mono.T)
            S = librosa.feature.melspectrogram(
                y=mono,
                sr=sample_rate,
                n_fft=_MEL_N_FFT,
                hop_length=_MEL_HOP,
                n_mels=_MEL_N_MELS,
            )
            db_specs.append(librosa.power_to_db(S, ref=np.max))

        global_vmin = min(s.min() for s in db_specs)
        global_vmax = max(s.max() for s in db_specs)

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
            title = f"Step {i + 1}"
            if alphas is not None:
                title += f"\nα={alphas[i]:.2f}"
            ax.set_title(title)
    else:
        # Numpy fallback: log-magnitude STFT.
        db_specs = [compute_stft_db(w) for w in waveforms]
        global_vmin = min(s.min() for s in db_specs)
        global_vmax = max(s.max() for s in db_specs)
        for i, S_db in enumerate(db_specs):
            ax = axes[0][i]
            ax.imshow(
                S_db,
                aspect="auto",
                origin="lower",
                vmin=global_vmin,
                vmax=global_vmax,
                cmap="viridis",
            )
            title = f"Step {i + 1}"
            if alphas is not None:
                title += f"\nα={alphas[i]:.2f}"
            ax.set_title(title)
            ax.set_xlabel("Frame")
            if i == 0:
                ax.set_ylabel("Frequency bin")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Spectrogram plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_metrics(
    alphas: np.ndarray,
    rms_values: list[float],
    lsd_values: list[float],
    metrics: dict | None = None,
    save_path: str | None = None,
) -> None:
    """Plot per-step RMS, log-spectral distance, and trajectory metrics.

    Layout: 1 row × 3 columns:
    - [0] RMS energy per step
    - [1] Log-spectral distance between consecutive steps
    - [2] Cumulative trajectory distance (with straight-line reference)

    Parameters
    ----------
    alphas     : Interpolation factors.
    rms_values : Per-step RMS energy (T values).
    lsd_values : LSD between consecutive steps (T-1 values).
    metrics    : Optional dict from :func:`compute_trajectory_metrics`.
    save_path  : If provided, save the figure to this path.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[plot] matplotlib not installed – skipping metrics plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("VAE interpolation metrics", fontsize=14)

    # [0] RMS energy.
    ax = axes[0]
    ax.plot(alphas, rms_values, marker="o")
    ax.set_xlabel("α (interpolation factor)")
    ax.set_ylabel("RMS energy")
    ax.set_title("RMS energy per step")
    ax.grid(True)

    # [1] Log-spectral distance between consecutive steps.
    ax = axes[1]
    if lsd_values:
        ax.plot(alphas[1:], lsd_values, marker="o", color="orange")
    ax.set_xlabel("α (interpolation factor)")
    ax.set_ylabel("Log-spectral distance")
    ax.set_title("LSD between consecutive steps")
    ax.grid(True)

    # [2] Cumulative trajectory distance.
    ax = axes[2]
    if metrics is not None:
        cumulative = metrics["cumulative_distances"]
        straight = metrics["straight_distance"]
        ax.plot(alphas, cumulative, marker="o", color="green")
        ax.axhline(
            straight,
            linestyle="--",
            color="red",
            label=f"Straight-line ({straight:.2f})",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("α (interpolation factor)")
    ax.set_ylabel("Cumulative spectral distance")
    ax.set_title("Cumulative trajectory distance")
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Metrics plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run_test(
    preset_a_path: str,
    preset_b_path: str,
    checkpoint:    str,
    config_path:   str,
    T:             int,
    device:        torch.device,
    data_dir:      str,
    save_audio:    bool,
    audio_out_dir: str,
    plot:          bool,
    save_plots:    bool,
    plots_dir:     str,
    play:          bool,
) -> None:
    """End-to-end VAE interpolation test.

    Parameters
    ----------
    preset_a_path  : Absolute path to preset A JSON file.
    preset_b_path  : Absolute path to preset B JSON file.
    checkpoint     : Path to the VAE checkpoint (.pt file).
    config_path    : Path to the YAML config file.
    T              : Number of interpolation steps.
    device         : Torch device.
    data_dir       : Root data directory.
    save_audio     : Whether to save each step's audio as a WAV file.
    audio_out_dir  : Directory for saved WAV files.
    plot           : Whether to show / save visualisation plots.
    save_plots     : Whether to save plots to disk instead of displaying.
    plots_dir      : Directory for saved plot images.
    play           : Whether to play back each step's audio.
    """
    cfg = _load_config(config_path)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train the VAE first with: python scripts/train_model.py"
        )
    print(f"\nLoading checkpoint: {checkpoint}")
    model = _load_model(checkpoint, cfg.get("model", {}), device)
    latent_dim = model.encoder.fc_mu.out_features
    print(
        f"Model: input_dim={model.encoder.fc1.in_features}, "
        f"hidden_dim={model.encoder.fc1.out_features}, "
        f"latent_dim={latent_dim}"
    )

    # ------------------------------------------------------------------
    # Load presets
    # ------------------------------------------------------------------
    print(f"\nPreset A: {preset_a_path}")
    print(f"Preset B: {preset_b_path}")
    params_a = _load_preset(preset_a_path)
    params_b = _load_preset(preset_b_path)

    tensor_a = _preset_to_tensor(params_a, device)
    tensor_b = _preset_to_tensor(params_b, device)

    # ------------------------------------------------------------------
    # Encode → interpolate in latent space → decode
    # ------------------------------------------------------------------
    print(f"\nInterpolating {T} steps in latent space …")
    decoded_tensors, alphas = interpolate_latent(model, tensor_a, tensor_b, T)

    # Convert decoded tensors to FM parameter dicts.
    decoded_presets: list[dict] = [
        _tensor_to_preset(t, params_a)
        for t in decoded_tensors
    ]

    # ------------------------------------------------------------------
    # Print decoded presets
    # ------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print(f"  {'Step':>4}  {'α':>6}  " + "  ".join(f"{n:>10}" for n in PARAM_NAMES))
    print(f"{'─' * 70}")
    for step_idx, (alpha, preset) in enumerate(zip(alphas, decoded_presets), 1):
        values = "  ".join(f"{preset[n]:>10.4f}" for n in PARAM_NAMES)
        print(f"  {step_idx:>4}  {alpha:>6.3f}  {values}")
    print(f"{'─' * 70}")

    # ------------------------------------------------------------------
    # Synthesise audio
    # ------------------------------------------------------------------
    print("\nSynthesising audio for each step …")
    waveforms: list[np.ndarray] = [render_audio(p) for p in decoded_presets]

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    rms_values = [rms(w) for w in waveforms]
    lsd_values = [
        log_spectral_distance(waveforms[i], waveforms[i + 1])
        for i in range(T - 1)
    ]
    metrics = compute_trajectory_metrics(waveforms)

    print(f"\n{'─' * 50}")
    print(f"  {'Step':>4}  {'α':>6}  {'RMS':>10}  {'LSD (→next)':>14}")
    print(f"{'─' * 50}")
    for step_idx, (alpha, rms_val) in enumerate(zip(alphas, rms_values), 1):
        lsd_str = f"{lsd_values[step_idx - 1]:>14.4f}" if step_idx <= len(lsd_values) else f"{'—':>14}"
        print(f"  {step_idx:>4}  {alpha:>6.3f}  {rms_val:>10.5f}  {lsd_str}")
    print(f"{'─' * 50}")
    print(
        f"\n  Linearity : {metrics['linearity']:.4f}  "
        f"(1.0 = perfectly linear trajectory)"
    )
    print(
        f"  Smoothness: {metrics['smoothness']:.4f}  "
        f"(lower std = more uniform step spacing)"
    )
    print(f"  Total path  : {metrics['cumulative_distances'][-1]:.4f}")
    print(f"  Straight-line: {metrics['straight_distance']:.4f}")

    # ------------------------------------------------------------------
    # Save audio
    # ------------------------------------------------------------------
    if save_audio:
        os.makedirs(audio_out_dir, exist_ok=True)
        for step_idx, (wave, preset) in enumerate(zip(waveforms, decoded_presets), 1):
            out_path = os.path.join(audio_out_dir, f"interp_step_{step_idx:02d}.wav")
            save_wav(wave, out_path)
            print(f"  Saved: {out_path}")

    # ------------------------------------------------------------------
    # Play audio
    # ------------------------------------------------------------------
    if play:
        print("\nPlaying back interpolated steps …")
        for step_idx, wave in enumerate(waveforms, 1):
            print(f"  Step {step_idx}/{T} …")
            play_audio(wave, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    if plot:
        wave_path = os.path.join(plots_dir, "vae_interp_waveforms.png") if save_plots else None
        spec_path = os.path.join(plots_dir, "vae_interp_spectrograms.png") if save_plots else None
        met_path  = os.path.join(plots_dir, "vae_interp_metrics.png") if save_plots else None
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)

        plot_waveforms(waveforms, alphas, SAMPLE_RATE, save_path=wave_path)
        plot_spectrograms(waveforms, SAMPLE_RATE, alphas=alphas, save_path=spec_path)
        plot_metrics(alphas, rms_values, lsd_values, metrics=metrics, save_path=met_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the Phase 2 VAE by interpolating between two FM presets "
            "in latent space."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset_a", type=str, required=True,
        help=(
            "Preset A: filename in data/params/ (e.g. sample_00001.json) "
            "or an absolute/relative path to a JSON file."
        ),
    )
    parser.add_argument(
        "--preset_b", type=str, required=True,
        help="Preset B: filename or path (same format as --preset_a).",
    )
    parser.add_argument(
        "--steps", type=int, default=9,
        help="Number of interpolation steps T (must be ≥ 2).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to a VAE checkpoint (.pt).  Defaults to "
            "checkpoints/phase2_final.pt (or phase3_final.pt if it exists)."
        ),
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Root data directory (contains params/ sub-folder).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help=(
            "Compute device (e.g. 'cpu', 'cuda', 'mps').  "
            "Auto-detected when not specified."
        ),
    )
    parser.add_argument(
        "--save_audio", action="store_true",
        help="Save each interpolation step as a WAV file.",
    )
    parser.add_argument(
        "--audio_out_dir", type=str, default="data/audio/interpolations",
        help="Output directory for saved WAV files (used with --save_audio).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show (or save with --save_plots) waveform and spectrogram plots.",
    )
    parser.add_argument(
        "--save_plots", action="store_true",
        help="Save plots to --plots_dir instead of displaying interactively.",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="data/plots",
        help="Output directory for saved plot images (used with --save_plots).",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play each step's audio in sequence (requires sounddevice).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.steps < 2:
        print("Error: --steps must be ≥ 2.")
        sys.exit(1)

    # ---- Device ---------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Resolve data directory (relative to repo root) -----------------
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_REPO_ROOT, data_dir)

    # ---- Resolve preset paths -------------------------------------------
    preset_a_path = _resolve_preset_path(args.preset_a, data_dir)
    preset_b_path = _resolve_preset_path(args.preset_b, data_dir)

    # ---- Resolve checkpoint ---------------------------------------------
    if args.checkpoint:
        checkpoint = args.checkpoint
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(_REPO_ROOT, checkpoint)
    else:
        ckpt_dir = os.path.join(_REPO_ROOT, "checkpoints")
        phase3 = os.path.join(ckpt_dir, "phase3_final.pt")
        phase2 = os.path.join(ckpt_dir, "phase2_final.pt")
        checkpoint = phase3 if os.path.isfile(phase3) else phase2

    # ---- Config path ----------------------------------------------------
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    # ---- Audio / plot output dirs ---------------------------------------
    audio_out_dir = args.audio_out_dir
    if not os.path.isabs(audio_out_dir):
        audio_out_dir = os.path.join(_REPO_ROOT, audio_out_dir)

    plots_dir = args.plots_dir
    if not os.path.isabs(plots_dir):
        plots_dir = os.path.join(_REPO_ROOT, plots_dir)

    run_test(
        preset_a_path  = preset_a_path,
        preset_b_path  = preset_b_path,
        checkpoint     = checkpoint,
        config_path    = config_path,
        T              = args.steps,
        device         = device,
        data_dir       = data_dir,
        save_audio     = args.save_audio,
        audio_out_dir  = audio_out_dir,
        plot           = args.plot,
        save_plots     = args.save_plots,
        plots_dir      = plots_dir,
        play           = args.play,
    )


if __name__ == "__main__":
    main()
