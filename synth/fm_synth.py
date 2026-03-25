"""
Simple 2-operator FM synthesizer.

Signal model:
    fm  = fc * ratio
    x(t) = sin(2π·fc·t + M·sin(2π·fm·t))

An ADSR envelope is applied to the amplitude.
"""

import numpy as np

try:
    from scipy.io import wavfile as _wavfile
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000   # Hz
DURATION    = 1.0     # seconds


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def midi_to_freq(midi_note: int) -> float:
    """Convert a MIDI note number (0–127) to its fundamental frequency in Hz.

    Uses the standard equal-temperament formula relative to A4 = 440 Hz
    (MIDI note 69).
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


# ---------------------------------------------------------------------------
# ADSR envelope
# ---------------------------------------------------------------------------

def _adsr_envelope(
    n_samples: int,
    attack:  float,
    decay:   float,
    sustain: float,
    release: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Build a linear ADSR amplitude envelope of length *n_samples*.

    Parameters
    ----------
    n_samples   : total number of samples in the envelope.
    attack      : attack time in seconds (ramp 0 → 1).
    decay       : decay time in seconds (ramp 1 → sustain).
    sustain     : sustain *level* (0–1), held until release begins.
    release     : release time in seconds (ramp sustain → 0).
    sample_rate : samples per second.

    The four phases are placed consecutively.  If their combined length
    exceeds *n_samples* the later phases are clipped gracefully.
    """
    n_attack  = int(attack  * sample_rate)
    n_decay   = int(decay   * sample_rate)
    n_release = int(release * sample_rate)
    # Sustain fills whatever is left between decay and release
    n_sustain = max(0, n_samples - n_attack - n_decay - n_release)

    segments = []

    if n_attack > 0:
        segments.append(np.linspace(0.0, 1.0, n_attack, endpoint=False))

    if n_decay > 0:
        segments.append(np.linspace(1.0, sustain, n_decay, endpoint=False))

    if n_sustain > 0:
        segments.append(np.full(n_sustain, sustain))

    if n_release > 0:
        segments.append(np.linspace(sustain, 0.0, n_release, endpoint=True))

    envelope = np.concatenate(segments) if segments else np.array([])

    # Pad or clip to exactly n_samples
    if len(envelope) < n_samples:
        envelope = np.pad(envelope, (0, n_samples - len(envelope)))
    else:
        envelope = envelope[:n_samples]

    return envelope


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def generate_fm(params: dict) -> np.ndarray:
    """Generate a 2-operator FM synthesis waveform.

    Parameters
    ----------
    params : dict with keys:
        midi_note  (int)   : MIDI note number (0–127).
        velocity   (int)   : MIDI velocity (0–127); scales output amplitude.
        mod_ratio  (float) : fm = fc * mod_ratio.
        mod_index  (float) : modulation depth M.
        attack     (float) : ADSR attack time in seconds.
        decay      (float) : ADSR decay time in seconds.
        sustain    (float) : ADSR sustain level (0–1).
        release    (float) : ADSR release time in seconds.
        duration   (float) : optional, default DURATION (seconds).
        sample_rate(int)   : optional, default SAMPLE_RATE (Hz).

    Returns
    -------
    waveform : np.ndarray, dtype float32, values in [-1, 1].
    """
    sr       = int(params.get("sample_rate", SAMPLE_RATE))
    duration = float(params.get("duration",    DURATION))

    # --- FM parameters ---
    fc        = midi_to_freq(int(params["midi_note"]))
    velocity  = float(params.get("velocity", 64))
    mod_ratio = float(params.get("mod_ratio", 1.0))
    mod_index = float(params.get("mod_index", 1.0))

    # --- ADSR parameters ---
    attack  = float(params.get("attack",  0.01))
    decay   = float(params.get("decay",   0.1))
    sustain = float(params.get("sustain", 0.7))
    release = float(params.get("release", 0.2))

    n_samples = int(sr * duration)
    t         = np.linspace(0.0, duration, n_samples, endpoint=False)

    # --- 2-operator FM equation: x(t) = sin(2π·fc·t + M·sin(2π·fm·t)) ---
    fm       = fc * mod_ratio
    modulator = mod_index * np.sin(2.0 * np.pi * fm * t)
    carrier   = np.sin(2.0 * np.pi * fc * t + modulator)

    # --- ADSR amplitude envelope ---
    envelope = _adsr_envelope(n_samples, attack, decay, sustain, release, sr)

    waveform = carrier * envelope

    # --- Velocity scaling (0–127 → 0–1) ---
    waveform *= velocity / 127.0

    # --- Normalize to [-1, 1] ---
    peak = np.max(np.abs(waveform))
    if peak > 0.0:
        waveform /= peak

    return waveform.astype(np.float32)


# ---------------------------------------------------------------------------
# Random parameter sampling
# ---------------------------------------------------------------------------

def sample_random_params(rng: np.random.Generator | None = None) -> dict:
    """Return a dictionary of randomly sampled, valid FM synthesis parameters.

    Parameters
    ----------
    rng : optional numpy random generator.  A fresh default_rng() is used
          when *None* is passed.

    Returns
    -------
    params : dict suitable for passing directly to ``generate_fm``.
    """
    if rng is None:
        rng = np.random.default_rng()

    return {
        "midi_note":  int(rng.integers(36, 85)),          # C2 – C6
        "velocity":   int(rng.integers(40, 128)),         # moderate to full
        "mod_ratio":  float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])),
        "mod_index":  float(rng.uniform(0.0, 10.0)),
        "attack":     float(rng.uniform(0.001, 0.1)),
        "decay":      float(rng.uniform(0.01,  0.3)),
        "sustain":    float(rng.uniform(0.3,   1.0)),
        "release":    float(rng.uniform(0.05,  0.4)),
        "duration":   DURATION,
        "sample_rate": SAMPLE_RATE,
    }


# ---------------------------------------------------------------------------
# Optional: WAV export
# ---------------------------------------------------------------------------

def save_wav(waveform: np.ndarray, filename: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a float32 waveform to a 16-bit PCM WAV file.

    Requires scipy.  Raises RuntimeError if scipy is not available.
    """
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required to save WAV files.  Install it with: pip install scipy")

    # Convert float32 [-1, 1] → int16
    int16_wave = (waveform * 32767).astype(np.int16)
    _wavfile.write(filename, sample_rate, int16_wave)


# ---------------------------------------------------------------------------
# Optional: Batch generation
# ---------------------------------------------------------------------------

def batch_generate(presets: list[dict]) -> list[np.ndarray]:
    """Generate waveforms for a list of parameter dictionaries.

    Parameters
    ----------
    presets : list of parameter dicts, each compatible with ``generate_fm``.

    Returns
    -------
    waveforms : list of np.ndarray, one per preset.
    """
    return [generate_fm(p) for p in presets]
