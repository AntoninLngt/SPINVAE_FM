"""
Real-time playable FM synthesizer with keyboard control.

Reuses generate_fm() from fm_synth.py to synthesize a short waveform on each
key-press and plays it immediately via sounddevice.

Key layout (AZERTY keyboard):

    Upper black:   é  "     (  '  è        (C#5 D#5  F#5 G#5 A#5)
    Upper white:  A  Z  E  R  T  Y  U      (C5  D5  E5  F5  G5  A5  B5)
    Lower black:   S  D     G  H  J        (C#4 D#4  F#4 G#4 A#4)
    Lower white:  W  X  C  V  B  N  ,      (C4  D4  E4  F4  G4  A4  B4)

Controls:
    UP / DOWN arrows   → modulation index (+/- 0.5)
    LEFT / RIGHT arrows → modulation ratio (cycles through presets)
    + / -              → volume (+/- 10 %)
    Esc                → quit

Usage:
    python synth/realtime_synth.py
"""

from __future__ import annotations

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except (ImportError, OSError):
    _SD_AVAILABLE = False

try:
    from pynput import keyboard as kb
    _KB_AVAILABLE = True
except (ImportError, Exception):
    _KB_AVAILABLE = False
    kb = None  # type: ignore[assignment]

from synth.fm_synth import generate_fm, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTE_DURATION  = 0.8          # seconds per generated note
VOLUME_STEP    = 0.10         # fractional volume change per keypress
MOD_INDEX_STEP = 0.5          # modulation index change per keypress
MOD_INDEX_MAX  = 20.0         # upper cap for modulation index
MOD_RATIO_PRESETS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# ---------------------------------------------------------------------------
# Key → MIDI note mapping  (AZERTY keyboard layout)
# ---------------------------------------------------------------------------
#
#  Piano-like two-octave layout.  Physical key positions match AZERTY:
#
#  Upper black:   é  "     (  '  è        (C#5 D#5  F#5 G#5 A#5)
#  Upper white:  A  Z  E  R  T  Y  U      (C5  D5  E5  F5  G5  A5  B5)
#  Lower black:   S  D     G  H  J        (C#4 D#4  F#4 G#4 A#4)
#  Lower white:  W  X  C  V  B  N  ,      (C4  D4  E4  F4  G4  A4  B4)
#

KEY_NOTE_MAP: dict[str, int] = {
    # Lower octave – white keys (AZERTY bottom row: W X C V B N ,)
    "w": 60,  # C4
    "x": 62,  # D4
    "c": 64,  # E4
    "v": 65,  # F4
    "b": 67,  # G4
    "n": 69,  # A4
    ",": 71,  # B4
    # Lower octave – black keys (AZERTY home row: S D G H J)
    "s": 61,  # C#4
    "d": 63,  # D#4
    "g": 66,  # F#4
    "h": 68,  # G#4
    "j": 70,  # A#4
    # Upper octave – white keys (AZERTY top row: A Z E R T Y U)
    "a": 72,  # C5
    "z": 74,  # D5
    "e": 76,  # E5
    "r": 77,  # F5
    "t": 79,  # G5
    "y": 81,  # A5
    "u": 83,  # B5
    # Upper octave – black keys (AZERTY number row: é " ( ' è)
    # Physical key 6 on AZERTY produces '-', which is already used for volume
    # decrease, so it cannot double as a note key.  Physical key 4 (apostrophe)
    # is used for G#5 in its place.
    "é": 73,  # C#5  (physical key 2)
    '"': 75,  # D#5  (physical key 3)
    "(": 78,  # F#5  (physical key 5)
    "'": 80,  # G#5  (physical key 4 — key 6 reserved for volume)
    "è": 82,  # A#5  (physical key 7)
}

# ---------------------------------------------------------------------------
# Global FM parameter dictionary
# ---------------------------------------------------------------------------

_mod_ratio_idx: int = 1        # index into MOD_RATIO_PRESETS

params: dict = {
    "midi_note":   60,
    "velocity":    100,
    "mod_ratio":   MOD_RATIO_PRESETS[_mod_ratio_idx],
    "mod_index":   2.0,
    "attack":      0.01,
    "decay":       0.05,
    "sustain":     0.8,
    "release":     0.2,
    "duration":    NOTE_DURATION,
    "sample_rate": SAMPLE_RATE,
}

# ---------------------------------------------------------------------------
# Note playback
# ---------------------------------------------------------------------------


def play_note(midi_note: int) -> None:
    """Generate and immediately play one note using the current parameters."""
    note_params = {**params, "midi_note": midi_note}
    waveform = generate_fm(note_params)
    # Scale by volume (params["velocity"] / 127 is already applied inside
    # generate_fm, so we use a separate volume scalar stored externally)
    waveform = waveform * _volume
    if _SD_AVAILABLE:
        sd.play(waveform, samplerate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Volume (kept separate from params so it doesn't interfere with generate_fm)
# ---------------------------------------------------------------------------

_volume: float = 0.7

# ---------------------------------------------------------------------------
# Keyboard event handling
# ---------------------------------------------------------------------------


def _key_char(key) -> str | None:  # noqa: ANN001
    """Return the printable character for *key*, or None."""
    if kb is not None and isinstance(key, kb.KeyCode):
        return key.char
    return None


def handle_key_press(key) -> bool | None:  # noqa: ANN001
    """Handle one key-press event.  Returns False to stop the listener."""
    global _mod_ratio_idx, _volume

    char = _key_char(key)

    # ---- note keys ----
    if char and char.lower() in KEY_NOTE_MAP:
        midi_note = KEY_NOTE_MAP[char.lower()]
        play_note(midi_note)
        return None

    if kb is None:
        return None

    # ---- parameter controls ----
    if key == kb.Key.up:
        params["mod_index"] = min(MOD_INDEX_MAX, max(0.0, params["mod_index"] + MOD_INDEX_STEP))
        print(f"[FM] mod_index  = {params['mod_index']:.2f}")

    elif key == kb.Key.down:
        params["mod_index"] = min(MOD_INDEX_MAX, max(0.0, params["mod_index"] - MOD_INDEX_STEP))
        print(f"[FM] mod_index  = {params['mod_index']:.2f}")

    elif key == kb.Key.right:
        _mod_ratio_idx = (_mod_ratio_idx + 1) % len(MOD_RATIO_PRESETS)
        params["mod_ratio"] = MOD_RATIO_PRESETS[_mod_ratio_idx]
        print(f"[FM] mod_ratio  = {params['mod_ratio']:.2f}")

    elif key == kb.Key.left:
        _mod_ratio_idx = (_mod_ratio_idx - 1) % len(MOD_RATIO_PRESETS)
        params["mod_ratio"] = MOD_RATIO_PRESETS[_mod_ratio_idx]
        print(f"[FM] mod_ratio  = {params['mod_ratio']:.2f}")

    elif char in ("+", "="):
        _volume = min(1.0, _volume + VOLUME_STEP)
        print(f"[FM] volume     = {_volume:.2f}")

    elif char == "-":
        _volume = max(0.0, _volume - VOLUME_STEP)
        print(f"[FM] volume     = {_volume:.2f}")

    elif key == kb.Key.esc:
        return False   # stop the listener

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not _SD_AVAILABLE:
        raise RuntimeError(
            "sounddevice (PortAudio) is not available. "
            "Install PortAudio and run: pip install sounddevice"
        )
    if not _KB_AVAILABLE:
        raise RuntimeError(
            "pynput is not available or no display is accessible. "
            "Run: pip install pynput  (and ensure a display is available)"
        )

    print("=== Real-time FM Synthesizer (AZERTY) ===")
    print("  é  \"     (  '  è     ← upper black keys (C#5 D#5  F#5 G#5 A#5)")
    print("A  Z  E  R  T  Y  U   ← upper white keys (C5  D5  E5  F5  G5  A5  B5)")
    print("  S  D     G  H  J     ← lower black keys (C#4 D#4  F#4 G#4 A#4)")
    print("W  X  C  V  B  N  ,   ← lower white keys (C4  D4  E4  F4  G4  A4  B4)")
    print()
    print("Params: ↑↓ mod_index  |  ←→ mod_ratio  |  +/- volume  |  Esc quit")
    print()
    print(
        f"Initial params: mod_index={params['mod_index']:.2f}  "
        f"mod_ratio={params['mod_ratio']:.2f}  volume={_volume:.2f}"
    )

    with kb.Listener(on_press=handle_key_press) as listener:
        listener.join()

    print("\nBye!")


if __name__ == "__main__":
    main()

