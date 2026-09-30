"""Choose distinct pitches from the C-major pentatonic scale."""
import numpy as np


def pentatonic_frequencies(count, min_hz=80.0, max_hz=8000.0):
    # C, D, E, G, A repeated across octaves; MIDI 69 is A4 = 440 Hz.
    midi_notes = np.arange(128)
    hz = 440.0 * 2.0 ** ((midi_notes - 69) / 12.0)
    in_scale = np.isin(midi_notes % 12, [0, 2, 4, 7, 9])
    candidates = hz[in_scale & (hz >= min_hz) & (hz <= max_hz)]
    if count < 1 or len(candidates) < count:
        raise ValueError("Pitch range must contain enough distinct pentatonic notes")
    indices = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    return candidates[indices]
