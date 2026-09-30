"""Continuous oscillators with sample-level envelopes and a bounded audio FIFO."""
from queue import Queue, Empty, Full
import threading
import numpy as np


class OscillatorBank:
    def __init__(self, frequencies, sample_rate, gain=0.25, smoothing_seconds=0.12, timbre="warm"):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.sample_rate = sample_rate
        self.gain = gain
        if timbre not in {"warm", "violin"}:
            raise ValueError(f"Unknown instrument: {timbre}")
        self.timbre = timbre
        self.smoothing_seconds = 0.18 if timbre == "violin" else smoothing_seconds
        self.release_seconds = 0.28 if timbre == "violin" else smoothing_seconds
        rng = np.random.default_rng(1729)
        voices = len(frequencies)
        # Independent rates, depths and phases avoid synchronized ensemble wobble.
        self.vibrato_phase = rng.uniform(0, 2 * np.pi, voices)
        self.vibrato_depth = 2 ** (rng.uniform(7, 11, voices) / 1200) - 1
        self.vibrato_rate = rng.uniform(4.8, 5.6, voices)
        # Three unrelated slow motions create bounded, irregular pitch drift.
        self.drift_phase = rng.uniform(0, 2 * np.pi, (voices, 3))
        self.drift_rate = rng.uniform(0.07, 0.31, (voices, 3))
        self.drift_depth = np.full((voices, 3), (2 ** (3 / 1200) - 1) / 3)
        self.glide_offset = np.zeros(voices)
        self.previous_target = np.zeros(voices)
        self.glide_seconds = 0.14
        self.phases = np.zeros(len(frequencies), dtype=np.float64)
        self.amplitudes = np.zeros(len(frequencies), dtype=np.float64)
        self.harmonics = []
        if timbre == "violin":
            # Bowed strings have many partials, shaped by the body resonance.
            for multiple in range(1, 17):
                partial_hz = self.frequencies * multiple
                body = (0.55 + 0.65 * np.exp(-0.5 * ((partial_hz - 500) / 260) ** 2)
                        + 0.5 * np.exp(-0.5 * ((partial_hz - 1800) / 650) ** 2)
                        + 0.3 * np.exp(-0.5 * ((partial_hz - 2800) / 700) ** 2))
                rolloff = 1 / np.sqrt(1 + (partial_hz / 3500) ** 4)
                weights = body * rolloff * np.exp(-(multiple - 1) / 12) / multiple
                weights *= partial_hz * (1 + self.vibrato_depth + np.sum(self.drift_depth, axis=1)) < sample_rate * 0.45
                self.harmonics.append((multiple, weights))
            normalization = max(float(np.max(sum(w for _, w in self.harmonics))), 1e-6)
        else:
            # Rounded sine-led tone with quiet second and third harmonics.
            for multiple, level in [(1, 1.0), (2, 0.12), (3, 0.03)]:
                partial_hz = self.frequencies * multiple
                warmth = 1 / np.sqrt(1 + (partial_hz / 1800) ** 2)
                weights = level * warmth * (partial_hz < sample_rate * 0.45)
                self.harmonics.append((multiple, weights))
            normalization = 1.15
        self.harmonics = [(n, weights / normalization) for n, weights in self.harmonics]
        self.voice_bounds = sum(weights for _, weights in self.harmonics)

    def render(self, target, frames):
        target = np.asarray(target, dtype=np.float64)
        indices = np.arange(frames, dtype=np.float64)
        smoothing = np.where(target >= self.amplitudes, self.smoothing_seconds, self.release_seconds)
        decay = np.exp(-(indices[None, :] + 1) / (self.sample_rate * smoothing[:, None]))
        envelopes = target[:, None] + (self.amplitudes - target)[:, None] * decay
        increments = 2 * np.pi * self.frequencies / self.sample_rate
        if self.timbre == "violin":
            # A returning note approaches its pentatonic center from slightly below.
            # Do not retrigger on ordinary level changes in a sustained voice.
            onset = (target > 0.01) & (self.previous_target <= 0.01) & (self.amplitudes < 0.02)
            self.glide_offset[onset] = 2 ** (-18 / 1200) - 1
            samples = np.arange(frames + 1, dtype=np.float64)
            positions = np.broadcast_to(samples, (len(target), frames + 1)).copy()
            omega = 2 * np.pi * self.vibrato_rate
            vibrato = self.vibrato_phase[:, None] + omega[:, None] * samples / self.sample_rate
            positions += (self.vibrato_depth * self.sample_rate / omega)[:, None] * (
                np.cos(self.vibrato_phase)[:, None] - np.cos(vibrato))
            for component in range(3):
                drift_omega = 2 * np.pi * self.drift_rate[:, component]
                phase = self.drift_phase[:, component]
                drift = phase[:, None] + drift_omega[:, None] * samples / self.sample_rate
                positions += (self.drift_depth[:, component] * self.sample_rate / drift_omega)[:, None] * (
                    np.cos(phase)[:, None] - np.cos(drift))
            glide_decay = np.exp(-samples / (self.sample_rate * self.glide_seconds))
            positions += self.glide_offset[:, None] * self.sample_rate * self.glide_seconds * (1 - glide_decay)
            advance = positions[:, -1].copy()
            positions = positions[:, :-1]
            self.vibrato_phase = vibrato[:, -1] % (2 * np.pi)
            self.drift_phase = (self.drift_phase + 2 * np.pi * self.drift_rate * frames / self.sample_rate) % (2 * np.pi)
            self.glide_offset *= glide_decay[-1]
        else:
            positions = indices[None, :]
            advance = frames
        self.previous_target = target.copy()
        angles = self.phases[:, None] + increments[:, None] * positions
        waves = np.zeros_like(angles)
        for multiple, weights in self.harmonics:
            waves += weights[:, None] * np.sin(multiple * angles)
        mixed = np.sum(envelopes * waves, axis=0)
        # Reserve headroom from the envelopes, avoiding waveform distortion
        # from passing the combined voices through a tanh limiter.
        peak_bound = np.sum(envelopes * self.voice_bounds[:, None], axis=0)
        mix_gain = self.gain / np.maximum(1.0, self.gain * peak_bound / 0.65)
        output = mixed * mix_gain
        # Every oscillator advances, including ones that are currently silent.
        self.phases = (self.phases + increments * advance) % (2 * np.pi)
        self.amplitudes = envelopes[:, -1].copy()
        return output.astype(np.float32)

    def state(self):
        return tuple(item.copy() for item in (
            self.phases, self.amplitudes, self.vibrato_phase, self.drift_phase,
            self.glide_offset, self.previous_target))

    def restore(self, state):
        (self.phases, self.amplitudes, self.vibrato_phase, self.drift_phase,
         self.glide_offset, self.previous_target) = (item.copy() for item in state)


class BufferedAudio:
    def __init__(self, frequencies, sample_rate, block_size, gain=0.25, buffer_seconds=0.20, timbre="warm"):
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.queue = Queue(maxsize=max(2, int(np.ceil(buffer_seconds * sample_rate / block_size))))
        self.synth = OscillatorBank(frequencies, sample_rate, gain, timbre=timbre)
        self.continuation = OscillatorBank(frequencies, sample_rate, gain, timbre=timbre)
        self.target = np.zeros(len(frequencies), dtype=np.float64)
        self.target_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.worker = None
        self.pending = None
        self.offset = 0
        self.in_fallback = False
        self.crossfade_left = 0
        self.crossfade_frames = max(1, int(sample_rate * 0.01))
        self.buffer_underruns = 0
        self.device_underruns = 0

    def set_target(self, amplitudes):
        with self.target_lock:
            self.target = np.asarray(amplitudes, dtype=np.float64).copy()

    def start(self):
        self.worker = threading.Thread(target=self._produce, name="audio-synthesis", daemon=True)
        self.worker.start()
        if not self.ready.wait(timeout=3):
            self.close()
            raise RuntimeError("Could not prefill the audio buffer")

    def _produce(self):
        while not self.stop_event.is_set():
            with self.target_lock:
                target = self.target.copy()
            samples = self.synth.render(target, self.block_size)
            packet = (samples, self.synth.state())
            while not self.stop_event.is_set():
                try:
                    self.queue.put(packet, timeout=0.05)
                    if self.queue.full():
                        self.ready.set()
                    break
                except Full:
                    pass

    def callback(self, outdata, frames, time_info, status):
        if status and status.output_underflow:
            self.device_underruns += 1
        written = 0
        while written < frames:
            if self.pending is None:
                try:
                    self.pending = self.queue.get_nowait()
                    self.offset = 0
                    if self.in_fallback:
                        self.crossfade_left = self.crossfade_frames
                        self.in_fallback = False
                except Empty:
                    if not self.in_fallback:
                        self.buffer_underruns += 1
                    self.in_fallback = True
                    # Continue the last played oscillators instead of inserting silence
                    # or repeating a waveform whose endpoints might not join.
                    outdata[written:, 0] = self.continuation.render(
                        self.continuation.amplitudes.copy(), frames - written)
                    return
            samples, state = self.pending
            take = min(frames - written, len(samples) - self.offset)
            outdata[written:written+take, 0] = samples[self.offset:self.offset+take]
            if self.crossfade_left:
                count = min(take, self.crossfade_left)
                continued = self.continuation.render(self.continuation.amplitudes.copy(), count)
                progress = self.crossfade_frames - self.crossfade_left
                mix = (np.arange(count) + progress + 1) / self.crossfade_frames
                destination = outdata[written:written+count, 0]
                destination[:] = continued * (1 - mix) + destination * mix
                self.crossfade_left -= count
            self.offset += take
            written += take
            if self.offset == len(samples):
                self.continuation.restore(state)
                self.pending = None

    def close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1)
