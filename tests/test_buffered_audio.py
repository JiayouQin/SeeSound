import unittest
import numpy as np
from seesound.buffered_audio import BufferedAudio, OscillatorBank


class BufferedAudioTests(unittest.TestCase):
    def test_waveform_is_continuous_across_render_blocks(self):
        whole = OscillatorBank([80, 440, 8000], 48000)
        split = OscillatorBank([80, 440, 8000], 48000)
        target = [0.3, 0.4, 0.2]
        expected = whole.render(target, 4096)
        actual = np.concatenate([split.render(target, count) for count in [513, 1024, 2559]])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_violin_vibrato_is_continuous_and_restores_from_buffer(self):
        whole = OscillatorBank([80, 440, 7040], 48000, timbre="violin")
        split = OscillatorBank([80, 440, 7040], 48000, timbre="violin")
        target = [0.8, 0.5, 0.2]
        expected = whole.render(target, 8192)
        actual = np.concatenate([split.render(target, n) for n in [1024, 3000, 4168]])
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        resumed = OscillatorBank([80, 440, 7040], 48000, timbre="violin")
        resumed.restore(split.state())
        np.testing.assert_allclose(resumed.render(target, 1024), split.render(target, 1024), atol=1e-7)
        self.assertEqual(len(resumed.harmonics), 16)

    def test_violin_voices_have_independent_motion_and_glide_only_at_onset(self):
        bank = OscillatorBank([440, 440], 48000, timbre="violin")
        self.assertNotEqual(bank.vibrato_rate[0], bank.vibrato_rate[1])
        self.assertFalse(np.array_equal(bank.drift_phase[0], bank.drift_phase[1]))
        bank.render([1, 1], 1024)
        initial_glide = bank.glide_offset.copy()
        self.assertTrue(np.all(initial_glide < 0))
        bank.render([0.5, 0.5], 1024)
        self.assertTrue(np.all(np.abs(bank.glide_offset) < np.abs(initial_glide)))
        bank.render([1, 1], 24000)
        self.assertTrue(np.all(np.abs(bank.glide_offset) < 0.001))
        self.assertNotAlmostEqual(bank.phases[0], bank.phases[1])

    def test_silenced_oscillator_keeps_advancing_phase(self):
        bank = OscillatorBank([440], 48000)
        bank.render([0], 1024)
        expected = (2 * np.pi * 440 * 1024 / 48000) % (2 * np.pi)
        self.assertAlmostEqual(bank.phases[0], expected)

    def test_amplitude_changes_are_smoothed_per_sample(self):
        bank = OscillatorBank([80], 48000)
        bank.render([1], 1)
        self.assertLess(bank.amplitudes[0], 0.001)
        bank.render([1], 48000)
        self.assertGreater(bank.amplitudes[0], 0.99)
        previous = bank.amplitudes[0]
        bank.render([0], 1)
        self.assertLess(previous - bank.amplitudes[0], 0.001)

    def test_dense_mix_has_headroom_without_clipping(self):
        bank = OscillatorBank(np.geomspace(80, 8000, 32), 48000)
        bank.amplitudes[:] = 1
        samples = bank.render(np.ones(32), 48000)
        self.assertLessEqual(float(np.max(np.abs(samples))), 0.650001)

    def test_warm_timbre_has_quiet_harmonics_and_softer_treble(self):
        low = OscillatorBank([440], 48000)
        low.amplitudes[:] = 1
        spectrum = np.abs(np.fft.rfft(low.render([1], 48000)))
        self.assertGreater(spectrum[880] / spectrum[440], 0.05)
        self.assertLess(spectrum[880] / spectrum[440], 0.15)
        self.assertLess(spectrum[1320] / spectrum[440], 0.04)
        high = OscillatorBank([7040], 48000)
        high.amplitudes[:] = 1
        high_samples = high.render([1], 48000)
        self.assertLess(float(np.sqrt(np.mean(high_samples ** 2))), 0.05)

    def test_buffer_handles_variable_reads_and_continues_when_empty(self):
        audio = BufferedAudio([80, 440], 48000, 1024)
        samples = audio.synth.render([0.8, 0.4], 1024)
        state = audio.synth.state()
        audio.queue.put((samples, state))
        first = np.empty((700, 1), dtype=np.float32)
        audio.callback(first, 700, None, None)
        second = np.empty((324, 1), dtype=np.float32)
        audio.callback(second, 324, None, None)
        np.testing.assert_array_equal(np.concatenate([first[:, 0], second[:, 0]]), samples)
        reference = OscillatorBank([80, 440], 48000)
        reference.restore(state)
        expected = reference.render(reference.amplitudes.copy(), 2048)
        continued = np.empty((2048, 1), dtype=np.float32)
        audio.callback(continued, 2048, None, None)
        np.testing.assert_allclose(continued[:, 0], expected, atol=1e-7)
        self.assertGreater(float(np.max(np.abs(continued))), 0.01)
        self.assertEqual(audio.buffer_underruns, 1)
        # Restored queued audio blends back in instead of switching abruptly.
        audio.queue.put((audio.synth.render([0.8, 0.4], 1024), audio.synth.state()))
        resumed = np.empty((1024, 1), dtype=np.float32)
        next_sample = reference.render(reference.amplitudes.copy(), 1)[0]
        audio.callback(resumed, 1024, None, None)
        self.assertLess(abs(resumed[0, 0] - next_sample), 0.002)
        self.assertTrue(np.all(np.isfinite(resumed)))


if __name__ == '__main__':
    unittest.main()
