# SeeSound

SeeSound turns the colors inside a movable screen frame into a continuous,
synthesized violin ensemble. It uses 32 distinct C-major pentatonic notes
(C, D, E, G, A across octaves), with red at the low end and violet/purple at
the high end. Brighter, more saturated colors contribute louder voices.

![SeeSound capture frame and live violin preview showing a cockatiel](<Pasted image.png>)

The teal frame selects the screen area; the preview below shows its color-frequency histogram.

Each voice has gentle independent vibrato, slow pitch drift, and a short
upward glide when it enters. Sample-level amplitude smoothing, a 200 ms
synthesis buffer, and additional device buffering keep playback continuous.
If synthesis falls behind, the audio continues the last played tones and
crossfades back to the queued audio.

## Requirements

- Python 3.10 or newer.
- An active Linux X11 desktop, including X11/Xext libraries. The transparent
  frame currently uses the X11 Shape extension; native Wayland, Windows,
  and macOS capture frames are not supported.
- Tkinter and PortAudio, plus an available audio output device.

On Ubuntu/Debian, install the system dependencies if needed:

```bash
sudo apt-get install python3-venv python3-tk libportaudio2 libx11-6 libxext6
```

## Setup and run

From the project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

If the existing project environment is already set up, run only the last
command. The app also recognizes the project-local PortAudio library at
`.venv/portaudio/usr/lib/x86_64-linux-gnu/libportaudio.so.2` when a system
installation is unavailable.

## Controls

- Drag the teal frame's top bar to move the capture region.
- Drag its lower-right corner to resize it.
- Only the clear area inside the frame supplies the sound.
- A separate preview shows the captured region and its frequency histogram.
  Keep the preview outside the capture frame to avoid visual feedback.
- Press **Esc** in either window, or close the preview, to stop.

If no audio output is available, the preview remains open and displays an
error in its status line. Grayscale and very low-saturation regions can
produce little or no sound.

## Configuration

Edit the settings near the top of `seesound/app.py` to change master volume,
image smoothing, or the permitted pitch range. The current 32 notes span
approximately 82–7,040 Hz. The range must contain at least 32 distinct
pentatonic notes to preserve all voices.

Instrument envelopes, harmonics, vibrato, drift, and buffering are defined
in `seesound/buffered_audio.py`. The application currently selects the
`violin` timbre; `warm` is also available.

The optical-frequency mapping is an approximation from display hue using
representative wavelengths and `f = c / wavelength`. Screen RGB values do
not uniquely identify a physical light spectrum. The audible pitches are
then mapped to pentatonic notes. The violin sound is synthesized, rather
than recorded instrument samples.

## Project layout

```text
main.py                    Application entry point
readme.md                  Setup, controls, and implementation notes
requirements.txt           Python dependencies
seesound/
    app.py                 Desktop UI and screen-to-audio pipeline
    screen_region.py       Movable transparent X11 capture frame
    light_frequency.py     Approximate color-to-optical-frequency mapping
    musical_scale.py       Distinct pentatonic pitches
    buffered_audio.py      Instrument synthesis and buffered playback
tests/
    test_buffered_audio.py Audio continuity and instrument tests
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests check waveform continuity, buffer fallback, smooth envelopes,
headroom, harmonics, and independent violin pitch motion without opening
windows or accessing audio devices.
