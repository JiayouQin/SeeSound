"""SeeSound desktop application and capture-to-audio pipeline."""

import cv2
import numpy as np
import ctypes.util
from pathlib import Path
from PIL import ImageGrab, Image, ImageTk
import tkinter as tk
import time
from .screen_region import CaptureRegion
from .light_frequency import hue_to_light_position, light_position_to_hue

from .buffered_audio import BufferedAudio
from .musical_scale import pentatonic_frequencies

# ----------------------------
# Settings
# ----------------------------

NUM_BINS = 32

AUDIO_MIN_HZ = 80.0
AUDIO_MAX_HZ = 8000.0

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024

SATURATION_MIN = 0.10

# Histogram temporal smoothing.
# Higher = smoother/slower response.
SMOOTHING = 0.85

# Overall loudness.
MASTER_GAIN = 0.25

# Reduce image resolution before histogram computation.
PROCESS_WIDTH = 320


def main():
    """Run the capture window and buffered violin synthesizer."""
    # Support the project-local PortAudio installation when the system lacks it.
    if ctypes.util.find_library("portaudio") is None:
        local_portaudio = Path(__file__).resolve().parents[1] / ".venv/portaudio/usr/lib/x86_64-linux-gnu/libportaudio.so.2"
        if local_portaudio.exists():
            original_find_library = ctypes.util.find_library
            ctypes.util.find_library = lambda name: str(local_portaudio) if name == "portaudio" else original_find_library(name)

    import sounddevice as sd

    # ----------------------------
    # Buffered audio synthesis
    # ----------------------------

    # Keep all 32 optical-frequency bins, assigning each a distinct ascending
    # C-major pentatonic note (C, D, E, G, A) within the existing pitch range.
    frequencies = pentatonic_frequencies(NUM_BINS, AUDIO_MIN_HZ, AUDIO_MAX_HZ)
    audio = BufferedAudio(frequencies, SAMPLE_RATE, BLOCK_SIZE, MASTER_GAIN, timbre="violin")

    # ----------------------------
    # Screen capture and preview window
    # ----------------------------

    WINDOW_NAME = "Capture Region -> Sound"

    root = tk.Tk()
    root.title(WINDOW_NAME)
    root.geometry("960x580+1800+200")
    status_label = tk.Label(root, text="Move the capture frame to choose the source | ESC: quit")
    status_label.pack()
    preview_label = tk.Label(root)
    preview_label.pack(fill="both", expand=True)
    running = True

    def close_window(event=None):
        nonlocal running
        running = False

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.bind("<Escape>", close_window)
    region = CaptureRegion(root, close_window)
    root.update()

    stream = None
    try:
        audio.start()
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=audio.callback,
            latency=0.10,
        )
        stream.start()
    except (sd.PortAudioError, OSError) as exc:
        if stream is not None:
            stream.close()
            stream = None
        audio.close()
        status_label.config(text=f"Capture region | Audio unavailable: {exc} | ESC: quit")
        print(f"Audio unavailable: {exc}", flush=True)

    smoothed_hist = np.zeros(NUM_BINS, dtype=np.float32)

    try:

        while running:

            screenshot = ImageGrab.grab(bbox=region.bbox()).convert("RGB")
            frame = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2BGR)

            # ----------------------------
            # Downsample
            # ----------------------------

            h, w = frame.shape[:2]

            scale = PROCESS_WIDTH / w

            small = cv2.resize(
                frame,
                (
                    PROCESS_WIDTH,
                    int(h * scale)
                )
            )

            # ----------------------------
            # BGR -> HSV
            # ----------------------------

            hsv = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2HSV
            ).astype(np.float32)

            # OpenCV hue uses half-degrees. Estimate wavelength, then compute f = c / wavelength.
            light_position = hue_to_light_position(hsv[:, :, 0] * 2.0)

            saturation = hsv[:, :, 1] / 255.0
            brightness = hsv[:, :, 2] / 255.0

            # ----------------------------
            # Pixel contribution
            # ----------------------------

            valid = saturation > SATURATION_MIN

            # Bright, saturated pixels contribute more strongly.
            weights = brightness * saturation

            weights[~valid] = 0.0

            # ----------------------------
            # Estimated optical-frequency histogram
            # ----------------------------

            hist, _ = np.histogram(
                light_position,
                bins=NUM_BINS,
                range=(0.0, 1.0),
                weights=weights
            )

            hist = hist.astype(np.float32)

            # Normalize relative spectral shape.
            total = hist.sum()

            if total > 0:
                hist /= total

            # Overall brightness controls total loudness.
            scene_brightness = float(np.mean(brightness))

            hist *= scene_brightness

            # ----------------------------
            # Temporal smoothing
            # ----------------------------

            smoothed_hist = (
                SMOOTHING * smoothed_hist
                + (1.0 - SMOOTHING) * hist
            )

            # sqrt makes weak bins more audible.
            amplitudes = np.sqrt(
                np.clip(smoothed_hist, 0.0, None)
            )

            audio.set_target(amplitudes)

            # ----------------------------
            # Visualize
            # ----------------------------

            display = frame.copy()

            display_h, display_w = display.shape[:2]

            max_bar_height = 150
            bar_width = max(1, display_w // NUM_BINS)

            max_value = max(
                smoothed_hist.max(),
                1e-6
            )

            for i, value in enumerate(smoothed_hist):

                height = int(
                    (value / max_value)
                    * max_bar_height
                )

                x1 = i * bar_width
                x2 = min(
                    display_w - 1,
                    x1 + bar_width - 1
                )

                y1 = display_h - height
                y2 = display_h

                # Color each bar by its estimated optical frequency.
                hsv_color = np.uint8([[
                    [
                        int(light_position_to_hue(i / (NUM_BINS - 1)) / 2),
                        255,
                        255
                    ]
                ]])

                bgr_color = cv2.cvtColor(
                    hsv_color,
                    cv2.COLOR_HSV2BGR
                )[0, 0]

                color = tuple(
                    int(v) for v in bgr_color
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    -1
                )

            cv2.putText(
                display,
                "Violin | Pentatonic | Red: low | Violet: high | ESC: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            preview = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            preview.thumbnail((max(1, preview_label.winfo_width()), max(1, preview_label.winfo_height())))
            photo = ImageTk.PhotoImage(preview)
            preview_label.configure(image=photo)
            preview_label.image = photo
            root.update()
            time.sleep(0.033)

    finally:

        if stream is not None:
            stream.stop()
            stream.close()
        audio.close()
        region.close()
        root.destroy()
