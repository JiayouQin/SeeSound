"""Approximate spectral position from display hue, not RGB channel magnitude.

RGB display colors do not uniquely specify an optical spectrum. These wavelength
anchors define a perceptual red-to-violet approximation; brightness weights volume.
"""
import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
RED_NM = 700.0
VIOLET_NM = 380.0
MIN_LIGHT_HZ = SPEED_OF_LIGHT / (RED_NM * 1e-9)
MAX_LIGHT_HZ = SPEED_OF_LIGHT / (VIOLET_NM * 1e-9)
HUE_DEGREES = np.array([0, 30, 60, 120, 180, 240, 270, 300, 330, 360])
WAVELENGTH_NM = np.array([700, 620, 580, 530, 490, 450, 400, 380, 620, 700])


def hue_to_light_position(hue_degrees):
    wavelength_nm = np.interp(np.asarray(hue_degrees) % 360, HUE_DEGREES, WAVELENGTH_NM)
    light_hz = SPEED_OF_LIGHT / (wavelength_nm * 1e-9)
    return np.clip((light_hz - MIN_LIGHT_HZ) / (MAX_LIGHT_HZ - MIN_LIGHT_HZ), 0, 1)


def light_position_to_hue(position):
    light_hz = MIN_LIGHT_HZ + np.asarray(position) * (MAX_LIGHT_HZ - MIN_LIGHT_HZ)
    wavelength_nm = SPEED_OF_LIGHT / light_hz * 1e9
    return np.interp(wavelength_nm, WAVELENGTH_NM[:8][::-1], HUE_DEGREES[:8][::-1])
