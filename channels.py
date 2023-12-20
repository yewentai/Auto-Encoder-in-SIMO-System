import numpy as np


def additive_white_gaussian_noise_channel(
    x, n_rx=4, attenuation=0.95, noise_power=1e-5
):
    """
    Add AWGN to a signal

    Parameters:
    - x: Signal
    - n_rx: Number of receive antennas
    - attenuation: Channel attenuation
    - noise_power: Noise power

    Returns:
    - y: Signal with AWGN
    """

    w = np.sqrt(noise_power / 2) * (
        np.random.normal(0, 1, n_rx) + 1j * np.random.normal(0, 1, n_rx)
    )

    y = attenuation * x + w

    return y


def line_of_sight_simo_channel(
    x,
    n_rx=4,
    delta_r=0.5,
    distance=3,
    wavelength=0.1,
    phi=0.1,
    attenuation=0.95,
    noise_power=1e-5,
):
    """
    Simulate a line-of-sight SIMO channel

    Parameters:
    - x: Transmitted symbol
    - n_rx: Number of receive antennas
    - delta_r: Normalized receive antenna separation
    - distance: Distance from transmit antenna to the first receive antenna (in meters)
    - wavelength: Wavelength (in meters)
    - phi: Angle of incidence of the line-of-sight onto the receive antenna array
    - attenuation: Channel attenuation
    - noise_power: Noise power

    Returns:
    - y: Received symbol
    - h: Channel spatial signature

    Reference:
    - [1] Manikas, A. (2019). Modelling of SIMO, MISO, and MIMO Antenna Array. Lecture slides presented in EE401: Advanced Communication Theory at Imperial College London.

    """

    c = 3e8  # Speed of light

    # Calculate spatial signature
    omega = np.cos(phi)
    h = (
        attenuation
        * np.exp(-1j * 2 * np.pi * distance / wavelength)
        * np.exp(-1j * 2 * np.pi * np.arange(n_rx) * delta_r * omega)
    )

    w = np.sqrt(noise_power / 2) * (
        np.random.normal(0, 1, n_rx) + 1j * np.random.normal(0, 1, n_rx)
    )

    y = h * x + w

    return y, h
