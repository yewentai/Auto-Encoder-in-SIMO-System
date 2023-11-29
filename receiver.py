import numpy as np


def maximal_ratio_combining(received_signals):
    """
    Maximal Ratio Combining (MRC) receiver.

    Parameters:
    - received_signals (list of complex arrays): List of received signals from different antennas.

    Returns:
    - combined_signal (complex array): Combined signal using MRC.
    """
    num_antennas = len(received_signals)

    # Calculate channel gains (amplitudes)
    channel_gains = [np.abs(signal) for signal in received_signals]

    # Normalize channel gains
    normalized_gains = [gain / np.linalg.norm(channel_gains) for gain in channel_gains]

    # Apply weights to each received signal
    weighted_signals = [
        normalized_gains[i] * received_signals[i] for i in range(num_antennas)
    ]

    # Sum up the weighted signals for MRC
    combined_signal = np.sum(weighted_signals, axis=0)

    return combined_signal
