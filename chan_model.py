from typing import Tuple
import numpy as np


def channel_model(
    carrier_freq_GHz: float = 2.1e9,
    distance: float = 51,
    num_antenna_tx: int = 10,
    num_antenna_rx: int = 1,
    path_loss_exponent: float = 2,
    K: float = 5,
) -> np.ndarray:
    c = 3e8
    lambda_ = c / carrier_freq_GHz
    antenna_spacing = 0.5 * lambda_

    def PL_dB(d: float, ple: float) -> float:
        return 20 * np.log10(lambda_ / (4 * np.pi * d)) + 10 * ple * np.log10(d)

    def dB_to_linear(dB_val: float) -> float:
        return 10 ** (dB_val / 10)

    pl = PL_dB(distance, path_loss_exponent)

    rice = []
    for _ in range(500):
        LOS_component = np.sqrt(K / (K + 1)) * np.random.normal(
            loc=0, scale=np.sqrt(2) / 2, size=(num_antenna_tx, 2)
        ).view(np.complex128)
        scattered_component = np.sqrt(1 / (K + 1)) * np.random.normal(
            loc=0, scale=np.sqrt(2) / 2, size=(num_antenna_tx, 2)
        ).view(np.complex128)
        rice.append((LOS_component + scattered_component).T)

    return np.sqrt(dB_to_linear(pl)) * np.mean(rice, 0)


# Example usage:
result = channel_model(
    carrier_freq_GHz=2.1e9,
    distance=51,
    num_antenna_tx=1,
    num_antenna_rx=4,
    path_loss_exponent=2,
    K=5,
)
print(result)
