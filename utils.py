# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import erfc

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("Inputs list has to be at least length:2")

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        x = x / torch.sqrt(2 * torch.mean(x**2))
        return x


class Decoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("Inputs list has to be at least lenght:2")
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        x = F.log_softmax(self.layers[-1](x), dim=-1)
        return x


def awgn(x, snr):
    """
    Add AWGN noise to the input signal.

    Parameters:
    - x: Input signal.
    - snr: Signal to Noise Ratio in dB.

    Returns:
    - x: Noisy signal.
    """

    sigma = torch.tensor(np.sqrt(0.5 / (10 ** (snr / 10)))).to(device)
    noise = sigma * torch.randn(x.shape).to(device)
    x = x + noise
    return x


def ser_mqam_awgn(M, SNR_dB):
    """
    Calculate the Symbol Error Rate (SER) for M-QAM modulation in AWGNC. Refer to https://dsplog.com/2012/01/01/symbol-error-rate-16qam-64qam-256qam/

    Parameters:
    - SNR_dB: Energy per symbol to noise power spectral density ratio in dB.
    - M: Modulation order (e.g., 16 for 16-QAM).

    Returns:
    - SER: Symbol Error Rate for M-QAM.
    """
    # Convert Es/N0 from dB to linear scale
    SNR_linear = 10 ** (SNR_dB / 10)

    # Normalization factor
    k = np.sqrt(1 / ((2 / 3) * (M - 1)))

    # Calculate SER
    SER = (
        2 * (1 - 1 / np.sqrt(M)) * erfc(k * np.sqrt(SNR_linear))
        - (1 - 2 / np.sqrt(M) + 1 / M) * erfc(k * np.sqrt(SNR_linear)) ** 2
    )

    return SER
