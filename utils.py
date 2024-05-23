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


def additive_white_gaussian_noise_channel(x, snr):
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


def rayleigh_channel_filtered_gaussian(fmT, Omgp=1, sample_num=3000, device="cpu"):
    """
    Generate the in-phase and quadrature components of a Rayleigh fading channel with filtered Gaussian noise.

    Parameters:
    - fmT: The normalized Doppler frequency, fmT = f_m * T.
    - Omgp: The power spectral density of the Gaussian noise source.
    - sample_num: The number of samples to generate.
    - device: The device (CPU or GPU) where the tensors should be located.

    Returns:
    - gI: The in-phase component of the Rayleigh fading channel.
    - gQ: The quadrature component of the Rayleigh fading channel.
    """
    sigma = (
        2
        - torch.cos(torch.tensor(np.pi * fmT / 2))
        - torch.sqrt((2 - torch.cos(torch.tensor(np.pi * fmT / 2))) ** 2 - 1)
    ).to(device)

    var = (1 + sigma) / (1 - sigma) * Omgp / 2

    # Generate two independent white Gaussian noise sources for Gi and Gq
    w1 = torch.normal(0, torch.sqrt(var), size=(sample_num,), device=device)
    w2 = torch.normal(0, torch.sqrt(var), size=(sample_num,), device=device)

    # Initialize the in-phase (Gi) and quadrature (Gq) output arrays
    gI = torch.zeros(sample_num, device=device)
    gQ = torch.zeros(sample_num, device=device)
    gI[0] = 1  # Initial condition
    gQ[0] = 1  # Initial condition

    # Apply the first-order lowpass filter to generate Gi and Gq
    for j in range(1, sample_num):
        gI[j] = sigma * gI[j - 1] + (1 - sigma) * w1[j - 1]
        gQ[j] = sigma * gQ[j - 1] + (1 - sigma) * w2[j - 1]

    return gI, gQ


def rayleigh_channel_sum_of_sinusoids(fmT, M, T=1, sample_num=3000, device="cpu"):
    """
    Generate the in-phase and quadrature components of a Rayleigh fading channel using the sum of sinusoids method.

    Parameters:
    - fmT: The normalized Doppler frequency, fmT = f_m * T.
    - M: The number of sinusoids to use in the sum.
    - T: The symbol period.
    - sample_num: The number of samples to generate.
    - device: The device (CPU or GPU) where the tensors should be located.

    Returns:
    - gI: The in-phase component of the Rayleigh fading channel.
    - gQ: The quadrature component of the Rayleigh fading channel.
    """
    fm = torch.tensor(fmT / T, device=device)
    m = torch.arange(1, M + 1, device=device)
    N = 4 * M + 2
    n = torch.arange(1, N + 1, device=device)
    theta_n = 2 * torch.pi * n / N  # theta_n is uniformly distributed
    theta_m = theta_n[:M]
    beta_m = torch.pi * m / M
    alpha = torch.tensor(0.0, device=device)  # Ensure alpha is a tensor

    # Calculate the Doppler shifts for different angles
    fn = torch.outer(fm, torch.cos(theta_m))

    gI = torch.zeros(sample_num + 1, device=device)
    gQ = torch.zeros(sample_num + 1, device=device)

    # Calculate gI and gQ using sum of sinusoids
    for t in range(sample_num + 1):
        cos_component = torch.cos(2 * torch.pi * t * fn)
        gI[t] = 2 * torch.sum(torch.cos(beta_m) * cos_component) + torch.sqrt(
            torch.tensor(2.0, device=device)
        ) * torch.cos(alpha) * torch.cos(2 * torch.pi * fm * t)
        gQ[t] = 2 * torch.sum(torch.sin(beta_m) * cos_component) + torch.sqrt(
            torch.tensor(2.0, device=device)
        ) * torch.sin(alpha) * torch.cos(2 * torch.pi * fm * t)

    return gI, gQ
