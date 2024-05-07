import torch
import numpy as np


def rrc_filter(roll_off, num_taps, upsampling_factor):
    """Generate a Root Raised Cosine (RRC) filter."""
    T = upsampling_factor
    t = np.linspace(-num_taps // 2, num_taps // 2, num_taps)
    eps = 1e-5  # To avoid division by zero

    rrc = np.zeros_like(t, dtype=np.float32)
    for i in range(len(t)):
        x = np.pi * t[i] / T
        if abs(x) < eps:
            rrc[i] = 1.0
        else:
            num = np.sin(np.pi * t[i] * (1 - roll_off) / T) + 4 * roll_off * t[
                i
            ] / T * np.cos(np.pi * t[i] * (1 + roll_off) / T)
            den = np.pi * t[i] * (1 - (4 * roll_off * t[i] / T) ** 2)
            rrc[i] = num / den
        rrc[i] *= T * np.sinc(t[i] / T)

    return rrc / np.sqrt(np.sum(rrc**2))


def stochastic_channel(x, gamma, alpha, L, tau_bound, fs, sigma_CFO, sigma_noise):
    """
    Stochastic channel model for a wireless communication system.

    Parameters:
    - x: Input signal.
    - gamma: Upsampling factor.
    - alpha: Roll-off factor of the RRC filter.
    - L: Number of taps of the RRC filter.
    - tau_bound: Maximum time offset.
    - fs: Sampling frequency.
    - sigma_CFO: Standard deviation of the CFO.
    - sigma_noise: Standard deviation of the AWGN.
    """

    device = x.device
    dtype = x.dtype

    # Upsampling & Pulse shaping
    N = x.shape[0]
    upsampling_factor = gamma
    x_upsampled = torch.zeros(upsampling_factor * N, device=device, dtype=dtype)
    x_upsampled[::gamma] = x
    # Convert the RRC filter to a complex tensor with zero imaginary part
    rrc_real = torch.tensor(
        rrc_filter(alpha, L, gamma), device=device, dtype=torch.float32
    )
    rrc = torch.complex(rrc_real, torch.zeros_like(rrc_real))
    x_rrc = torch.conv1d(x_upsampled.view(1, 1, -1), rrc.view(1, 1, -1)).view(-1)

    # Time Offset
    tau_off = torch.tensor(
        np.random.uniform(-tau_bound, tau_bound), device=device, dtype=torch.float32
    )
    # Simple approximation for small offsets
    x_rrc = torch.roll(x_rrc, int(tau_off * fs))

    # Phase Offset & CFO
    varphi_off = torch.tensor(
        np.random.uniform(0, 2 * np.pi), device=device, dtype=torch.float32
    )
    f_cfo = torch.tensor(
        np.random.normal(0, sigma_CFO), device=device, dtype=torch.float32
    )
    delta_varphi = f_cfo / fs
    k = torch.arange(len(x_rrc), device=device, dtype=torch.float32)
    phase = 2 * np.pi * k * delta_varphi + varphi_off
    x_cfo = x_rrc * torch.exp(1j * phase)

    # AWGN
    noise = (
        torch.randn_like(x_cfo, dtype=torch.float32)
        + 1j * torch.randn_like(x_cfo, dtype=torch.float32)
    ) * sigma_noise
    y = x_cfo + noise

    return y


# Example usage
n_symbols = 100
symbols = torch.randn(n_symbols, dtype=torch.cfloat)
gamma = 4
alpha = 0.25
L = 101
tau_bound = 0.1
fs = 1.0  # Sampling frequency (normalized)
sigma_CFO = 0.001
sigma_noise = 0.1

received_signal = stochastic_channel(
    symbols, gamma, alpha, L, tau_bound, fs, sigma_CFO, sigma_noise
)


print(symbols)
print(received_signal)
