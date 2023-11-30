import numpy as np
import matplotlib.pyplot as plt


def generate_channel_coefficients(num_antennas):
    # Generate complex channel coefficients (h_i)
    return np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas)


def calculate_snr_mrc(h, w, sigma_n):
    # Calculate SNR for Maximal-ratio combining
    snr_mrc = (np.abs(np.dot(w, h)) ** 2) / (np.linalg.norm(w) ** 2 * sigma_n**2)
    return snr_mrc


def maximize_snr_mrc(h):
    # Maximize SNR for Maximal-ratio combining
    w = h / np.linalg.norm(h)  # Cauchy-Schwartz Inequality
    return w


def main():
    # Simulation parameters
    num_antennas = 4
    P = 1  # Signal power
    sigma_n = 0.1  # Noise standard deviation

    # Generate random channel coefficients
    h = generate_channel_coefficients(num_antennas)

    # Maximize SNR using Cauchy-Schwartz Inequality
    w_max_snr = maximize_snr_mrc(h)

    # Calculate SNR for Maximal-ratio combining
    snr_mrc = calculate_snr_mrc(h, w_max_snr, P, sigma_n)

    # Print results
    print("Channel Coefficients (h):", h)
    print("Optimal Weight Vector (w) for Max SNR:", w_max_snr)
    print("Maximal-ratio Combining SNR:", snr_mrc)

    # Visualization (Optional)
    plt.figure(figsize=(8, 6))
    plt.scatter(np.real(h), np.imag(h), label="Channel Coefficients (h)")
    plt.scatter(
        np.real(w_max_snr),
        np.imag(w_max_snr),
        marker="*",
        color="red",
        s=200,
        label="Optimal Weight (w)",
    )
    plt.title("Maximal-ratio Combining: Channel Coefficients and Optimal Weight")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
