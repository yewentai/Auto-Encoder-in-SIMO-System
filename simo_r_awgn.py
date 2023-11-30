import numpy as np
import matplotlib.pyplot as plt


def generate_rayleigh_fading(num_samples):
    # Generate Rayleigh fading coefficients
    fading_coefficients = (
        np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    ) / np.sqrt(2)
    return fading_coefficients


def simulate_simo_channel(signal, fading_coefficients, snr_db):
    # Simulate SIMO channel with AWGN
    awgn = np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    received_signal = fading_coefficients * signal + awgn / np.sqrt(10 ** (snr_db / 10))
    return received_signal


def main():
    # Simulation parameters
    num_samples = 1000
    snr_db = 20  # Signal-to-noise ratio in dB

    # Generate input signal (e.g., QPSK symbols)
    input_signal = np.random.choice([-1, 1], size=num_samples) + 1j * np.random.choice(
        [-1, 1], size=num_samples
    )

    # Generate Rayleigh fading coefficients
    fading_coefficients = generate_rayleigh_fading(num_samples)

    # Simulate SIMO channel with AWGN
    received_signal = simulate_simo_channel(input_signal, fading_coefficients, snr_db)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(
        np.real(received_signal), np.imag(received_signal), label="Received Signal"
    )
    plt.scatter(
        np.real(fading_coefficients),
        np.imag(fading_coefficients),
        marker="x",
        color="red",
        label="Rayleigh Fading Coefficients",
    )
    plt.title("Simulation of Rayleigh SIMO Channel with AWGN")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
