import numpy as np
import matplotlib.pyplot as plt


def ofdm_modulation(data_symbols, num_subcarriers, cp_length):
    # OFDM modulation
    num_symbols = len(data_symbols)

    # Map data symbols to subcarriers (zero-padding if necessary)
    subcarrier_symbols = np.zeros(num_subcarriers, dtype=complex)
    subcarrier_symbols[:num_symbols] = data_symbols

    # Inverse Fourier Transform to get time-domain signal
    time_domain_signal = np.fft.ifft(subcarrier_symbols)

    # Add cyclic prefix
    cp = time_domain_signal[-cp_length:]
    ofdm_signal = np.concatenate((cp, time_domain_signal))

    return ofdm_signal


def ofdm_demodulation(received_signal, num_subcarriers, cp_length):
    # Remove cyclic prefix
    received_signal = received_signal[cp_length:]

    # Fourier Transform to convert back to frequency domain
    frequency_domain_signal = np.fft.fft(received_signal)

    # Extract data symbols from subcarriers
    data_symbols = frequency_domain_signal[:num_subcarriers]

    return data_symbols


def main():
    # Simulation parameters
    num_subcarriers = 64
    cp_length = 16
    num_symbols = 48

    # Generate random data symbols (QPSK modulation for simplicity)
    data_symbols = np.random.choice(
        [-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], size=num_symbols
    )

    # Perform OFDM modulation
    transmitted_signal = ofdm_modulation(data_symbols, num_subcarriers, cp_length)

    # Add channel noise (for simplicity, assuming AWGN channel)
    snr_db = 20
    noise_power = 10 ** (-snr_db / 10)
    received_signal = transmitted_signal + np.sqrt(noise_power / 2) * (
        np.random.randn(len(transmitted_signal))
        + 1j * np.random.randn(len(transmitted_signal))
    )

    # Perform OFDM demodulation
    received_data_symbols = ofdm_demodulation(
        received_signal, num_subcarriers, cp_length
    )

    # Visualization (optional)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        np.real(data_symbols), np.imag(data_symbols), label="Transmitted Symbols"
    )
    plt.scatter(
        np.real(received_data_symbols),
        np.imag(received_data_symbols),
        marker="x",
        color="red",
        label="Received Symbols",
    )
    plt.title("Single Antenna OFDM Modulation and Demodulation")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
