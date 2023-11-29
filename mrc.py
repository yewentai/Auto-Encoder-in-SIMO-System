import numpy as np
import matplotlib.pyplot as plt

def generate_signal(symbol_duration, sampling_rate, snr_db):
    # Generate a simple QPSK signal
    num_symbols = int(symbol_duration * sampling_rate)
    symbols = np.random.choice([-1, 1], size=num_symbols)
    signal = np.repeat(symbols, int(sampling_rate / 2))

    # Add noise to the signal
    noise_power = 10**(-snr_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    received_signal = signal + noise

    return signal, received_signal

def maximal_ratio_combining(received_signals):
    # Maximal-ratio combining
    weights = np.conj(received_signals) / np.abs(received_signals)
    combined_signal = np.sum(weights * received_signals, axis=0)
    return combined_signal

def main():
    # Simulation parameters
    symbol_duration = 1.0  # seconds
    sampling_rate = 1000  # samples per second
    snr_db = 10  # signal-to-noise ratio in dB

    # Generate multiple received signals
    num_antennas = 2
    received_signals = [generate_signal(symbol_duration, sampling_rate, snr_db)[1] for _ in range(num_antennas)]

    # Apply maximal-ratio combining
    combined_signal = maximal_ratio_combining(received_signals)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(np.real(combined_signal), label='Combined Signal (Real)')
    plt.plot(np.imag(combined_signal), label='Combined Signal (Imaginary)')
    plt.legend()
    plt.title('Maximal-Ratio Combining Receiver Simulation')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

if __name__ == "__main__":
    main()
