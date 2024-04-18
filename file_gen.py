import numpy as np


def generate_IQ_samples(num_samples, sample_rate, frequency, amplitude):
    # Time array
    t = np.arange(num_samples) / sample_rate

    # Generate I and Q components
    I = amplitude * np.cos(2 * np.pi * frequency * t)
    Q = amplitude * np.sin(2 * np.pi * frequency * t)

    # Interleave I and Q samples
    IQ_samples = np.column_stack((I, Q))

    return IQ_samples


def save_IQ_samples_to_file(IQ_samples, filename):
    # Save to binary file
    with open(filename, "wb") as f:
        f.write(IQ_samples.astype(np.int16).tobytes())
