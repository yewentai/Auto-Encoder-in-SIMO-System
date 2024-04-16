import numpy as np

# Parameters
num_samples = 1000e3  # Number of samples
sample_rate = 100e3  # Sample rate in Hz
frequency = 10e3  # Frequency of the signal in Hz
amplitude = 10000  # Amplitude of the signal

# Time array
t = np.arange(num_samples) / sample_rate

# Generate I and Q components
I = amplitude * np.cos(2 * np.pi * frequency * t)
Q = amplitude * np.sin(2 * np.pi * frequency * t)

# Interleave I and Q samples
IQ_samples = np.column_stack((I, Q))

# Save to binary file
with open("IQ_samples.bin", "wb") as f:
    f.write(IQ_samples.astype(np.int16).tobytes())
