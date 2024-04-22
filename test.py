import numpy as np

# # Parameters
# sampling_rate = 1000  # Sampling rate in Hz
# duration = 1  # Duration of the signal in seconds
# frequency = 10  # Frequency of the complex sinusoid in Hz
# amplitude = 1  # Amplitude of the complex sinusoid

# # Time array
# t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# # Generate complex sinusoid
# complex_signal = amplitude * np.exp(1j * 2 * np.pi * frequency * t)

# # Save as binary file
# file_path = "./file/complex_signal.bin"
# complex_signal.astype(np.complex64).tofile(file_path)

# # print the messages
# print(complex_signal)

# read complex_signal.bin
complex_signal = np.fromfile("./file/tx.bin")
# print(complex_signal[0:10])
print(len(complex_signal))

# read rx_samples.bin
rx_samples = np.fromfile("./file/rx.bin")
# print(rx_samples[0:10])
