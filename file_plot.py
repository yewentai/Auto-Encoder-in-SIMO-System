import numpy as np
import matplotlib.pyplot as plt

# Read IQ samples from binary file
with open("file/IQ_samples.bin", "rb") as f:
    IQ_samples = np.frombuffer(f.read(), dtype=np.int16)

# Separate I and Q components
I = IQ_samples[::2]
Q = IQ_samples[1::2]

# Parameters
num_samples = len(I)
sample_rate = 100e3  # Sample rate in Hz

# Time array
t = np.arange(num_samples) / sample_rate

# Plot I and Q components
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t, I)
plt.title("I Component")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, Q)
plt.title("Q Component")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
