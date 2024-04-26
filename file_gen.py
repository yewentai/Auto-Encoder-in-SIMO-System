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


def generate_message(num_messages, M):
    return np.random.randint(0, M, num_messages)


def save_message_to_file(messages, filename):
    # Save to binary file
    with open(filename, "wb") as f:
        f.write(messages.astype(np.int16).tobytes())


def save_IQ_samples_to_file(IQ_samples, filename):
    # Save to binary file
    with open(filename, "wb") as f:
        f.write(IQ_samples.astype(np.int16).tobytes())


# # Generate IQ samples
# num_samples = 100000
# sample_rate = 1e6
# frequency = 100e3
# amplitude = 10000
# IQ_samples = generate_IQ_samples(num_samples, sample_rate, frequency, amplitude)

# # Save to file
# filename = "./file/tx.bin"
# save_IQ_samples_to_file(IQ_samples, filename)
# print(f"Generated {num_samples} IQ samples and saved to {filename}")

# Generate messages
num_messages = 100000
M = 16
messages = generate_message(num_messages, M)

# Save to file
filename = "./file/tx.dat"
save_message_to_file(messages, filename)
print(f"Generated {num_messages} messages and saved to {filename}")

# read tx.bin and rx.bin
with open("./file/tx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.int16)
    print(messages[0:20])

with open("./file/rx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.int16)
    print(messages[0:20])
