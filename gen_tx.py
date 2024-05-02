import numpy as np


def generate_and_save_messages(num_messages, M):
    """
    Generate a number of messages and save them to a specified file.

    Parameters:
        num_messages (int): The number of messages to generate.
        M (int): The maximum value for message generation (exclusive).
        filename (str): The path to the file where the messages will be saved.
    """
    # Generate messages
    messages = np.random.randint(0, M, num_messages)

    # Save messages to a binary file
    with open("./file/tx.dat", "wb") as f:
        f.write(messages.astype(np.int16).tobytes())

    print(f"Generated {num_messages} messages")


# Usage
num_messages = 10000
M = 256
generate_and_save_messages(num_messages, M)


# read tx.bin and rx.bin
with open("./file/tx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.int16)
    print(messages[0:20])
