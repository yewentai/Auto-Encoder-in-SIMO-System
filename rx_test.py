import numpy as np
import os

with open("./file/tx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.complex64)
    print(messages[3:-5])

print(len(messages))
with open("./file/rx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.complex64)
    # messages = messages.astype(np.int16)
    print(messages[0:9992])

print(len(messages))
