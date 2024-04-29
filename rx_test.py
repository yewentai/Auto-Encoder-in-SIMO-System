import numpy as np
import os

with open("./file/tx_float.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.int16)
    print(messages[0:20])

print(len(messages))
with open("./file/rx_float.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.byte)
    # messages = messages.astype(np.int16)
    print(messages[0:20])

print(len(messages))
