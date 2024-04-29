import numpy as np
import os

with open("./file/tx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.byte)
    print(messages[0:20])

print(len(messages))
with open("./file/rx.dat", "rb") as f:
    messages = np.frombuffer(f.read(), dtype=np.byte)
    # messages = messages.astype(np.int16)
    print(messages[7:27])

print(len(messages))
