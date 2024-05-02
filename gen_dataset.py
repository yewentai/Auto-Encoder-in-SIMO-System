import numpy as np
import torch
import torch.nn.functional as F
import os

from utils import Encoder, Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
encoder = Encoder([16, 10, 10, 2]).to(device)
decoder = Decoder([2, 20, 20, 16]).to(device)

encoder_state_dict = torch.load("./model/ae_siso_awgn_16qam_best_encoder.pth")
encoder.load_state_dict(encoder_state_dict)
decoder_state_dict = torch.load("./model/ae_siso_awgn_16qam_best_decoder.pth")
decoder.load_state_dict(decoder_state_dict["Decoder"])

num_messages = int(1e4 + 8)  # Number of messages to use for training (batch size)
messages = torch.randint(0, 16, size=(num_messages,), device=device)
one_hot = F.one_hot(messages, 16).float()
tx = encoder(one_hot)
# write tx to a binary file
tx = tx.detach().numpy()
with open("./file/tx.dat", "wb") as f:
    f.write(tx.tobytes())
# run channel.py to generate rx
os.system("python3 ./gnuradio/channel_model.py")
# read rx from a binary file
with open("./file/rx.dat", "rb") as f:
    rx = np.frombuffer(f.read(), dtype=np.float32)
rx = torch.from_numpy(rx).to(device)
rx = rx.view(-1, 2)
y_pred_one_hot = decoder(rx)
y_pred = torch.argmax(y_pred_one_hot, -1)
tx = tx[3:-5]
rx = rx.detach().numpy()

messages = messages[3:-5]
print(messages)
print(y_pred)
print(tx)
print(rx)
