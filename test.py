import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam

# from keras.utils import to_categorical
from scipy.special import erfc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from utils import Encoder, Decoder, awgn
from file_gen import *

CONFIG_TRAIN = {
    "M": 16,  # Number of constellation points
    "flag_train_model": True,  # Flag to control training
    "training_snr": 12,  # Training SNR (dB)
    "best_model_path": "./model/ae_simo_rayleigh_16qam_best_model.pth",  # Path to save the best model
    "latest_checkpoint_path": "./model/ae_simo_rayleigh_16qam_latest_checkpoint.pth",  # Path to save the latest checkpoint
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

encoder = Encoder([CONFIG_TRAIN["M"], 64, 64, 64, 2]).to(device)
decoder = Decoder([8, 512, 512, 512, CONFIG_TRAIN["M"]]).to(device)

criterion = nn.NLLLoss()  # negative log likelihood loss
best_loss = float("inf")  # Initialize the best loss to infinity

# Parameters
sample_rate = 100e3  # Sample rate in Hz
frequency = 10e3  # Frequency of the signal in Hz
amplitude = 10000  # Amplitude of the signal
num_messages = 1000  # Number of messages to generate

import numpy as np


def ls_channel_estimation(tx, rx_samples):
    # Detach the tensor from the computation graph
    tx_detached = tx.detach().cpu().numpy()

    # Compute the pseudo-inverse of X
    X_inv = np.linalg.pinv(tx_detached)

    # Compute the LS estimate of the channel response H
    H = np.dot(X_inv, rx_samples)

    return H


messages = torch.randint(
    0, CONFIG_TRAIN["M"], size=(num_messages,), device=device
)  # generate random messages
one_hot = F.one_hot(messages, CONFIG_TRAIN["M"]).float()  # convert to one hot encoding
tx = encoder(one_hot)  # type of tx is torch.float32

# # Generate IQ samples from the transmitted signal
# I = tx[:, 0].cpu().detach().numpy()
# Q = tx[:, 1].cpu().detach().numpy()
# IQ_samples = np.stack((I, Q), axis=1)

# # Save to binary file
# with open("file/iq_samples.bin", "wb") as f:
#     f.write(IQ_samples.tobytes())

# # Run GNURadio flowgraph to generate the received signal
# os.system("python3 channel.py")

# # Read the received signal from the binary file
# with open("rx_samples.bin", "rb") as f:
#     rx_samples = np.frombuffer(f.read(), dtype=np.float32)

# # Perform LS channel estimation
# H_ls = ls_channel_estimation(tx, rx_samples)

# # Combine H and rx
# rx_csi = torch.cat(
#     (
#         torch.tensor(rx_samples, device=device),
#         torch.tensor(H_ls, device=device),
#     ),
#     dim=1,
# )

# y_pred_ae = decoder(rx_csi)
