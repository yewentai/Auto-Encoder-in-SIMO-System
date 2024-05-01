#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os

from utils import Encoder, Decoder, awgn, ser_mqam_awgn


# In[14]:


CONFIG_TRAIN = {
    "M": 16,  # Number of constellation points
    "flag_train_model": True,  # Flag to control training
    "training_snr": 12,  # Training SNR (dB)
    "best_model_path": "./model/ae_siso_awgn_16qam_best_model.pth",  # Path to save the best model
    "latest_checkpoint_path": "./model/ae_siso_awgn_16qam_latest_checkpoint.pth",  # Path to save the latest checkpoint
    "best_encoder_path": "./model/ae_siso_awgn_16qam_best_encoder.pth",  # Path to save the best encoder
    "best_decoder_path": "./model/ae_siso_awgn_16qam_best_decoder.pth",  # Path to save the best decoder
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# In[15]:


encoder = Encoder([CONFIG_TRAIN["M"], 10, 10, 2]).to(device)
decoder = Decoder([2, 20, 20, CONFIG_TRAIN["M"]]).to(device)


# In[16]:


checkpoint = torch.load(CONFIG_TRAIN["best_encoder_path"], map_location=device)
encoder.load_state_dict(checkpoint)


# In[26]:


def save_encoder(encoder, loss, optimizer, model_type="latest"):
    """
    Save the model to a file.
    - encoder: the encoder model
    - loss: the loss history
    - optimizer: the optimizer
    - model_type: "latest"(default) or "best"
    """

    if model_type == "best":  # Decide the filename based on model_type
        filename = CONFIG_TRAIN["best_model_path"]
    else:
        filename = CONFIG_TRAIN["latest_checkpoint_path"]
    torch.save(
        {
            "Encoder": encoder.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
        },
        filename,
    )


def save_decoder(decoder, loss, optimizer):
    """
    Save the model to a file.
    - decoder: the decoder model
    - loss: the loss history
    - optimizer: the optimizer
    - model_type: "latest"(default) or "best"
    """
    torch.save(
        {
            "Decoder": decoder.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
        },
        CONFIG_TRAIN["best_decoder_path"],
    )


def early_stopping(loss_hist, patience=5):
    """
    Check if the training should be stopped early.
    - loss_hist: list of loss values
    - patience: number of epochs to wait before stopping
    """

    if len(loss_hist) > patience:
        if all(
            x > loss_hist[-1] for x in loss_hist[-(patience + 1) :]
        ):  # if the last patience losses are decreasing
            return True
    return False


def train_decoder(
    encoder, decoder, optimizer, iterations, loss_hist, num_messages, device
):
    criterion = nn.NLLLoss()  # negative log likelihood loss
    best_loss = float("inf")  # Initialize the best loss to infinity
    try:
        for iterator in tqdm(
            range(len(loss_hist), iterations), desc="training process"
        ):
            # option1: train encoder and decoder simultaneously
            messages = torch.randint(
                0, CONFIG_TRAIN["M"], size=(num_messages,), device=device
            )
            one_hot = F.one_hot(messages, CONFIG_TRAIN["M"]).float()
            tx = encoder(one_hot)
            # write tx to a binary file
            tx = tx.detach().numpy()
            with open("./file/tx.dat", "wb") as f:
                f.write(tx.tobytes())
            # run channel.py to generate rx
            os.system("python3 ./gnuradio/channel1.py")
            # read rx from a binary file
            with open("./file/rx.dat", "rb") as f:
                rx = np.frombuffer(f.read(), dtype=np.float32)
            rx = torch.from_numpy(rx).to(device)
            rx = rx.view(-1, 2)
            rx_constant = (
                rx.clone().detach().requires_grad_(False)
            )  # no gradients in the channel model
            y_pred = decoder(rx_constant)

            if rx.shape[0] != num_messages:
                # 如果不匹配，裁剪或填充rx以匹配num_messages
                messages = messages[: rx.shape[0]]  # 假设rx长度大于num_messages
                # 如果rx长度小于num_messages，则需要其他逻辑处理

            loss = criterion(y_pred, messages)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_decoder(decoder, loss_hist, optimizer)

            if early_stopping(loss_hist):
                print("Early stopping")
                break
        print("Training complete")

    except KeyboardInterrupt:
        save_decoder(decoder, loss_hist, optimizer)
        print("Training interrupted")

    save_decoder(decoder, loss_hist, optimizer)
    # Plot the loss
    plt.semilogy(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


# In[18]:


if CONFIG_TRAIN["flag_train_model"]:
    # check if there is a checkpoint to resume training
    if os.path.exists(CONFIG_TRAIN["best_decoder_path"]):
        checkpoint = torch.load(CONFIG_TRAIN["best_decoder_path"], map_location=device)
        # load the model, optimizer a loss history
        decoder.load_state_dict(checkpoint["Decoder"])

    parameters = list(decoder.parameters())
    optimizer = Adam(parameters, lr=0.1)

    if os.path.exists(CONFIG_TRAIN["best_decoder_path"]):
        # optimizer.load_state_dict(checkpoint["optimizer"])
        loss_hist = checkpoint["loss"]
        print(f"Resuming training from iterator {len(loss_hist)}")
    else:
        loss_hist = []
        print("Training from scratch")

    iterations = int(3e2)  # Total number of epochs to train
    num_messages = int(1e4)  # Number of messages to use for training (batch size)
    if iterations > len(loss_hist):
        train_decoder(
            encoder,
            decoder,
            optimizer,
            iterations,
            loss_hist,
            num_messages,
            device,
        )
    else:
        print("epochs already completed")
else:
    # check if there is a checkpoint to load the model
    if os.path.exists(CONFIG_TRAIN["best_decoder_path"]):
        checkpoint = torch.load(CONFIG_TRAIN["best_decoder_path"], map_location=device)
        decoder.load_state_dict(checkpoint["Decoder"])
        print("Model loaded. Training iterator: ", len(checkpoint["loss"]))
    else:
        print(
            "Model not found, please set flag_train_model to True and train the model"
        )
        exit(1)


# In[21]:


SNR_dB = np.arange(0, 22, 2)
SER = np.array([])

iteration = tqdm(range(len(SNR_dB)), desc="simulation process")
for iterator in iteration:
    snr = SNR_dB[iterator]
    num_mess = 6400  # number of messages to test
    minErr = 1  # minimum number of errors
    minSym = 1e6  # minimum number of symbols
    totSym = 0  # total number of symbols
    totErr = 0  # total number of errors
    while totErr < minErr or totSym < minSym:
        messages = torch.randint(0, CONFIG_TRAIN["M"], size=(num_mess,)).to(device)
        one_hot = F.one_hot(messages).float()
        tx = encoder(one_hot)
        rx = awgn(tx, snr)
        rx_constant = (
            rx.clone().detach().requires_grad_(False)
        )  # no gradients in the channel model

        y_pred = decoder(rx_constant)

        m_hat = torch.argmax(y_pred, -1)

        err = torch.sum(torch.not_equal(messages, m_hat)).to("cpu").detach().numpy()

        totErr += err
        totSym += num_mess
    SER = np.append(SER, totErr / totSym)


# In[22]:


SER_theory = ser_mqam_awgn(CONFIG_TRAIN["M"], SNR_dB)
# plot the SER-SNR curve
plt.figure()
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.semilogy(SNR_dB, SER_theory, label="Standard_16QAM_AWGN")
plt.semilogy(SNR_dB, SER, "-*", label="AE_GS_{}QAM_AGWN".format(CONFIG_TRAIN["M"]))
plt.legend()
plt.grid(True)


# In[23]:


# Generate transmitted symbols using the encoder
num_mess = 6400  # number of messages to test
messages = torch.randint(0, CONFIG_TRAIN["M"], size=(num_mess,)).to(device)
one_hot = F.one_hot(messages).float()
tx = encoder(one_hot)
tx = tx.to("cpu").detach().numpy()

# Define the coordinates for 16-QAM constellation points
points_I = np.array([1, 1, 1, 1, -1, -1, -1, -1, 3, 3, 3, 3, -3, -3, -3, -3]) / np.sqrt(
    10
)
points_Q = np.array([3, 1, -1, -3, 3, 1, -1, -3, 3, 1, -1, -3, 3, 1, -1, -3]) / np.sqrt(
    10
)

# Plot both constellations
plt.figure(figsize=(10, 5))

# Plot transmitted symbols
plt.subplot(1, 2, 1)
plt.scatter(tx[:, 0], tx[:, 1])
plt.xlabel("I")
plt.ylabel("Q")
plt.title("Transmitted Constellation")
plt.grid(True)

# Plot standard 16-QAM constellation
plt.subplot(1, 2, 2)
plt.scatter(points_I, points_Q, c="r")
plt.xlabel("I")
plt.ylabel("Q")
plt.title("Standard 16-QAM Constellation")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[24]:


num_messages = int(1e4)  # Number of messages to use for training (batch size)
messages = torch.randint(0, CONFIG_TRAIN["M"], size=(num_messages,), device=device)
one_hot = F.one_hot(messages, CONFIG_TRAIN["M"]).float()
tx = encoder(one_hot)
# write tx to a binary file
tx = tx.detach().numpy()
with open("./file/tx.dat", "wb") as f:
    f.write(tx.tobytes())
# run channel.py to generate rx
os.system("python3 ./gnuradio/channel1.py")
# read rx from a binary file
with open("./file/rx.dat", "rb") as f:
    rx = np.frombuffer(f.read(), dtype=np.float32)
rx = torch.from_numpy(rx).to(device)
rx = rx.view(-1, 2)
y_pred_one_hot = decoder(rx)
y_pred = torch.argmax(y_pred_one_hot, -1)

if rx.shape[0] != num_messages:
    # 如果不匹配，裁剪或填充rx以匹配num_messages
    messages = messages[: rx.shape[0]]  # 假设rx长度大于num_messages
    # 如果rx长度小于num_messages，则需要其他逻辑处理

print(messages[0:20])
print(y_pred[0:20])
