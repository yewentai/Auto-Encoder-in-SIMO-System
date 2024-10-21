# Deep Learning-Based Physical Layer Communication: Concept and Prototyping

## Overview

This project explores the application of deep learning techniques in physical-layer communication systems, focusing on the use of **autoencoders** to jointly optimize transmitter and receiver performance. Traditional communication systems are typically broken down into distinct blocks (source coding, channel coding, modulation), each optimized separately. In contrast, the **channel autoencoder** approach treats the communication system as an end-to-end learning problem, offering new ways to improve communication performance.

The project compares **traditional communication methods** with **autoencoder-based approaches** in both simulation and real-world environments, using **SISO** (Single Input Single Output) and **SIMO** (Single Input Multiple Output) systems. The experimental setup is based on **GNU Radio** and **USRP B210 hardware**.

## Project Goals

1. **Explore Factors Impacting Deep Learning in Communication**: Investigating key factors that influence the performance of deep learning models in physical-layer communication systems.
  
2. **Compare with Traditional Baselines**: Benchmarking autoencoder-based methods against traditional communication systems in SISO and SIMO setups.

3. **Test in Real-World Scenarios**: Deploying the models in over-the-air transmissions using GNU Radio and USRP B210 hardware.

## System Architecture

### Autoencoder Model

The autoencoder used in this project consists of:
- **Encoder**: Converts the one-hot encoded message into a complex IQ symbol for transmission.
- **Channel**: Simulated or real channel (AWGN, Rayleigh fading, etc.) over which the symbol is transmitted.
- **Decoder**: Receives the IQ samples and estimates the transmitted message.

The **negative log-likelihood loss** is used, and the model is optimized using the **Adam optimizer**.

### Channel Models

1. **AWGN Channel**: Additive White Gaussian Noise channel, used as the simplest case.
2. **Rayleigh Fading Channel**: Modeled using filtered Gaussian noise and sum-of-sinusoids methods.
3. **Pulse Shaping**: Implemented using a **root-raised cosine filter** to reduce the signal bandwidth and make the model more realistic for over-the-air transmission.

### Two-Phase Training Strategy

1. **Phase 1: Simulated Channel Training**: The autoencoder is trained using a simulated channel model, aiming to approximate real-world channel behavior.
   
2. **Phase 2: Fine-tuning on Real Channel Data**: After deployment, the receiver is fine-tuned using actual IQ samples collected from over-the-air transmissions to account for any mismatch between the simulated and real channels.

## Experimental Setup

The hardware setup includes:
- **USRP B210**: SDR board used for transmitting and receiving over-the-air signals.
- **GNU Radio**: Software platform for signal processing.
- **Zadoff-Chu Sequence**: Used for frame synchronization, taking advantage of its low autocorrelation sidelobes.

Experiments are conducted in an office environment, with the following configuration:
- **SISO System**: Evaluated using 16-QAM modulation, compared with minimum distance decoding as a baseline.
- **SIMO System**: Evaluated with two receiver antennas, compared with Maximal Ratio Combining (MRC) as the baseline.

## Results

### SISO System
- **AWGN Channel**: The autoencoder outperforms traditional minimum distance decoding across all Signal-to-Noise Ratios (SNRs).
- **Rayleigh Channel**: The autoencoder performs better than traditional Zero Forcing techniques at higher SNRs but underperforms at lower SNRs.

### SIMO System
- **AWGN Channel**: The autoencoder lags behind MRC at lower SNRs but surpasses it at higher SNRs.
- **Rayleigh Channel**: The autoencoder outperforms MRC across all SNRs.

### Over-the-Air Results
The real-world experiments revealed higher **Symbol Error Rates (SER)** than simulations. This discrepancy is mainly due to the lack of symbol synchronization and unmodeled real-world factors like **sampling time offsets** and **carrier frequency offsets**.

## Future Work

- **Symbol Synchronization**: Implementing symbol synchronization to reduce over-the-air error rates.
- **Improved Channel Models**: Incorporating carrier frequency offset and sample time offset to bridge the gap between simulation and real-world performance.
- **Advanced Architectures**: Exploring the use of more sophisticated deep learning architectures, such as **convolutional neural networks**, **recurrent neural networks**, or **transformer-based models** for further improvement.
