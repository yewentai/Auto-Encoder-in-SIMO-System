import numpy as np
import matplotlib.pyplot as plt


def commlteMIMO_Ch(txSig, prmLTEPDSCH, prmMdl):
    snr_dB = prmMdl["snrdB"]

    # MIMO Fading channel
    rxFade, chPathG = MIMOFadingChan(txSig, prmLTEPDSCH, prmMdl)

    # Add AWG noise
    sigPow = 10 * np.log10(np.var(rxFade))
    nVar = 10 ** (0.1 * (sigPow - snr_dB))
    rxSig = AWGNChannel(rxFade, nVar)

    return rxSig, chPathG, nVar


def MIMOFadingChan(txSig, prmLTEPDSCH, prmMdl):
    # Placeholder for MIMO fading channel simulation
    # You can replace this with your specific implementation
    # For simplicity, just copying the input signal for now
    rxFade = txSig.copy()
    chPathG = np.ones_like(txSig)  # Placeholder for channel gains
    return rxFade, chPathG


def AWGNChannel(signal, nVar):
    # Add AWGN to the signal
    noise = np.random.normal(0, np.sqrt(nVar), len(signal))
    rxSignal = signal + noise
    return rxSignal


# Example usage
prmLTEPDSCH = {}  # Fill in with your LTE parameters
prmMdl = {"snrdB": 20}  # Set your desired SNR in dB

# Example input signal (replace this with your actual input signal)
txSig = np.random.randn(1000) + 1j * np.random.randn(1000)

# Call the MIMO channel function
rxSig, chPathG, nVar = commlteMIMO_Ch(txSig, prmLTEPDSCH, prmMdl)

# Print or visualize the results as needed
print("Received Signal:", rxSig)
print("Channel Gains:", chPathG)
print("Noise Variance:", nVar)

# Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(np.real(rxSig), np.imag(rxSig), label="Received Signal")
plt.title("Received Signal in the Complex Plane")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.grid(True)
plt.show()
