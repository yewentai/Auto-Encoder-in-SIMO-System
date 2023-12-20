def mrc_receiver(received_signal, channel_gains):
    """
    Maximum Ratio Combining (MRC) receiver

    Parameters:
    - received_signal: Received symbol vector
    - channel_gains: Channel gains vector

    Returns:
    - decoded_symbol: Decoded symbol
    """

    mrc_weights = (
        np.conj(channel_gains) / np.linalg.norm(channel_gains) / 2
    )  # MRC weights

    decoded_symbol = np.dot(mrc_weights, received_signal)

    return decoded_symbol
