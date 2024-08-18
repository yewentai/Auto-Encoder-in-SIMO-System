"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


def zcsequence(u, seq_length, q=0):
    """
    Generate a Zadoff-Chu (ZC) sequence.

    Parameters
    ----------
    u : int
        Root index of the the ZC sequence: u>0.

    seq_length : int
        Length of the sequence to be generated. Usually a prime number:
        u<seq_length, greatest-common-denominator(u,seq_length)=1.

    q : int
        Cyclic shift of the sequence (default 0).

    Returns
    -------
    zcseq : 1D ndarray of complex floats
        ZC sequence generated.
    """
    for el in [u, seq_length, q]:
        if not float(el).is_integer():
            raise ValueError("{} is not an integer".format(el))
    if u <= 0:
        raise ValueError("u is not stricly positive")
    if u >= seq_length:
        raise ValueError("u is not stricly smaller than seq_length")
    if np.gcd(u, seq_length) != 1:
        raise ValueError("the greatest common denominator of u and seq_length is not 1")

    cf = seq_length % 2
    n = np.arange(seq_length)
    zcseq = np.exp(-1j * np.pi * u * n * (n + cf + 2.0 * q) / seq_length)

    return zcseq


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block - add a zadoff-chu sequence to the input signal"""

    def __init__(self, u=5.0, zc_len=62):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Embedded Python Block',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.u = int(u)
        self.zc_len = int(zc_len)
        self.zc_seq = zcsequence(self.u, self.zc_len)
        self.normalized_zc_seq = self.zc_seq / np.sqrt(np.mean(np.abs(self.zc_seq) ** 2))

    def work(self, input_items, output_items):
        """add a zadoff-chu sequence to the input signal"""
        input_signal = input_items[0]
        # output_signal = np.concatenate([self.normalized_zc_seq, input_signal])
        output_items[0][:len(input_signal)] = input_signal.astype(np.complex64)
        return len(output_items[0])
