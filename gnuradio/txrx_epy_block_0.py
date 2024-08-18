import numpy as np
from gnuradio import gr

def generate_zadoff_chu_sequence(root, length):
    """Generates a Zadoff-Chu sequence with specified root and length."""
    n = np.arange(length)
    zc_seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    return zc_seq

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - Add a Zadoff-Chu sequence with the same power as input to the beginning"""

    def __init__(self, seq_length=64, root=1):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Embedded Python Block',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.seq_length = seq_length
        self.root = root

    def work(self, input_items, output_items):
        """Calculate input signal power, generate Zadoff-Chu sequence, and prepend it to the input signal"""
        # Calculate the average power of the input signal
        input_signal = input_items[0]
        power = np.mean(np.abs(input_signal) ** 2)
        
        # Generate a Zadoff-Chu sequence
        zc_sequence = generate_zadoff_chu_sequence(self.root, self.seq_length)
        
        # Scale the Zadoff-Chu sequence to match the power of the input signal
        current_power = np.mean(np.abs(zc_sequence) ** 2)
        scaling_factor = np.sqrt(power / current_power)
        zc_sequence = zc_sequence * scaling_factor
        
        # Prepend the Zadoff-Chu sequence to the input signal
        output_signal = np.concatenate([zc_sequence, input_signal])
        
        # Ensure the output length matches the expected length
        output_items[0][:] = output_signal[:len(output_items[0])]
        return len(output_items[0])
