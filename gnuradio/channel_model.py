#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: yewentai
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window


class channel(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        # The sample rate of the signal
        self.samp_rate = samp_rate = 3200000
        # The AWGN noise level as a voltage (to be calculated externally to meet, say, a desired SNR).
        self.noise_voltage = noise_voltage = 0.1
        # The normalized frequency offset. 0 is no offset; 0.25 would be, for example, one quarter of a sample time.
        self.frequency_offset = frequency_offset = 0
        # The sample timing offset to emulate the different rates between the sample clocks of the transmitter and receiver. 1.0 is no difference.
        self.epsilon = epsilon = 1.0
        # Taps of a FIR filter to emulate a multipath delay profile. Default is 1+0j meaning a single tap, and thus no multipath.
        self.taps = taps = [1.0]
        # A random number generator seed for the noise source.
        self.noise_seed = noise_seed = 0
        self.file_source_path = file_source_path = "./file/tx.dat"
        self.file_sink_path = file_sink_path = "./file/rx1.dat"

        ##################################################
        # Blocks
        ##################################################

        self.channels_channel_model_0_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=frequency_offset,
            epsilon=epsilon,
            taps=taps,
            noise_seed=noise_seed,
        )
        self.blocks_throttle2_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1,
            samp_rate,
            True,
            (
                0
                if "auto" == "auto"
                else max(
                    int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1
                )
            ),
        )
        self.blocks_file_source_0_0 = blocks.file_source(
            gr.sizeof_gr_complex * 1, file_source_path, False, 0, 0
        )
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0_0 = blocks.file_sink(
            gr.sizeof_gr_complex * 1, file_sink_path, False
        )
        self.blocks_file_sink_0_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.channels_channel_model_0_0, 0))
        self.connect(
            (self.channels_channel_model_0_0, 0), (self.blocks_file_sink_0_0, 0)
        )

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)


def main(top_block_cls=channel, options=None):
    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == "__main__":
    main()
