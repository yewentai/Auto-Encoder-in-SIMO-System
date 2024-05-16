#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
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
        self.samp_rate = samp_rate = 32000
        self.num_sinusoids = num_sinusoids = 8
        self.normalized_max_doppler = normalized_max_doppler = 4
        self.flag_los = flag_los = True
        self.rician_factor = rician_factor = 4.0
        self.seed = seed = 0
        self.noise_voltage = noise_voltage = 0.2
        self.frequency_offset = frequency_offset = 0.0
        self.epsilon = epsilon = 1.0
        self.taps = taps = [1.0]
        self.noise_seed = noise_seed = 0

        ##################################################
        # Blocks
        ##################################################

        self.channels_fading_model_0 = channels.fading_model(
            num_sinusoids,
            (normalized_max_doppler / samp_rate),
            flag_los,
            rician_factor,
            seed,
        )
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=frequency_offset,
            epsilon=epsilon,
            taps=taps,
            noise_seed=noise_seed,
            block_tags=False,
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
            gr.sizeof_gr_complex * 1, "./file/tx.dat", False, 0, 0
        )
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(
            gr.sizeof_gr_complex * 1, "./file/rx.dat", False
        )
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.channels_fading_model_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_file_sink_0, 0))
        self.connect(
            (self.channels_fading_model_0, 0), (self.channels_channel_model_0, 0)
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
        self.channels_fading_model_0.set_fDTs(
            (self.normalized_max_doppler / self.samp_rate)
        )


def main(top_block_cls=channel, options=None):

    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == "__main__":
    main()
