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
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window


class channel(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        # The sample rate of the signal
        self.samp_rate = samp_rate = 32000
        # The number of sinusoids to use in simulating the channel; 8 is a good value
        self.num_sinusoids = num_sinusoids = 8
        # Normalized maximum doppler frequency (f_doppler / f_samprate)
        self.normalized_max_doppler = normalized_max_doppler = 0.01 / samp_rate
        # Include Line-of-Site path? selects between Rayleigh (NLOS) and Rician (LOS) models. False is Rayleigh, True is Rician
        self.los_model = los_model = True
        # Rician factor (ratio of the specular power to the scattered power)
        self.rician_factor = rician_factor = 4.0
        # Number to seed the noise generators
        self.seed = seed = 0
        # Time delay in the fir filter (in samples) for each arriving Wide-Sense Stationary Uncorrelated Scattering (WSSUS) Ray (defalut, min, max)
        self.pdp_delays = pdp_delays = [0.0]
        # Magnitude corresponding to each WSSUS Ray (linear) (defalut, min, max)
        self.pdp_magnitudes = pdp_magnitudes = [1.0]
        # Number of FIR taps to use in selective fading model
        self.num_taps = num_taps = 8
        self.file_source_path = file_source_path = "./file/tx.dat"
        self.file_sink_path = file_sink_path = "./file/rx3.dat"

        ##################################################
        # Blocks
        ##################################################

        self.channels_selective_fading_model_0 = channels.selective_fading_model(
            num_sinusoids,
            normalized_max_doppler,
            los_model,
            rician_factor,
            seed,
            pdp_delays,
            pdp_magnitudes,
            num_taps,
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
        self.blocks_file_sink_0_0_0_0 = blocks.file_sink(
            gr.sizeof_gr_complex * 1, file_sink_path, False
        )
        self.blocks_file_sink_0_0_0_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_throttle2_0, 0))
        self.connect(
            (self.blocks_throttle2_0, 0), (self.channels_selective_fading_model_0, 0)
        )
        self.connect(
            (self.channels_selective_fading_model_0, 0),
            (self.blocks_file_sink_0_0_0_0, 0),
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
        self.channels_selective_fading_model_0.set_fDTs((0.01 / self.samp_rate))


def main(top_block_cls=channel, options=None):

    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == "__main__":
    main()
