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

        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################

        self.channels_channel_model_0_0 = channels.channel_model(
            noise_voltage=0,
            frequency_offset=0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0,
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
        self.blocks_file_sink_0_0 = blocks.file_sink(
            gr.sizeof_gr_complex * 1, "./file/rx.dat", False
        )
        self.blocks_file_sink_0_0.set_unbuffered(True)

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
