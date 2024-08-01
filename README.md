# Auto-encoder in wireless communication system

## Explanation of the code

### [Step 1](./AE_SISO_MQAM_AWGN_python.ipynb)

Auto-encoder in SISO system with 16QAM modulation, AWGN channel, without CSI has shown fair performance in terms of BER. Even 16*8*2 encoder and 2*16*16 decoder has outperformed the traditional 16QAM modulation.

### [Step 2](AE_SISO_MQAM_Rayleigh_python.ipynb)

Then I transferred the code to the rayleigh fading channel. The rayleigh fading channel is simulated by generating random complex channel coefficients, performing complex multiplication, adding white Gaussian noise.

First, I enlarge the size of the encoder and decoder to 16*16*16*2 and 4*512*512*16. Then I add perfect CSI to the decoder. The result looks good.

### [Step 3](AE_SISO_MQAM_Rayleigh_python_no_CSI.ipynb)

Then I try to remove the CSI from the decoder. The result is not good. The BER is very high. The reason is that the channel is time-varying. The channel is different for each symbol. The decoder cannot decode the symbol without knowing the channel.

**Try to add a channel estimator.**

NN channel estimator doesn't need do design the "pilot"

### [Step 4](AE_SISO_MQAM_Rayleigh_GNU.ipynb)

Then I try to use GNU Radio's channel block to generate the dataset and train the model. In this way the gradient infomation of the channel is lost.
