# read wav file and get the sample rate
import wave


def get_sample_rate(wav_file):
    with wave.open(wav_file, "rb") as wf:
        sample_rate = wf.getframerate()
    return sample_rate


# Replace 'input.wav' with the path to your WAV file
wav_file = "GNU Radio/text.wav"
sample_rate = get_sample_rate(wav_file)
print("Sample rate:", sample_rate)
