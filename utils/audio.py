import torch
import soundfile


def load_audio(audiofile):
    """
    Read speech from audio file and return the sample rate and
    decoded PCM audio as a float32 torch tensor.

    Returns (samples, sample_rate).
    """
    samples, sample_rate = soundfile.read(audiofile, dtype="float32", always_2d=True)
    samples = torch.from_numpy(samples.T)
    return samples, sample_rate
