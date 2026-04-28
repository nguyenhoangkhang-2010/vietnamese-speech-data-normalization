import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, cfg):
        self.sr = cfg["audio"]["sample_rate"]

    def process(self, path):
        audio, sr = librosa.load(path, sr=self.sr)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio