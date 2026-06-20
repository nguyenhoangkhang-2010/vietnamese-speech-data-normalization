import torch
import numpy as np
from nemo.collections.asr.models import EncDecCTCModel

class NemoAligner:
    def __init__(self, cfg):
        self.use_nemo = cfg["nemo"].get("use_pretrained", False)

        if self.use_nemo:
            self.model = EncDecCTCModel.from_pretrained(cfg["nemo"]["model"])
            self.model.eval()
            self.model.preprocessor.featurizer.sample_rate = 16000
            self.model.sample_rate = 16000

    def align(self, audio, text):
        pred = text
        
        if isinstance(audio, np.ndarray):
            duration = audio.shape[0] / 16000
        else:
            duration = audio.shape[-1] / 16000

        return {
            "text": pred,
            "duration": duration
        }