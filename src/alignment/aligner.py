import torch
import numpy as np
from nemo.collections.asr.models import EncDecCTCModel

class NemoAligner:
    def __init__(self, cfg):
        self.use_nemo = cfg["nemo"]["use_pretrained"]

        if self.use_nemo:
            self.model = EncDecCTCModel.from_pretrained(
                cfg["nemo"]["model"]
            )
            self.model.preprocessor.featurizer.sample_rate = 16000
            self.model.sample_rate = 16000

    def align(self, audio, text):
        if self.use_nemo:
            with torch.no_grad():
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                
                audio = audio.reshape(-1)
                
                audio = audio.to(self.model.device)
                
                pred = self.model.transcribe([audio], batch_size=1)[0]
                
                if not isinstance(pred, str) and hasattr(pred, 'text'):
                    pred = pred.text
        else:
            pred = text

        return {
            "text": pred,
            "duration": audio.shape[0] / 16000
        }