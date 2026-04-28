import torch
from nemo.collections.asr.models import EncDecCTCModel

class NemoAligner:
    def __init__(self, cfg):
        self.use_nemo = cfg["nemo"]["use_pretrained"]

        if self.use_nemo:

            self.model = EncDecCTCModel.from_pretrained(
                cfg["nemo"]["model"]
            )

    def align(self, audio, text):
        if self.use_nemo:
            with torch.no_grad():
                pred = self.model.transcribe([audio])[0]
        else:
            pred = text

        return {
            "text": pred,
            "duration": len(audio) / 16000
        }