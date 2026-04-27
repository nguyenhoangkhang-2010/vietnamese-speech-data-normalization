import nemo.collections.asr as nemo_asr

class Aligner:
    def __init__(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="stt_en_conformer_ctc_large"
        )
        pass