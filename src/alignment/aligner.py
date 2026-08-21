from pathlib import Path

import numpy as np
import torch

from nemo.collections.asr.models import EncDecCTCModel


class NemoAligner:
    """NeMo-based ASR inference for English and Vietnamese."""

    def __init__(self, cfg):
        nemo_cfg = cfg["nemo"]

        self.use_nemo = nemo_cfg.get("use_pretrained", False)

        self.sample_rate = cfg["audio"]["sample_rate"]

        self.default_language = nemo_cfg.get(
            "default_language",
            "vi",
        )

        self.device = nemo_cfg.get(
            "device",
            "cpu",
        )

        self.models = {}

        if not self.use_nemo:
            return

        model_configs = nemo_cfg.get(
            "models",
            {},
        )

        if not model_configs:
            raise ValueError(
                "Không tìm thấy cấu hình NeMo models."
            )

        self._load_models(model_configs)

    def _load_models(self, model_configs):
        """Load local NeMo .nemo checkpoints."""

        for language, model_path in model_configs.items():
            path = Path(model_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy NeMo model "
                    f"cho language='{language}': {path}"
                )

            if path.suffix.lower() != ".nemo":
                raise ValueError(
                    f"NeMo model phải là file .nemo: {path}"
                )

            print(
                f"Loading NeMo model for "
                f"'{language}': {path}"
            )

            model = EncDecCTCModel.restore_from(
                restore_path=str(path)
            )

            model.eval()
            
            model.sample_rate = self.sample_rate

            if self.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA được yêu cầu nhưng "
                        "không khả dụng."
                    )

                model = model.cuda()

            else:
                model = model.cpu()

            self.models[language] = model

    def transcribe(self, audio, language=None):

        if not self.use_nemo:
            raise RuntimeError(
                "NeMo ASR đang bị disable trong config."
            )

        language = (
            language
            or self.default_language
        )

        if language not in self.models:
            available = ", ".join(
                self.models.keys()
            )

            raise ValueError(
                f"Không có NeMo model cho "
                f"language='{language}'. "
                f"Available: {available}"
            )

        audio = self._prepare_audio(audio)

        model = self.models[language]

        with torch.inference_mode():
            hypotheses = model.transcribe(
                [audio],
                batch_size=1,
            )

        if not hypotheses:
            raise ValueError(
                "NeMo không trả về kết quả "
                "transcription."
            )

        hypothesis = hypotheses[0]

        if hasattr(hypothesis, "text"):
            text = hypothesis.text

        elif isinstance(hypothesis, str):
            text = hypothesis

        else:
            text = str(hypothesis)

        text = text.strip()

        if not text:
            raise ValueError(
                "NeMo không nhận diện được "
                "nội dung audio."
            )

        duration = (
            len(audio) / self.sample_rate
        )

        return {
            "text": text,
            "duration": duration,
            "language": language,
        }

    def _prepare_audio(self, audio):
        """Convert audio to mono float32 NumPy array."""

        if isinstance(audio, torch.Tensor):
            audio = (
                audio
                .detach()
                .cpu()
                .numpy()
            )

        else:
            audio = np.asarray(audio)

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim > 1:
            audio = np.squeeze(audio)

        if audio.ndim != 1:
            raise ValueError(
                "Audio phải là mono 1-D array. "
                f"Received shape={audio.shape}"
            )

        if audio.size == 0:
            raise ValueError(
                "Audio không được rỗng."
            )

        return audio