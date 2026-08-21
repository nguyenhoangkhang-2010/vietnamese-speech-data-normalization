import numpy as np
import pytest

from src.alignment.aligner import NemoAligner


def test_nemo_disabled():
    cfg = {
        "audio": {
            "sample_rate": 16000,
        },
        "nemo": {
            "use_pretrained": False,
            "default_language": "vi",
        },
    }

    aligner = NemoAligner(cfg)

    assert aligner.use_nemo is False
    assert aligner.models == {}


def test_prepare_audio_numpy():
    cfg = {
        "audio": {
            "sample_rate": 16000,
        },
        "nemo": {
            "use_pretrained": False,
            "default_language": "vi",
        },
    }

    aligner = NemoAligner(cfg)

    audio = np.zeros(16000, dtype=np.float32)

    result = aligner._prepare_audio(audio)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (16000,)


def test_prepare_audio_empty():
    cfg = {
        "audio": {
            "sample_rate": 16000,
        },
        "nemo": {
            "use_pretrained": False,
            "default_language": "vi",
        },
    }

    aligner = NemoAligner(cfg)

    with pytest.raises(ValueError):
        aligner._prepare_audio(
            np.array([], dtype=np.float32)
        )
        
import numpy as np
import yaml

from src.alignment.aligner import NemoAligner


def load_config():
    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def test_transcribe_requires_nemo():
    cfg = load_config()

    cfg["nemo"]["use_pretrained"] = False

    aligner = NemoAligner(cfg)

    audio = np.zeros(
        16000,
        dtype=np.float32,
    )

    try:
        aligner.transcribe(audio, "vi")
        assert False
    except RuntimeError:
        assert True