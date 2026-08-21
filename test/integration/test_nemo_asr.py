from pathlib import Path

import yaml

from src.alignment.aligner import NemoAligner
from src.audio.processor import AudioProcessor


def load_config():
    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def test_nemo_vietnamese_real_audio():
    cfg = load_config()

    audio_path = (
        "data/raw/audio/Chodoicodangso.mp4"
    )

    assert Path(audio_path).exists()

    audio_processor = AudioProcessor(cfg)

    audio = audio_processor.process(
        audio_path
    )

    aligner = NemoAligner(cfg)

    result = aligner.transcribe(
        audio,
        language="vi",
    )

    assert result["language"] == "vi"
    assert result["text"].strip()
    assert result["duration"] > 0
    
def test_nemo_english_real_audio():
    cfg = load_config()

    audio_path = (
        "data/raw/audio/love-yourself.mp4"
    )

    assert Path(audio_path).exists()

    audio_processor = AudioProcessor(cfg)

    audio = audio_processor.process(
        audio_path
    )

    aligner = NemoAligner(cfg)

    result = aligner.transcribe(
        audio,
        language="en",
    )

    assert result["language"] == "en"
    assert result["text"].strip()
    assert result["duration"] > 0