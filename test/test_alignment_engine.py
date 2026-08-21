import numpy as np
import pytest

from src.alignment.engine import AlignmentEngine


def test_alignment_engine_basic():
    engine = AlignmentEngine(sample_rate=16000)

    audio = np.zeros(16000, dtype=np.float32)

    result = engine.align(
        audio=audio,
        transcript="xin chào",
        language="vi",
    )

    assert result.text == "xin chào"
    assert result.language == "vi"
    assert result.duration == pytest.approx(1.0)

    assert len(result.words) == 2

    assert result.words[0].word == "xin"
    assert result.words[1].word == "chào"

    assert result.words[0].start == pytest.approx(0.0)
    assert result.words[0].end == pytest.approx(0.5)

    assert result.words[1].start == pytest.approx(0.5)
    assert result.words[1].end == pytest.approx(1.0)


def test_alignment_engine_normalizes_transcript():
    engine = AlignmentEngine()

    audio = np.zeros(16000, dtype=np.float32)

    result = engine.align(
        audio=audio,
        transcript="  xin    chào   bạn  ",
        language="vi",
    )

    assert result.text == "xin chào bạn"
    assert [word.word for word in result.words] == [
        "xin",
        "chào",
        "bạn",
    ]


def test_alignment_engine_empty_audio():
    engine = AlignmentEngine()

    with pytest.raises(ValueError, match="Audio không được rỗng"):
        engine.align(
            audio=np.array([], dtype=np.float32),
            transcript="xin chào",
            language="vi",
        )


def test_alignment_engine_empty_transcript():
    engine = AlignmentEngine()

    audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ValueError, match="Transcript không được rỗng"):
        engine.align(
            audio=audio,
            transcript="   ",
            language="vi",
        )


def test_alignment_engine_invalid_language():
    engine = AlignmentEngine()

    audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ValueError, match="Language không được rỗng"):
        engine.align(
            audio=audio,
            transcript="xin chào",
            language="",
        )


def test_alignment_engine_multilingual():
    engine = AlignmentEngine()

    audio = np.zeros(32000, dtype=np.float32)

    result = engine.align(
        audio=audio,
        transcript="hello world",
        language="en",
    )

    assert result.language == "en"
    assert result.duration == pytest.approx(2.0)
    assert len(result.words) == 2