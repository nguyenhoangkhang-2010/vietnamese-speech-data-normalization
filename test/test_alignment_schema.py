from src.alignment.schema import (
    AlignmentResult,
    WordAlignment,
)


def test_word_alignment_duration():
    word = WordAlignment(
        word="hello",
        start=1.0,
        end=2.5,
    )

    assert word.word == "hello"
    assert word.start == 1.0
    assert word.end == 2.5
    assert word.duration == 1.5


def test_alignment_result():
    words = [
        WordAlignment(
            word="xin",
            start=0.0,
            end=0.5,
        ),
        WordAlignment(
            word="chào",
            start=0.6,
            end=1.2,
        ),
    ]

    result = AlignmentResult(
        text="xin chào",
        language="vi",
        duration=1.5,
        words=words,
    )

    assert result.text == "xin chào"
    assert result.language == "vi"
    assert result.duration == 1.5
    assert len(result.words) == 2
    assert result.words[0].word == "xin"
    assert result.words[1].word == "chào"