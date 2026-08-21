import torch
import pytest

from src.alignment.forced_aligner import CTCForcedAligner


def test_ctc_forced_alignment_basic():
    blank_id = 0

    aligner = CTCForcedAligner(
        blank_id=blank_id,
        sample_rate=16000,
    )

    # Vocabulary:
    # 0 = blank
    # 1 = token A
    # 2 = token B
    #
    # Acoustic sequence:
    # blank A A blank B B blank
    log_probs = torch.tensor(
        [
            [-0.01, -5.0, -5.0],
            [-5.0, -0.01, -5.0],
            [-5.0, -0.01, -5.0],
            [-0.01, -5.0, -5.0],
            [-5.0, -5.0, -0.01],
            [-5.0, -5.0, -0.01],
            [-0.01, -5.0, -5.0],
        ],
        dtype=torch.float32,
    )

    result = aligner.align(
        log_probs=log_probs,
        token_ids=[1, 2],
        duration=7.0,
    )

    assert len(result) == 2

    assert result[0].token == "1"
    assert result[1].token == "2"

    assert result[0].start < result[0].end
    assert result[1].start < result[1].end

    assert result[0].end <= result[1].start
    assert result[1].end <= 7.0

    assert 0.0 <= result[0].confidence <= 1.0
    assert 0.0 <= result[1].confidence <= 1.0


def test_ctc_forced_alignment_empty_transcript():
    aligner = CTCForcedAligner(
        blank_id=0,
        sample_rate=16000,
    )

    log_probs = torch.zeros(
        (5, 3),
        dtype=torch.float32,
    )

    result = aligner.align(
        log_probs=log_probs,
        token_ids=[],
        duration=5.0,
    )

    assert result == []


def test_ctc_forced_alignment_invalid_shape():
    aligner = CTCForcedAligner(
        blank_id=0,
        sample_rate=16000,
    )

    log_probs = torch.zeros(
        (5,),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        aligner.align(
            log_probs=log_probs,
            token_ids=[1],
            duration=5.0,
        )


def test_ctc_forced_alignment_invalid_duration():
    aligner = CTCForcedAligner(
        blank_id=0,
        sample_rate=16000,
    )

    log_probs = torch.zeros(
        (5, 3),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        aligner.align(
            log_probs=log_probs,
            token_ids=[1],
            duration=0,
        )


def test_ctc_forced_alignment_not_enough_frames():
    aligner = CTCForcedAligner(
        blank_id=0,
        sample_rate=16000,
    )

    log_probs = torch.zeros(
        (1, 3),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        aligner.align(
            log_probs=log_probs,
            token_ids=[1, 2],
            duration=1.0,
        )


def test_ctc_forced_alignment_invalid_token():
    aligner = CTCForcedAligner(
        blank_id=0,
        sample_rate=16000,
    )

    log_probs = torch.zeros(
        (5, 3),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        aligner.align(
            log_probs=log_probs,
            token_ids=[99],
            duration=5.0,
        )