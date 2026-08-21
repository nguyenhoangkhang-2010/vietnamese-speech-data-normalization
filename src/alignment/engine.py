import re
from typing import List

import numpy as np

from .schema import AlignmentResult, WordAlignment


class AlignmentEngine:
    """Prepare transcript tokens for forced-alignment."""

    def __init__(self, sample_rate: int = 16000):
        if sample_rate <= 0:
            raise ValueError("sample_rate phải lớn hơn 0.")

        self.sample_rate = sample_rate

    def align(
        self,
        audio: np.ndarray,
        transcript: str,
        language: str,
    ) -> AlignmentResult:

        audio = self._prepare_audio(audio)

        transcript = self._normalize_transcript(transcript)

        if not transcript:
            raise ValueError("Transcript không được rỗng.")

        if not language:
            raise ValueError("Language không được rỗng.")

        duration = len(audio) / self.sample_rate

        tokens = self._tokenize(transcript)

        words = self._create_baseline_alignment(
            tokens=tokens,
            duration=duration,
        )

        return AlignmentResult(
            text=transcript,
            language=language,
            duration=duration,
            words=words,
        )

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validate and normalize audio to mono float32."""

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
            raise ValueError("Audio không được rỗng.")

        return audio

    @staticmethod
    def _normalize_transcript(transcript: str) -> str:
        """Normalize whitespace in transcript."""

        if not isinstance(transcript, str):
            raise TypeError("Transcript phải là string.")

        return re.sub(r"\s+", " ", transcript).strip()

    @staticmethod
    def _tokenize(transcript: str) -> List[str]:

        return transcript.split()

    @staticmethod
    def _create_baseline_alignment(
        tokens: List[str],
        duration: float,
    ) -> List[WordAlignment]:

        if not tokens:
            return []

        token_duration = duration / len(tokens)

        words = []

        for index, token in enumerate(tokens):
            start = index * token_duration

            if index == len(tokens) - 1:
                end = duration
            else:
                end = (index + 1) * token_duration

            words.append(
                WordAlignment(
                    word=token,
                    start=start,
                    end=end,
                )
            )

        return words