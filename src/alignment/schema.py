from dataclasses import dataclass, field
from typing import List


@dataclass
class WordAlignment:
    """Alignment information for a single word."""

    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AlignmentResult:
    """Result returned by the alignment pipeline."""

    text: str
    language: str
    duration: float
    words: List[WordAlignment] = field(default_factory=list)