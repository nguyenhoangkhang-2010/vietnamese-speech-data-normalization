from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class TokenAlignment:

    token: str
    start: float
    end: float
    confidence: float


class CTCForcedAligner:

    def __init__(
        self,
        blank_id: int,
        sample_rate: int = 16000,
    ) -> None:
        if blank_id < 0:
            raise ValueError("blank_id must be >= 0.")

        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")

        self.blank_id = blank_id
        self.sample_rate = sample_rate

    def align(
        self,
        log_probs: torch.Tensor,
        token_ids: Sequence[int],
        duration: float,
    ) -> list[TokenAlignment]:
        if not isinstance(log_probs, torch.Tensor):
            raise TypeError("log_probs must be a torch.Tensor.")

        if log_probs.ndim != 2:
            raise ValueError(
                "log_probs must have shape [time, vocabulary]."
            )

        if len(token_ids) == 0:
            return []

        if duration <= 0:
            raise ValueError("duration must be > 0.")

        time_steps, vocab_size = log_probs.shape

        if time_steps == 0:
            raise ValueError("log_probs contains no time steps.")

        for token_id in token_ids:
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    f"Token ID {token_id} is outside vocabulary "
                    f"range [0, {vocab_size})."
                )

        log_probs = log_probs.detach().float().cpu()

        states = self._build_ctc_states(token_ids)

        num_states = len(states)

        if time_steps < len(token_ids):
            raise ValueError(
                "Not enough acoustic time steps to align transcript."
            )

        scores = torch.full(
            (time_steps, num_states),
            float("-inf"),
            dtype=torch.float32,
        )

        backpointers = torch.full(
            (time_steps, num_states),
            -1,
            dtype=torch.long,
        )

        scores[0, 0] = log_probs[0, self.blank_id]

        if num_states > 1:
            scores[0, 1] = log_probs[0, states[1]]

        for t in range(1, time_steps):
            for s in range(num_states):
                token_id = states[s]

                candidates = [
                    (
                        scores[t - 1, s],
                        s,
                    )
                ]

                if s > 0:
                    candidates.append(
                        (
                            scores[t - 1, s - 1],
                            s - 1,
                        )
                    )

                if (
                    s > 1
                    and token_id != self.blank_id
                    and token_id != states[s - 2]
                ):
                    candidates.append(
                        (
                            scores[t - 1, s - 2],
                            s - 2,
                        )
                    )

                best_score, best_state = max(
                    candidates,
                    key=lambda item: item[0],
                )

                scores[t, s] = (
                    best_score
                    + log_probs[t, token_id]
                )

                backpointers[t, s] = best_state

        final_candidates = [
            (scores[-1, num_states - 1], num_states - 1)
        ]

        if num_states > 1:
            final_candidates.append(
                (scores[-1, num_states - 2], num_states - 2)
            )

        final_score, state = max(
            final_candidates,
            key=lambda item: item[0],
        )

        if not torch.isfinite(final_score):
            raise ValueError(
                "CTC forced alignment failed: "
                "transcript cannot be aligned to audio."
            )

        path = [state]

        for t in range(time_steps - 1, 0, -1):
            state = int(backpointers[t, state])

            if state < 0:
                raise ValueError(
                    "Invalid CTC backpointer encountered."
                )

            path.append(state)

        path.reverse()

        frame_duration = duration / time_steps

        return self._extract_alignments(
            states=states,
            path=path,
            log_probs=log_probs,
            frame_duration=frame_duration,
        )

    def _build_ctc_states(
        self,
        token_ids: Sequence[int],
    ) -> list[int]:

        states: list[int] = []

        for token_id in token_ids:
            states.append(self.blank_id)
            states.append(int(token_id))

        states.append(self.blank_id)

        return states

    def _extract_alignments(
        self,
        states: Sequence[int],
        path: Sequence[int],
        log_probs: torch.Tensor,
        frame_duration: float,
    ) -> list[TokenAlignment]:

        alignments: list[TokenAlignment] = []

        current_state: int | None = None
        token_frames: list[int] = []

        for frame, state in enumerate(path):
            token_id = states[state]

            if token_id == self.blank_id:
                if current_state is not None:
                    alignment = self._create_alignment(
                        state=current_state,
                        frames=token_frames,
                        states=states,
                        log_probs=log_probs,
                        frame_duration=frame_duration,
                    )

                    if alignment is not None:
                        alignments.append(alignment)

                    current_state = None
                    token_frames = []

                continue

            if current_state != state:
                if current_state is not None:
                    alignment = self._create_alignment(
                        state=current_state,
                        frames=token_frames,
                        states=states,
                        log_probs=log_probs,
                        frame_duration=frame_duration,
                    )

                    if alignment is not None:
                        alignments.append(alignment)

                current_state = state
                token_frames = [frame]
            else:
                token_frames.append(frame)

        if current_state is not None:
            alignment = self._create_alignment(
                state=current_state,
                frames=token_frames,
                states=states,
                log_probs=log_probs,
                frame_duration=frame_duration,
            )

            if alignment is not None:
                alignments.append(alignment)

        return alignments

    def _create_alignment(
        self,
        state: int,
        frames: Sequence[int],
        states: Sequence[int],
        log_probs: torch.Tensor,
        frame_duration: float,
    ) -> TokenAlignment | None:

        if not frames:
            return None

        if state < 0 or state >= len(states):
            raise ValueError(
                f"Invalid alignment state: {state}. "
                f"Expected range [0, {len(states) - 1}]"
            )

        token_id = states[state]

        # Blank states are not returned as token alignments.
        if token_id == self.blank_id:
            return None

        if token_id < 0 or token_id >= log_probs.shape[1]:
            raise ValueError(
                f"Invalid token_id={token_id} for acoustic vocabulary "
                f"size={log_probs.shape[1]}"
            )

        frame_indices = list(frames)

        probabilities = torch.exp(
            log_probs[frame_indices, token_id]
        )

        start_frame = frame_indices[0]
        end_frame = frame_indices[-1] + 1

        start = start_frame * frame_duration
        end = end_frame * frame_duration

        return TokenAlignment(
            token=str(token_id),
            start=start,
            end=end,
            confidence=float(probabilities.mean().item()),
        )