import numpy as np

from m3_duplexbench.inference.events import EventSample
from m3_duplexbench.inference.models.base import BaseInferenceModel

class DummyModel(BaseInferenceModel):
    def __init__(self, output_duration: float = 10.0):
        self.output_duration = output_duration
        self._context_mode = "user_audio"

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def sample_rate(self) -> int | None:
        return None

    @property
    def context_mode(self) -> str:
        return self._context_mode

    def generate(
        self,
        context_audio: np.ndarray,
        pre_event_audio: np.ndarray | None,
        eval_window_audio: np.ndarray,
        sample_rate: int,
        event: EventSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray|None]:
        if pre_event_audio is not None:
            user_audio = np.concatenate(
                [pre_event_audio[:, event.user_channel], eval_window_audio]
            )
        else:
            user_audio = eval_window_audio

        if context_audio.ndim != 2:
            raise ValueError(
                f"context_audio must be 2D, got shape {context_audio.shape}"
            )
        if user_audio.ndim != 1:
            raise ValueError(f"user_audio must be mono, got shape {user_audio.shape}")

        model_audio = np.zeros_like(user_audio, dtype=np.float32)
        return user_audio.astype(np.float32), model_audio, None
