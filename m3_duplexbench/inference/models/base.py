import numpy as np

from m3_duplexbench.inference.events import EventSample
from m3_duplexbench.utils import resample_mono_np, pad_or_trim

class BaseInferenceModel:
    def __init__(
        self,
        use_context: bool = False,
        use_teacher_forcing: bool = False,
    ):
        self.use_context = use_context
        self.use_teacher_forcing = use_teacher_forcing

    @property
    def name(self) -> str:
        ...

    @property
    def sample_rate(self) -> int:
        """Return output sample rate."""
        ...

    @property
    def context_mode(self) -> str:
        ...

    def generate(
        self,
        context_audio: np.ndarray,
        pre_event_audio: np.ndarray | None,
        eval_window_audio: np.ndarray,
        sample_rate: int,
        event: EventSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray|None]:
        """Generate model output for an event."""
        assert context_audio.ndim == 2

# <<< Common process
        if pre_event_audio is not None:
            pre_and_eval_user_audio = np.concatenate(
                [pre_event_audio[:, event.user_channel], eval_window_audio]
            )
            # only used for teacher-foring (for Moshi and its variants)
            pre_event_system_audio = pre_event_audio[:, event.system_channel]
        else: # BC, BARGE_IN
            pre_and_eval_user_audio = eval_window_audio
            pre_event_system_audio = None

        if len(pre_and_eval_user_audio) == 0:
            return pre_and_eval_user_audio, pre_and_eval_user_audio, None

        target_len = int(self.sample_rate * len(pre_and_eval_user_audio) / sample_rate)

        # with teacher-forcing
        if self.use_teacher_forcing:
            raise NotImplementedError

        # without teacher-forcing
        else:
            input_user_audio = pre_and_eval_user_audio
            context_len = 0

            # with context
            if self.use_context:
                context_user_audio = context_audio[:, event.user_channel]
                context_len = int(self.sample_rate * len(context_user_audio) / sample_rate)
                input_user_audio = np.concatenate([context_user_audio, input_user_audio])
# Common process >>>

# <<< Model specific process
        system_out_audio = np.zeros(target_len)
# Model specific process >>>

# <<< Common process
        context_system_out_audio = None
        if self.use_teacher_forcing:
            raise NotImplementedError
        else:
            pre_and_eval_system_out_audio = system_out_audio[context_len : context_len + target_len]
            if context_len > 0:
                context_system_out_audio = system_out_audio[:context_len]

        pre_and_eval_system_out_audio = pad_or_trim(pre_and_eval_system_out_audio, target_len)

        if sample_rate != self.sample_rate:
            pre_and_eval_user_audio = resample_mono_np(
                pre_and_eval_user_audio, sample_rate, self.sample_rate,
            )
        pre_and_eval_user_audio = pad_or_trim(pre_and_eval_user_audio, target_len)

        return pre_and_eval_user_audio, pre_and_eval_system_out_audio, context_system_out_audio
# Common process >>>
