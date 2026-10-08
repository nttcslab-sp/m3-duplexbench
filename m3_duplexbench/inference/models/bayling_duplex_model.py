from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from m3_duplexbench.inference.events import EventSample
from m3_duplexbench.inference.models.base import BaseInferenceModel
from m3_duplexbench.utils import resample_mono_np, pad_or_trim

from bayling_duplex import BayLingDuplex


# BayLingDuplex sampling rate
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 22050


class BayLingDuplexModel(BaseInferenceModel):
    def __init__(
        self,
        model_path: str,
        speech_tokenizer_path: str,
        decoder_path: str,
        device: str = "cuda",
        use_context: bool = False,
        use_teacher_forcing: bool = False,
        interleave_ratio: str = "10:5:10",
        max_duration: float | None = None,
        temperature: float = 0.8,
        top_p: float = 0.8,
        max_epad_count: int | None = 50,
    ):
        super().__init__()

        if use_teacher_forcing:
            raise NotImplementedError(
                "BayLing-Duplex wrapper does not support teacher forcing."
            )

        self.model = BayLingDuplex(
            model_path=model_path,
            speech_tokenizer_path=speech_tokenizer_path,
            decoder_path=decoder_path,
            interleave_ratio=interleave_ratio,
            device=device,
        )

        self.use_context = use_context
        self.use_teacher_forcing = use_teacher_forcing

        self.max_duration = max_duration
        self.temperature = temperature
        self.top_p = top_p
        self.max_epad_count = max_epad_count
        self.synthesize = "all" # if "response", only first segment is returned. 

        self.use_context = use_context
        self.use_teacher_forcing = use_teacher_forcing and self.use_context
        self._context_mode = (
            "teacher_forcing"
            if use_teacher_forcing
            else "user_audio"
            if use_context
            else "none"
        )

    @property
    def name(self) -> str:
        return "bayling_duplex"

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    @property
    def context_mode(self) -> str:
        return self._context_mode

    def _run_model(self, input_audio: np.ndarray) -> np.ndarray:
        """Run BayLing-Duplex on mono audio.
        input_audio is expected to be mono float ndarray at INPUT_SAMPLE_RATE.
        Returns mono float ndarray at self.sample_rate.
        """
        assert input_audio.ndim == 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_wav = tmpdir / "input.wav"
            output_wav = tmpdir / "output.wav"

            # Make input audio file emporarily.
            sf.write(str(input_wav), input_audio, INPUT_SAMPLE_RATE)


            kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.max_duration is None: # default
                input_duration = input_audio.shape[0] / INPUT_SAMPLE_RATE
                kwargs["max_duration"] = input_duration
            else:
                kwargs["max_duration"] = self.max_duration
            if self.max_epad_count is not None:
                kwargs["max_epad_count"] = self.max_epad_count
            print(kwargs)
            result = self.model.generate(str(input_wav), **kwargs)

            # BayLing-Duplex official API saves audio from response_audio_tokens.
            if self.synthesize == "all":
                tokens = result.audio_tokens
            else:
                tokens = result.response_audio_tokens
            self.model.save_audio(tokens, str(output_wav))

            output_audio, output_sr = sf.read(
                str(output_wav),
                dtype="float32",
                always_2d=False,
            )

        # Add time_offset
        offset_samples = round(self.model.time_offset * output_sr)
        output_audio = np.concatenate([
            np.zeros(offset_samples),
            output_audio,
        ])
        
        if output_sr != self.sample_rate:
            output_audio = resample_mono_np(
                output_audio,
                output_sr,
                self.sample_rate,
            )

        return output_audio

    def generate(
        self,
        context_audio: np.ndarray,
        pre_event_audio: np.ndarray | None,
        eval_window_audio: np.ndarray,
        sample_rate: int,
        event: EventSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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

        # with teacher-forcing
        else:
            input_user_audio = pre_and_eval_user_audio
            context_len = 0

            # with context
            if self.use_context:
                context_user_audio = context_audio[:, event.user_channel]
                context_len = int(self.sample_rate * len(context_user_audio) / sample_rate)
                input_user_audio = np.concatenate([context_user_audio, input_user_audio])
# Common process >>>

# <<< Model specific
        # resampling
        input_user_audio = resample_mono_np(input_user_audio, sample_rate, INPUT_SAMPLE_RATE)
        # run BayLing-Duplex
        system_out_audio = self._run_model(input_user_audio)
# Model specific >>>

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
