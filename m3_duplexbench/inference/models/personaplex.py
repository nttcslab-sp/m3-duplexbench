# Original: https://github.com/NVIDIA/personaplex/blob/main/moshi/moshi/offline.py
import numpy as np

import argparse
import os
import tarfile
from pathlib import Path
import json
from typing import Optional, List

import numpy as np
import torch
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

# personaplex/moshi
from moshi.client_utils import make_log
from moshi.models import loaders, LMGen, MimiModel
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.offline import (
    log,
    seed_all,
    wrap_with_system_tags,
    warmup,
    decode_tokens_to_pcm,
    _get_voice_prompt_dir,
    encode_audio_tensor_to_codes,
)
from moshi.utils.textproc import load_alignments
from moshi.utils.interleaver import CustomInterleaver

from m3_duplexbench.inference.events import EventSample
from m3_duplexbench.inference.models.base import BaseInferenceModel
from m3_duplexbench.utils import resample_mono_np, pad_or_trim


class PersonaPlexModel(BaseInferenceModel):
    """PersonaPlex wrapper."""

    def __init__(
        self,
        hf_repo: str,
        voice_prompt: str,
        voice_prompt_dir: str | None = None,
        prioritize_voice_prompt: bool = False,
        voice_prompt_duration: float | None = None,
        domain: str = "assistant",
        text_prompt: str | None = None,
        tokenizer: str | None = None,
        moshi_weight: str | None = None,
        mimi_weight: str | None = None,
        device: str = "cuda",
        seed: int = 42424242,
        use_context: bool = True,
        use_teacher_forcing: bool = False,
        use_text_conditioning: bool = False,
        temp_audio: float = 0.8,
        temp_text: float = 0.7,
        topk_audio: int = 250,
        topk_text: int = 25,
        greedy: bool = False,
        save_voice_prompt_embeddings: bool = False,
        cpu_offload: bool = False,
        print_text_tokens: bool = False
    ):
        seed_all(seed)
        log("info", f"seed: {seed}")

        self.device = device
        self.moshi_weight = moshi_weight
        self.mimi_weight = mimi_weight

        self.prioritize_voice_prompt = prioritize_voice_prompt
        self.voice_prompt_duration = voice_prompt_duration

        self.use_context = use_context
        self.use_teacher_forcing = use_teacher_forcing and self.use_context
        self.use_text_conditioning = use_text_conditioning
        self._context_mode = (
            "teacher_forcing_with_text"
            if use_teacher_forcing and use_text_conditioning
            else "teacher_forcing_without_text"
            if use_teacher_forcing
            else "user_audio"
            if use_context
            else "none"
        )

        self.print_text_tokens = print_text_tokens

        if text_prompt is None:
            log("info", f"domain: {domain}")
            if domain == "assistant":
                text_prompt = (
                    "You are a wise and friendly teacher. "
                    "Answer questions or provide advice in a clear and engaging way."
                )
            elif domain == "chat":
                text_prompt = (
                    "You enjoy having a good conversation."
                )
            else:
                log("warning", "Unknown domain is specified. Using default text prompt.")
                text_prompt = (
                    "You are a wise and friendly teacher. "
                    "Answer questions or provide advice in a clear and engaging way."
                )
        log("info", f"text_prompt: {text_prompt}")

        # Download config.json to increment download counter
        # No worries about double-counting since config.json will be cached the second time
        hf_hub_download(hf_repo, "config.json")

        # 1) Load Mimi encoders/decoders (same as server.py)
        log("info", "loading mimi")
        if self.mimi_weight is None:
            self.mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
        self.mimi = loaders.get_mimi(self.mimi_weight, device)
        self.other_mimi = loaders.get_mimi(self.mimi_weight, device)
        log("info", "mimi loaded")

        # 2) Load tokenizer
        if tokenizer is None:
            tokenizer = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore

        # 3) Load Moshi LM and eval mode
        log("info", "loading moshi")
        if self.moshi_weight is None:
            self.moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
        self.lm = loaders.get_moshi_lm(self.moshi_weight, device=device, cpu_offload=cpu_offload)
        self.lm.eval()
        log("info", "moshi loaded")
        # 4) Construct LMGen like server.py's ServerState does
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            self.lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),  # spacer after prompts
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
            use_sampling=not greedy,
            temp=temp_audio,
            temp_text=temp_text,
            top_k=topk_audio,
            top_k_text=topk_text,
        )
        # Keep models in streaming mode similar to the server
        #RM self.mimi.streaming_forever(1)
        #RM self.other_mimi.streaming_forever(1)
        #RM self.lm_gen.streaming_forever(1)

        # 5) Warmup
        log("info", "warming up the model")
        #RM warmup(self.mimi, self.other_mimi, self.lm_gen, self.device, self.frame_size)
        with self.mimi.streaming(1), self.other_mimi.streaming(1), self.lm_gen.streaming(1):
            warmup(
                self.mimi,
                self.other_mimi,
                self.lm_gen,
                self.device,
                self.frame_size,
            )

        # 6) Prompt configuration (text + voice)
        if os.path.exists(voice_prompt):
            self.voice_prompt_path = voice_prompt
        else:
            voice_prompt_dir = _get_voice_prompt_dir(
                voice_prompt_dir,
                hf_repo,
            )
            assert voice_prompt_dir is not None
            self.voice_prompt_path = os.path.join(voice_prompt_dir, voice_prompt)
            if not os.path.exists(self.voice_prompt_path):
                raise FileNotFoundError(f"Voice prompt {self.voice_prompt_path} not found.")

        if self.use_teacher_forcing and not self.prioritize_voice_prompt:
            log("info", "Skip loading voice prompt.")
        else:
            log("info", "Loading voice prompt.")
           # System text tokens (k=0) and agent voice-prompt audio (k=1..dep_q) are forced
            if self.voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(self.voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(self.voice_prompt_path)

        log("info", "Loading text prompt.")
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
        )

        # prepare interleaver
        if self.use_text_conditioning:
            self.interleaver = CustomInterleaver(
                self.text_tokenizer,
                self.mimi.frame_rate,
                self.lm.text_padding_token_id,
                self.lm.end_of_text_padding_id,
                self.lm.zero_token_id,
                keep_main_only=True,
                resegment_by_tokenizer=False,
                device='cpu',
            )

    @property
    def name(self) -> str:
        return "personaplex"

    @property
    def sample_rate(self) -> int:
        return int(self.mimi.sample_rate)

    @property
    def context_mode(self) -> str:
        return self._context_mode

    def apply_stereo_conditioning(
        self,
        user_audio: np.ndarray | None,
        system_audio: np.ndarray | None,
        system_text_tokens: torch.Tensor | None,
    ):
        """Condition PersonaPlex on paired user/system audio."""
        log("info", "Start stereo conditioning")

        if user_audio is not None and system_audio is not None:
            num_samples = max(user_audio.shape[-1], system_audio.shape[-1])
        elif user_audio is not None:
            num_samples = user_audio.shape[-1]
        else:
            assert system_audio is not None
            num_samples = system_audio.shape[-1]

        if num_samples == 0:
            log("info", "Conditioning audio is empty. Skipped.")
            return

        if user_audio is None:
            user_audio = np.zeros((1, num_samples), dtype=np.float32)
        elif user_audio.shape[-1] < num_samples:
            pad_len = num_samples - user_audio.shape[-1]
            user_audio = np.pad(user_audio, ((0, 0), (0, pad_len)))

        if system_audio is None:
            system_audio = np.zeros((1, num_samples), dtype=np.float32)
        elif system_audio.shape[-1] < num_samples:
            pad_len = num_samples - system_audio.shape[-1]
            system_audio = np.pad(system_audio, ((0, 0), (0, pad_len)))

        user_codes = encode_audio_tensor_to_codes(
            user_audio,
            self.mimi,
            self.frame_size,
        )
        system_codes = encode_audio_tensor_to_codes(
            system_audio,
            self.other_mimi,
            self.frame_size,
        )

        steps = min(user_codes.shape[-1], system_codes.shape[-1])
        if system_text_tokens is not None:
            steps = min(steps, system_text_tokens.shape[-1])

        for t in range(steps):
            user_step = user_codes[:, :, t : t + 1]
            system_step = system_codes[:, :, t : t + 1]
            text_step = None
            if system_text_tokens is not None:
                # text_step = text_tokens[:, 0, t]
                text_step = system_text_tokens[t].unsqueeze(0)
            tokens = self.lm_gen.step(
                input_tokens=user_step,
                moshi_tokens=system_step,
                text_token=text_step,
            )
            _ = tokens

        # Reset codec streaming states before the real input wav.
        # Do not reset lm_gen, because that would erase the conditioning.
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        log("info", f"Audio conditioning finished: {steps} steps")

    def run_inference(
        self,
        input_user_audio: np.ndarray,
        conditioning_user_audio: np.ndarray | None = None,
        conditioning_audio: np.ndarray | None = None,
        conditioning_text_tokens: torch.Tensor | None = None,
    ):
        if len(input_user_audio) == 0:
            return np.zeros(0, dtype=np.float32)

        if conditioning_user_audio is not None:
            conditioning_user_audio = conditioning_user_audio.reshape(1, -1) # (1, T)
        if conditioning_audio is not None:
            conditioning_audio = conditioning_audio.reshape(1, -1) # (1, T)

        # 6) Prompt configuration (voice) * use context audio as prompt
        if self.use_teacher_forcing and not self.prioritize_voice_prompt:
            assert conditioning_audio is not None
            if len(conditioning_audio) > self.sample_rate and \
                np.max(np.abs(conditioning_audio)) > 0.0:
                log("info", "Setting conditioning audio as voice prompt.")
                self.lm_gen.load_voice_prompt_audio(
                    conditioning_audio,
                    self.voice_prompt_duration,
                )
            else:
                log("info", "Conditioning audio is silence. Loading voice prompt.")
                if self.voice_prompt_path.endswith('.pt'):
                    self.lm_gen.load_voice_prompt_embeddings(self.voice_prompt_path)
                else:
                    self.lm_gen.load_voice_prompt(self.voice_prompt_path)

        # 7) Reset streaming and run initial prompt phases
        #    - Voice prompt injection
        #    - Audio silence
        #    - Text prompt injection
        #    - Final audio silence

        #RM self.mimi.reset_streaming()
        #RM self.other_mimi.reset_streaming()
        #RM self.lm_gen.reset_streaming()
        # automatically reset the streaming state
        with self.mimi.streaming(1), self.other_mimi.streaming(1), self.lm_gen.streaming(1):
            self.lm_gen.step_system_prompts(self.mimi)
            # Reset mimi streaming after voice prompt encoding
            self.mimi.reset_streaming()

            # add conditioning here
            if conditioning_user_audio is not None or conditioning_audio is not None:
                self.apply_stereo_conditioning(
                    user_audio=conditioning_user_audio,
                    system_audio=conditioning_audio,
                    system_text_tokens=conditioning_text_tokens,
                )

            user_audio = input_user_audio.astype(np.float32)[None, :] # (C, T)

            generated_frames: List[np.ndarray] = []
            generated_text_tokens: List[str] = []
            total_target_samples = user_audio.shape[-1]

            for user_encoded in lm_encode_from_sphn(
                self.mimi,
                lm_iterate_audio(
                    user_audio, sample_interval_size=self.lm_gen._frame_size, pad=True
                ),
                max_batch=1,
            ):
                # user_encoded: [1, K, T]. Feed one step at a time (usually T==1)
                steps = user_encoded.shape[-1]
                for c in range(steps):
                    step_in = user_encoded[:, :, c : c + 1]
                    # Feed user-side input channels; text + agent audio are sampled
                    tokens = self.lm_gen.step(step_in)
                    if tokens is None:
                        continue
                    # Decode current sampled agent frame to PCM
                    pcm = decode_tokens_to_pcm(self.mimi, self.other_mimi, self.lm_gen, tokens)
                    generated_frames.append(pcm)
                    # Decode text token
                    if self.print_text_tokens:
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            log("info", f"text token '{_text}'")
                            generated_text_tokens.append(_text)
                        #else:
                        #    text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                        #    log("info", f"text token '{text_token_map[text_token]}'")
                        #    generated_text_tokens.append(text_token_map[text_token])

            if len(generated_frames) == 0:
                log("error", "No audio frames were generated. Check input file and configuration.")
                return np.zeros(total_target_samples, dtype=np.float32)

            # 10) Concatenate frames and trim/pad to match input duration
            output_pcm = np.concatenate(generated_frames, axis=-1)
            output_pcm = pad_or_trim(output_pcm, total_target_samples)

            return output_pcm

    def generate(
        self,
        context_audio: np.ndarray, # (La, 2)
        pre_event_audio: np.ndarray | None, # (Lb, 2) | None
        eval_window_audio: np.ndarray, # (Lc, 1)
        sample_rate: int,
        event: EventSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray|None]:

        model_sr = self.sample_rate

        if pre_event_audio is not None:
            pre_and_eval_user_audio = np.concatenate(
                [pre_event_audio[:, event.user_channel], eval_window_audio]
            )
            pre_event_system_audio = pre_event_audio[:, event.system_channel]
        else: # BC, BARGE_IN
            pre_and_eval_user_audio = eval_window_audio
            pre_event_system_audio = None

        if len(pre_and_eval_user_audio) == 0:
            return pre_and_eval_user_audio, pre_and_eval_user_audio, None

        # target_len = len(pre_and_eval_user_audio)
        target_len = int(self.sample_rate * len(pre_and_eval_user_audio) / sample_rate)

        # with teacher-forcing
        cond_user_audio = None
        cond_system_audio = None
        cond_text_token = None
        if self.use_teacher_forcing:
            context_len = 0
            assert self.use_context, "Set --use-context to use teacher-forcing."

            input_user_audio = eval_window_audio

            # Extend context
            if pre_event_audio is not None:
                context_audio = np.concatenate([context_audio, pre_event_audio])

            # Load conditioning audio
            context_user_audio = context_audio[:, event.user_channel]
            context_system_audio = context_audio[:, event.system_channel]

            # resampling
            input_user_audio = resample_mono_np(input_user_audio, sample_rate, model_sr)
            cond_user_audio = resample_mono_np(context_user_audio, sample_rate, model_sr)
            cond_system_audio = resample_mono_np(context_system_audio, sample_rate, model_sr)

            # Load conditioning tokens
            cond_text_token = None
            if self.use_text_conditioning and event.alignment_path is not None:
                # get aligns
                aligns, chs = load_alignments(event.alignment_path)
                assert len(chs) == 1 and chs[0] == '0', '--conditioning_txt must include only channel "0"'

                # tokenize
                cond_audio_dur = cond_system_audio.shape[0] / self.mimi.sample_rate
                cond_text_token = self.interleaver.prepare_item(
                    aligns,
                    cond_audio_dur,
                    main_speaker="0"
                )[0].squeeze()     # [L, ]-shaped torch.tensor

        # without teacher-forcing
        else:
            input_user_audio = pre_and_eval_user_audio
            context_len = 0

            # with context
            if self.use_context:
                context_user_audio = context_audio[:, event.user_channel]
                # context_len = len(context_user_audio)
                context_len = int(self.sample_rate * len(context_user_audio) / sample_rate)
                input_user_audio = np.concatenate([context_user_audio, input_user_audio])

            # resampling
            input_user_audio = resample_mono_np(input_user_audio, sample_rate, model_sr)

        system_out_audio = self.run_inference(
            input_user_audio = input_user_audio,
            conditioning_user_audio = cond_user_audio,
            conditioning_audio = cond_system_audio,
            conditioning_text_tokens = cond_text_token,
        )

        context_system_out_audio = None
        if self.use_teacher_forcing:
            eval_system_out_audio = system_out_audio[:target_len]
            if pre_event_system_audio is not None:
                if sample_rate != self.sample_rate:
                    pre_event_system_audio = resample_mono_np(
                        pre_event_system_audio, sample_rate, self.sample_rate
                    )
                pre_and_eval_system_out_audio = np.concatenate(
                    [pre_event_system_audio, eval_system_out_audio]
                )
            else:
                pre_and_eval_system_out_audio = eval_system_out_audio
        else:
            pre_and_eval_system_out_audio = system_out_audio[context_len : context_len + target_len]
            if context_len > 0:
                context_system_out_audio = system_out_audio[:context_len]

        pre_and_eval_system_out_audio = pad_or_trim(pre_and_eval_system_out_audio, target_len)

        if sample_rate != self.sample_rate:
            pre_and_eval_user_audio = resample_mono_np(
                pre_and_eval_user_audio, sample_rate, self.sample_rate
            )

        return pre_and_eval_user_audio, pre_and_eval_system_out_audio, context_system_out_audio
