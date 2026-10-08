# Original: moshi/moshi/run_inference.py

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sentencepiece
import torch
import torch.nn.functional as F

from transformers import T5TokenizerFast, AutoTokenizer

from moshi.client_utils import AnyPrinter, Printer, RawPrinter
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import get_condition_tensors, seed_all
from moshi.utils.audioproc import load_audio
from moshi.utils.textproc import load_alignments
from moshi.utils.interleaver import CustomInterleaver

from m3_duplexbench.inference.events import EventSample
from m3_duplexbench.inference.models.base import BaseInferenceModel
from m3_duplexbench.utils import resample_mono_np, pad_or_trim


class CustomMoshiInferenceState:
    """Moshi inference state that resets streaming state for every run."""

    def __init__(
        self,
        checkpoint_info: loaders.CheckpointInfo,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        cfg_coef: float,
        device: str | torch.device,
        cond_audio: torch.Tensor | None = None,
        cond_user_audio: torch.Tensor | None = None,
        cond_text_token: torch.Tensor | None = None,
        **kwargs,
    ):
        self.checkpoint_info = checkpoint_info
        self.model_type = checkpoint_info.model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        if isinstance(self.text_tokenizer, T5TokenizerFast):
            self.text_eos_id = self.text_tokenizer.convert_tokens_to_ids(['</s>'])[0]
            self.get_text = lambda text_token: self.text_tokenizer.convert_ids_to_tokens([text_token])[0]
        else:
            self.text_eos_id = self.text_tokenizer.eos_id()
            self.get_text = lambda text_token: self.text_tokenizer.id_to_piece(text_token)
        condition_tensors = get_condition_tensors(self.model_type, lm, batch_size, cfg_coef)
        self.lm_gen = LMGen(
            lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs
        )

        self.cond_audio = cond_audio
        self.cond_text_token = cond_text_token
        self.cond_user_audio = cond_user_audio

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        #RM: self.mimi.streaming_forever(batch_size)
        #RM: self.lm_gen.streaming_forever(batch_size)
        self.printer: AnyPrinter
        if sys.stdout.isatty():
            self.printer = Printer()
        else:
            self.printer = RawPrinter()

        self.printer.log("info", f"LMGen config: {str(self.lm_gen.get_generate_config())}")

    def run(self, in_pcms: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of tupel `(text_tokens, audio_tokens)`"""

        # moshi/moshi/modules/streaming.py#L59
        with self.mimi.streaming(self.batch_size), self.lm_gen.streaming(
            self.batch_size
        ):
            out_pcms_per_item: list[list[torch.Tensor]] = [
                [] for _ in range(self.batch_size)
            ]
            out_text_tokens_per_item: list[list[torch.Tensor]] = [
                [] for _ in range(self.batch_size)
            ]
            # For the Hibiki translation model, we feed a special token for the end of the input stream,
            # which corresponds to `2048` on all the codebooks of the audio stream, and wait
            # for the EOS on the output text stream to be emitted, as indication that the model is done.
            eos_reached: list[bool] = [False] * self.batch_size
            need_eos_input: bool = True
            device = self.lm_gen.lm_model.device

            if self.cond_audio is not None or self.cond_user_audio is not None:
                self.apply_stereo_conditioning()

            start_time = time.time()
            ntokens = 0
            first_frame = self.cond_audio is None and self.cond_user_audio is None
            # first_frame = True
            if self.model_type == "stt":
                stt_config = self.checkpoint_info.stt_config
                pad_right = stt_config.get("audio_delay_seconds", 0.0)
                pad_left = stt_config.get("audio_silence_prefix_seconds", 0.0)
                pad_left = int(pad_left * 24000)
                pad_right = int((pad_right + 1.0) * 24000)
                in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")
            # We keep only fully frames.
            chunks = deque(
                [
                    chunk
                    for chunk in in_pcms.split(self.frame_size, dim=2)
                    if chunk.shape[-1] == self.frame_size
                ]
            )

            self.printer.print_header()
            while not all(eos_reached):
                if chunks:
                    chunk = chunks.popleft()
                    codes = self.mimi.encode(chunk)
                else:
                    if self.model_type == "hibiki":
                        if need_eos_input:
                            # First frame after the end of the file, we feed a code full of 2048
                            # to indicate the end of stream.
                            need_eos_input = False
                            eos_value = self.mimi.cardinality
                            codes = torch.full(
                                (self.batch_size, self.mimi.num_codebooks, 1),
                                eos_value,
                                device=device,
                                dtype=torch.long,
                            )
                        else:
                            silence = torch.zeros(
                                (self.batch_size, self.mimi.channels, self.frame_size),
                                device=device,
                            )
                            codes = self.mimi.encode(silence)
                    else:
                        # For other models, we stop as soon as we are reaching the end of the audio.
                        break
                if first_frame:
                    # Ensure that the first slice of codes is properly seen by the transformer
                    # as otherwise the first slice is replaced by the initial tokens.
                    tokens = self.lm_gen.step(codes)
                    if max(self.lm_gen.lm_model.delays) > 0:
                        assert tokens is None
                    first_frame = False
                tokens = self.lm_gen.step(codes)
                if tokens is None:
                    continue
                assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                if self.lm_gen.lm_model.dep_q > 0:
                    # ad-hoc
                    audio_tokens = tokens[:, 1:]
                    bad = (audio_tokens < 0) | (audio_tokens >= self.mimi.cardinality)
                    if bad.any():
                        print(f"[WARNING] invalid Mimi token ids: {audio_tokens[bad]}")
                        continue

                    out_pcm = self.mimi.decode(tokens[:, 1:]).cpu()
                    for b, (one_text, one_pcm) in enumerate(
                        zip(tokens[:, 0].cpu(), out_pcm)
                    ):
                        if eos_reached[b]:
                            continue
                        elif one_text.item() == self.text_eos_id:
                            if need_eos_input:
                                # We sampled the EOS before the end of the file! Not possible.
                                self.printer.log("warning", "EOS sampled too early.")
                            else:
                                eos_reached[b] = True

                        out_text_tokens_per_item[b].append(one_text)
                        out_pcms_per_item[b].append(one_pcm)
#                        if b == 0:
#                            if one_text.item() not in [0, 3]:
#                                text = self.text_tokenizer.id_to_piece(one_text.item())  # pyright: ignore
#                                text = text.replace("▁", " ")
#                                self.printer.print_token(text)
#                else:
#                    one_text = tokens[0, 0].cpu()
#                    if one_text.item() not in [0, 3]:
#                        text = self.text_tokenizer.id_to_piece(one_text.item())  # pyright: ignore
#                        text = text.replace("▁", " ")
#                        self.printer.print_token(text)
                ntokens += 1
            dt = time.time() - start_time
            self.printer.log(
                "info",
                f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step",
            )
#RM            if ntokens > 0:
#RM                print(
#RM                    f"[Moshi] processed {ntokens} steps in {dt:.1f}s "
#RM                    f"({1000 * dt / ntokens:.2f} ms/step)"
#RM                )
            if self.lm_gen.lm_model.dep_q > 0:
                out = [
                    (torch.cat(one_texts, dim=0), torch.cat(one_pcms, dim=1))
                    for one_texts, one_pcms in zip(
                        out_text_tokens_per_item, out_pcms_per_item
                    )
                ]
                return out
            else:
                return []

    def apply_stereo_conditioning(self):
        assert self.cond_audio is not None or self.cond_user_audio is not None
        self.printer.log("info", "start stereo conditioning")

        if self.cond_text_token is not None:
            self.printer.log("info", "text conditioning: true")
        else:
            self.printer.log("info", "text conditioning: false")

        self.lm_gen.use_stereo_input = True
        self.lm_gen.dialog_prompt_size = 100000

        tokens = None
        zero_chunk = torch.zeros(self.batch_size, 1, self.frame_size, dtype=torch.float32, device=self.device)

        system_len = 0 if self.cond_audio is None else len(self.cond_audio)
        user_len = 0 if self.cond_user_audio is None else len(self.cond_user_audio)
        total_len = max(system_len, user_len)

        for n, i in enumerate(range(0, total_len, self.frame_size)):
            # system chunk
            if self.cond_audio is None:
                chunk_sys = zero_chunk
            else:
                sys_part = self.cond_audio[i: i + self.frame_size]
                if sys_part.size(0) < self.frame_size:
                    sys_part = F.pad(sys_part, (0, self.frame_size - sys_part.size(0)))
                chunk_sys = sys_part.view(1, 1, self.frame_size).to(self.device)
                chunk_sys = chunk_sys.expand(self.batch_size, -1, -1).contiguous()
            codes_sys = self.mimi.encode(chunk_sys)

            # user chunk
            if self.cond_user_audio is None:
                chunk_usr = zero_chunk
            else:
                usr_part = self.cond_user_audio[i: i + self.frame_size]
                if usr_part.size(0) < self.frame_size:
                    usr_part = F.pad(usr_part, (0, self.frame_size - usr_part.size(0)))
                chunk_usr = usr_part.view(1, 1, self.frame_size).to(self.device)
                chunk_usr = chunk_usr.expand(self.batch_size, -1, -1).contiguous()
            codes_usr = self.mimi.encode(chunk_usr)

            # create concatenated input codes
            if self.cond_text_token is not None and n < len(self.cond_text_token):
                # w/ text tokens
                text_tok = self.cond_text_token[n:n+1].reshape(1, -1, 1).to(self.device)
                text_tok = text_tok.expand(self.batch_size, -1, -1).contiguous()
                codes = torch.cat((text_tok, codes_sys, codes_usr), dim=1)
            else:
                # w/o text tokens
                if tokens is None:
                    assert self.lm_gen._streaming_state is not None
                    text_in = self.lm_gen._streaming_state.initial[:, :1, :]
                else:
                    # use text token generated in the previous frame
                    text_in = tokens[:, :1, :]
                if text_in.shape[0] != self.batch_size:
                    text_in = text_in.expand(self.batch_size, -1, -1).contiguous()
                codes = torch.cat((text_in, codes_sys, codes_usr), dim=1)

            # decode step-by-step
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue

        self.lm_gen.use_stereo_input = False
        self.lm_gen.dialog_prompt_size = 0
        torch.cuda.synchronize()
        self.printer.log("info", "stereo conditioning finished")


class MoshiModel(BaseInferenceModel):
    """Moshi wrapper."""

    def __init__(
        self,
        hf_repo: str,
        moshi_weight: str | None = None,
        mimi_weight: str | None = None,
        tokenizer: str | None = None,
        config: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cfg_coef: float = 1.0,
        seed: int = 4242,
        use_context: bool = True,
        use_teacher_forcing: bool = True,
        use_text_conditioning: bool = True,
        use_sampling_text: bool = True,
        temp_text: float = 0.7,
        top_k_text: int = 25,
        use_sampling: bool = True,
        temp: float = 0.8,
        top_k: int = 250,
        prepend_silence_sec: float = 0.0,
    ) -> None:
        seed_all(seed)

        self.device = device
        self.dtype = dtype
        self.cfg_coef = cfg_coef
        self.batch_size = 1
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

        self.prepend_silence_sec = prepend_silence_sec
        if self.prepend_silence_sec > 0:
            assert not self.use_teacher_forcing

        print("[MoshiModel] Retrieving checkpoint...")
        self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo,
            moshi_weight,
            mimi_weight,
            tokenizer,
            config,
        )

        print("[MoshiModel] Loading Mimi...")
        self.mimi = self.checkpoint_info.get_mimi(device=device)
        print("[MoshiModel] Mimi loaded.")

        print("[MoshiModel] Loading text tokenizer...")
        # If tokenizer is explicitly provided, use AutoTokenizer so that
        # resegment_by_tokenizer=True is available for non-English models.
        if tokenizer:
            repo = Path(tokenizer.replace("hf://", "")).parent
            self.text_tokenizer = AutoTokenizer.from_pretrained(str(repo))
            self.resegment_text_by_tokenizer = True
        else:
            self.text_tokenizer = self.checkpoint_info.get_text_tokenizer()
            self.resegment_text_by_tokenizer = False

        print("[MoshiModel] Loading Moshi LM...")
        self.lm = self.checkpoint_info.get_moshi(device=device, dtype=dtype)
        print("[MoshiModel] Moshi LM loaded.")

        if self.lm.dep_q == 0:
            self.batch_size = 1

        # prepare interleaver
        if use_text_conditioning:
            self.interleaver = CustomInterleaver(
                self.text_tokenizer,
                self.mimi.frame_rate,
                self.lm.text_padding_token_id,
                self.lm.end_of_text_padding_id,
                self.lm.zero_token_id,
                keep_main_only=True,
                resegment_by_tokenizer=self.resegment_text_by_tokenizer,
                device='cpu',
            )

        # Update config
        self.checkpoint_info.lm_gen_config.update(
            {
                "use_sampling_text": use_sampling_text,
                "temp_text": temp_text,
                "top_k_text": top_k_text,
                "use_sampling": use_sampling,
                "temp": temp,
                "top_k": top_k,
            }
        )

    @property
    def name(self) -> str:
        return "moshi"

    @property
    def sample_rate(self) -> int:
        return int(self.mimi.sample_rate)

    @property
    def context_mode(self) -> str:
        return self._context_mode

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
            context_user_audio = resample_mono_np(context_user_audio, sample_rate, model_sr)
            context_system_audio = resample_mono_np(context_system_audio, sample_rate, model_sr)

            # ndarray -> tensor
            cond_user_audio = torch.from_numpy(context_user_audio.astype(np.float32)).to(self.device)
            cond_system_audio = torch.from_numpy(context_system_audio.astype(np.float32)).to(self.device)

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

        # Prepending silence
        if self.use_teacher_forcing:
            prepend_len = 0
            prepend_len_resamp = 0
        else:
            prepend_len = int(round(self.prepend_silence_sec * sample_rate))
            prepend_len_resamp = int(round(self.prepend_silence_sec * model_sr))
            if prepend_len_resamp > 0:
                silence = np.zeros(prepend_len_resamp, dtype=np.float32)
                input_user_audio = np.concatenate([silence, input_user_audio], axis=0)
                print(f"{self.prepend_silence_sec} sec silence prepended.")

        in_pcms = torch.from_numpy(input_user_audio)[None, None, :].to(self.device)

        if self.batch_size > 1:
            in_pcms = in_pcms.expand(self.batch_size, -1, -1)

        with torch.no_grad():
            state = CustomMoshiInferenceState(
                self.checkpoint_info,
                self.mimi,
                self.text_tokenizer,
                self.lm,
                self.batch_size,
                self.cfg_coef,
                self.device,
                cond_audio=cond_system_audio,
                cond_user_audio=cond_user_audio,
                cond_text_token=cond_text_token,
                **self.checkpoint_info.lm_gen_config,
            )
            out_items = state.run(in_pcms)

        if len(out_items) == 0:
            model_audio = np.zeros(target_len, dtype=np.float32)
            return pre_and_eval_user_audio, model_audio, None

        _, out_pcm = out_items[0]
        system_out_audio = out_pcm[0].detach().cpu().numpy().astype(np.float32)

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
            # system_out_audio = system_out_audio[prepend_len:]
            system_out_audio = system_out_audio[prepend_len_resamp:]
            pre_and_eval_system_out_audio = system_out_audio[context_len : context_len + target_len]
            if context_len > 0:
                context_system_out_audio = system_out_audio[:context_len]

        pre_and_eval_system_out_audio = pad_or_trim(pre_and_eval_system_out_audio, target_len)

        if sample_rate != self.sample_rate:
            pre_and_eval_user_audio = resample_mono_np(
                pre_and_eval_user_audio, sample_rate, self.sample_rate
            )

        return pre_and_eval_user_audio, pre_and_eval_system_out_audio, context_system_out_audio
