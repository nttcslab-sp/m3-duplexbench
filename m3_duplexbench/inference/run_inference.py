#!/usr/bin/env python3

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from m3_duplexbench.inference.events import (
    DataStats,
    build_event_samples,
    slice_event_audio,
)
from m3_duplexbench.utils import (
    load_audio,
    pad_or_trim,
    read_jsonl,
    write_jsonl,
    resample_mono_np,
)

@dataclass
class InferenceOutput:
    sample_id: str
    event_id: str
    model: str
    task: str
    event_type: str
    user_channel: int
    system_channel: int
    event_start: float
    event_end: float
    trigger_time: float
    input_start: float
    input_end: float
    context_start: float
    context_end: float
    input_audio_path: str
    context_audio_path: str
    alignment_path: str | None
    output_audio_path: str
    use_context: bool
    context_mode: str


def save_stereo_output(
    output_audio_path: Path,
    user_audio: np.ndarray,
    model_audio: np.ndarray,
    sample_rate: int,
) -> None:
    if user_audio.ndim != 1:
        raise ValueError(f"user_audio must be mono, got shape {user_audio.shape}")
    if model_audio.ndim != 1:
        raise ValueError(f"model_audio must be mono, got shape {model_audio.shape}")
    if len(user_audio) != len(model_audio):
        raise ValueError(
            f"Length mismatch: user_audio={len(user_audio)}, "
            f"model_audio={len(model_audio)}"
        )

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    output_stereo = np.stack([user_audio, model_audio], axis=1).astype(np.float32)
    sf.write(str(output_audio_path), output_stereo, sample_rate)


def create_model(args: argparse.Namespace):
    if args.model == "dummy":
        from m3_duplexbench.inference.models.dummy import DummyModel
        return DummyModel(output_duration=args.eval_window_len)

    if args.model == "moshi":
        from m3_duplexbench.inference.models.moshi import MoshiModel
        dtype = torch.float16 if args.half else torch.bfloat16
        return MoshiModel(
            hf_repo=args.hf_repo,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            tokenizer=args.tokenizer,
            config=args.config,
            device=args.device,
            dtype=dtype,
            cfg_coef=args.cfg_coef,
            use_context=args.use_context,
            use_teacher_forcing=args.teacher_forcing,
            use_text_conditioning=args.conditioning_text,
            seed=args.seed,
            use_sampling_text=args.use_sampling_text,
            temp_text=args.temp_text,
            top_k_text=args.top_k_text,
            use_sampling=args.use_sampling_audio,
            temp=args.temp_audio,
            top_k=args.top_k_audio,
            prepend_silence_sec=args.prepend_silence_sec,
        )

    if args.model == "personaplex":
        from m3_duplexbench.inference.models.personaplex import PersonaPlexModel
        assert args.voice_prompt is not None, "--voice-prompt is required"
        return PersonaPlexModel(
            hf_repo=args.hf_repo,
            voice_prompt=args.voice_prompt,
            voice_prompt_dir=args.voice_prompt_dir,
            prioritize_voice_prompt=args.prioritize_voice_prompt,
            voice_prompt_duration=args.voice_prompt_duration,
            text_prompt=args.text_prompt,
            domain=args.domain,
            tokenizer=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            device=args.device,
            seed=args.seed,
            use_context=args.use_context,
            use_teacher_forcing=args.teacher_forcing,
            use_text_conditioning=args.conditioning_text,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.top_k_audio,
            topk_text=args.top_k_text,
            greedy=args.greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
        )

    if args.model == "freeze_omni":
        from m3_duplexbench.inference.models.freeze_omni import FreezeOmniModel
        return FreezeOmniModel(
            server_url=args.server_url,
            ip=args.ip,
            port=args.port,
            use_context=args.use_context,
        )

    if args.model == "duplexcascade":
        from m3_duplexbench.inference.models.duplexcascade import DuplexCascadeModel

        return DuplexCascadeModel(
            server_url=args.server_url,
            use_context=args.use_context,
            use_teacher_forcing=args.teacher_forcing,
        )

    if args.model == "freeze_omni_offline":
        from m3_duplexbench.inference.models.freeze_omni_offline import FreezeOmniOfflineModel
        return FreezeOmniOfflineModel(
            model_path=args.model_path,
            llm_path=args.llm_path,
            use_context=args.use_context,
            use_teacher_forcing=args.teacher_forcing,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            device=args.device,
        )

    if args.model == "bayling_duplex":
        from m3_duplexbench.inference.models.bayling_duplex_model import BayLingDuplexModel
        return BayLingDuplexModel(
            model_path=args.model_path,
            speech_tokenizer_path=args.speech_tokenizer_path,
            decoder_path=args.decoder_path,
            device=args.device,
            use_context=args.use_context,
            use_teacher_forcing=args.teacher_forcing,
            interleave_ratio=args.interleave_ratio,
            max_duration=args.max_duration,
            temperature=args.temperature,
            top_p=args.top_p,
            max_epad_count=args.max_epad_count,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def run_inference(
    audio_dir: Path,
    data_path: Path,
    output_dir: Path,
    ignore_short_pause: bool,
    ignored_events: None | list[str],
    include_channels: list[int],
    model,
    alignment_dir: Path | None = None,
    eval_window_len: float = 10.0,
    use_context: bool = True,
    use_teacher_forcing: bool = False,
    context_max_len: float | None = None,
    overwrite: bool = False,
) -> None:
    output_audio_dir = output_dir / "audio"
    output_alignment_root = output_dir / "alignments"

    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_alignment_root.mkdir(parents=True, exist_ok=True)

    all_outputs: list[InferenceOutput] = []
    stats = DataStats()
    stats.channels_to_evaluate = include_channels

    records = list(read_jsonl(data_path))

    records_event_samples = []
    for i, record in enumerate(records):
        event_samples = build_event_samples(
            record=record,
            audio_dir=audio_dir,
            ignore_short_pause=ignore_short_pause,
            ignored_events=ignored_events,
            stats=stats,
            alignment_dir=alignment_dir,
            output_alignment_root=output_alignment_root,
            eval_window_len=eval_window_len,
            use_context=use_context,
            use_teacher_forcing=use_teacher_forcing,
            context_max_len=context_max_len,
        )
        records_event_samples.append(event_samples)

    stats_path = output_dir / "data_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)
    print(f"Data stats: {stats_path}")
    print("Data summary:")
    for k, v in asdict(stats).items():
        print(f"  {k}: {v}")

    for i, record in enumerate(records):
        print(f"[run_inference] processing dialogue {i+1}/{len(records)}")

        event_samples = records_event_samples[i]

        if len(event_samples) == 0:
            continue

        audio_path = Path(event_samples[0].audio_path)
        audio, sample_rate = load_audio(audio_path)

        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(
                f"Expected 2-channel audio for {audio_path}, got shape {audio.shape}"
            )

        for j, event in enumerate(event_samples):
            print(f"[run_inference] processing event {j+1}/{len(event_samples)} ({event.event_type})")
            if sample_rate != event.sample_rate:
                raise RuntimeError(
                    f"Sample rate mismatch for {event.sample_id}: "
                    f"{sample_rate} != {event.sample_rate}"
                )


            event_output_dir = output_audio_dir / event.sample_id
            output_audio_path = event_output_dir / f"{event.event_id}.wav"
            context_audio_path = ""
            if use_context:
                context_audio_path = event_output_dir / f"{event.event_id}.context.wav"

            all_outputs.append(
                InferenceOutput(
                    sample_id=event.sample_id,
                    event_id=event.event_id,
                    model=model.name,
                    task=event.task,
                    event_type=event.event_type,
                    user_channel=event.user_channel,
                    system_channel=event.system_channel,
                    event_start=event.event_start,
                    event_end=event.event_end,
                    trigger_time=event.trigger_time,
                    input_start=event.input_start,
                    input_end=event.input_end,
                    context_start=event.context_start,
                    context_end=event.context_end,
                    input_audio_path=event.audio_path,
                    context_audio_path=str(context_audio_path),
                    alignment_path=event.alignment_path,
                    output_audio_path=str(output_audio_path),
                    use_context=use_context,
                    context_mode=model.context_mode,
                )
            )

            if output_audio_path.exists() and not overwrite:
                print(f"Skipping existing file: {output_audio_path}")
                continue

            context_audio, pre_event_audio, eval_window_audio = slice_event_audio(
                audio=audio,
                sample_rate=sample_rate,
                event=event,
            )

            # Run model inference
            user_audio_out, model_audio_out, model_context_out = model.generate(
                context_audio=context_audio,
                pre_event_audio=pre_event_audio,
                eval_window_audio=eval_window_audio,
                sample_rate=sample_rate,
                event=event,
            )

            output_sample_rate = model.sample_rate or sample_rate

            # save output
            print(f"Saving audio to: {output_audio_path}")
            save_stereo_output(
                output_audio_path=output_audio_path,
                user_audio=user_audio_out,
                model_audio=model_audio_out,
                sample_rate=output_sample_rate,
            )
            
            # save context
            if use_context:
                context_user_audio = context_audio[:, event.user_channel]
                context_system_audio = context_audio[:, event.system_channel]
                if sample_rate != output_sample_rate:
                    context_user_audio = resample_mono_np(context_user_audio, sample_rate, output_sample_rate)
                    context_system_audio = resample_mono_np(context_system_audio, sample_rate, output_sample_rate)
                # use model's output
                if model_context_out is not None:
                    context_system_audio = model_context_out
                    context_system_audio = pad_or_trim(
                        context_system_audio,
                        len(context_user_audio)
                    )

                assert len(context_user_audio) == len(context_system_audio), \
                    f"Different context lengths ({len(context_user_audio)}, {len(context_system_audio)})"
                context_audio = np.stack([context_user_audio, context_system_audio], axis=1).astype(np.float32)
                sf.write(str(context_audio_path), context_audio, output_sample_rate)

    metadata_path = output_dir / "metadata.jsonl"
    write_jsonl(metadata_path, (asdict(o) for o in all_outputs))

    print("Done.")
    print(f"# output samples: {len(all_outputs)}")
    print(f"Output audio dir: {output_audio_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Data stats: {stats_path}")
    print()
    print("Data summary:")
    for k, v in asdict(stats).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "dummy",
            "moshi",
            "freeze_omni",
            "personaplex",
            "freeze_omni_offline",
            "bayling_duplex",
            "duplexcascade",
        ],
        required=True,
        help="Inference model.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/en/chat/toy/audio"),
        help="Directory containing input 2-channel wav files.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/en/chat/toy/test.jsonl"),
        help="Path to jsonl data file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory to save inference outputs.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="0,1",
        help="Comma-separated channel ids to evaluate, e.g., '0' or '0,1'.",
    )
    parser.add_argument(
        "--eval-window-len",
        type=float,
        default=10.0,
        help="Evaluation window length in seconds.",
    )
    parser.add_argument("--use-context", action='store_true')
    #RM parser.add_argument("--save-context", action='store_true')
    parser.add_argument(
        "--context-max-len",
        type=float,
        default=None,
        help=(
            "Maximum context length in seconds. "
            "If specified, context starts from the earliest TURN start "
            "within [context_end - context_max_len, context_end]. "
            "If omitted, context starts from 0.0."
        ),
    )
    parser.add_argument(
        "--ignore-short-pause", action="store_true",
        help="Ignore short pauses for pause handling task.",
    )
    parser.add_argument(
        "--ignored-events", type=str, default=None,
        help="Comma-separated events to be ignored, e.g., 'TURN_SHIFT,TURN_HOLD,BC'"
    )

    # Moshi-specific args.
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--moshi-weight", type=str, default=None)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument(
        "--hf-repo", type=str, default="kyutai/moshiko-pytorch-bf16",
        help="HF repo for Moshi.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for Moshi inference.",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Run Moshi inference with float16 instead of bfloat16.",
    )
    parser.add_argument(
        "--config", "--lm-config", dest="config", type=str, default=None,
        help="Moshi LM config json file.",
    )
    parser.add_argument(
        "--cfg-coef", type=float, default=1.0,
        help="Moshi CFG coefficient.",
    )
    parser.add_argument(
        "--prepend-silence-sec", type=float, default=0.0,
        help="Seconds of silence to prepend to input audio before inference.",
    )
    parser.add_argument(
        "--teacher-forcing", action="store_true",
        help="Enable Moshi teacher-forced context conditioning.",
    )
    parser.add_argument(
        "--conditioning-text", action="store_true",
        help="Use system-side word alignments for Moshi text conditioning.",
    )
    parser.add_argument(
        "--alignment-dir", type=Path, default=None,
        help="Directory containing word alignment files for conditioning text.",
    )
    parser.add_argument(
        "--seed", type=int, default=4242,
        help="Random seed.",
    )
    parser.add_argument("--use-sampling-text", action='store_true')
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--top-k-text", type=int, default=25)
    parser.add_argument("--use-sampling-audio", action='store_true')
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--top-k-audio", type=int, default=250)

    # PersonaPlex specific args.
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling (greedy decoding)"
    )
    parser.add_argument("--text-prompt", default=None, type=str, help="Text prompt")
    parser.add_argument(
        "--domain", choices=["assistant", "chat"], default="assistant",
        help="Selecting default text prompt by domain.")

    parser.add_argument(
        "--voice-prompt", default=None, type=str,
        help=(
            "Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt'). "
            "If omitted, --conditioning-system-wav can be used."
        )
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted. "
            "Voice prompt filenames from -voice-prompt arg will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--prioritize-voice-prompt",
        action="store_true", help="Do not use conditioning wav as voice prompt."
    )
    parser.add_argument(
        "--voice-prompt-duration", type=float, default=None,
        help="Optional maximum duration in seconds for the voice prompt audio.",
    )
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    #RM    # https://github.com/DanielTomaro13/personaplex/tree/pr-creator-1775629538666
    #RM    parser.add_argument("--multi-gpu", action="store_true",
    #RM                        help="Distribute the LM model across all available CUDA GPUs. "
    #RM                             "Requires 'accelerate' package and multiple CUDA devices. "
    #RM                             "CUDA graphs are automatically disabled in this mode.")

    # Server-client model args (Freeze-Omni and DuplexCascade).
    # parser.add_argument('--model_path', default=None, help='model_path to load')
    # parser.add_argument('--llm_path', default=None, help='llm_path to load')
    # parser.add_argument('--top_k', type=int, default=5)
    # parser.add_argument('--top_p', type=float, default=0.8)
    # parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument("--server-url", default=None,
        help="Model server URL, for example ws://127.0.0.1:31606.")
    parser.add_argument("--ip", default="127.0.0.1", help="server ip, used when --server-url is not set")
    parser.add_argument("--port", default="8000", help="server port, used when --server-url is not set")

    # Bayling-specific args.
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--speech_tokenizer_path", type=str, default=None)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--interleave_ratio", type=str, default="10:5:10")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--max_epad_count", type=int, default=50)


    args = parser.parse_args()

    if not args.audio_dir.exists():
        raise FileNotFoundError(f"audio_dir does not exist: {args.audio_dir}")
    if not args.data.exists():
        raise FileNotFoundError(f"data file does not exist: {args.data}")

    include_channels = [int(v.strip()) for v in args.channels.split(",")]

    model = create_model(args)

    ignored_events = None
    if args.ignored_events:
        ignored_events = args.ignored_events.split(",")
    print(f"Events to be ignored: {ignored_events}")

    run_inference(
        audio_dir=args.audio_dir,
        data_path=args.data,
        output_dir=args.output_dir,
        ignore_short_pause=args.ignore_short_pause,
        ignored_events=ignored_events,
        include_channels=include_channels,
        model=model,
        alignment_dir=args.alignment_dir,
        eval_window_len=args.eval_window_len,
        use_context=args.use_context,
        use_teacher_forcing=args.teacher_forcing,
        context_max_len=args.context_max_len,
    )
