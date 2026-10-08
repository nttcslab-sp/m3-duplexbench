import os
import argparse
import json
import time
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import BatchedInferencePipeline, WhisperModel
from tqdm import tqdm
from tqdm.contrib import tenumerate

from m3_duplexbench.utils import (
    read_jsonl,
    write_json,
    write_jsonl,
)

def load_audio(audio_path: str | Path):
    data, fs = sf.read(str(audio_path))
    data = data.astype(np.float32)

    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    elif len(data.shape) == 2:
        data = data.T
    else:
        raise ValueError(f"Unsupported audio shape: {data.shape}")

    return data, fs


class WhisperASR:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        word_timestamps: bool = True,
        vad_filter: bool = True,
        vad_min_speech_duration_ms: int = 150,
        vad_min_silence_duration_ms: int = 100,
        vad_speech_pad_ms: int = 300,
        batch_size: int = 1,
    ) -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.batch_size = batch_size
        self.batched_model = None
        if batch_size > 1:
            self.batched_model = BatchedInferencePipeline(model=self.model)
        self.sample_rate = self.model.feature_extractor.sampling_rate
        self.language = language
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.vad_parameters = {
            "min_speech_duration_ms": vad_min_speech_duration_ms,
            "min_silence_duration_ms": vad_min_silence_duration_ms,
            "speech_pad_ms": vad_speech_pad_ms,
        }

    def _decode_mono(
        self,
        audio: np.ndarray,
        beam_size: int = 5,
    ):
        if self.batch_size > 1:
            assert self.batched_model is not None
            segments, info = self.batched_model.transcribe(
                audio,
                language=self.language,
                beam_size=beam_size,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters,
                batch_size=self.batch_size,
            )
        else:
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=beam_size,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters,
            )
        return segments, info

    def transcribe_file(
        self,
        audio_path: str | Path,
        channels: list[int],
        verbose: bool = True,
    ) -> dict:
        """Transcribe audio file and return JSON."""
        data, fs = load_audio(audio_path)

        if fs != self.sample_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=self.sample_rate, axis=1)

        t_start = time.time()
        output_segments = []

        for ch in channels:
            segments, _ = self._decode_mono(data[ch])

            for segment in segments:
                segment_text = str(segment.text).strip()
                segment_start = float(segment.start)
                segment_end = float(segment.end)

                chunks = []

                if self.word_timestamps:
                    assert segment.words is not None
                    for word in segment.words:
                        word_text = str(word.word).strip()
                        if not word_text:
                            continue

                        word_start = float(word.start)
                        word_end = float(word.end)

                        # text_words.append(word_text)
                        chunks.append({
                            "text": word_text,
                            "timestamp": [word_start, word_end],
                        })

                if not segment_text:
                    continue

                output_segments.append({
                    "channel": ch,
                    "text": segment_text,
                    "timestamp": [segment_start, segment_end],
                    "chunks": chunks,
                })

        output_segments = sorted(
            output_segments,
            key=lambda x: (
                x["timestamp"][0],
                x["channel"],
                x["timestamp"][1],
            ),
        )

        if verbose:
            t_proc = time.time() - t_start
            n_ch = len(channels)
            dur = data.shape[1] / self.sample_rate
            rtf = t_proc / max(dur * n_ch, 1e-6)
            print(f"{audio_path}: ch={n_ch}, dur={dur:.1f}s, ASR-RTF={rtf:.2f}")

        return {"segments": output_segments}


def json_to_list(
    segdata: dict,
    num_channels: int = 2,
):
    results = [[] for _ in range(num_channels)]
    for seg in segdata["segments"]:
        ch = seg["channel"]
        for word in seg["chunks"]:
            results[ch].append("\t".join([
                str(word["timestamp"][0]),
                str(word["timestamp"][1]),
                word["text"],
            ]))

    return results

def save_audio_with_context(
    target_audio_path: Path,
    context_audio_path: Path | None,
    out_audio_path: Path,
    context_length: float,
):
    """Write [short context + target] wav."""
    target, target_sr = load_audio(target_audio_path)
    if context_audio_path is not None:
        context, context_sr = load_audio(context_audio_path)
        if target_sr != context_sr:
            context = librosa.resample(context, orig_sr=context_sr, target_sr=target_sr, axis=1)

        context_nsamples = int(round(context_length) * target_sr)
        context_nsamples = min(context_nsamples, context.shape[1])
    else:
        context = None
        context_nsamples = 0

    if context_nsamples == 0:
        # without context
        # out_audio_path.symlink_to(target_audio_path.absolute())
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)
        target_audio_path = target_audio_path.resolve(strict=True)
        if out_audio_path.exists():
            out_audio_path.unlink()
        os.symlink(target_audio_path, out_audio_path)
        exact_context_length = 0.0
    else:
        assert context is not None
        context = context[:, -context_nsamples:]
        merged = np.concatenate([context, target], axis=1)
        exact_context_length = context_nsamples / float(target_sr)
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_audio_path), merged.T, target_sr)

    return exact_context_length


def run_whisper_asr(
    root_dir: str | Path,
    output_dir: str | Path | None,
    audio_glob: str,
    metadata: str | None,
    channels: list[int],
    model_name: str,
    language: str,
    word_timestamps: bool,
    vad_filter: bool,
    vad_min_speech_duration_ms: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    batch_size: int = 1,
    overwrite: bool = True,
    write_audacity_labels: bool = False,
    exclude_context_file: bool = True,
    context_length: float = 0.0,
):
    # get audio paths
    root_dir = Path(root_dir)
    records = None
    if metadata is not None:
        metadata_path = root_dir / metadata
        assert metadata_path.exists()
        print(f"Reading audio paths from metadata ({metadata}) ...")
        audio_paths = []
        path_to_idx = {}
        context_paths = []
        records = list(read_jsonl(metadata_path))
        for i, r in enumerate(records):
            p = Path(r["output_audio_path"])
            audio_paths.append(p)
            path_to_idx[str(p)] = i
            if context_length > 0:
                p = Path(r["context_audio_path"])
                if not p.exists() or r["context_audio_path"] == "":
                    print(f"[WARGING] Context {p} does not exist.")
                    context_paths.append(None)
                else:
                    context_paths.append(p)
    else:
        context_paths = []
        print(f"Reading audio paths from directory ({audio_glob}) ...")
        audio_paths = sorted(Path(p) for p in glob(str(root_dir / audio_glob)))
        if exclude_context_file:
            audio_paths = [p for p in audio_paths if not p.name.endswith(".context.wav")]
    if not output_dir:
        output_dir = root_dir / "asr"

    print(f"# audio files: {len(audio_paths)}")
    print(f"model: {model_name}")
    print(f"language: {language}")
    print(f"channels: {channels}")
    print(f"word_timestamps: {word_timestamps}")
    print(f"vad_filter: {vad_filter}")
    print(f"output_dir: {output_dir}")

    asr = WhisperASR(
        model_name=model_name,
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        vad_min_speech_duration_ms=vad_min_speech_duration_ms,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        batch_size=batch_size,
    )

    exact_context_length = 0.0
    for i, audio_path in tenumerate(audio_paths, desc="Transcribing"):
        # set output path
        sample_id = audio_path.parent.name
        output_sample_dir = Path(output_dir) / sample_id
        json_path = Path(output_sample_dir) / f"{audio_path.stem}.json"


        asr_audio_path = audio_path
        if json_path.exists() and not overwrite:
            print(f"{json_path} exists. skipped.")
        else:
            # (Optional) prepend context to audio file.
            if context_length and context_length > 0:
                context_audio_path = context_paths[i]
                asr_audio_path = Path(output_sample_dir) / f"{audio_path.stem}.asr.wav"
                exact_context_length = save_audio_with_context(
                    audio_path,
                    context_audio_path,
                    asr_audio_path,
                    context_length,
                )
    
            # run whisper
            result = asr.transcribe_file(
                audio_path=asr_audio_path,
                channels=channels,
            )
            result["context_length"] = exact_context_length
            result["asr_audio_path"] = str(asr_audio_path)
    
            # output json
            write_json(json_path, result)
    
            # output audacity-format word timestamps
            if word_timestamps and write_audacity_labels:
                result_list = json_to_list(result)
                for ch in channels:
                    with json_path.with_suffix(f".ch{ch}.txt").open("w") as f:
                        f.write("\n".join(result_list[ch]) + "\n")

        # update metadata
        if metadata is not None and records is not None:
            idx = path_to_idx[str(audio_path)]
            records[idx]["output_asr_path"] = str(json_path)
            records[idx]["asr_audio_path"] = str(asr_audio_path)
            records[idx]["asr_context_length"] = exact_context_length

    # write metadata with asr_path info
    if metadata is not None and records is not None:
        out_metadata_path = Path(output_dir) / metadata
        write_jsonl(out_metadata_path, records)
        print(f"Wrote metadata to: {out_metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faster-whisper ASR.")
    parser.add_argument(
        "--root-dir", type=str, required=True,
        help="Root directory containing audio/ directory."
    )
    parser.add_argument(
        "--audio-glob", type=str, default="audio/*/*.wav",
        help="Glob pattern under root-dir.",
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help=(
            "Name of metadata under root-dir. "
            "If specified, audio files are read from the metadata."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="(Optional) Output directory for transcription files."
    )
    parser.add_argument(
        "--write-audacity-labels", action="store_true",
        help="If True, output audacity-format files." 
    )
    parser.add_argument(
        "--transcribe-context", action="store_true",
        help="Transcribe *context.wav"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--channels", type=str, default="0,1",
        help="Channels to transcribe, e.g., '0,1' or '1'.",
    )

    parser.add_argument("--model-name", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-word-timestamps", action="store_true")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--vad-min-speech-duration-ms", type=int, default=150)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument(
        "--context-length", type=float, default=0.0,
        help=(
            "Adds a few seconds of context before ASR."
            "Recommend using this when evaluating user barge-in or backchanneling."
        ),
    )

    args = parser.parse_args()

    channels = [int(v.strip()) for v in args.channels.split(",")]

    run_whisper_asr(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        audio_glob=args.audio_glob,
        metadata=args.metadata,
        channels=channels,
        model_name=args.model_name,
        language=args.lang,
        word_timestamps=not args.no_word_timestamps,
        vad_filter=args.vad,
        vad_min_speech_duration_ms=args.vad_min_speech_duration_ms,
        vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        write_audacity_labels=args.write_audacity_labels,
        exclude_context_file=not args.transcribe_context,
        context_length=args.context_length
    )
