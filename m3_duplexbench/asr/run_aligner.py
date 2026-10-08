import argparse
import json
import re
import time
from glob import glob
from pathlib import Path
from multiprocessing import Pool

import subprocess
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from m3_duplexbench.utils import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    load_audio_channel_first,
    detect_non_silent_region,
)


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


def make_textgrid(dur, uttdic, channels, fid, margin=0.0, trim_regions=None):
    def fix_boundary(uttdic, margin, dur):
        uttdic_margin = {}
        for ch, utts in uttdic.items():
            buf = None

            trim_start, trim_end = 0.0, dur
            if trim_regions is not None and ch in trim_regions:
                trim_start, trim_end = trim_regions[ch]
            for st, en, w in utts:

                # (Optional) Add margin
                st = max(0.0, float(st)-margin)
                en = min(float(en)+margin, dur)

                # (Default) Energy-based trim
                st = max(st, trim_start)
                en = min(en, trim_end)

                if buf is None:
                    buf = [st, en, w]
                else:
                    if st <= buf[1]:
                        # joint utt to buf
                        buf[1] = en
                        buf[2] = f"{buf[2]} {w}".strip()
                    else:
                        # save buffer and initialize
                        uttdic_margin.setdefault(ch, []).append(buf)
                        buf = [st, en, w]
            if buf:
                uttdic_margin.setdefault(ch, []).append(buf)
        return uttdic_margin

    tg_format = [
        "File type = \"ooTextFile\"",
        "Object class = \"TextGrid\"",
        "",
        "xmin = 0.0",
        f"xmax = {dur:.3f}",
        "tiers? <exists>",
        f"size = {channels}",
        "item []:",
    ]

    uttdic = fix_boundary(uttdic, margin, dur)

    for ch in sorted(uttdic):
        tg_format.extend([
            f"\titem [{ch}]:",
            "\t\tclass = \"IntervalTier\"",
            f"\t\tname = \"{fid}_spk{ch}\"",
            "\t\txmin = 0",
            f"\t\txmax = {dur}",
            f"\t\tintervals: size = {len(uttdic[ch])}"
            ])

        for i, (stime, etime, content) in enumerate(uttdic[ch]):
            tg_format.extend([
                    f"\t\tintervals [{i+1}]:",
                    f"\t\t\txmin = {stime:.2f}",
                    f"\t\t\txmax = {etime:.2f}",
                    f"\t\t\ttext = \"{content}\"",
                ])
    return '\n'.join(tg_format)


def create_textgrid(_args: tuple):
    item, textgrid_in_dir, margin, ignore_segment_boundary = _args

    uid = item["event_id"]
    wav_path = Path(item["output_audio_path"])
    orig_alignment_path = Path(item["output_asr_path"])

    out_wav = textgrid_in_dir / f"{uid}.wav"
    out_textgrid = textgrid_in_dir / f"{uid}.TextGrid"

    assert wav_path.exists()
    assert orig_alignment_path.exists()

    # load json
    segment_list = []
    orig_alignment = read_json(orig_alignment_path)
    segments = orig_alignment.get("segments", [])

    # Get context length (prepended during ASR with --context-length)
    context_length = float(orig_alignment.get("context_length", 0.0))
    item["context_length"] = context_length

    # Get audio path used in ASR (context prepended)
    asr_audio_path = orig_alignment.get("asr_audio_path")
    asr_audio_path = Path(asr_audio_path) if asr_audio_path else None
    if asr_audio_path:
        wav_path = asr_audio_path

    # get duration
    info = sf.info(str(wav_path))
    dur = float(info.frames) / float(info.samplerate)
    channels = int(info.channels)

    # Detect non-silent region for each channel
    audio_cf, sr = load_audio_channel_first(wav_path)
    trim_regions = {}
    for ch0 in range(min(channels, audio_cf.shape[0])):
        speech_start, speech_end = detect_non_silent_region(
            audio=audio_cf[ch0],
            sr=sr,
        )
        # TextGrid speaker indices are 1-based.
        trim_regions[ch0 + 1] = (speech_start, speech_end)

    for seg in segments:
        ts = seg["timestamp"]
        st, en = float(ts[0]), float(ts[1])
        assert len(ts) == 2
        text = str(seg.get("text", ""))
        if not text:
            continue
        chunks = str(seg.get("chunks", []))
        segment_list.append({
            "channel": int(seg["channel"]),
            "timestamp": [st, en],
            "text": text,
            "chunks": chunks,
        })
    if len(segment_list) == 0:
        return None

    # get channel dict
    ch_outputs = {}
    for seg in segment_list:
        ch = int(seg["channel"]) + 1
        st, en = seg["timestamp"]
        snt = seg["text"]
        ch_outputs.setdefault(ch, []).append([st, en, snt])

    if ignore_segment_boundary:
        ch_outputs_all = {}
        for ch, outputs in ch_outputs.items():
            whole_text = " ".join([w[-1] for w in outputs])
            ch_outputs_all[ch] = [[0.0, dur, whole_text]]
        ch_outputs = ch_outputs_all

    # create textgrid content
    tg_format = make_textgrid(dur, ch_outputs, channels, uid, margin, trim_regions)
    with out_textgrid.open('w', newline='\n') as fp:
        fp.write(tg_format)
    if not out_wav.exists():
        out_wav.symlink_to(wav_path.absolute())

    return item


def prepare_mfa_data(
    items: list[dict],
    out_dir: Path,
    margin: float,
    nj: int,
    ignore_segment_boundary: bool = True,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = []

    arglist = [(item, textgrid_in_dir, margin, ignore_segment_boundary) for item in items]
    print('start create textgrid ...')
    pool = Pool(processes=nj)
    with tqdm(total=len(arglist)) as pbar:
        for ret_item in pool.imap_unordered(create_textgrid, arglist):
            pbar.update(1)
            if ret_item is not None:
                prepared.append(ret_item)

    return prepared


def parse_textgrid(textgrid_path: str | Path) -> dict[int, list]:
    """Parse MFA TextGrid output."""
    path = Path(textgrid_path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    channel_items = {}

    current_channel = None
    current_is_words_tier = False
    current_start = None
    current_end = None

    #   name = "xxx_spk1 - words"
    #   name = "xxx_spk2 - words"
    #   name = "xxx_spk1 - phones"
    name_re = re.compile(r'name\s*=\s*".*_spk(\d+)\s*-\s*(words|phones)"')
    xmin_re = re.compile(r"xmin\s*=\s*([0-9.]+)")
    xmax_re = re.compile(r"xmax\s*=\s*([0-9.]+)")
    text_re = re.compile(r'text\s*=\s*"(.*)"')

    for line in lines:
        line = line.strip()

        m = name_re.match(line)
        if m:
            tier_ch = int(m.group(1))
            tier_type = m.group(2)

            current_channel = tier_ch - 1
            current_is_words_tier = tier_type == "words"

            if current_is_words_tier:
                channel_items.setdefault(current_channel, [])

            current_start = None
            current_end = None
            continue

        if current_channel is None or not current_is_words_tier:
            continue

        m = xmin_re.match(line)
        if m:
            current_start = float(m.group(1))
            continue

        m = xmax_re.match(line)
        if m:
            current_end = float(m.group(1))
            continue

        m = text_re.match(line)
        if m:
            word = m.group(1).strip()

            if word and current_start is not None \
                and current_end is not None and current_end > current_start:
                channel_items[current_channel].append(
                    {
                        "channel": current_channel,
                        "start": current_start,
                        "end": current_end,
                        "word": word,
                    }
                )

            current_start = None
            current_end = None

    for ch in channel_items:
        channel_items[ch].sort(key=lambda x: (x["start"], x["end"]))

    return channel_items


def word_in_segment(
    w: dict,
    seg_start: float,
    seg_end: float,
    small_margin: float = 0.1
):
    return w["end"] > seg_start - small_margin \
        and w["start"] < seg_end + small_margin


def fix_alignment_depre(
    original_json_path: Path,
    aligned_words: dict[int, list],
    context_length: float = 0.0,
) -> dict:
    """(Deprecated) Replace Whisper word timestamps with MFA word timestamps.

    Segment boundaries are recomputed from the first/last MFA word assigned
    to each segment.

    If ASR was run on context-appended audio, context_length is used to:
      - remove words in the prepended context
      - shift remaining timestamps so that target audio starts at 0.0
    """
    original = read_json(original_json_path)
    original_segments = original.get("segments", [])

    output_segments = []

    for seg in original_segments:
        ch = int(seg["channel"])
        seg_start, seg_end = map(float, seg["timestamp"])
        words = aligned_words.get(ch, [])

        seg_words = [w for w in words if word_in_segment(w, seg_start, seg_end)]

        chunks = []
        for w in seg_words:
            shifted_start = float(w["start"]) - context_length
            shifted_end = float(w["end"]) - context_length

            # remove words fully contained in context
            if shifted_end <= 0.0:
                continue

            shifted_start = max(0.0, shifted_start)

            chunks.append(
                {
                    "text": w["word"],
                    "timestamp": [float(shifted_start), float(shifted_end)],
                }
            )

        if chunks:
            text = " ".join(c["text"] for c in chunks).strip()
            timestamp = [
                float(chunks[0]["timestamp"][0]),
                float(chunks[-1]["timestamp"][1]),
            ]

            output_segments.append(
                {
                    "channel": ch,
                    "text": text,
                    "timestamp": timestamp,
                    "chunks": chunks,
                }
            )

    output_segments.sort(
        key=lambda x: (
            x["timestamp"][0],
            x["channel"],
            x["timestamp"][1],
        )
    )

    return {
        "context_length": context_length,
        "segments": output_segments,
    }


def segment_words(
    words: list,
    max_gap: float = 0.5,
    context_length: float = 0.0,
) -> list[dict]:
    """Segment aligned words to IPUs."""
    shifted_words: list = []

    for w in sorted(words, key=lambda x: (x["start"], x["end"])):
        start = float(w["start"]) - context_length
        end = float(w["end"]) - context_length

        # Fully inside prepended context.
        if end <= 0.0:
            continue

        # Boundary-crossing word: keep it and clip start.
        start = max(0.0, start)

        if end <= start:
            continue

        shifted_words.append(
            {
                "channel": w["channel"],
                "start": start,
                "end": end,
                "word": w["word"],
            }
        )

    if len(shifted_words) == 0:
        return []

    segments: list[dict] = []
    buf = [shifted_words[0]]

    for w in shifted_words[1:]:
        prev = buf[-1]
        gap = w["start"] - prev["end"]

        if gap >= max_gap: # IPU<500ms
            segments.append(make_segment_from_words(buf))
            buf = [w]
        else:
            buf.append(w)

    if buf:
        segments.append(make_segment_from_words(buf))

    return segments


def make_segment_from_words(words: list) -> dict:
    ch = int(words[0]["channel"])

    chunks = [{
        "text": w["word"],
        "timestamp": [float(w["start"]), float(w["end"])],
    } for w in words ]

    text = " ".join(w["word"] for w in words).strip()
    timestamp = [
        float(words[0]["start"]),
        float(words[-1]["end"]),
    ]

    return {
        "channel": ch,
        "text": text,
        "timestamp": timestamp,
        "chunks": chunks,
    }


def fix_alignment(
    aligned_words_by_channel: dict[int, list],
    context_length: float = 0.0,
    max_gap: float = 0.5,
) -> dict:
    """Build ASR JSON from MFA word alignment."""
    output_segments = []

    for ch in sorted(aligned_words_by_channel):
        words = aligned_words_by_channel[ch]

        segs = segment_words(
            words=words,
            max_gap=max_gap,
            context_length=context_length,
        )
        output_segments.extend(segs)

    output_segments.sort(
        key=lambda x: (
            x["timestamp"][0],
            x["channel"],
            x["timestamp"][1],
        )
    )

    return {
        "context_length": context_length,
        "segments": output_segments,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MFA to refine word timestamps.")

    parser.add_argument("--metadata", type=str, required=True,
        help="Path to metadata.jsonl including key for ASR JSON path (output_asr_path).")
    parser.add_argument("--output-dir", type=str, default=None,
        help="(Optional) Output directory for JSON files.")
    parser.add_argument("--output-metadata", type=str, default=None,
        help="(Optional) Output metadata.jsonl.")
    parser.add_argument("--work-dir", type=str, required=True,
        help="Working directory for MFA.")
    parser.add_argument('--margin', type=float, default=0.0,
                        help='Margin for start/end time (sec)')
    parser.add_argument("--use-whisper-segments", action="store_true",
        help="Use whisper timestamps (deprecated)")
    parser.add_argument('--nj', type=int, default=8, help='Number of jobs.')
    parser.add_argument("--dictionary", type=str, default="english_us_arpa")
    parser.add_argument("--acoustic-model", type=str, default="english_us_arpa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-audacity-labels", action="store_true")

    args = parser.parse_args()
    work_dir = Path(args.work_dir)

    records = list(read_jsonl(Path(args.metadata)))
    path_to_idx = {}
    for i, r in enumerate(records):
        path_to_idx[str(r["output_audio_path"])] = i

    output_dir = args.output_dir
    if not output_dir:
        output_dir = Path(records[0]["output_asr_path"]).parent.parent
    output_metadata = args.output_metadata
    if not output_metadata:
        output_metadata = Path(output_dir) / "metadata_mfa.jsonl"
    print(f"Output directory: {str(output_dir)}")
    print(f"Output metadata: {str(output_metadata)}")

    # Step 1: prepare textgrid_in/*TextGrid
    textgrid_in_dir = work_dir / "textgrid_in"
    prepared_items = prepare_mfa_data(
        records,
        textgrid_in_dir,
        margin=args.margin,
        nj=args.nj,
        ignore_segment_boundary=not args.use_whisper_segments,
    )
    if len(prepared_items) == 0:
        print("No valid items prepared for MFA.")
        exit()

    # Step 2: run MFA to generate textgrid_align/*TextGrid
    textgrid_align_dir = work_dir / "textgrid_align"
    cmd = [
        "mfa",
        "align",
        str(textgrid_in_dir),
        args.dictionary,
        args.acoustic_model,
        str(textgrid_align_dir),
        "--num_jobs",
        str(args.nj),
        "--clean"
    ]
    print("[MFA]", " ".join(cmd))
    if textgrid_align_dir.exists() and not args.overwrite:
        print(f"MFA output already exists, skipped.")
    else:
        subprocess.run(cmd, check=True)

    # Step 3: generate json and jsonl files
    for item in prepared_items:
        uid = item["event_id"]
        wav_path = Path(item["output_audio_path"])
        sample_id = wav_path.parent.name

        orig_alignment_path = Path(item["output_asr_path"])
        output_sample_dir = Path(output_dir) / sample_id
        new_alignment_path = output_sample_dir / f"{orig_alignment_path.stem}.mfa.json"
    
        textgrid_path = textgrid_align_dir / f"{uid}.TextGrid"
        if not textgrid_path.exists():
            print(f"[WARN] TextGrid file not found: {textgrid_path}")
            continue
   
        aligned_words = parse_textgrid(textgrid_path)
        if args.use_whisper_segments:
            new_alignment_json = fix_alignment_depre(
                original_json_path=orig_alignment_path,
                aligned_words=aligned_words,
                context_length=item.get("context_length", 0.0)
            )
        else:
            new_alignment_json = fix_alignment(
                aligned_words_by_channel=aligned_words,
                context_length=item.get("context_length", 0.0),
                max_gap=0.5,
            )

        # output json
        write_json(new_alignment_path, new_alignment_json)

        # output audacity-format word timestamps
        if args.write_audacity_labels:
            result_list = json_to_list(new_alignment_json)
            for ch in [0, 1]:
                with new_alignment_path.with_suffix(f".ch{ch}.txt").open("w") as f:
                    f.write("\n".join(result_list[ch]) + "\n")

        # update metadata
        records[path_to_idx[str(wav_path)]]["output_asr_path"] = str(new_alignment_path)

    write_jsonl(Path(output_metadata), records)
    print(f"Wrote metadata to: {output_metadata}")
