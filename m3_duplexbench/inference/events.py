import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

EVENT_TYPES = {
    "TURN",
    "TURN_SHIFT",
    "TURN_HOLD",
    "SHORTPAUSE",
    "BC",
    "BARGE_IN",
}

EVENT_TO_TASK = {
    "TURN_SHIFT": "smooth_turn_taking",
    "TURN_HOLD": "pause_handling",
    "SHORTPAUSE": "pause_handling",
    "BC": "backchanneling",
    "BARGE_IN": "barge_in",
}

CONTEXT_TYPES = {
    "continuous",
    "sequential",
}

@dataclass
class EventSample:
    sample_id: str
    event_id: str
    user_channel: int
    system_channel: int
    event_type: str
    task: str
    # Original event span
    event_start: float
    event_end: float
    trigger_time: float # deprecated
    # User input span
    input_start: float
    input_end: float
    # Context input span
    context_start: float
    context_end: float
    # Relative event span
    # relative_event_start: float
    # relative_event_end: float
    audio_path: str
    sample_rate: int
    input_duration: float
    alignment_path: str | None = None


@dataclass
class DataStats:
    channels_to_evaluate: list[int] = field(default_factory=lambda: [0, 1])
    total_events: int = 0
    kept_events: int = 0
    skipped_invalid_event: int = 0
    skipped_filtered_event: int = 0
    skipped_short_user_audio: int = 0
    num_events: dict = field(
        default_factory=lambda: {
            "smooth_turn_taking": 0,
            "pause_handling": 0,
            "backchanneling": 0,
            "barge_in": 0,
        }
    )


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {path}") from e


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_audio_info(audio_path: Path) -> tuple[int, float]:
    info = sf.info(str(audio_path))
    sample_rate = int(info.samplerate)
    duration = float(info.frames) / float(info.samplerate)
    return sample_rate, duration


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    audio = audio.astype(np.float32)
    return audio, int(sample_rate)


def is_time_in_interval(
    t: float,
    start: float,
    end: float,
    include_start: bool = True,
    include_end: bool = True,
) -> bool:
    if include_start and include_end:
        return start <= t <= end
    if include_start and not include_end:
        return start <= t < end
    if not include_start and include_end:
        return start < t <= end
    return start < t < end


def is_time_in_any_event(
    events: list[dict[str, Any]],
    t: float,
    event_type: str,
) -> bool:
    for ev in events:
        if ev.get("event") != event_type:
            continue

        time_span = ev.get("time")
        if not isinstance(time_span, list) or len(time_span) != 2:
            continue

        start, end = float(time_span[0]), float(time_span[1])
        if is_time_in_interval(t, start, end):
            return True

    return False


def get_event_channels(
    raw_channel: int,
    event_type: str,
) -> tuple[int, int]:
    """
    Return (user_channel, system_channel) for an evaluation event.
    - raw_channel -> system_channel for TURN_SHIFT
    - raw_channel -> user_channel for others
    """
    if event_type == "TURN_SHIFT":
        system_channel = raw_channel
        user_channel = 1 - raw_channel
        return user_channel, system_channel

    user_channel = raw_channel
    system_channel = 1 - raw_channel
    return user_channel, system_channel


def get_turn_events(
    # e.g. {"0": [{"event": "TURN", "time": [0.16, 1.04]}]}
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
) -> list[tuple[float, float]]:
    turns: list[tuple[float, float]] = []
    for ev in events_by_channel.get(str(channel), []):
        if ev.get("event") != "TURN":
            continue
        time_span = ev.get("time")
        if not isinstance(time_span, list) or len(time_span) != 2:
            continue
        start, end = float(time_span[0]), float(time_span[1])
        if start <= end:
            turns.append((start, end))
    turns.sort(key=lambda x: (x[0], x[1]))
    return turns


def get_context_interval(
    events_by_channel: dict[str, list[dict[str, Any]]],
    context_end: float,
    use_context: bool = True,
    context_max_len: float | None = None,
) -> tuple[float, float]:
    """Get context interval.

    If context_max_len is specified, find the earliest TURN start within
    [context_end - context_max_len, context_end].
    This gives the longest possible context not exceeding context_max_len.
    """

    if not use_context:
        return 0.0, 0.0

    if context_max_len is None:
        return 0.0, context_end

    lower_bound = max(0.0, context_end - context_max_len)
    turns = get_turn_events(events_by_channel, 0)
    turns += get_turn_events(events_by_channel, 1)
    turns = sorted(turns, key=lambda x: x[0])

    turn_starts = [t[0] for t in turns if lower_bound <= t[0] < context_end]

    if len(turn_starts) > 0:
        context_start = turn_starts[0]
    else:
        context_start = lower_bound

    return context_start, context_end

def find_turn_start_at_time(
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
    time: float,
) -> tuple[float, float] | None:
    """
    Find a TURN on channel whose start is at time.
    Used to find a TURN corresponding to the BARGE_IN.
    """
    for start, end in get_turn_events(events_by_channel, channel):
        if start == time:
            return start, end

    print("No corresponding TURN found.")
    return None


def find_turn_end_at_time(
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
    time: float,
) -> tuple[float, float] | None:
    """
    Find a TURN on channel whose end is at time.
    Used to find a TURN before TURN_SHIFT or TURN_HOLD.
    """
    for start, end in get_turn_events(events_by_channel, channel):
        if end == time:
            return start, end

    print("No corresponding TURN found.")
    return None

def find_parent_turn(
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
    event_start: float,
    event_end: float,
) -> tuple[float, float] | None:
    """Find a TURN containing SHORTPAUSE [event_start, event_end]."""
    for start, end in get_turn_events(events_by_channel, channel):
        if start <= event_start and event_end <= end:
            return start, end


def get_input_times(
    event_type: str,
    channel: int,
    event_start: float,
    event_end: float,
    events_by_channel: dict[str, list[dict[str, Any]]],
    eval_window_len: float,
    use_context: bool = True,
    context_max_len: float | None = None,
) -> dict[str, float] | None:
    """Compute input/context times."""

    if event_type in {"TURN_SHIFT", "TURN_HOLD"}:
        prev_turn = find_turn_end_at_time(
            events_by_channel=events_by_channel,
            channel=channel,
            time=event_start,
        )
        if prev_turn is None:
            return None
    
        turn_start, turn_end = prev_turn
        input_start = turn_start
        input_end = turn_end + eval_window_len
    elif event_type == "SHORTPAUSE":
        parent_turn = find_parent_turn(
            events_by_channel=events_by_channel,
            channel=channel,
            event_start=event_start,
            event_end=event_end,
        )
        if parent_turn is None:
            return None

        turn_start, turn_end = parent_turn
        input_start = turn_start
        input_end = turn_end + eval_window_len
    elif event_type in {"BC", "BARGE_IN"}:
        input_start = event_start
        event_duration = max(0.0, event_end - event_start)
        input_end = event_start + max(eval_window_len, event_duration)
    else:
        return None

    # Context
    context_start, context_end = get_context_interval(
        events_by_channel=events_by_channel,
        context_end=input_start,
        use_context=use_context,
        context_max_len=context_max_len,
    )

    return {
        "input_start": input_start,
        "input_end": input_end,
        "context_start": context_start,
        "context_end": context_end,
    }


def is_speaking_at(
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
    t: float,
) -> bool:
    """
    Return whether t is inside a turn and outside a short pause.
    """
    channel_events = events_by_channel.get(str(channel), [])
    in_turn = is_time_in_any_event(channel_events, t, "TURN")
    in_pause = is_time_in_any_event(channel_events, t, "SHORTPAUSE")
    return in_turn and not in_pause


def is_in_turn_at(
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
    t: float,
) -> bool:
    """
    Return whether t is inside a turn of the given channel.
    t can be inside a short pause.
    """
    channel_events = events_by_channel.get(str(channel), [])
    in_turn = is_time_in_any_event(channel_events, t, "TURN")
    return in_turn


def is_in_long_turn_at(
    events_by_channel: dict[str, list[dict[str, Any]]],
    channel: int,
    t: float,
) -> bool:
    """
    Return whether t is inside a turn or a turn-holding.
    """
    channel_events = events_by_channel.get(str(channel), [])
    in_turn = is_time_in_any_event(channel_events, t, "TURN")
    in_hold = is_time_in_any_event(channel_events, t, "TURN_HOLD")
    return in_turn or in_hold


def event_filter(
    event_type: str,
    trigger_time: float,
    channel: int,
    events_by_channel: dict[str, list[dict[str, Any]]],
) -> bool:
    """Apply event-specific filtering."""
    system_channel = 1 - channel

    if event_type in {"BC", "BARGE_IN"}:
        return is_in_turn_at(
            events_by_channel=events_by_channel,
            channel=system_channel,
            t=trigger_time,
        )

    if event_type in {"SHORTPAUSE", "TURN_HOLD", "TURN_SHIFT"}:
        return not is_in_long_turn_at(
            events_by_channel=events_by_channel,
            channel=system_channel,
            t=trigger_time,
        )

    return False


def write_event_alignment(
    source_alignment_path: Path,
    output_alignment_path: Path,
    system_channel: int,
    context_start: float,
    context_end: float,
) -> str | None:
    """
    Saves event-specific alignment file.
     - keeps only words from system_channel
     - keeps words overlapping [context_start, context_end]
     - rewrites channel id as 0
     - saves the event-specific alignment file
    """
    if not source_alignment_path.exists():
        return None

    if context_start == context_end:
        return None

    rows  = []
    with source_alignment_path.open("r") as f:
        for line in f:
            line = line.strip()

            parts = line.split("\t")
            assert len(parts) == 4
            ch_str, start_str, end_str, word = parts

            start = float(start_str)
            end = float(end_str)

            if ch_str != str(system_channel):
                continue

            # Keep words overlapping [context_start, context_end].
            if end <= context_start or start >= context_end:
                continue

            clipped_start = max(context_start, start)
            clipped_end = min(context_end, end)
            assert clipped_end > clipped_start

            relative_start = clipped_start - context_start
            relative_end = clipped_end - context_start

            # Moshi conditioning text expects system-side channel to be "0".
            rows.append(f"0\t{relative_start:.4f}\t{relative_end:.4f}\t{word}")

    if len(rows) == 0:
        return None

    output_alignment_path.parent.mkdir(parents=True, exist_ok=True)
    with output_alignment_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    return str(output_alignment_path)


def build_event_samples(
    record: dict[str, Any],
    audio_dir: Path,
    ignore_short_pause: bool,
    ignored_events: None | list[str],
    stats: DataStats,
    alignment_dir: Path | None = None,
    output_alignment_root: Path | None = None,
    eval_window_len: float = 10.0,
    use_context: bool = True,
    use_teacher_forcing: bool = False,
    context_max_len: float | None = None,
) -> list[EventSample]:
    """Build event-level samples from a dialogue."""
    sample_id = str(record["id"])
    events_by_channel = record["events"]

    # Load audio
    audio_path = audio_dir / f"{sample_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load word alignment (optional)
    source_alignment_path  = None
    if alignment_dir is not None:
        source_alignment_path = alignment_dir / f"{sample_id}.txt"
        if not source_alignment_path.exists():
            raise FileNotFoundError(f"Alignment file not found: {source_alignment_path}")

    sample_rate, input_duration = load_audio_info(audio_path)

    samples = []
    event_counter = 0

    for raw_channel_str, events in events_by_channel.items():
        raw_channel = int(raw_channel_str)

        for ev in events:
            event_type = ev.get("event")

            # Check event type
            assert event_type in EVENT_TYPES, f"Unknown event: {event_type}"
            if event_type not in EVENT_TO_TASK:
                continue
            if ignore_short_pause and event_type == "SHORTPAUSE":
                continue
            if ignored_events is not None and event_type in ignored_events:
                continue
            if event_type in {"BC", "BARGE_IN"} and not use_context:
                # BC, BARGE_IN cannot be evaluated without context.
                continue

            # raw_channel -> system_channel for TURN_SHIFT
            # raw_channel -> user_channel for others
            user_channel, system_channel = get_event_channels(
                raw_channel=raw_channel,
                event_type=event_type,
            )

            # Check user channel
            if user_channel not in stats.channels_to_evaluate:
                continue

            # Calculate event span
            time_span = ev.get("time")
            assert isinstance(time_span, list)
            assert len(time_span) == 2
            event_start = float(time_span[0])
            event_end = float(time_span[1])
            if event_type == "BARGE_IN":
                curr_turn = find_turn_start_at_time(
                    events_by_channel,
                    user_channel,
                    event_start
                )
                if curr_turn:
                    event_end = curr_turn[1]
                else:
                    stats.skipped_invalid_event += 1
                    continue

            stats.total_events += 1

            task = EVENT_TO_TASK[event_type]

            windows = get_input_times(
                event_type=event_type,
                channel=user_channel,
                event_start=event_start,
                event_end=event_end,
                events_by_channel=events_by_channel,
                eval_window_len=eval_window_len,
                use_context=use_context,
                context_max_len=context_max_len,
            )
            if windows is None:
                stats.skipped_invalid_event += 1
                continue

            if not event_filter(
                event_type=event_type,
                trigger_time=event_start,
                channel=user_channel,
                events_by_channel=events_by_channel,
            ):
                stats.skipped_filtered_event += 1
                continue

            event_id = f"{sample_id}_ch{user_channel}_{event_counter:04d}_{task}"
            event_counter += 1

            # Write word alignment for event
            event_alignment_path = None
            if (
                use_context
                and source_alignment_path is not None
                and output_alignment_root is not None
            ):
                context_start = windows["context_start"]
                context_end = windows["context_end"]
                if use_teacher_forcing:
                    # Include prev_turn interval for teacher-forcing
                    context_end = event_end

                event_alignment_path = write_event_alignment(
                    source_alignment_path=source_alignment_path,
                    output_alignment_path=(
                        output_alignment_root
                        / sample_id
                        / f"{event_id}.alignment.txt"
                    ),
                    system_channel=system_channel,
                    context_start=context_start,
                    context_end=context_end,
                )

            samples.append(
                EventSample(
                    sample_id=sample_id,
                    event_id=event_id,
                    user_channel=user_channel,
                    system_channel=system_channel,
                    event_type=event_type,
                    task=task,
                    event_start=event_start,
                    event_end=event_end,
                    trigger_time=windows["input_start"],
                    input_start=windows["input_start"],
                    input_end=windows["input_end"],
                    context_start=windows["context_start"],
                    context_end=windows["context_end"],
                    audio_path=str(audio_path),
                    sample_rate=sample_rate,
                    input_duration=input_duration,
                    alignment_path=event_alignment_path,
                )
            )
            stats.kept_events += 1
            stats.num_events[EVENT_TO_TASK[event_type]] += 1

    return samples


def seconds_to_sample(
    t: float,
    sample_rate: int,
    *,
    clamp_min: int = 0,
    clamp_max: int | None = None,
) -> int:
    idx = int(round(t * sample_rate))
    idx = max(clamp_min, idx)
    if clamp_max is not None:
        idx = min(idx, clamp_max)
    return idx


def slice_event_audio(
    audio: np.ndarray,
    sample_rate: int,
    event: EventSample,
    mute_eval_window: bool = True,
) -> tuple[np.ndarray, np.ndarray|None, np.ndarray]:
    """
    Slice context audio and user audio for evaluation.

    - context_audio: 2 channels
    - pre_event_audio: 2 channels
    - eval_window_audio: 1 channel
    """
    num_samples, num_channels = audio.shape
    assert num_channels == 2, f"Expected 2-channel audio, got {num_channels}"

    # Context audio
    context_start_sample = seconds_to_sample(event.context_start, sample_rate)
    context_end_sample = seconds_to_sample(event.context_end, sample_rate)
    context_audio = audio[context_start_sample:context_end_sample, :]
    context_audio = context_audio.astype(np.float32)

    # # User audio with silence
    # input_start_sample = seconds_to_sample(event.input_start, sample_rate)
    # input_end_sample = seconds_to_sample(event.input_end, sample_rate)
    # user_input_len = input_end_sample - input_start_sample
    # user_audio = np.zeros(user_input_len, dtype=np.float32)
    #
    # # input_end may exceed the original audio length because silence can be appended.
    # available_end_sample = min(input_end_sample, num_samples)
    # available_len = max(0, available_end_sample - input_start_sample)
    #
    # user_audio[:available_len] = audio[
    #     input_start_sample:available_end_sample, event.user_channel,
    # ]

    # Previous turn (after context ~ before event)
    turn_start_sample = seconds_to_sample(event.input_start, sample_rate)
    turn_end_sample = seconds_to_sample(event.event_start, sample_rate)
    if turn_start_sample == turn_end_sample:  # BC, BARGE_IN
        pre_event_audio = None
    else: # TURN_SHIFT, TURN_HOLD
        pre_event_audio = audio[turn_start_sample:turn_end_sample, :] # stereo
        pre_event_audio = pre_event_audio.astype(np.float32)

    # Eval window
    eval_start_sample = seconds_to_sample(event.event_start, sample_rate)
    eval_end_sample = seconds_to_sample(event.input_end, sample_rate)
    eval_window_audio = np.zeros(eval_end_sample - eval_start_sample, dtype=np.float32)
    audio_end_sample = min(eval_end_sample, num_samples)
    eval_window_audio[:audio_end_sample - eval_start_sample] = audio[
        eval_start_sample:audio_end_sample, event.user_channel
    ] # mono


    # Mute user audio
    if mute_eval_window:
        if event.event_type in {"TURN_SHIFT", "TURN_HOLD", "SHORTPAUSE"}:
            mute_start_time = event.event_start
        elif event.event_type in {"BC", "BARGE_IN"}:
            mute_start_time = event.event_end
        else:
            raise ValueError
        mute_start_sample = seconds_to_sample(mute_start_time, sample_rate)
        # user_audio[mute_start_sample - input_start_sample:] = 0.0

        eval_window_audio[mute_start_sample - eval_start_sample:] = 0.0
        eval_window_audio = eval_window_audio.astype(np.float32)

    return context_audio, pre_event_audio, eval_window_audio
