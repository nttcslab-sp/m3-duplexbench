def get_channel_segments(
    transcript: dict,
    channel: int,
) -> list[dict]:
    segments = transcript.get("segments", [])
    return [s for s in segments if int(s["channel"]) == channel]


def get_timestamp(segment: dict) -> tuple[float, float]:
    ts = segment["timestamp"]
    return float(ts[0]), float(ts[1])


def is_backchannel(
    segment: dict,
    turn_duration_threshold: float,
    turn_num_words_threshold: int,
) -> bool:
    start, end = get_timestamp(segment)
    duration = max(0.0, end - start)
    num_words = len(segment.get("chunks", []))

    return (
        duration < turn_duration_threshold and
        num_words < turn_num_words_threshold
    )


class BaseTimingEvaluator:
    is_llmaj = False

    def __init__(
        self,
        model_channel: int = 1,
        max_overlap: float = 0.4,
        turn_duration_threshold: float = 1.0,
        turn_num_words_threshold: int = 4,
    ):
        self.model_channel = model_channel
        self.max_overlap = max_overlap
        self.turn_duration_threshold = turn_duration_threshold
        self.turn_num_words_threshold = turn_num_words_threshold

    def evaluate(
        self,
        metadata: dict,
        transcript: dict,
    ):
        raise NotImplementedError


class SmoothTurnTakingEvaluator(BaseTimingEvaluator):
    is_llmaj = False

    def __init__(
        self,
        model_channel: int = 1,
        max_overlap: float = 0.4,
        turn_duration_threshold: float = 1.0,
        turn_num_words_threshold: int = 4,
        allow_short_utterance: bool = True,
    ):
        if allow_short_utterance:
            turn_duration_threshold = 0
            turn_num_words_threshold = 0

        super().__init__(
            model_channel,
            max_overlap,
            turn_duration_threshold,
            turn_num_words_threshold,
        )


    """Evaluate TOR and latency for TURN_SHIFT."""
    def find_turn_taking_segment(
        self,
        transcript: dict,
        user_turn_end: float,
    ) -> tuple[dict|None, str]:
        """Find a target model segment.
    
        1. find candidates:
          - segment.end > user_turn_end
          - segment.start >= user_turn_end - max_overlap
        2. choose the segment whose start time is closest to user_turn_end.

        Return:
            target_segment
            miss_type
        """
        candidates = []
        is_silence = True
    
        for segment in get_channel_segments(
            transcript,
            channel=self.model_channel
        ):
            seg_start, seg_end = get_timestamp(segment)
    
            if seg_end <= user_turn_end:
                continue

            if is_backchannel(
                segment,
                self.turn_duration_threshold,
                self.turn_num_words_threshold,
            ):
                continue

            is_silence = False
    
            if seg_start < user_turn_end - self.max_overlap:
                continue
    
            candidates.append(segment)
    
        if candidates:
            target = min(
                candidates,
                key=lambda s: abs(get_timestamp(s)[0] - user_turn_end),
            )
            return target, "none"

        if is_silence:
            return None, "silence"

        return None, "long_overlap"

    def evaluate(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:
        user_turn_end = float(metadata["event_start"]) - float(metadata["input_start"])

        target_segment, miss_type = self.find_turn_taking_segment(
            transcript,
            user_turn_end,
        )

        if target_segment is None:
            results = {
                "tor ↑": 0,
                "latency ↓": None,
                "tor_err_silence": 1 if miss_type == "silence" else 0,
                "tor_err_long_overlap": 1 if miss_type == "long_overlap" else 0,
            }
            info = {
                "user_turn_end": user_turn_end,
                "target_segment": None,
                "miss_type": miss_type,
            }
            return results, info

        seg_start, seg_end = get_timestamp(target_segment)
        latency = seg_start - user_turn_end

        results = {
            "tor ↑": 1,
            "latency ↓": latency,
            "tor_err_silence": 0,
            "tor_err_long_overlap": 0,
        }
        info = {
            "user_turn_end": user_turn_end,
            "target_segment": {
                "channel": target_segment.get("channel"),
                "text": target_segment.get("text"),
                "timestamp": target_segment.get("timestamp"),
            },
            "miss_type": miss_type,
        }
        return results, info


class PauseHandlingEvaluator(BaseTimingEvaluator):
    is_llmaj = False

    """Evaluate TOR and latency for TURN_HOLD and SHORTPAUSE."""
    def find_overlap_segments(
        self,
        transcript: dict,
        pause_start: float,
        pause_end: float,
    ) -> tuple[list[dict], bool]:
        """Find segmens overlapping user pause window."""
        overlap_segments = []
        has_long_overlap = False
    
        for segment in get_channel_segments(
            transcript,
            channel=self.model_channel
        ):
            seg_start, seg_end = get_timestamp(segment)
  
            # non-overlapping
            if seg_end <= pause_start or seg_start >= pause_end:
                continue

            # not a turn
            if is_backchannel(
                segment,
                self.turn_duration_threshold,
                self.turn_num_words_threshold,
            ):
                continue

            overlap_segments.append(segment)

            if seg_start < pause_start - self.max_overlap:
                has_long_overlap = True

        overlap_segments = sorted(
            overlap_segments, key=lambda x: get_timestamp(x)[0]
        )
    
        return overlap_segments, has_long_overlap

    def get_segment_start_after_pause(
        self,
        transcript: dict,
        pause_end: float,
    ):
        starts = []
        for segment in get_channel_segments(
            transcript,
            channel=self.model_channel
        ):
            seg_start, seg_end = get_timestamp(segment)
            if seg_start >= pause_end:
                starts.append(seg_start)

        if not starts:
            return None

        return min(starts)

    def evaluate(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:

        pause_start = float(metadata["event_start"]) - float(metadata["input_start"])
        pause_end = float(metadata["event_end"]) - float(metadata["input_start"])

        overlap_segments, miss_long_overlap = self.find_overlap_segments(
            transcript,
            pause_start,
            pause_end,
        )

        tor = 1 if len(overlap_segments) > 0 else 0
        miss_normal = True if tor == 1 and not miss_long_overlap else False

        if tor == 1:
            seg_start, seg_end = get_timestamp(overlap_segments[0])
            latency = seg_start - pause_start
        else:
            assert miss_long_overlap == False
            assert miss_normal == False
            seg_start = self.get_segment_start_after_pause(
                transcript,
                pause_end,
            )
            if seg_start is None:
                latency = float(metadata["input_end"]) - float(metadata["event_start"])
            else:
                latency = seg_start - pause_start

        latency_wo_long_overlap = None if miss_long_overlap else latency

        results = {
            "tor ↓": tor,
            "latency ↑": latency,
            "latency_wo_long_overlap": latency_wo_long_overlap,
            "tor_err_normal": 1 if miss_normal else 0,
            "tor_err_long_overlap": 1 if miss_long_overlap else 0,
        }
        info = {
            "pause_start": pause_start,
            "pause_end": pause_end,
            "miss_type": (
                "long_overlap" if miss_long_overlap
                else "takeover" if miss_normal
                else "none"
            ),
        }
        return results, info


class UserBackchannelingEvaluator(BaseTimingEvaluator):
    is_llmaj = False

    """Evaluate stop latency for BC."""

    def find_speaking_segment_at_time(
        self,
        transcript: dict,
        time: float,
        max_short_pause_len: float = 0.5,
        ignore_bc: bool = False,
    ):
        candidates = []

        for segment in get_channel_segments(
            transcript,
            channel=self.model_channel,
        ):
            seg_start, seg_end = get_timestamp(segment)

            # if not (seg_start <= time < seg_end):
            # Allow short pause
            if not (seg_start < time + max_short_pause_len):
                continue

            if ignore_bc and is_backchannel(
                segment,
                self.turn_duration_threshold,
                self.turn_num_words_threshold,
            ):
                continue

            candidates.append(segment)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda s: get_timestamp(s)[0]
        )

    def evaluate(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:
        # usually, bc_start will be 0
        bc_start = float(metadata["event_start"]) - float(metadata["input_start"])

        target_segment = self.find_speaking_segment_at_time(
            transcript=transcript,
            time=bc_start,
        )

        if target_segment is None:
            results = {
                "stop_latency ↑": None,
                "is_speaking": 0,
            }
            info = {
                "target_segment": None,
            }
            return results, info

        seg_start, seg_end = get_timestamp(target_segment)
        stop_latency = seg_end - bc_start

        results = {
            "stop_latency ↑": stop_latency,
            "is_speaking": 1,
        }
        info = {
            "target_segment": {
                "channel": target_segment.get("channel"),
                "text": target_segment.get("text"),
                "timestamp": target_segment.get("timestamp"),
            },
        }
        return results, info


class UserBargeInEvaluator(UserBackchannelingEvaluator):
    is_llmaj = False

    """Evaluate stop latency for Barge-in."""
    def evaluate(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:
        # usually, barge_in_start will be 0
        barge_in_start = float(metadata["event_start"]) - float(metadata["input_start"])

        target_segment = self.find_speaking_segment_at_time(
            transcript=transcript,
            time=barge_in_start,
        )

        if target_segment is None:
            results = {
                "stop_latency ↓": None,
                "is_speaking": 0,
            }
            info = {
                "target_segment": None,
            }
            return results, info

        seg_start, seg_end = get_timestamp(target_segment)
        stop_latency = seg_end - barge_in_start

        results = {
            "stop_latency ↓": stop_latency,
            "is_speaking": 1,
        }
        info = {
            "target_segment": {
                "channel": target_segment.get("channel"),
                "text": target_segment.get("text"),
                "timestamp": target_segment.get("timestamp"),
            },
        }
        return results, info
