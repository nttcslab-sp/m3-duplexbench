import json
import os
import ssl
import threading
import time
from urllib.parse import urlparse, urlunparse

import numpy as np

from m3_duplexbench.inference.events import EventSample
from m3_duplexbench.inference.models.base import BaseInferenceModel
from m3_duplexbench.utils import pad_or_trim, resample_mono_np


SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1_920  # 80 ms at 24 kHz


def _websocket_url(url: str) -> str:
    """Normalize HTTP and bare server addresses to a WebSocket URL."""
    if "://" not in url:
        url = f"wss://{url}"

    parsed = urlparse(url)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    if scheme not in {"ws", "wss"}:
        raise ValueError(f"Unsupported DuplexCascade server URL: {url}")
    return urlunparse(parsed._replace(scheme=scheme))


def _trim_segments(
    segments: list[tuple[int, np.ndarray]],
    stop_sample: int,
) -> list[tuple[int, np.ndarray]]:
    """Drop audio that would still be queued when playback is stopped."""
    trimmed = []
    for offset, chunk in segments:
        keep = min(len(chunk), stop_sample - offset)
        if keep > 0:
            trimmed.append((offset, chunk[:keep]))
    return trimmed


def _build_timeline(
    segments: list[tuple[int, np.ndarray]],
    length: int,
) -> np.ndarray:
    timeline = np.zeros(length, dtype=np.float32)
    for offset, chunk in segments:
        if offset >= length:
            continue
        end = min(offset + len(chunk), length)
        timeline[offset:end] = chunk[: end - offset]
    return np.clip(timeline, -1.0, 1.0)


class DuplexCascadeModel(BaseInferenceModel):
    """WebSocket client for the DuplexCascade server."""

    def __init__(
        self,
        server_url: str | None = None,
        use_context: bool = False,
        use_teacher_forcing: bool = False,
        connect_timeout: float = 30.0,
        close_timeout: float = 10.0,
        startup_delay: float = 1.0,
        ssl_verify: bool = False,
        use_proxy: bool = False,
    ) -> None:
        super().__init__(
            use_context=use_context,
            use_teacher_forcing=use_teacher_forcing,
        )
        self.server_url = _websocket_url(server_url or "wss://127.0.0.1:31606")
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout
        self.startup_delay = startup_delay
        self.ssl_verify = ssl_verify
        self.use_proxy = use_proxy

    @property
    def name(self) -> str:
        return "duplexcascade"

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def context_mode(self) -> str:
        return "user_audio" if self.use_context else "none"

    # Original: https://github.com/sbintuitions/DuplexCascade/blob/main/web/client.js 
    def client(self, input_audio: np.ndarray) -> np.ndarray:
        try:
            import websocket
        except ImportError as e:
            raise ImportError(
                "DuplexCascade inference requires the 'websocket-client' package."
            ) from e

        segments: list[tuple[int, np.ndarray]] = []
        playhead = 0
        started_at: float | None = None
        output_lock = threading.Lock()
        receiver_errors: list[Exception] = []

        def receive(ws) -> None:
            nonlocal playhead, segments
            try:
                while True:
                    message = ws.recv()
                    if message is None or message == "":
                        return

                    if isinstance(message, bytes):
                        if len(message) % np.dtype("<f4").itemsize != 0:
                            raise ValueError("Received malformed float32 audio data.")

                        chunk = np.frombuffer(message, dtype="<f4").copy()
                        if len(chunk) == 0 or started_at is None:
                            continue

                        received_at = int(
                            (time.monotonic() - started_at) * SAMPLE_RATE
                        )
                        with output_lock:
                            offset = max(received_at, playhead)
                            segments.append((offset, chunk))
                            playhead = offset + len(chunk)
                        continue

                    try:
                        event = json.loads(message)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(event, dict):
                        continue

                    # stop playback immediately
                    if (
                        event.get("type") == "audio_control"
                        and event.get("action") == "stop"
                        and started_at is not None
                    ):
                        stop_sample = int(
                            (time.monotonic() - started_at) * SAMPLE_RATE
                        )
                        with output_lock:
                            segments = _trim_segments(segments, stop_sample)
                            playhead = stop_sample
            except websocket.WebSocketConnectionClosedException:
                return
            except Exception as e:
                receiver_errors.append(e)

        connect_options = {}
        if self.server_url.startswith("wss://"):
            connect_options["sslopt"] = {
                "cert_reqs": (
                    ssl.CERT_REQUIRED if self.ssl_verify else ssl.CERT_NONE
                ),
                "check_hostname": self.ssl_verify,
            }

        no_proxy = {key: os.environ.get(key) for key in ("no_proxy", "NO_PROXY")}
        if not self.use_proxy:
            os.environ["no_proxy"] = "*"
            os.environ["NO_PROXY"] = "*"
        try:
            ws = websocket.create_connection(
                self.server_url,
                timeout=self.connect_timeout,
                **connect_options,
            )
        finally:
            if not self.use_proxy:
                for key, value in no_proxy.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        ws.settimeout(None)
        receiver: threading.Thread | None = None
        try:
            # The server initializes its STT/TTS connections after the handshake.
            if self.startup_delay > 0:
                time.sleep(self.startup_delay)

            started_at = time.monotonic()
            receiver = threading.Thread(target=receive, args=(ws,), daemon=True)
            receiver.start()

            for offset in range(0, len(input_audio), FRAME_SAMPLES):
                if receiver_errors:
                    raise receiver_errors[0]

                chunk = input_audio[offset : offset + FRAME_SAMPLES]
                ws.send_binary(chunk.astype("<f4", copy=False).tobytes())

                deadline = started_at + (offset + len(chunk)) / SAMPLE_RATE
                time.sleep(max(0.0, deadline - time.monotonic()))

            ws.send("Done")
            receiver.join(timeout=self.close_timeout)
        finally:
            ws.close()
            if receiver is not None and receiver.is_alive():
                receiver.join(timeout=1.0)

        if receiver_errors:
            raise receiver_errors[0]

        with output_lock:
            return _build_timeline(list(segments), len(input_audio))

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
        system_out_audio = self.client(
            input_user_audio.astype(np.float32, copy=False)
        )
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
