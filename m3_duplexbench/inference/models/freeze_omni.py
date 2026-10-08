# # Before execution, set PYTHONPATH to Freeze-Omni root directory.
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import socketio

from .base import EventSample
from m3_duplexbench.inference.models.base import BaseInferenceModel
from m3_duplexbench.utils import resample_mono_np, pad_or_trim


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


@dataclass
class FreezeOmniClientConfig:
    server_url: Optional[str] = None
    ip: str = "127.0.0.1"
    port: str = "8000"
    chunk_size: int = 256
    tail_silence: float = 3.0
    poll_interval: Optional[float] = None
    output_idle_timeout: float = 4.0
    timeout: float = 480.0
    connect_timeout: float = 30.0
    connect_retries: int = 10
    connect_retry_interval: float = 2.0
    ssl_verify: bool = False
    prompt: Optional[str] = None
    teacher_text: Optional[str] = None

    def __post_init__(self) -> None:
        if self.server_url is None:
            self.server_url = f"https://{self.ip}:{self.port}"
        if self.poll_interval is None:
            self.poll_interval = self.chunk_size / INPUT_SAMPLE_RATE


def load_input_audio_from_array(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert float mono audio array to 16 kHz int16 audio."""
    import librosa

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    wav = audio.astype(np.float32)

    if sample_rate != INPUT_SAMPLE_RATE:
        wav = librosa.resample(
            wav,
            orig_sr=sample_rate,
            target_sr=INPUT_SAMPLE_RATE,
        )

    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16)


def iter_chunks(audio: np.ndarray, chunk_size: int):
    pad_len = int(np.ceil(len(audio) / chunk_size) * chunk_size)
    padded = np.zeros(pad_len, dtype=np.int16)
    padded[:len(audio)] = audio
    for start in range(0, len(padded), chunk_size):
        yield padded[start:start + chunk_size]


def audio_payload(chunk: np.ndarray) -> str:
    return json.dumps({
        "sample_rate": INPUT_SAMPLE_RATE,
        "audio": np.frombuffer(chunk.tobytes(), dtype=np.uint8).tolist(),
    })


def resample_int16_audio(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    import librosa

    audio = audio.astype(np.float32) / 32768.0
    if original_sample_rate != target_sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
        )
    return np.clip(audio, -1.0, 1.0)


def connect_with_retries(
    sio: socketio.Client,
    args: FreezeOmniClientConfig,
) -> None:
    last_error = None
    for attempt in range(1, args.connect_retries + 1):
        try:
            print(
                "Connecting to {} ({}/{})".format(
                    args.server_url,
                    attempt,
                    args.connect_retries,
                )
            )
            sio.connect(
                args.server_url,
                transports=["polling", "websocket"],
                wait_timeout=args.connect_timeout,
            )
            return
        except socketio.exceptions.ConnectionError as e:
            last_error = e
            if sio.connected:
                sio.disconnect()
            if attempt >= args.connect_retries:
                break
            print(
                "Connection failed: {}. Retrying in {:.1f}s.".format(
                    e,
                    args.connect_retry_interval,
                )
            )
            time.sleep(args.connect_retry_interval)

    raise socketio.exceptions.ConnectionError(
        "Failed to connect to {} after {} attempts. Last error: {}".format(
            args.server_url,
            args.connect_retries,
            last_error,
        )
    )


def build_conversation_arrays(
    input_audio: np.ndarray,
    user_offset_samples: int,
    output_segments: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-length user/model waveforms on a 24 kHz timeline."""
    user_audio = resample_int16_audio(
        input_audio,
        INPUT_SAMPLE_RATE,
        OUTPUT_SAMPLE_RATE,
    )

    # Fixed output length.
    fixed_len = user_offset_samples + len(user_audio)

    user_timeline = np.zeros(fixed_len, dtype=np.float32)
    model_timeline = np.zeros(fixed_len, dtype=np.float32)

    user_end = user_offset_samples + len(user_audio)
    user_timeline[user_offset_samples:user_end] = user_audio

    for offset, chunk in output_segments:
        if offset >= fixed_len:
            # Model output starts after the evaluation window
            continue

        model_audio = chunk.astype(np.float32) / 32768.0
        end = min(offset + len(model_audio), fixed_len)
        valid_len = end - offset

        if valid_len > 0:
            model_timeline[offset:end] += model_audio[:valid_len]

    user_timeline = np.clip(user_timeline, -1.0, 1.0)
    model_timeline = np.clip(model_timeline, -1.0, 1.0)

    return user_timeline.astype(np.float32), model_timeline.astype(np.float32)


class FreezeOmniModel:
    """Freeze-Omni client wrapper."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        ip: str = "127.0.0.1",
        port: str = "8000",
        chunk_size: int = 256,
        tail_silence: float = 0.0,
        poll_interval: Optional[float] = None,
        output_idle_timeout: float = 4.0,
        timeout: float = 480.0,
        connect_timeout: float = 30.0,
        connect_retries: int = 10,
        connect_retry_interval: float = 2.0,
        ssl_verify: bool = False,
        prompt: Optional[str] = None,
        teacher_text: Optional[str] = None,
        use_context: bool = True,
        use_teacher_forcing: bool = False,
    ) -> None:
        self.args = FreezeOmniClientConfig(
            server_url=server_url,
            ip=ip,
            port=port,
            chunk_size=chunk_size,
            tail_silence=tail_silence,
            poll_interval=poll_interval,
            output_idle_timeout=output_idle_timeout,
            timeout=timeout,
            connect_timeout=connect_timeout,
            connect_retries=connect_retries,
            connect_retry_interval=connect_retry_interval,
            ssl_verify=ssl_verify,
            prompt=prompt,
            teacher_text=teacher_text,
        )
        self.no_output_timeout = 20.0

        if not self.args.ssl_verify:
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

        self._context_mode = "server_client_no_audio_context"

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
        return "freeze_omni"

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    @property
    def context_mode(self) -> str:
        return self._context_mode

    def _run_client(self, input_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        args = self.args
    
        teacher_text = args.teacher_text
    
        output_chunks = []
        output_segments = []
        output_lock = threading.Lock()
        output_started = threading.Event()
        stop_requested = threading.Event()
        prompt_success = threading.Event()
        teacher_done = threading.Event()
    
        last_audio_time = {"value": None}
        conversation_started_at = {"value": None}
        user_offset_samples = {"value": 0}
        model_playhead_samples = {"value": None}
        error_message = {"value": None}
    
        sio = socketio.Client(ssl_verify=args.ssl_verify, reconnection=False)
    
        def check_server_error() -> None:
            if error_message["value"] is not None:
                raise RuntimeError(error_message["value"])
    
        @sio.event
        def connect():
            print("Connected to {}".format(args.server_url))
    
        @sio.event
        def disconnect():
            print("Disconnected")
    
        @sio.on("audio")
        def on_audio(data):
            chunk = np.frombuffer(data, dtype=np.int16).copy()
            if len(chunk) == 0:
                return
    
            received_at = time.monotonic()
    
            if conversation_started_at["value"] is None:
                received_offset = 0
            else:
                received_offset = int(
                    (received_at - conversation_started_at["value"])
                    * OUTPUT_SAMPLE_RATE
                )
    
            with output_lock:
                if model_playhead_samples["value"] is None:
                    output_offset = received_offset
                else:
                    output_offset = max(
                        received_offset,
                        model_playhead_samples["value"],
                    )
    
                model_playhead_samples["value"] = output_offset + len(chunk)
    
                output_chunks.append(chunk)
                output_segments.append((output_offset, chunk))
                last_audio_time["value"] = received_at
    
            output_started.set()
            print("Received {} output samples".format(len(chunk)))
    
        @sio.on("stop_tts")
        def on_stop_tts():
            print("Received stop_tts")
            stop_requested.set()
    
        @sio.on("too_many_users")
        def on_too_many_users():
            error_message["value"] = (
                "Server rejected the connection because too many users are connected."
            )
            stop_requested.set()
    
        @sio.on("out_time")
        def on_out_time():
            error_message["value"] = (
                "Server closed the connection because the session timed out."
            )
            stop_requested.set()
    
        @sio.on("prompt_success")
        def on_prompt_success():
            prompt_success.set()
            print("Prompt set successfully")
    
        @sio.on("teacher_force_done")
        def on_teacher_force_done():
            teacher_done.set()
            print("Teacher-forced speech completed")
    
        @sio.on("teacher_force_error")
        def on_teacher_force_error(message):
            error_message["value"] = "Teacher forcing failed: {}".format(message)
            stop_requested.set()
    
        connect_with_retries(sio, args)
        started_at = time.monotonic()

        def poll_for_output(
            phase: str,
            phase_start_count: int,
            done_event=None,
            allow_no_output: bool = False,
        ) -> None:
            silent_chunk = np.zeros(args.chunk_size, dtype=np.int16)
            phase_started_at = time.monotonic()
        
            while True:
                check_server_error()
                now = time.monotonic()
        
                if now - started_at > args.timeout:
                    if allow_no_output:
                        print(
                            f"[FreezeOmni] Global timeout while waiting for {phase}; "
                            "continue with silence."
                        )
                        break
                    raise TimeoutError(f"Timed out waiting for {phase} output.")
        
                with output_lock:
                    phase_has_output = len(output_chunks) > phase_start_count
                    last_time = last_audio_time["value"]
        
                # If server explicitly stops TTS, do not keep waiting forever.
                if stop_requested.is_set() and allow_no_output:
                    print(
                        f"[FreezeOmni] stop_tts received during {phase}; "
                        "continue with silence."
                    )
                    break
        
                if phase_has_output and stop_requested.is_set():
                    break
        
                if phase_has_output and last_time is not None:
                    if done_event is None and now - last_time >= args.output_idle_timeout:
                        break
        
                    if (
                        done_event is not None
                        and done_event.is_set()
                        and now - last_time >= args.output_idle_timeout
                    ):
                        break
        
                # If no output ever arrives, treat it as a no-response case.
                if not phase_has_output and allow_no_output:
                    if now - phase_started_at >= args.no_output_timeout:
                        print(
                            f"[FreezeOmni] No {phase} output for "
                            f"{args.no_output_timeout:.1f}s; continue with silence."
                        )
                        break
        
                time.sleep(args.poll_interval)
    
        try:
            if args.prompt is not None:
                sio.emit("prompt_text", args.prompt)
                if not prompt_success.wait(timeout=10):
                    raise TimeoutError("Timed out waiting for prompt_success.")
    
            # Start the conversation timeline before optional teacher-forced speech.
            conversation_started_at["value"] = time.monotonic()
    
            if teacher_text is not None:
                if len(teacher_text) == 0:
                    raise ValueError("Teacher-forcing text is empty.")
    
                with output_lock:
                    teacher_start_count = len(output_chunks)
    
                print("Requesting teacher-forced speech")
                sio.emit("teacher_force_text", teacher_text)
    
                poll_for_output(
                    "teacher-forced speech",
                    teacher_start_count,
                    teacher_done,
                )
    
                # User audio begins after teacher-forced speech.
                user_offset_samples["value"] = int(
                    (time.monotonic() - conversation_started_at["value"])
                    * OUTPUT_SAMPLE_RATE
                )
                stop_requested.clear()
    
            with output_lock:
                response_start_count = len(output_chunks)
            
            sio.emit("recording-started")
            stop_requested.clear()
            
            print("Sending {} input samples".format(len(input_audio)))
            for chunk in iter_chunks(input_audio, args.chunk_size):
                check_server_error()
                # sio.emit("audio", audio_payload(chunk))
                if not sio.connected:
                    print("[FreezeOmni] socket disconnected while sending input audio; stop sending.")
                    break
                try:
                    sio.emit("audio", audio_payload(chunk))
                except socketio.exceptions.BadNamespaceError:
                    print("[FreezeOmni] namespace disconnected while sending input audio; stop sending.")
                    break
                time.sleep(args.poll_interval)
            
                if time.monotonic() - started_at > args.timeout:
                    raise TimeoutError("Timed out while sending input audio.")
            
            print("Waiting for server output")
            poll_for_output(
                "server response",
                response_start_count,
                allow_no_output=True,
            )
    
        finally:
            if sio.connected:
                sio.disconnect()
                print("disconnect() called")
                time.sleep(6)
    
        with output_lock:
            segments = list(output_segments)
            user_offset = int(user_offset_samples["value"])
        
        user_timeline, model_timeline = build_conversation_arrays(
            input_audio=input_audio,
            user_offset_samples=user_offset,
            output_segments=segments,
        )
        
        return user_timeline, model_timeline

    def generate(
        self,
        context_audio: np.ndarray,
        pre_event_audio: np.ndarray | None,
        eval_window_audio: np.ndarray,
        sample_rate: int,
        event: EventSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray|None]:
        assert context_audio.ndim == 2

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

            # resampling
            input_user_audio = resample_mono_np(input_user_audio, sample_rate, INPUT_SAMPLE_RATE)

        # Convert float to int16 for freeze-omni
        input_user_audio = load_input_audio_from_array(input_user_audio, INPUT_SAMPLE_RATE)
    
        # Get aligned 24 kHz 2-channel audio
        user_out_audio, system_out_audio = self._run_client(input_user_audio)
        # print(f"user/model_timeline: {user_timeline.shape[0] / OUTPUT_SAMPLE_RATE} (s)")

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
                pre_and_eval_user_audio, sample_rate, self.sample_rate
            )

        return pre_and_eval_user_audio, pre_and_eval_system_out_audio, context_system_out_audio
