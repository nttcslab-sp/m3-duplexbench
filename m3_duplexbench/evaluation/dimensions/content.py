import re
import json
from pathlib import Path

from m3_duplexbench.evaluation.prompts import (
    RESPONSE_RELEVANCE_SYSTEM_PROMPT,
    QA_ACCURACY_SYSTEM_PROMPT,
    INSTRUCTION_FOLLOWING_SYSTEM_PROMPT_BINARY,
    CONTEXTUAL_CONSISTENCY_SYSTEM_PROMPT,
)


def load_batch_api_outputs(
    batch_output_path: str | Path,
) -> dict[str, dict]:
    """Load Batch API output JSONL."""
    outputs = {}

    with Path(batch_output_path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            custom_id = item.get("custom_id")
            event_id, _ = parse_custom_id(custom_id)
            outputs[event_id] = item

    return outputs


def parse_custom_id(custom_id: str) -> tuple[str, str]:
    """Parse custom_id formatted as {event_id}::{dimension}."""
    if "::" not in custom_id:
        raise ValueError(f"Invalid custom_id format: {custom_id}")

    event_id, dimension = custom_id.rsplit("::", 1)
    return event_id, dimension


def build_batch_api_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "minimal",
) -> dict:
    """Build one OpenAI Batch API request for Chat Completions."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning_effort": reasoning_effort,
        },
    }


def load_word_transcript(
    transcript_dir: str | Path,
    sample_id: str,
) -> list[dict]:
    """Load timestamped word transcript.
    Format:
        channel start end word
    """
    path = Path(transcript_dir) / f"{sample_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Reference transcript not found: {path}")

    words = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue

            ch, start, end, word = parts

            words.append({
                    "channel": int(ch),
                    "start": float(start),
                    "end": float(end),
                    "word": word,
                    "line_idx": line_idx,
            })

    return words


def build_dialogue_context(
    words: list[dict],
    end_time: float,
    user_channel: int,
    model_channel: int,
    max_turns: int = -1,
) -> str:
    """Build dialogue context up to end_time from word-level transcript."""
    words_by_channel: dict[int, list[dict]] = {}

    speaker_by_channel = {
        str(user_channel): "User",
        str(model_channel): "Model",
    }

    for w in words:
        if float(w["start"]) >= end_time:
            continue
        ch = int(w["channel"])
        words_by_channel.setdefault(ch, []).append(w)

    utterances: list[dict] = []
    for ch, ch_words in words_by_channel.items():
        ch_words = sorted(
            ch_words,
            key=lambda x: (float(x["start"]), float(x["end"]), int(x.get("line_idx", 0))),
        )

        current_words = []
        utt_start = None
        utt_end = None

        for w in ch_words:
            start = float(w["start"])
            end = float(w["end"])
            word = str(w["word"])

            if utt_start is None:
                utt_start = start
                utt_end = end
                current_words = [word]
                continue

            assert utt_end is not None
            if start - utt_end == 0.0:
                current_words.append(word)
                utt_end = end
            else:
                utterances.append({
                    "channel": ch,
                    "start": utt_start,
                    "end": utt_end,
                    "text": " ".join(current_words),
                })
                utt_start = start
                utt_end = end
                current_words = [word]

        if utt_start is not None and utt_end is not None and current_words:
            utterances.append({
                "channel": ch,
                "start": utt_start,
                "end": utt_end,
                "text": " ".join(current_words),
            })

    utterances = sorted(
        utterances,
        key=lambda x: (float(x["start"]), float(x["end"]), int(x["channel"])),
    )

    if max_turns > 0:
        utterances = utterances[-max_turns:]

    return "\n".join(
        f"{speaker_by_channel[str(u['channel'])]}: {u['text']}"
        for u in utterances
    ).strip()


def extract_reference_answer(
    transcript_dir: str | Path,
    sample_id: str,
    system_channel: int,
    answer_start_time: float,
) -> tuple[str, list[dict]]:
    """Extract reference answer from a timestamped transcript."""
    words = load_word_transcript(
        transcript_dir=transcript_dir,
        sample_id=sample_id,
    )
    words = sorted(words, key=lambda w: (w["start"], w["end"], w["channel"], w["line_idx"]))

    start_idx = None
    for i, w in enumerate(words):
        if w["channel"] != system_channel:
            continue
        if w["start"] - answer_start_time == 0.0:
            start_idx = i
            break

    if start_idx is None:
        return "", []

    reference_words = [words[start_idx]]

    prev_end = words[start_idx]["end"]
    for w in words[start_idx + 1:]:
        if w["channel"] != system_channel:
            continue

        if not w["start"] - prev_end == 0.0:
            break

        reference_words.append(w)
        prev_end = w["end"]

    reference_answer = " ".join(w["word"] for w in reference_words).strip()
    return reference_answer, reference_words


def load_reference_transcript(
    transcript_dir: str | Path,
    sample_id: str,
) -> str:
    """Load reference timestamped transcript for a dialogue sample.

    Expected path:
        transcript_dir / f"{sample_id}.txt"

    The extraction of the exact reference answer is intentionally left to
    QAAccuracyEvaluator for later refinement. For now, this returns the full
    transcript text.
    """
    path = Path(transcript_dir) / f"{sample_id}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Reference transcript not found: {path}")

    return path.read_text(encoding="utf-8").strip()


def concat_channel_text(
    transcript: dict,
    channel: int,
) -> str:
    """Concatenate segment texts from a given channel."""
    texts = []

    for segment in transcript.get("segments", []):
        if int(segment.get("channel", -1)) != channel:
            continue

        text = str(segment.get("text", "")).strip()
        if text:
            texts.append(text)

    return " ".join(texts).strip()


def extract_chat_completion_text(batch_output: dict) -> str:
    """Extract assistant message text from one Batch API output item."""
    response = batch_output.get("response", {})
    status_code = response.get("status_code")

    body = response.get("body", {})
    choices = body.get("choices", [])

    if not choices:
        raise ValueError(f"No choices found in batch output: {batch_output}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    return str(content).strip()


def parse_score(
    text: str,
    valid_scores: set[int],
) -> int:
    match = re.search(r"\b[0-9]\b$", text.strip())

    if match is None:
        raise ValueError(f"Could not parse score from output: {text!r}")

    score = int(match.group(0))

    if score not in valid_scores:
        raise ValueError(f"Invalid score: {score}")

    return score


def evaluate_batch_output_score(
    event_id: str,
    batch_output: dict | None,
    dimension: str,
    valid_scores: set[int],
) -> tuple[dict, dict]:
    """Evaluate one Batch API output item."""
    try:
        if batch_output:
            custom_id = batch_output["custom_id"]
            _event_id, _dimension = parse_custom_id(custom_id)
            assert event_id == _event_id
            assert dimension == _dimension

            raw_output = extract_chat_completion_text(batch_output)
            score = parse_score(raw_output, valid_scores)

            results = {
                "gpt-score ↑": score,
            }
            info = {
                "raw_output": raw_output,
                "parse_error": None,
            }
        else:
            raise Exception

    except Exception as e:
        results = {
            "score ↑": None,
        }
        info = {
            "raw_output": None,
            "parse_error": str(e),
        }

    return results, info


class InstructionFollowingEvaluator:
    name = "instruction_following"
    category = "content"
    is_llmaj = True

    def __init__(
        self,
        model: str = "gpt-5-nano",
        user_channel: int = 0,
        model_channel: int = 1,
        reasoning_effort: str = "minimal",
    ):
        self.model = model
        self.user_channel = user_channel
        self.model_channel = model_channel
        self.reasoning_effort = reasoning_effort

        self.system_prompt = INSTRUCTION_FOLLOWING_SYSTEM_PROMPT_BINARY

    def build_prompt(
        self,
        transcript: dict,
    ) -> tuple[str, str, str]:
        user_question = concat_channel_text(
            transcript,
            channel=self.user_channel,
        )
        model_answer = concat_channel_text(
            transcript,
            channel=self.model_channel,
        )

        user_prompt = (
            f"User Question: {user_question}\n"
            f"Model Answer: {model_answer}\n"
            "Your output (0 or 1): "
        )

        return user_prompt, user_question, model_answer

    def prepare(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:
        """Build a Batch API request."""
        event_id = metadata["event_id"]
        custom_id = f"{event_id}::{self.name}"

        user_prompt, user_question, model_answer = self.build_prompt(transcript)

        request = build_batch_api_request(
            custom_id=custom_id,
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            reasoning_effort=self.reasoning_effort,
        )

        info = {
            "user_question": user_question,
            "model_answer": model_answer,
        }

        return request, info

    def evaluate(
        self,
        metadata: dict,
        batch_output: dict | None,
    ) -> tuple[dict, dict]:
        return evaluate_batch_output_score(
            event_id=metadata["event_id"],
            batch_output=batch_output,
            dimension=self.name,
            valid_scores={0, 1},
        )


class ResponseRelevanceEvaluator(InstructionFollowingEvaluator):
    name = "response_relevance"
    category = "content"
    is_llmaj = True

    def __init__(
        self,
        model: str = "gpt-5-nano",
        user_channel: int = 0,
        model_channel: int = 1,
        reasoning_effort: str = "minimal",
    ):
        super().__init__(
            model=model,
            user_channel=user_channel,
            model_channel=model_channel,
            reasoning_effort=reasoning_effort,
        )
        self.system_prompt = RESPONSE_RELEVANCE_SYSTEM_PROMPT

    def build_prompt(
        self,
        transcript: dict,
    ) -> tuple[str, str, str]:
        user_utterance = concat_channel_text(
            transcript,
            channel=self.user_channel,
        )
        model_answer = concat_channel_text(
            transcript,
            channel=self.model_channel,
        )

        user_prompt = (
            f"User Utterance: {user_utterance}\n"
            f"Model Response: {model_answer}\n"
            "Your output (0, 1, or 2):"
        )

        return user_prompt, user_utterance, model_answer

    def evaluate(
        self,
        metadata: dict,
        batch_output: dict | None,
    ) -> tuple[dict, dict]:
        return evaluate_batch_output_score(
            event_id=metadata["event_id"],
            batch_output=batch_output,
            dimension=self.name,
            valid_scores={0, 1, 2},
        )


class QAAccuracyEvaluator:
    name = "qa_accuracy"
    category = "content"
    is_llmaj = True

    def __init__(
        self,
        transcript_dir: str | Path | None,
        model: str = "gpt-5-nano",
        user_channel: int = 0,
        model_channel: int = 1,
        reasoning_effort: str = "minimal",
    ):
        self.model = model
        self.user_channel = user_channel
        self.model_channel = model_channel
        self.reasoning_effort = reasoning_effort
        if transcript_dir:
            self.transcript_dir = Path(transcript_dir)
            if not self.transcript_dir.exists():
                raise FileNotFoundError(f"{transcript_dir} does not exist.")

        self.system_prompt = QA_ACCURACY_SYSTEM_PROMPT

    def build_prompt(
        self,
        transcript: dict,
        reference_answer: str,
    ) -> tuple[str, str, str, str]:
        user_question = concat_channel_text(
            transcript,
            channel=self.user_channel,
        )
        model_answer = concat_channel_text(
            transcript,
            channel=self.model_channel,
        )

        user_prompt = (
            f"User Question: {user_question}\n"
            f"Model Answer: {model_answer}\n"
            f"Reference Answer: {reference_answer}\n"
            "Your output (0, 1, or 2): "
        )

        return user_prompt, user_question, model_answer, reference_answer

    def prepare(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:
        """Build a Batch API request."""
        event_id = metadata["event_id"]
        sample_id = metadata["sample_id"]
        orig_system_channel = int(metadata["system_channel"])
        answer_start_time = float(metadata["event_end"])

        custom_id = f"{event_id}::{self.name}"

        reference_answer, reference_words = extract_reference_answer(
            transcript_dir=self.transcript_dir,
            sample_id=sample_id,
            system_channel=orig_system_channel,
            answer_start_time=answer_start_time,
        )

        user_prompt, user_question, model_answer, reference_answer = self.build_prompt(
            transcript=transcript,
            reference_answer=reference_answer,
        )

        request = build_batch_api_request(
            custom_id=custom_id,
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            reasoning_effort=self.reasoning_effort,
        )

        info = {
            "user_question": user_question,
            "model_answer": model_answer,
            "reference_answer": reference_answer,
            "reference_transcript_path": str(self.transcript_dir / f"{sample_id}.txt"),
        }

        return request, info

    def evaluate(
        self,
        metadata: dict,
        batch_output: dict | None,
    ) -> tuple[dict, dict]:
        return evaluate_batch_output_score(
            event_id=metadata["event_id"],
            batch_output=batch_output,
            dimension=self.name,
            valid_scores={0, 1, 2},
        )

class ContextualConsistencyEvaluator(QAAccuracyEvaluator):
    name = "contextual_consistency"
    category = "content"
    is_llmaj = True

    def __init__(
        self,
        transcript_dir: str | Path | None,
        model: str = "gpt-5-nano",
        user_channel: int = 0,
        model_channel: int = 1,
        reasoning_effort: str = "minimal",
        context_num_turns: int = -1,
    ):
        super().__init__(
            model=model,
            transcript_dir=transcript_dir,
            user_channel=user_channel,
            model_channel=model_channel,
            reasoning_effort=reasoning_effort,
        )

        self.context_num_turns = context_num_turns
        self.system_prompt = CONTEXTUAL_CONSISTENCY_SYSTEM_PROMPT

    def build_prompt(
        self,
        transcript: dict,
        dialogue_context: str,
    ) -> tuple[str, str]:
        model_response = concat_channel_text(
            transcript,
            channel=self.model_channel,
        )

        user_prompt = (
            f"Dialogue Context:\n{dialogue_context}\n"
            f"Model Response: {model_response}\n"
            "Your output (0, 1, or 2): "
        )

        return user_prompt, model_response

    def prepare(
        self,
        metadata: dict,
        transcript: dict,
    ) -> tuple[dict, dict]:

        event_id = metadata["event_id"]
        sample_id = metadata["sample_id"]
        orig_user_channel = int(metadata["user_channel"])
        orig_system_channel = int(metadata["system_channel"])
        context_end_time = float(metadata["event_start"])
        custom_id = f"{event_id}::{self.name}"

        words = load_word_transcript(
            transcript_dir=self.transcript_dir,
            sample_id=sample_id,
        )

        dialogue_context = build_dialogue_context(
            words=words,
            end_time=context_end_time,
            user_channel=orig_user_channel,
            model_channel=orig_system_channel,
            max_turns=self.context_num_turns,
        )

        user_prompt, model_response = self.build_prompt(
            transcript=transcript,
            dialogue_context=dialogue_context,
        )

        request = build_batch_api_request(
            custom_id=custom_id,
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            reasoning_effort=self.reasoning_effort,
        )

        info = {
            "dialogue_context": dialogue_context,
            "model_response": model_response,
            "context_num_turns": self.context_num_turns,
            "reference_transcript_path": str(self.transcript_dir / f"{sample_id}.txt"),
        }

        return request, info

    def evaluate(
        self,
        metadata: dict,
        batch_output: dict | None,
    ) -> tuple[dict, dict]:
        return evaluate_batch_output_score(
            event_id=metadata["event_id"],
            batch_output=batch_output,
            dimension=self.name,
            valid_scores={0, 1, 2},
        )
