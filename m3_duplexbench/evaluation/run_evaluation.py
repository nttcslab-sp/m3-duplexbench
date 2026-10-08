import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from m3_duplexbench.utils import (
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from .registry import (
    CATEGORIES,
    get_dimensions_for_event,
)
from .dimensions.content import load_batch_api_outputs


def build_evaluator(
    dimension: str,
    transcript_dir: str | None = None,
):
    if dimension == "smooth_turn_taking": 
        from .dimensions.timing import SmoothTurnTakingEvaluator
        return SmoothTurnTakingEvaluator()
    elif dimension == "pause_handling":
        from .dimensions.timing import PauseHandlingEvaluator
        return PauseHandlingEvaluator()
    elif dimension == "user_backchanneling":
        from .dimensions.timing import UserBackchannelingEvaluator
        return UserBackchannelingEvaluator()
    elif dimension == "user_barge_in":
        from .dimensions.timing import UserBargeInEvaluator
        return UserBargeInEvaluator()
    elif dimension == "instruction_following":
        from .dimensions.content import InstructionFollowingEvaluator
        return InstructionFollowingEvaluator()
    elif dimension == "response_relevance":
        from .dimensions.content import ResponseRelevanceEvaluator
        return ResponseRelevanceEvaluator()
    elif dimension == "contextual_consistency":
        from .dimensions.content import ContextualConsistencyEvaluator
        return ContextualConsistencyEvaluator(transcript_dir=transcript_dir)
    elif dimension == "qa_accuracy":
        from .dimensions.content import QAAccuracyEvaluator
        return QAAccuracyEvaluator(transcript_dir=transcript_dir)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

def build_evaluators(
    records: list[dict[str, Any]],
    domain: str,
    lang: str,
    categories: list[str],
    transcript_dir: str | None = None,
) -> dict[str, Any]:

    # Get unique events
    events = set()
    for record in records:
        events.add(record["event_type"])
 
    # Get unique dimensions
    dimensions = []
    for e in events:
        dimensions += get_dimensions_for_event(
            event_type=e,
            domain=domain,
            lang=lang,
            categories=categories,
        )
    dimensions = set(dimensions)

    # Get evaluators
    evaluators = {}
    for d in dimensions:
        evaluators[d] = build_evaluator(
            d, transcript_dir
        )

    return evaluators
    

def evaluate_event(
    evaluators: dict[str, Any],
    record: dict[str, Any],
    transcript: dict[str, Any],
    domain: str,
    lang: str,
    categories: list[str],
    eval_mode: str | None = None,
    batch_api_outputs: dict | None = None,
) -> tuple[dict, dict]:
    event_type = record["event_type"]

    dimensions = get_dimensions_for_event(
        event_type=event_type,
        domain=domain,
        lang=lang,
        categories=categories,
    )

    result = {
        "sample_id": record.get("sample_id"),
        "event_id": record.get("event_id"),
        "event_type": event_type,
        "results": {},
        "eval_info": {},
    }

    api_requests = {}

    for dim in dimensions:
        evaluator = evaluators[dim]

        # Prepare OpenAI Batch API request
        if eval_mode == "prepare" and evaluator.is_llmaj:
            request, _ = evaluator.prepare(
                metadata=record,
                transcript=transcript,
            )
            api_requests[dim] = [request]
            continue
        else:
            if evaluator.is_llmaj:
                batch_output = None
                if batch_api_outputs:
                    batch_output = batch_api_outputs.get(dim, None)
                eval_results = evaluator.evaluate(
                    metadata=record,
                    batch_output=batch_output,
                )
            else:
                eval_results = evaluator.evaluate(
                    metadata=record,
                    transcript=transcript,
                )

            if isinstance(eval_results, dict):
                result["results"][dim] = eval_results
            elif isinstance(eval_results, tuple):
                assert len(eval_results) == 2
                result["results"][dim] = eval_results[0]
                result["eval_info"][dim] = eval_results[1]

    return result, api_requests


def aggregate_results(
    event_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    value_rows = [] # one evaluation value per element
    for r in event_results:
        # e.g., 'results': {'smooth_turn_taking': {'tor': 1}}
        results = r["results"]
        for dimension, metrics in results.items():
            for metric, value in metrics.items():
                if value is None:
                    continue

                if isinstance(value, bool):
                    value = int(value)

                if isinstance(value, (int, float)):
                    value_rows.append({
                        "event_type": r["event_type"],
                        "dimension": dimension,
                        "metric": metric,
                        "value": float(value),
                    })
                    

    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in value_rows:
        key = (
            row["event_type"],
            row["dimension"],
            row["metric"],
        )
        groups[key].append(row["value"])

    summary_rows = []

    for key, values in sorted(groups.items()):
        event_type, dimension, metric = key
        n = len(values)
        mean = sum(values) / n if n > 0 else None

        row = {
            "event_type": event_type,
            "dimension": dimension,
            "metric": metric,
            "n": n,
            "mean": mean,
        }

        summary_rows.append(row)

    return summary_rows


def merge_dict(d1, d2):
    for k, v in d2.items():
        d1.setdefault(k, []).extend(v)
    return d1


def run_evaluation(
    root_dir: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    domain: str,
    lang: str,
    categories_str: str,
    transcript_key: str = "output_asr_path",
    eval_mode: str | None = None,
    transcript_dir: str | None = None,
) -> None:

    # evaluation categories. (e.g., ["timing", "content"]
    categories = [v.strip().lower() for v in categories_str.split(",")]
    for c in categories:
        if c not in CATEGORIES:
            raise ValueError(f"Unsupported cattegory: {c}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(Path(metadata_path)))

    event_results = []

    api_requests = {}

    evaluators = build_evaluators(
        records,
        domain=domain,
        lang=lang,
        categories=categories,
        transcript_dir=transcript_dir,
    )
    print("Evaluators:")
    for k, v in evaluators.items():
        print(f" - {k}: {type(v).__name__}")

    # Load Batch API results
    batch_api_outputs = {} # {"dim": {"event_id: item"}}
    batch_api_outputs_by_events = {} # {"event_id": {"dim: item"}}
    if eval_mode == "evaluate" and "content" in categories:
        for dim in evaluators.keys():
            batch_api_path = output_dir / dim / "batch_output.jsonl"
            if batch_api_path.exists():
                print(f"Loading Batch API output file: {batch_api_path}")
                batch_api_outputs[dim] = load_batch_api_outputs(batch_api_path)
            else:
                print(f"[WARN] File not found: {batch_api_path}. Skipping.")

        event_ids = set().union(*(v.keys() for v in batch_api_outputs.values()))
        batch_api_outputs_by_events = {
            event_id: {
                dim: batch_api_outputs[dim].get(event_id, None)
                for dim in batch_api_outputs.keys()
            }
            for event_id in event_ids
        }

    for record in records:
        # load ASR transcript
        transcript_path = Path(record[transcript_key])
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        transcript = read_json(transcript_path)

        event_id = record.get("event_id", None)
        event_batch_api_outputs = batch_api_outputs_by_events.get(event_id, None)

        # evaluate
        event_result, event_api_requests = evaluate_event(
            evaluators=evaluators,
            record=record,
            transcript=transcript,
            domain=domain,
            lang=lang,
            categories=categories,
            eval_mode=eval_mode,
            batch_api_outputs=event_batch_api_outputs,
        )
        if event_result["results"]:
            event_results.append(event_result)

        # update requests dict
        if len(event_api_requests) > 0:
            api_requests = merge_dict(api_requests, event_api_requests)

    # Save requests
    if eval_mode == "prepare" and len(api_requests) > 0:
        request_paths = []
        print("**********")
        print("Prepared content batch requests for LLMAJ using OpenAI Batch API.")
        for dim, batch in api_requests.items():
            request_path = output_dir / dim / "batch_requests.jsonl"
            request_paths.append(Path(request_path))
            write_jsonl(request_path, batch, avoid_overwrite=True)
            print(f" - {dim}: {request_path}")
        print()
        print("Next steps:")
        print("  1. Submit the batch requests (you need to set OPENAI_API_KEY):")
        for path in request_paths:
            print(
                f"     python m3_duplexbench/evaluation/utils/openai_batch.py submit "
                f"--request-jsonl {str(path)} --output-dir {str(path.parent)}"
            )
        print("  2. Wait requests and download results:")
        for path in request_paths:
            print(
                f"     python m3_duplexbench/evaluation/utils/openai_batch.py wait "
                f"--batch-info {str(path.parent / 'batch_info.json')} --output-dir {str(path.parent)}"
            )
        print(
            "  3. After the batch is completed, run this script again with "
            "--content-mode merge-batch."
        )
        print("**********")
        return

    # Output results
    event_results_path = output_dir / f"event_results_{categories_str}.jsonl"
    aggregate_path = output_dir / f"aggregate_results_{categories_str}.json"
    summary_path = output_dir / f"summary_{categories_str}.csv"

    # output event results
    write_jsonl(event_results_path, event_results, avoid_overwrite=True)

    # output aggregated results
    summary_rows = aggregate_results(event_results)
    aggregate = {}
    aggregate["summary"] = summary_rows
    aggregate["metadata"] = {
        "metadata_path": str(metadata_path),
        "event_results_path": str(event_results_path),
        "domain": domain,
        "language": lang,
        "categories": categories,
        "n_input_records": len(records),
        "n_evaluated_samples": len(event_results),
    }
    write_json(aggregate_path, aggregate, avoid_overwrite=True)

    # output csv
    write_csv(summary_path, summary_rows, avoid_overwrite=True)

    print("Evaluation finished.")
    print(f" - Evaluated samples: {len(event_results)}")
    print(f" - Sample results: {event_results_path}")
    print(f" - Aggregate results: {aggregate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation.")

    parser.add_argument(
        "--root-dir", type=str, default=None,
        help="Optional root directory to resolve relative transcript/audio paths.",
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Path to metadata jsonl. Usually metadata.asr.jsonl.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--domain", type=str, required=True, choices=["chat", "task"],
        help="Evaluation domain.",
    )
    parser.add_argument(
        "--lang", type=str, required=True, choices=["en", "ja"],
        help="Evaluation language.",
    )
    parser.add_argument(
        "--eval-category", type=str, default="timing",
        help="Comma-separated categories to evaluate.",
    )
    parser.add_argument(
        "--eval-mode", type=str, default="evaluate",
        choices=["prepare", "evaluate"],
        help=(
            "Evaluation mode. Setting `prepare` will create a batch file "
            "for OpenAI API (for content category metrics)."
        )
    )
    parser.add_argument(
        "--transcript-dir", type=str, default=None,
        help="Directory containing reference timestamped transcripts. ",
    )
    args = parser.parse_args()

    run_evaluation(
        root_dir=args.root_dir,
        metadata_path=args.metadata,
        output_dir=args.output,
        domain=args.domain,
        lang=args.lang,
        categories_str=args.eval_category,
        eval_mode=args.eval_mode,
        transcript_dir=args.transcript_dir,
    )
