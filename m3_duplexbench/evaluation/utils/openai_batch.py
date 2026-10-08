import argparse
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def object_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def extract_chat_completion_text(batch_output: dict) -> str:
    """Extract assistant message text from one Batch API output item."""
    response = batch_output.get("response", {})
    status_code = response.get("status_code")

    if status_code != 200:
        raise ValueError(
            f"Batch request failed: status_code={status_code}, "
            f"body={response.get('body')}"
        )

    body = response.get("body", {})
    choices = body.get("choices", [])

    if not choices:
        raise ValueError(f"No choices found in batch output: {batch_output}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    return str(content).strip()


def submit_batch(
    *,
    request_jsonl: str | Path,
    output_dir: str | Path,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
) -> None:
    """Upload request JSONL and create an OpenAI Batch job."""
    client = OpenAI()

    request_jsonl = Path(request_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not request_jsonl.exists():
        raise FileNotFoundError(f"Request JSONL not found: {request_jsonl}")

    print(f"Uploading request file: {request_jsonl}")
    with request_jsonl.open("rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="batch",
        )

    print(f"Uploaded file id: {uploaded_file.id}")

    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )

    batch_info = object_to_dict(batch)
    batch_info["request_jsonl"] = str(request_jsonl)
    batch_info["uploaded_file_id"] = uploaded_file.id
    batch_info["endpoint"] = endpoint
    batch_info["completion_window"] = completion_window

    info_path = output_dir / "batch_info.json"
    write_json(info_path, batch_info)

    print(f"Batch id: {batch.id}")
    print(f"Batch info written to: {info_path}")


def wait_for_batch(
    *,
    batch_id: str,
    polling_interval: int = 60,
) -> Any:
    """Wait until a batch reaches a terminal status."""
    client = OpenAI()

    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")

        if batch.status == "completed":
            if not batch.output_file_id:
                raise RuntimeError(
                    f"Batch completed but output_file_id is missing. "
                    f"error_file_id={batch.error_file_id}"
                )
            return batch

        if batch.status == "failed":
            raise RuntimeError(f"Batch failed: {batch.errors}")

        if batch.status in {"cancelled", "expired"}:
            raise RuntimeError(f"Batch ended with status: {batch.status}")

        time.sleep(polling_interval)


def download_file(
    *,
    file_id: str,
    output_path: str | Path,
) -> None:
    """Download an OpenAI file to output_path."""
    client = OpenAI()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = client.files.content(file_id)

    # openai-python returns a response-like object with write_to_file().
    if hasattr(content, "write_to_file"):
        content.write_to_file(str(output_path))
    else:
        data = content.read() if hasattr(content, "read") else content
        with output_path.open("wb") as f:
            f.write(data)

    print(f"Downloaded file {file_id} to: {output_path}")


def wait_and_download(
    *,
    batch_id: str,
    output_dir: str | Path,
    polling_interval: int = 60,
) -> None:
    """Wait for completion and download output/error files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = wait_for_batch(
        batch_id=batch_id,
        polling_interval=polling_interval,
    )
    batch_info = object_to_dict(batch)

    write_json(output_dir / "batch_final.json", batch_info)

    if batch.output_file_id:
        download_file(
            file_id=batch.output_file_id,
            output_path=output_dir / "batch_output.jsonl",
        )

    if batch.error_file_id:
        download_file(
            file_id=batch.error_file_id,
            output_path=output_dir / "batch_errors.jsonl",
        )


def status(
    batch_id: str,
    output_dir: str | Path | None = None,
) -> None:
    """Print current batch status."""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    batch_info = object_to_dict(batch)

    print(json.dumps(batch_info, ensure_ascii=False, indent=2))

    if output_dir is not None:
        write_json(Path(output_dir) / "batch_status.json", batch_info)


def get_batch_id_from_info(path: str | Path) -> str:
    info = read_json(path)
    if "id" not in info:
        raise KeyError(f"'id' is missing in batch info: {path}")
    return str(info["id"])


def main():
    parser = argparse.ArgumentParser(
        description="Submit, monitor, and download OpenAI Batch API jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument(
        "--request-jsonl",
        type=str,
        required=True,
        help="Path to Batch API request JSONL.",
    )
    submit_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save batch_info.json.",
    )
    submit_parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="Batch endpoint. Default: /v1/chat/completions.",
    )
    submit_parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="Batch completion window. Default: 24h.",
    )

    status_parser = subparsers.add_parser("status")
    status_group = status_parser.add_mutually_exclusive_group(required=True)
    status_group.add_argument("--batch-id", type=str)
    status_group.add_argument("--batch-info", type=str)
    status_parser.add_argument("--output-dir", type=str, default=None)

    wait_parser = subparsers.add_parser("wait")
    wait_group = wait_parser.add_mutually_exclusive_group(required=True)
    wait_group.add_argument("--batch-id", type=str)
    wait_group.add_argument("--batch-info", type=str)
    wait_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save batch_output.jsonl and batch_final.json.",
    )
    wait_parser.add_argument(
        "--polling-interval",
        type=int,
        default=60,
        help="Polling interval in seconds. Default: 60.",
    )

    download_parser = subparsers.add_parser("download-file")
    download_parser.add_argument("--file-id", type=str, required=True)
    download_parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    if args.command == "submit":
        submit_batch(
            request_jsonl=args.request_jsonl,
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            completion_window=args.completion_window,
        )

    elif args.command == "status":
        batch_id = args.batch_id or get_batch_id_from_info(args.batch_info)
        status(
            batch_id=batch_id,
            output_dir=args.output_dir,
        )

    elif args.command == "wait":
        batch_id = args.batch_id or get_batch_id_from_info(args.batch_info)
        wait_and_download(
            batch_id=batch_id,
            output_dir=args.output_dir,
            polling_interval=args.polling_interval,
        )

    elif args.command == "download-file":
        download_file(
            file_id=args.file_id,
            output_path=args.output_path,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
