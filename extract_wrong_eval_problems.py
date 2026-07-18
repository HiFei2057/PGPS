#!/usr/bin/env python3
"""Extract wrong problems from a Qwen PGPS eval_results.json file.

The eval log stores one entry per problem in ``details`` and can contain many
very long candidate responses per entry.  This script streams the ``details``
array so it does not need to load the whole log into memory, and it writes only
compact candidate summaries by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


IMPORTANT_FIELDS = (
    "id",
    "diagram",
    "text",
    "answer",
    "choices",
    "ground_truth_expression",
    "number_values",
    "metrics",
)

CANDIDATE_SUMMARY_FIELDS = (
    "rank",
    "answer",
    "program_value",
    "value",
    "value_source",
    "parse_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the id and useful metadata for problems marked wrong in "
            "a PGPS eval_results.json file."
        )
    )
    parser.add_argument("eval_results", type=Path, help="Path to eval_results.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--metric",
        default="completion",
        help=(
            "Metric used to decide wrongness: completion, choice, top3, any, "
            "or all. Default: completion."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidate summaries to include per problem. Default: all.",
    )
    parser.add_argument(
        "--response-chars",
        type=int,
        default=0,
        help=(
            "Include this many characters of each raw candidate response as "
            "response_preview. Default: 0, which omits raw responses."
        ),
    )
    parser.add_argument(
        "--max-program-tokens",
        type=int,
        default=80,
        help=(
            "Maximum program_tokens entries to include per candidate. Use 0 to "
            "omit tokens, or -1 for no limit. Default: 80."
        ),
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Output only the wrong problem ids.",
    )
    return parser.parse_args()


def iter_detail_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Yield objects from the top-level ``details`` array without loading all JSON.

    This assumes the eval output shape produced by ``run_eval_qwen35.py``:
    ``{"summary": ..., "args": ..., "details": [ ... ]}``.
    """

    decoder = json.JSONDecoder()
    marker = '"details"'
    buffer = ""
    eof = False

    with path.open("r", encoding="utf-8") as fp:
        while marker not in buffer:
            chunk = fp.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find top-level {marker} array in {path}")
            buffer += chunk
            if len(buffer) > len(marker) + chunk_size:
                buffer = buffer[-(len(marker) + chunk_size) :]

        marker_index = buffer.index(marker) + len(marker)
        buffer = buffer[marker_index:]

        while "[" not in buffer:
            chunk = fp.read(chunk_size)
            if not chunk:
                raise ValueError(f"Found {marker}, but not its array in {path}")
            buffer += chunk

        array_start = buffer.index("[") + 1
        buffer = buffer[array_start:]

        while True:
            stripped = buffer.lstrip()
            buffer = stripped

            while buffer.startswith(","):
                buffer = buffer[1:].lstrip()

            if buffer.startswith("]"):
                return

            try:
                detail, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = fp.read(chunk_size)
                if not chunk:
                    eof = True
                buffer += chunk
                continue

            if not isinstance(detail, dict):
                raise ValueError(f"Expected a detail object in {path}, got {type(detail).__name__}")

            yield detail
            buffer = buffer[end:]


def wrong_metrics_for(detail: Dict[str, Any], metric: str) -> Tuple[bool, List[str]]:
    metrics = detail.get("metrics") or {}

    if metric == "any":
        wrong = [name for name, result in metrics.items() if not result.get("correct", False)]
        return bool(wrong), wrong

    if metric == "all":
        wrong = [name for name, result in metrics.items() if not result.get("correct", False)]
        return bool(metrics) and len(wrong) == len(metrics), wrong

    result = metrics.get(metric)
    if result is None:
        return True, [metric]
    if not result.get("correct", False):
        return True, [metric]
    return False, []


def compact_response_preview(response: Any, max_chars: int) -> Optional[str]:
    if max_chars <= 0 or not isinstance(response, str):
        return None

    compact = " ".join(response.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def summarize_candidates(
    candidates: Iterable[Dict[str, Any]],
    max_candidates: Optional[int],
    response_chars: int,
    max_program_tokens: int,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        if max_candidates is not None and index >= max_candidates:
            break

        summary = {
            field: candidate.get(field)
            for field in CANDIDATE_SUMMARY_FIELDS
            if field in candidate
        }
        program_tokens = candidate.get("program_tokens")
        if isinstance(program_tokens, list):
            summary["program_token_count"] = len(program_tokens)
            if max_program_tokens != 0:
                if max_program_tokens < 0:
                    summary["program_tokens"] = program_tokens
                else:
                    summary["program_tokens"] = program_tokens[:max_program_tokens]
                    if len(program_tokens) > max_program_tokens:
                        summary["program_tokens_truncated"] = True

        preview = compact_response_preview(candidate.get("response"), response_chars)
        if preview is not None:
            summary["response_preview"] = preview
        summaries.append(summary)

    return summaries


def build_record(
    detail: Dict[str, Any],
    wrong_metric_names: List[str],
    max_candidates: Optional[int],
    response_chars: int,
    max_program_tokens: int,
    ids_only: bool,
) -> Any:
    if ids_only:
        return detail.get("id")

    record = {field: detail.get(field) for field in IMPORTANT_FIELDS if field in detail}
    candidates = detail.get("candidates") or []
    record["wrong_metrics"] = wrong_metric_names
    record["candidate_count"] = len(candidates)
    record["candidate_summaries"] = summarize_candidates(
        candidates,
        max_candidates=max_candidates,
        response_chars=response_chars,
        max_program_tokens=max_program_tokens,
    )
    return record


def write_json(records: List[Any], args: argparse.Namespace, total_seen: int) -> None:
    payload = {
        "source": str(args.eval_results),
        "metric_filter": args.metric,
        "total_details_seen": total_seen,
        "wrong_count": len(records),
        "problems": records,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def write_jsonl(records: List[Any], args: argparse.Namespace) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    text = "\n".join(lines)
    if text:
        text += "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main() -> int:
    args = parse_args()
    if args.max_candidates is not None and args.max_candidates < 0:
        raise SystemExit("--max-candidates must be non-negative")
    if args.response_chars < 0:
        raise SystemExit("--response-chars must be non-negative")
    if args.max_program_tokens < -1:
        raise SystemExit("--max-program-tokens must be -1 or greater")

    records: List[Any] = []
    total_seen = 0

    for detail in iter_detail_objects(args.eval_results):
        total_seen += 1
        is_wrong, wrong_metric_names = wrong_metrics_for(detail, args.metric)
        if not is_wrong:
            continue
        records.append(
            build_record(
                detail,
                wrong_metric_names=wrong_metric_names,
                max_candidates=args.max_candidates,
                response_chars=args.response_chars,
                max_program_tokens=args.max_program_tokens,
                ids_only=args.ids_only,
            )
        )

    if args.format == "jsonl":
        write_jsonl(records, args)
    else:
        write_json(records, args, total_seen)

    print(
        f"Extracted {len(records)} wrong problem(s) out of {total_seen} "
        f"using metric={args.metric}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
