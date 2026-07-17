"""
Evaluate Qwen3-VL/Qwen3.5-VL models on PGPS/PGPS9K.

The PGPS repository defines three answer-accuracy protocols:
  - completion: score the first usable generated candidate.
  - choice: choose the first candidate whose result matches a listed option;
    if no candidate matches an option, optionally fall back to a random option
    as in the original PGPS helper.
  - top3: score correct if any of the first three candidates is correct.

This script asks Qwen for a ranked set of solution candidates, parses each
candidate's numeric answer and optional operator program, and computes all three
metrics in one pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import traceback
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

try:
    from func_timeout import func_timeout
except Exception:  # pragma: no cover - optional runtime dependency
    func_timeout = None


ANSWER_TOL = 5e-3
CHOICE_TOL = 5e-2
PROGRAM_TIMEOUT_SECONDS = 2.0
FLOAT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PGPS_OPERATORS_MODULE = None
STATIC_ARITH_OP_LIST = [
    "Get",
    "Iso_Tri_Ang",
    "Gsin",
    "Gcos",
    "Gtan",
    "Geo_Mean",
    "Ratio",
    "TanSec_Ang",
    "Chord2_Ang",
    "Tria_BH_Area",
    "Para_Area",
    "Kite_Area",
    "Circle_R_Circum",
    "Circle_D_Circum",
    "Circle_R_Area",
    "Circle_D_Area",
    "ArcSeg_Area",
    "Ngon_Angsum",
    "RNgon_B_Area",
    "RNgon_L_Area",
    "RNgon_H_Area",
    "Sum",
    "Multiple",
    "Equal",
    "Gougu",
    "Cos_Law",
    "Sin_Law",
    "Median",
    "Proportion",
    "Tria_SAS_Area",
    "PRK_Perim",
    "Rect_Area",
    "Rhom_Area",
    "Trap_Area",
]


@dataclass
class Candidate:
    rank: int
    response: str
    answer: Optional[float]
    program_tokens: List[str]
    program_value: Optional[float]
    value: Optional[float]
    value_source: str
    parse_error: Optional[str] = None


def _load_pgps_operators_module():
    global _PGPS_OPERATORS_MODULE
    if _PGPS_OPERATORS_MODULE is not None:
        return _PGPS_OPERATORS_MODULE

    operators_path = Path(__file__).resolve().parent / "datasets" / "operators.py"
    spec = importlib.util.spec_from_file_location("pgps_operators_for_qwen_eval", operators_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PGPS operators from {operators_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PGPS_OPERATORS_MODULE = module
    return module


def arith_op_list() -> List[str]:
    try:
        return list(_load_pgps_operators_module().arith_op_list)
    except Exception:
        return STATIC_ARITH_OP_LIST[:]


def normalize_exp(exp: List[str]) -> List[str]:
    try:
        return _load_pgps_operators_module().normalize_exp(exp)
    except Exception:
        return exp


def result_compute(num_all_list: List[str], exp_tokens: List[str]) -> str:
    return _load_pgps_operators_module().result_compute(num_all_list, exp_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL models on PGPS")
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Local path or Hugging Face id for a Qwen3-VL/Qwen3.5-VL model.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./datasets/PGPS9K_all",
        help=(
            "Dataset root. The script expects either "
            "<dataset_dir>/<dataset>/<split>.json plus <dataset_dir>/Diagram, "
            "or <dataset_dir>/<split>.json plus <dataset_dir>/Diagram."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="PGPS9K",
        choices=["PGPS9K", "Geometry3K"],
        help="Dataset folder/name to evaluate.",
    )
    parser.add_argument("--split", default="test", help="Split json name without .json.")
    parser.add_argument("--diagram_dir", default=None, help="Override diagram image directory.")
    parser.add_argument("--output_dir", default="./eval_results_qwen3vl", help="Output directory.")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 evaluates the full split.")
    parser.add_argument("--sample_offset", type=int, default=0, help="Skip this many samples first.")
    parser.add_argument(
        "--input_mode",
        default="image_text",
        choices=["image_text", "image_text_clauses"],
        help=(
            "image_text uses only diagram + problem text. image_text_clauses also "
            "adds PGPS parsing_stru_seqs/parsing_sem_seqs to the prompt."
        ),
    )
    parser.add_argument(
        "--show_choices_in_prompt",
        action="store_true",
        help="Show answer choices to the model. Disabled by default for PGPS-style scoring.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["completion", "choice", "top3"],
        choices=["completion", "choice", "top3"],
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--choice_fallback",
        default="random",
        choices=["random", "none"],
        help="When no candidate matches a choice, random reproduces the PGPS helper.",
    )
    parser.add_argument("--num_candidates", type=int, default=3, help="Ranked candidates to generate.")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam count for deterministic decoding.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--visible_cuda_devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument("--device_map", default="auto", help="Transformers device_map.")
    parser.add_argument("--seed", type=int, default=202302)
    parser.add_argument("--save_every", type=int, default=25, help="Write partial results every N samples.")
    return parser.parse_args()


def _split_expression(expr: Any) -> List[str]:
    if isinstance(expr, list):
        return [str(token) for token in expr]
    if isinstance(expr, str):
        return [token for token in expr.strip().split() if token]
    return []


def _dataset_json_path(dataset_dir: str, dataset: str, split: str) -> Path:
    root = Path(dataset_dir)
    candidates = [
        root / dataset / f"{split}.json",
        root / f"{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find split file. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def _normalise_choice_list(choices: Any) -> List[float]:
    if choices is None:
        return []
    if isinstance(choices, str):
        choices = re.findall(FLOAT_RE, choices)
    out: List[float] = []
    for choice in choices:
        value = _safe_float(choice)
        if value is not None:
            out.append(value)
    return out


def _fallback_number_values(sample: Dict[str, Any]) -> List[str]:
    text_parts = [str(sample.get("text", ""))]
    for key in ("parsing_stru_seqs", "parsing_sem_seqs"):
        value = sample.get(key, [])
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
    nums = FLOAT_RE.findall(" ".join(text_parts))
    return nums


def _load_number_values_from_pgps_preprocess(json_path: Path) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    """Use the repo's preprocessing to recover the N0/N1/... order used by programs."""
    try:
        from datasets.preprossing import SN, get_raw_pairs
        from datasets.utils import get_combined_text, get_var_arg
    except Exception as exc:
        print(f"Warning: PGPS preprocessing imports failed; program execution fallback will be weaker: {exc}")
        return {}

    try:
        pairs = get_raw_pairs(str(json_path))
        preprocess_args = SimpleNamespace(without_stru=False)
        values: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
        for pair in pairs:
            combined = SN()
            get_combined_text(
                pair["text"],
                pair["parsing_stru_seqs"],
                pair["parsing_sem_seqs"],
                combined,
                preprocess_args,
            )
            _, var_values, arg_values = get_var_arg(combined, preprocess_args)
            values[str(pair["id"])] = (list(var_values), list(arg_values), list(pair["expression"]))
        return values
    except Exception as exc:
        print(f"Warning: PGPS preprocessing failed; program execution fallback will be weaker: {exc}")
        return {}


def load_dataset(dataset_dir: str, dataset: str, split: str) -> Tuple[List[Dict[str, Any]], Path]:
    json_path = _dataset_json_path(dataset_dir, dataset, split)
    with json_path.open("r", encoding="utf-8") as fp:
        raw_data = json.load(fp)

    preprocessed_values = _load_number_values_from_pgps_preprocess(json_path)

    samples: List[Dict[str, Any]] = []
    for problem_id, content in raw_data.items():
        sample = dict(content)
        sample["id"] = str(problem_id)
        sample["expression"] = _split_expression(sample.get("expression", []))
        sample["choices"] = _normalise_choice_list(sample.get("choices", []))

        pp = preprocessed_values.get(str(problem_id))
        if pp is not None:
            var_values, arg_values, pp_expression = pp
            sample["number_values"] = var_values
            sample["arg_values"] = arg_values
            sample["expression"] = pp_expression
        else:
            sample["number_values"] = _fallback_number_values(sample)
            sample["arg_values"] = []

        answer = _safe_float(sample.get("answer"))
        if answer is None:
            raise ValueError(f"Sample {problem_id} has a non-numeric answer: {sample.get('answer')!r}")
        sample["answer_float"] = answer
        samples.append(sample)

    return samples, json_path


def resolve_diagram_dir(args: argparse.Namespace, json_path: Path) -> Path:
    if args.diagram_dir:
        return Path(args.diagram_dir)
    dataset_root = Path(args.dataset_dir)
    candidates = [
        dataset_root / "Diagram",
        dataset_root / args.dataset / "Diagram",
        json_path.parent / "Diagram",
        json_path.parent.parent / "Diagram",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_model_and_processor(args: argparse.Namespace):
    import torch

    if args.visible_cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices

    import transformers
    from transformers import AutoProcessor

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Loading model: {args.model_path}")
    model_path_lower = str(args.model_path).lower()
    preferred_model_classes = [
        "Qwen3VLForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    if "3.5" in model_path_lower or "3_5" in model_path_lower:
        preferred_model_classes = [
            "Qwen3_5ForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ]

    load_errors: List[str] = []
    for class_name in preferred_model_classes:
        ModelClass = getattr(transformers, class_name, None)
        if ModelClass is None:
            load_errors.append(f"{class_name}: not available in installed transformers")
            continue
        try:
            model = ModelClass.from_pretrained(
                args.model_path,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            load_errors.append(f"{class_name}: {exc}")
    else:
        raise RuntimeError(
            "Could not load the requested Qwen VL model. Tried:\n"
            + "\n".join(f"  - {error}" for error in load_errors)
        )

    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    return model, processor


def build_prompt(sample: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str]:
    system_prompt = (
        "You are a careful plane-geometry solver. Use the diagram and problem text to solve the problem. "
        "Return a ranked solution candidate with a concise derivation, an optional PGPS operator program, "
        "and one final numeric answer."
    )

    operators = "\n".join(
        [
            "PGPS operators you may use in the Program section:",
            "Sum a b c: a + b = c",
            "Multiple a b c: a * b = c",
            "Equal a b: a = b",
            "Gougu a b c: a^2 + b^2 = c^2",
            "Gsin/Gcos/Gtan a b c: trigonometric ratio with angle c in degrees",
            "Cos_Law a b c d and Sin_Law a b c d: cosine/sine laws",
            "Iso_Tri_Ang, Median, Geo_Mean, Proportion, Ratio",
            "Chord2_Ang, TanSec_Ang",
            "Tria_BH_Area, Tria_SAS_Area, PRK_Perim, Para_Area, Rect_Area, Rhom_Area, Kite_Area, Trap_Area",
            "Circle_R_Circum, Circle_D_Circum, Circle_R_Area, Circle_D_Area, ArcSeg_Area",
            "Ngon_Angsum, RNgon_B_Area, RNgon_L_Area, RNgon_H_Area",
            "Get x: report the requested value x",
        ]
    )

    user_parts = [
        "Solve this PGPS geometry problem.",
        "",
        f"Problem:\n{sample.get('text', '')}",
    ]
    if args.input_mode == "image_text_clauses":
        stru = sample.get("parsing_stru_seqs", [])
        sem = sample.get("parsing_sem_seqs", [])
        if stru:
            user_parts.append("\nDiagram structural clauses:\n" + "\n".join(map(str, stru)))
        if sem:
            user_parts.append("\nDiagram semantic clauses:\n" + "\n".join(map(str, sem)))
    if args.show_choices_in_prompt and sample.get("choices"):
        choices = ", ".join(format_float(choice) for choice in sample["choices"])
        user_parts.append(f"\nChoices: {choices}")

    user_parts.extend(
        [
            "",
            operators,
            "",
            "Output exactly this structure:",
            "### Reasoning",
            "Briefly explain the key geometry facts and arithmetic.",
            "",
            "### Program",
            "One PGPS operator step per line when you can express the solution this way. "
            "Use N0, N1, ... for numbers in reading order and V0, V1, ... for intermediate values. "
            "End with Get <value>. If you are unsure of the operator program, write NONE.",
            "",
            "### Answer",
            "ANSWER: <one numeric value>",
        ]
    )
    return system_prompt, "\n".join(user_parts)


def build_messages(sample: Dict[str, Any], image: Optional[Image.Image], args: argparse.Namespace) -> List[Dict[str, Any]]:
    system_prompt, user_prompt = build_prompt(sample, args)
    user_content: List[Dict[str, Any]] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_inputs(processor, messages: List[Dict[str, Any]], image: Optional[Image.Image], device):
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = processor(**kwargs)
    return inputs.to(device)


def generate_responses(model, processor, sample: Dict[str, Any], diagram_dir: Path, args: argparse.Namespace) -> List[str]:
    import torch

    diagram_path = diagram_dir / str(sample.get("diagram", ""))
    image: Optional[Image.Image] = None
    if diagram_path.exists():
        image = Image.open(diagram_path).convert("RGB")
    else:
        print(f"Warning: missing diagram for {sample['id']}: {diagram_path}")

    messages = build_messages(sample, image, args)
    inputs = prepare_inputs(processor, messages, image, _model_input_device(model))

    do_sample = args.temperature > 0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": args.num_candidates,
        "repetition_penalty": args.repetition_penalty,
    }
    if do_sample:
        gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    else:
        gen_kwargs["num_beams"] = max(args.num_beams, args.num_candidates)

    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    trimmed = generated[:, prompt_len:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        pass
    match = FLOAT_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def extract_answer(response: str) -> Optional[float]:
    clean = strip_ansi(response).replace(",", "")
    answer_patterns = [
        r"###\s*Answer\b(?P<section>.*)$",
        r"\bANSWER\s*:\s*(?P<section>[-+0-9.,eE\s]+)",
        r"\bfinal\s+answer\s*(?:is|:)?\s*(?P<section>[-+0-9.,eE\s]+)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, clean, re.IGNORECASE | re.DOTALL)
        if match:
            nums = FLOAT_RE.findall(match.group("section"))
            if nums:
                return _safe_float(nums[-1])

    nums = FLOAT_RE.findall(clean)
    if nums:
        return _safe_float(nums[-1])
    return None


def _program_search_regions(response: str) -> List[str]:
    clean = strip_ansi(response)
    regions: List[str] = []
    program_match = re.search(
        r"###\s*Program\b(?P<section>.*?)(?:\n\s*###\s*Answer\b|$)",
        clean,
        re.IGNORECASE | re.DOTALL,
    )
    if program_match:
        regions.append(program_match.group("section"))
    regions.extend(re.findall(r"```(?:text|plaintext|program|python)?\s*\n(.*?)\n```", clean, re.DOTALL))
    regions.append(clean)
    return regions


def _clean_program_line(line: str) -> Optional[str]:
    line = strip_ansi(line).strip()
    if not line:
        return None
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
    line = re.sub(r"^`+|`+$", "", line.strip())
    line = re.sub(r"\s*(?:#|//|->).*$", "", line)
    line = re.sub(r"\s*\([^)]*(?:=|therefore|so)[^)]*\)\s*$", "", line, flags=re.IGNORECASE)
    line = line.strip()
    if not line or line.upper() == "NONE":
        return None
    return line


def extract_program_tokens(response: str) -> List[str]:
    operators = set(arith_op_list())
    for region in _program_search_regions(response):
        tokens: List[str] = []
        for raw_line in region.splitlines():
            line = _clean_program_line(raw_line)
            if not line:
                continue
            parts = line.split()
            if parts and parts[0] in operators:
                tokens.extend(parts)
        if tokens:
            return tokens
    return []


def compute_program_value(sample: Dict[str, Any], program_tokens: List[str]) -> Tuple[Optional[float], Optional[str]]:
    if not program_tokens:
        return None, None
    try:
        program = normalize_exp(program_tokens[:])
        number_values = list(sample.get("number_values") or [])
        if func_timeout is not None:
            value = func_timeout(
                PROGRAM_TIMEOUT_SECONDS,
                result_compute,
                kwargs={"num_all_list": number_values, "exp_tokens": program},
            )
        else:
            value = result_compute(num_all_list=number_values, exp_tokens=program)
        return _safe_float(value), None
    except Exception as exc:
        return None, str(exc)


def parse_candidate(sample: Dict[str, Any], response: str, rank: int) -> Candidate:
    answer = extract_answer(response)
    program_tokens = extract_program_tokens(response)
    program_value, parse_error = compute_program_value(sample, program_tokens)
    if program_value is not None:
        value, source = program_value, "program"
    elif answer is not None:
        value, source = answer, "answer"
    else:
        value, source = None, "none"
    return Candidate(
        rank=rank,
        response=response,
        answer=answer,
        program_tokens=program_tokens,
        program_value=program_value,
        value=value,
        value_source=source,
        parse_error=parse_error,
    )


def is_correct_value(value: Optional[float], target: float, tol: float = ANSWER_TOL) -> bool:
    return value is not None and abs(value - target) < tol


def normalized_program_equal(pred_tokens: Sequence[str], target_tokens: Sequence[str]) -> bool:
    if not pred_tokens or not target_tokens:
        return False
    try:
        pred = normalize_exp(list(pred_tokens))
        target = normalize_exp(list(target_tokens))
    except Exception:
        pred = list(pred_tokens)
        target = list(target_tokens)
    return pred == target


def metric_completion(candidates: Sequence[Candidate], target: float) -> Tuple[bool, Optional[int]]:
    for candidate in candidates:
        if candidate.value is not None:
            return is_correct_value(candidate.value, target), candidate.rank
    return False, None


def metric_topk(candidates: Sequence[Candidate], target: float, k: int = 3) -> Tuple[bool, Optional[int]]:
    for candidate in candidates[:k]:
        if is_correct_value(candidate.value, target):
            return True, candidate.rank
    return False, None


def metric_choice(
    sample: Dict[str, Any],
    candidates: Sequence[Candidate],
    rng: random.Random,
    fallback: str,
) -> Tuple[bool, Optional[int], Optional[float], str]:
    target = sample["answer_float"]
    choices = list(sample.get("choices") or [])

    if not choices:
        ok, rank = metric_completion(candidates, target)
        selected = candidates[rank - 1].value if rank is not None else None
        return ok, rank, selected, "no_choices_completion"

    for candidate in candidates:
        if normalized_program_equal(candidate.program_tokens, sample.get("expression", [])):
            return True, candidate.rank, target, "exact_program"
        if candidate.value is None:
            continue
        if any(abs(candidate.value - choice) < CHOICE_TOL for choice in choices):
            return is_correct_value(candidate.value, target), candidate.rank, candidate.value, "candidate_choice"

    if fallback == "random":
        selected = rng.choice(choices)
        return abs(selected - target) < CHOICE_TOL, None, selected, "random_fallback"
    return False, None, None, "no_candidate_choice"


def evaluate_sample(
    sample: Dict[str, Any],
    candidates: Sequence[Candidate],
    metrics: Iterable[str],
    rng: random.Random,
    choice_fallback: str,
) -> Dict[str, Any]:
    metric_results: Dict[str, Dict[str, Any]] = {}
    target = sample["answer_float"]

    if "completion" in metrics:
        ok, rank = metric_completion(candidates, target)
        metric_results["completion"] = {"correct": ok, "rank": rank}

    if "top3" in metrics:
        ok, rank = metric_topk(candidates, target, 3)
        metric_results["top3"] = {"correct": ok, "rank": rank}

    if "choice" in metrics:
        ok, rank, selected, source = metric_choice(sample, candidates, rng, choice_fallback)
        metric_results["choice"] = {
            "correct": ok,
            "rank": rank,
            "selected_value": selected,
            "source": source,
        }

    return metric_results


def format_float(value: float) -> str:
    return f"{value:.10g}"


def save_results(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_json: Path,
    details: List[Dict[str, Any]],
    started_at: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "total_samples": len(details),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "dataset_json": str(dataset_json),
        "model_path": args.model_path,
        "num_candidates": args.num_candidates,
        "metrics": {},
    }

    for metric in args.metrics:
        correct = sum(1 for item in details if item["metrics"].get(metric, {}).get("correct"))
        total = len(details)
        summary["metrics"][metric] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }

    result_path = output_dir / "eval_results.json"
    with result_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "summary": summary,
                "args": vars(args),
                "details": details,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    return result_path


def print_summary(result_path: Path) -> None:
    with result_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    summary = data["summary"]

    print("\n" + "=" * 72)
    print("Qwen-VL PGPS Evaluation")
    print("=" * 72)
    print(f"Model: {summary['model_path']}")
    print(f"Samples: {summary['total_samples']}")
    for metric, item in summary["metrics"].items():
        print(f"{metric:>10}: {item['accuracy']:.4f} ({item['correct']}/{item['total']})")
    print(f"Results: {result_path}")
    print("=" * 72)


def run_evaluation(args: argparse.Namespace) -> Path:
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    samples, dataset_json = load_dataset(args.dataset_dir, args.dataset, args.split)
    if args.sample_offset:
        samples = samples[args.sample_offset :]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if args.num_candidates < 1:
        raise ValueError("--num_candidates must be at least 1")
    if "top3" in args.metrics and args.num_candidates < 3:
        print("Warning: top3 requested with fewer than 3 candidates; using available candidates only.")

    diagram_dir = resolve_diagram_dir(args, dataset_json)
    print(f"Dataset: {dataset_json}")
    print(f"Diagram directory: {diagram_dir}")
    print(f"Samples: {len(samples)}")

    model, processor = load_model_and_processor(args)
    started_at = time.time()
    details: List[Dict[str, Any]] = []
    output_dir = Path(args.output_dir)

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating"), start=1):
        try:
            responses = generate_responses(model, processor, sample, diagram_dir, args)
            candidates = [
                parse_candidate(sample, response, rank=rank)
                for rank, response in enumerate(responses, start=1)
            ]
        except Exception as exc:
            print(f"\nInference failed for sample {sample.get('id', idx)}: {exc}")
            traceback.print_exc()
            candidates = [
                Candidate(
                    rank=1,
                    response="",
                    answer=None,
                    program_tokens=[],
                    program_value=None,
                    value=None,
                    value_source="none",
                    parse_error=str(exc),
                )
            ]

        metric_results = evaluate_sample(
            sample,
            candidates,
            args.metrics,
            rng,
            args.choice_fallback,
        )
        details.append(
            {
                "id": sample["id"],
                "diagram": sample.get("diagram"),
                "text": sample.get("text", ""),
                "answer": sample["answer_float"],
                "choices": sample.get("choices", []),
                "ground_truth_expression": sample.get("expression", []),
                "number_values": sample.get("number_values", []),
                "metrics": metric_results,
                "candidates": [asdict(candidate) for candidate in candidates],
            }
        )

        if args.save_every > 0 and idx % args.save_every == 0:
            save_results(output_dir, args, dataset_json, details, started_at)

    result_path = save_results(output_dir, args, dataset_json, details, started_at)
    print_summary(result_path)
    return result_path


if __name__ == "__main__":
    run_evaluation(parse_args())
