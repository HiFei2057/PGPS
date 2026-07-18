#!/usr/bin/env python3
"""Generate a weakness-targeted ``--families`` prompt for draw-task generation.

The prompt is derived from a PGPS-style eval_results.json file.  It uses the
full eval log when possible so category counts have denominators, then emits a
hard diversity contract that targets the categories with the highest wrong
rates and largest wrong-case share.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple


OPS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a weakness-targeted families prompt for generate_verifiable_draw_tasks.py."
    )
    parser.add_argument(
        "eval_results",
        nargs="?",
        type=Path,
        default=Path("eval_results_qwen3_0717_full/eval_results.json"),
        help="Full eval_results.json. Default: eval_results_qwen3_0717_full/eval_results.json",
    )
    parser.add_argument(
        "--metric",
        default="completion",
        help="Metric used to estimate weakness. Default: completion.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of task templates to request in the generated family contract. Default: 8.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=10,
        help="Minimum category/operator support before reporting a wrong rate. Default: 10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the families prompt.",
    )
    parser.add_argument(
        "--shell-quote",
        action="store_true",
        help="Print the prompt shell-quoted for direct use after --families.",
    )
    parser.add_argument(
        "--include-evidence",
        action="store_true",
        help="Include compact eval evidence inside the generated prompt.",
    )
    return parser.parse_args()


def iter_detail_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
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

        buffer = buffer[buffer.index("[") + 1 :]

        while True:
            buffer = buffer.lstrip()
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

            if isinstance(detail, dict):
                yield detail
            buffer = buffer[end:]


def expression_ops(item: Dict[str, Any]) -> List[str]:
    return [token for token in item.get("ground_truth_expression") or [] if token in OPS and token != "Get"]


def is_wrong(item: Dict[str, Any], metric: str) -> bool:
    return not (item.get("metrics") or {}).get(metric, {}).get("correct", False)


@dataclass(frozen=True)
class Category:
    name: str
    tag: str
    predicate: Callable[[Dict[str, Any]], bool]


def text_of(item: Dict[str, Any]) -> str:
    return (item.get("text") or "").lower()


def numbers_of(item: Dict[str, Any]) -> str:
    return " ".join(str(value).lower() for value in item.get("number_values") or [])


def has_op(item: Dict[str, Any], candidates: Iterable[str]) -> bool:
    present = set(expression_ops(item))
    return any(candidate in present for candidate in candidates)


CATEGORIES = [
    Category(
        "trig/law of sines/cosines",
        "trig",
        lambda item: has_op(item, ["Gsin", "Gcos", "Gtan", "Sin_Law", "Cos_Law"])
        or bool(re.search(r"\bsin\b|\bcos\b|\btan\b|law of sines|law of cosines|right triangle", text_of(item))),
    ),
    Category(
        "circle geometry",
        "circle",
        lambda item: bool(
            re.search(
                r"\\odot|circle|diameter|radius|arc|widehat|chord|tangent|secant|circumference",
                text_of(item),
            )
        )
        or any(op.startswith("Circle") or op in {"ArcSeg_Area", "Chord2_Ang", "TanSec_Ang"} for op in expression_ops(item)),
    ),
    Category(
        "angle/arc chasing",
        "angle",
        lambda item: bool(re.search(r"angle|\\angle|widehat|arc|m ", text_of(item)))
        or has_op(item, ["Ngon_Angsum", "Chord2_Ang", "TanSec_Ang", "Iso_Tri_Ang"]),
    ),
    Category(
        "area/perimeter/formula",
        "formula",
        lambda item: bool(re.search(r"area|perimeter|circumference", text_of(item)))
        or any("Area" in op or "Perim" in op or "Circum" in op for op in expression_ops(item)),
    ),
    Category(
        "triangle properties",
        "triangle",
        lambda item: bool(re.search(r"triangle|\\triangle|tri", text_of(item)))
        or any(op.startswith("Tria") or op in {"Gougu", "Sin_Law", "Cos_Law", "Median", "Iso_Tri_Ang"} for op in expression_ops(item)),
    ),
    Category(
        "symbolic x/y algebra",
        "algebra",
        lambda item: bool(re.search(r"find [xyz]|value of [xyz]|solve", text_of(item)))
        or bool(re.search(r"[xyz]", numbers_of(item))),
    ),
    Category(
        "quadrilateral properties",
        "quadrilateral",
        lambda item: bool(re.search(r"parallelogram|kite|rectangle|rhombus|trapezoid|quadrilateral|square", text_of(item)))
        or has_op(item, ["Para_Area", "Kite_Area", "Rect_Area", "Rhom_Area", "Trap_Area"]),
    ),
    Category(
        "ratio/proportion/similarity",
        "proportion",
        lambda item: bool(re.search(r"similar|proportion|ratio|scale", text_of(item)))
        or has_op(item, ["Proportion", "Ratio"]),
    ),
]


@dataclass(frozen=True)
class TemplateSpec:
    title: str
    tags: Tuple[str, ...]
    text: str


TEMPLATES = [
    TemplateSpec(
        "oblique trig altitude web",
        ("trig", "triangle", "angle", "algebra"),
        "Template A: oblique trig-altitude triangle web. Construct a scalene acute or obtuse triangle ABC with non-axis-aligned integer coordinates. Draw side lines/segments AB, BC, AC. Add a coordinate-defined altitude segment from C to AB meeting AB at H via add_intersect, add midpoint M of AB, and draw median CM. Add one angle-bisector from B meeting AC at D. Checks must include CH perpendicular AB, H/M/D exist, AM=MB, two angle checks around the altitude or bisector, and at least two nontrivial distances.",
    ),
    TemplateSpec(
        "law-of-cosines diagonal triangle",
        ("trig", "triangle", "formula", "angle"),
        "Template B: law-of-cosines-style diagonal figure. Construct four coordinate points A,B,C,D so triangles ABC and ACD share AC, with one obtuse angle and no right angle unless explicitly checked. Draw AB, BC, AC, CD, AD and one diagonal/connector. Add midpoint M of AC and intersection O of two cross-lines. Checks must include O/M exist, one obtuse or acute angle measure, three side distances from the two linked triangles, and one true perpendicular or true parallel relation introduced by coordinates.",
    ),
    TemplateSpec(
        "circle tangent chord midpoint",
        ("circle", "trig", "angle"),
        "Template C: circle chord-midpoint-radius-tangent. Construct circle cO with center O, radius point A, and another integer-coordinate point B on the circle from a 3-4-5 or 5-12-13 relation. Draw chord AB, midpoint M of AB, radius segments OA/OB/OM, and a coordinate-defined tangent line t through A using point P. Checks must include OA=OB radius, t tangent cO, t perpendicular OA, OM perpendicular AB, M exists, chord length AB, and one angle involving t and AB.",
    ),
    TemplateSpec(
        "circle two-chord secant angle web",
        ("circle", "angle", "algebra"),
        "Template D: circle two-chord/secant angle web. Construct circle cO and four non-cardinal points A,B,C,D on or tied to the circle using integer coordinates. Draw chords AB and CD plus secant-like lines AC and BD, define intersection E of AC and BD, add midpoint M of one chord, and draw OM. Checks must include E/M exist, OM perpendicular to its chord, at least two chord/radius distances, one true tangent or perpendicular final check, and two angle checks from the intersecting-line web.",
    ),
    TemplateSpec(
        "incircle bisector midsegment",
        ("circle", "triangle", "angle", "proportion"),
        "Template E: incircle plus bisector plus midsegment. Construct scalene triangle ABC, side lines lAB/lBC/lCA, incircle inc, angle bisector from one vertex meeting the opposite side at D, and midpoints E and F on two sides. Draw EF. Checks must include three incircle tangencies, angle split at the bisected vertex, EF parallel to the third side, D/E/F exist, midpoint distance equalities, and one side or midsegment length.",
    ),
    TemplateSpec(
        "parallel transversal angle chase",
        ("angle", "quadrilateral", "algebra"),
        "Template F: parallel transversal angle chase. Construct a non-rectangle parallelogram or trapezoid ABCD with non-axis-aligned coordinates. Draw both diagonals and at least two transversals, define intersection O of diagonals, add midpoints M and N on different segments. Checks must include the true parallel side pair(s), O/M/N exist, two angle checks created by transversals, one diagonal bisection or midpoint equality, one distance, and one true perpendicular or parallel final check.",
    ),
    TemplateSpec(
        "similarity/proportion nested triangle",
        ("proportion", "triangle", "algebra", "angle"),
        "Template G: similarity/proportion nested triangle. Construct triangle ABC, choose points D on AB and E on AC by coordinates so DE is parallel BC without using a parallel_line tool. Draw AD, AE, DE, BC, and one cross-line BE or CD. Add midpoint M on BC and intersection O of the cross-lines. Checks must include DE parallel BC, O/M exist, two proportional-looking distance checks, one angle correspondence check, one midpoint equality, and one true perpendicular or parallel final check.",
    ),
    TemplateSpec(
        "complete quadrilateral line-web",
        ("angle", "quadrilateral", "formula"),
        "Template H: complete quadrilateral line-web. Construct four base points A,B,C,D with varied integer coordinates, draw lines lAB,lCD,lAC,lBD and selected side segments, define intersections E and F from non-adjacent/cross lines, and add midpoint objects on two segments. Checks must include E/F/midpoints exist, one pair of true parallel or perpendicular lines, two midpoint distance equalities, at least two angle checks, and one nontrivial segment length.",
    ),
]


def categories_for(item: Dict[str, Any]) -> List[Category]:
    matched = [category for category in CATEGORIES if category.predicate(item)]
    return matched or [Category("other/simple arithmetic", "other", lambda _: True)]


def first_op(item: Dict[str, Any]) -> str:
    ops = expression_ops(item)
    return ops[0] if ops else "None"


def analyze(details: Sequence[Dict[str, Any]], metric: str, min_total: int) -> Dict[str, Any]:
    total = len(details)
    wrong_total = sum(is_wrong(item, metric) for item in details)

    cat_total: Counter[str] = Counter()
    cat_wrong: Counter[str] = Counter()
    tag_scores: Dict[str, float] = {}
    for item in details:
        for category in categories_for(item):
            cat_total[category.name] += 1
            if is_wrong(item, metric):
                cat_wrong[category.name] += 1

    category_rows = []
    for category in CATEGORIES:
        total_for_cat = cat_total[category.name]
        if total_for_cat == 0:
            continue
        wrong_for_cat = cat_wrong[category.name]
        wrong_rate = wrong_for_cat / total_for_cat
        wrong_share = wrong_for_cat / wrong_total if wrong_total else 0.0
        score = (0.7 * wrong_rate) + (0.3 * wrong_share)
        tag_scores[category.tag] = score
        if total_for_cat >= min_total:
            category_rows.append((category.name, category.tag, wrong_for_cat, total_for_cat, wrong_rate, wrong_share, score))

    op_total: Counter[str] = Counter()
    op_wrong: Counter[str] = Counter()
    for item in details:
        op = first_op(item)
        op_total[op] += 1
        if is_wrong(item, metric):
            op_wrong[op] += 1

    op_rows = []
    for op, total_for_op in op_total.items():
        if total_for_op < min_total:
            continue
        wrong_for_op = op_wrong[op]
        op_rows.append((op, wrong_for_op, total_for_op, wrong_for_op / total_for_op))

    op_count_total: Counter[str] = Counter()
    op_count_wrong: Counter[str] = Counter()
    for item in details:
        count = len(expression_ops(item))
        bucket = "4+" if count >= 4 else str(count)
        op_count_total[bucket] += 1
        if is_wrong(item, metric):
            op_count_wrong[bucket] += 1

    complexity_rows = []
    for bucket in ("0", "1", "2", "3", "4+"):
        total_for_bucket = op_count_total[bucket]
        if total_for_bucket:
            wrong_for_bucket = op_count_wrong[bucket]
            complexity_rows.append((bucket, wrong_for_bucket, total_for_bucket, wrong_for_bucket / total_for_bucket))

    return {
        "total": total,
        "wrong_total": wrong_total,
        "category_rows": sorted(category_rows, key=lambda row: (-row[6], -row[2], row[0])),
        "op_rows": sorted(op_rows, key=lambda row: (-row[3], -row[2], row[0])),
        "complexity_rows": complexity_rows,
        "tag_scores": tag_scores,
    }


def select_templates(analysis: Dict[str, Any], count: int) -> List[TemplateSpec]:
    tag_scores: Dict[str, float] = analysis["tag_scores"]

    def template_score(template: TemplateSpec) -> float:
        return sum(tag_scores.get(tag, 0.0) for tag in template.tags) / len(template.tags)

    ranked = sorted(TEMPLATES, key=lambda template: (-template_score(template), template.title))
    selected = ranked[: max(1, min(count, len(ranked)))]

    title_to_template = {template.title: template for template in TEMPLATES}
    must_have = [
        "oblique trig altitude web",
        "circle tangent chord midpoint",
        "incircle bisector midsegment",
        "parallel transversal angle chase",
    ]
    for title in reversed(must_have):
        template = title_to_template[title]
        if template not in selected and len(selected) >= count:
            selected[-1] = template
        elif template not in selected:
            selected.append(template)

    deduped: List[TemplateSpec] = []
    for template in selected:
        if template not in deduped:
            deduped.append(template)

    for template in ranked:
        if len(deduped) >= count:
            break
        if template not in deduped:
            deduped.append(template)
    return deduped[:count]


def evidence_text(analysis: Dict[str, Any]) -> str:
    total = analysis["total"]
    wrong_total = analysis["wrong_total"]
    accuracy = 1 - (wrong_total / total) if total else 0.0
    lines = [
        f"Evidence from the eval log: {wrong_total}/{total} wrong on the selected metric, accuracy {accuracy:.1%}.",
        "Highest-priority weak categories: "
        + "; ".join(
            f"{name} {wrong}/{total_for_cat} wrong ({wrong_rate:.0%})"
            for name, _tag, wrong, total_for_cat, wrong_rate, _share, _score in analysis["category_rows"][:4]
        )
        + ".",
        "Weak first operators: "
        + "; ".join(
            f"{op} {wrong}/{total_for_op} ({wrong_rate:.0%})"
            for op, wrong, total_for_op, wrong_rate in analysis["op_rows"][:5]
        )
        + ".",
    ]
    return " ".join(lines)


def relabel_templates(templates: Sequence[TemplateSpec]) -> str:
    parts = []
    for index, template in enumerate(templates):
        label = chr(ord("A") + index)
        body = re.sub(r"^Template [A-Z]:\s*", "", template.text)
        parts.append(f"Template {label}: {body}")
    return " ".join(parts)


def build_prompt(analysis: Dict[str, Any], templates: Sequence[TemplateSpec], include_evidence: bool) -> str:
    template_text = relabel_templates(templates)
    evidence = (evidence_text(analysis) + " ") if include_evidence else ""
    template_names = ", ".join(f"Template {chr(ord('A') + index)}" for index in range(len(templates)))

    return (
        f"{evidence}"
        "Weakness-targeted hard diversity contract for challenging a geometry model. "
        "Prioritize the failure modes indicated by the eval analysis: trig/law-style reasoning, circle tangent/chord/secant reasoning, angle chasing, symbolic x/y algebra, and multi-step constructions. "
        "Each accepted task must use 12-16 oracle_tool_calls, 8-11 checks, at least 4 check kinds, and at least 3 derived objects from add_midpoint, add_intersect, add_angle_bisector, or add_incircle. "
        "Every task must combine at least two weak modes, for example circle+angle, trig+triangle, parallel+angle, incircle+midsegment, or similarity+algebra. "
        "Every task must include at least one true tangent, true perpendicular, or true parallel final check. "
        "At least 5 of every 8 accepted tasks must include non-axis-aligned integer coordinates, at least 3 must contain an explicit circle or incircle, at least 3 must contain an angle-bisector or angle-split check, and at least 4 must require an intersection object that is later checked. "
        "Do not use false tangent/perpendicular/parallel checks. Do not generate simple O,T,P circle-tangent-only tasks. "
        f"Cycle through these templates without repeating a template within any {len(templates)} accepted tasks: {template_names}. "
        f"{template_text} "
        "Use varied integer coordinates in [-10,10], including acute, obtuse, trapezoid, parallelogram, nested-triangle, and circle-chord layouts. "
        "Use only add_point, add_line, add_segment, add_midpoint, add_intersect, add_angle_bisector, add_circle, and add_incircle in oracle_tool_calls. "
        "Avoid arcs, sectors, polygons, perpendicular_line, parallel_line, tangent tools, hidden object names, copied few-shot examples, area-only checks, and simple circle-only tangent tasks."
    )


def main() -> int:
    args = parse_args()
    details = list(iter_detail_objects(args.eval_results))
    analysis = analyze(details, args.metric, args.min_total)
    templates = select_templates(analysis, args.count)
    prompt = build_prompt(analysis, templates, args.include_evidence)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt + "\n", encoding="utf-8")

    print(shlex.quote(prompt) if args.shell_quote else prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
