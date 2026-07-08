"""
Qwen3.5-9B 多模态大模型在 PGPS 几何问题数据集上的评测脚本。

使用方法:
    python run_eval_qwen35.py \\
        --model_path /home/jianda.syf/helix_data/qwen3_5_9B \\
        --dataset_dir /home/jianda.syf/data_pgps9k/PGPS9K \\
        --dataset PGPS9K \\
        --output_dir ./eval_results \\
        --max_samples 100

PGPS 数据集每个样本包含:
    - diagram: 几何图形图片 (Diagram/xxx.png)
    - text: 自然语言描述的几何问题文本
    - expression: 解题表达式 (算子序列)
    - answer: 数值答案
    - choices: 选择题选项 (PGPS9K 有 choices 字段)

本脚本使用 Qwen3.5-9B 的多模态能力 (Qwen3_5ForConditionalGeneration)，
将几何图形图片和文本问题一起输入模型，让模型生成解题表达式并计算结果，
与标准答案进行比较，计算准确率。
"""

import os
import json
import argparse
import re
import traceback
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# 尝试导入 sympy 用于表达式计算验证
try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-9B PGPS Evaluation")
    parser.add_argument("--model_path", type=str, default="/home/jianda.syf/helix_data/qwen3_5_9B",
                        help="本地 Qwen3.5-9B 模型路径")
    parser.add_argument("--dataset_dir", type=str, default="/home/jianda.syf/data_pgps9k/PGPS9K",
                        help="PGPS 数据集根目录（包含 PGPS9K/Geometry3K 子目录和 Diagram 图片目录）")
    parser.add_argument("--dataset", type=str, default="PGPS9K", choices=["PGPS9K", "Geometry3K"],
                        help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="评测结果输出目录")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最大评测样本数，-1 表示全部")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小（当前仅支持 batch_size=1）")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="生成温度，0 表示贪婪解码，推荐 0.1-0.3 避免重复循环")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="nucleus sampling 概率阈值")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="重复惩罚系数，1.0 表示不启用")
    parser.add_argument("--visible_cuda_devices", type=str, default="0",
                        help="指定可见的 GPU 卡号，如 '0'、'0,1'、'0,2,3'。"
                             "设置后 CUDA_VISIBLE_DEVICES 会被设为该值，"
                             "模型将自动分布到所有可见卡上")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="模型推理精度")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def load_dataset(dataset_dir: str, dataset_name: str) -> List[Dict]:
    """
    加载 PGPS 数据集的 test.json。
    
    PGPS 数据格式 (test.json):
    {
        "problem_id": {
            "diagram": "img_3909.png",              # 几何图形图片文件名
            "text": "In triangle ABC...",           # 自然语言问题文本
            "parsing_stru_seqs": [...],             # 结构解析序列
            "parsing_sem_seqs": [...],              # 语义解析序列
            "expression": "Multiple V0 C2 N2 Get V0",  # 解题表达式 (空格分隔的字符串)
            "answer": "25.000",                     # 答案数值字符串
            "choices": [25.0, 30.0, 40.0, 80.0]    # 选择题选项 (PGPS9K)
        }
    }
    
    注意: expression 是字符串格式，用空格分隔算子 token。
    """
    test_path = os.path.join(dataset_dir, dataset_name, "test.json")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"数据集文件不存在: {test_path}\n"
            f"请确保已下载 PGPS 数据集并解压到 {dataset_dir} 目录下。"
        )
    
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 转换为列表格式，并统一 expression 为列表
    samples = []
    for problem_id, content in data.items():
        content["id"] = problem_id
        # expression 是字符串 (空格分隔)，转为列表
        expr = content.get("expression", "")
        if isinstance(expr, str):
            content["expression"] = expr.strip().split()
        samples.append(content)
    
    return samples


def load_model_and_processor(model_path: str, torch_dtype: str):
    """
    加载 Qwen3.5-9B 多模态模型和处理器。
    模型会通过 device_map="auto" 自动分布到所有可见 GPU 上。
    用 --visible_cuda_devices 控制可见哪些 GPU。
    """
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    print(f"正在加载模型: {model_path}")
    print(f"可见 GPU 数量: {torch.cuda.device_count()}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print("正在加载 Processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    return model, processor


def build_prompt(text: str, expression: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    构建给 Qwen3.5 的 prompt（v2 优化版）。
    
    关键改进：
    1. 使用结构化输出格式（### Reasoning / ### Program / ### Answer），
       允许模型推理，但通过标记清晰分隔各部分，解析器可精确提取
    2. 增加 3 个覆盖不同算子类型的 Few-shot 示例
    3. 提示模型避免重复循环推理
    """
    system_prompt = (
        "You are a geometry problem solver. Given a geometry diagram and a text problem, "
        "you must solve it step by step using the operators listed below.\n\n"
        "Operators (each line is one step, tokens separated by spaces):\n"
        "- Sum a b c : a + b = c\n"
        "- Multiple a b c : a * b = c\n"
        "- Equal a b : a = b\n"
        "- Gougu a b c : a^2 + b^2 = c^2 (Pythagorean theorem)\n"
        "- Gsin a b c : sin(c) = a / b\n"
        "- Gcos a b c : cos(c) = a / b\n"
        "- Gtan a b c : tan(c) = a / b\n"
        "- Cos_Law a b c d : a^2 = b^2 + c^2 - 2*b*c*cos(d)\n"
        "- Sin_Law a b c d : sin(a)/b = sin(c)/d\n"
        "- Iso_Tri_Ang a b : a + 2*b = 180 (isosceles triangle base angles)\n"
        "- Median a b c : a + c = 2*b\n"
        "- Geo_Mean a b c : a * b = c^2\n"
        "- Proportion a b c d : a/b = c/d\n"
        "- Ratio a b c : a/b = c\n"
        "- Chord2_Ang a b c : a = (b+c)/2\n"
        "- TanSec_Ang a b c : a = (c-b)/2\n"
        "- Tria_BH_Area a b c : a*b/2 = c (triangle area: base*height/2)\n"
        "- Tria_SAS_Area a b c d : a*c*sin(b)/2 = d (triangle area: two sides * sin(angle)/2)\n"
        "- PRK_Perim a b c : (a+b)*2 = c (parallelogram perimeter)\n"
        "- Para_Area a b c : a*b = c\n"
        "- Rect_Area a b c : a*b = c\n"
        "- Rhom_Area a b c : a*b*2 = c\n"
        "- Kite_Area a b c : a*b/2 = c\n"
        "- Trap_Area a b c d : (a+b)*c/2 = d\n"
        "- Circle_R_Circum a b : 2*pi*a = b\n"
        "- Circle_D_Circum a b : pi*a = b\n"
        "- Circle_R_Area a b : pi*a^2 = b\n"
        "- Circle_D_Area a b : pi*(a/2)^2 = b\n"
        "- Ngon_Angsum a b : (a-2)*180 = b\n"
        "- RNgon_B_Area a b c : a*b^2/tan(180/a)/4 = c\n"
        "- RNgon_L_Area a b c : a*b^2*sin(360/a)/2 = c\n"
        "- RNgon_H_Area a b c : a*b^2*tan(180/a) = c\n\n"
        "Variables use lowercase letters (a, b, c, ...) for unknown values.\n"
        "Numbers from the problem text use N0, N1, N2, ... format.\n\n"
        "You MUST structure your output using EXACTLY these three sections in order:\n"
        "### Reasoning\n"
        "Analyze the diagram and problem. Identify known values (N0, N1, ...), "
        "choose the right operators, and explain your solution plan briefly. "
        "Be concise — do NOT repeat the same reasoning.\n\n"
        "### Program\n"
        "Write the operator program, one operator per line. "
        "Each line must start with the operator name followed by its arguments. "
        "Use plain text only — no backticks, no code blocks, no extra text.\n\n"
        "### Answer\n"
        "ANSWER: <number>\n\n"
        "IMPORTANT: Do NOT repeat reasoning loops. Once you have a solution, output it immediately. "
        "If you find yourself repeating the same analysis, STOP and output the program.\n\n"
    )
    
    # Few-shot 示例：覆盖 Sum/Ratio（简单算术）、Gougu（勾股定理）、Tria_BH_Area（面积）
    examples = [
        # 示例 1: 简单加法 + 比例
        {
            "problem": "In triangle ABC, AB = 5, BC = 8, and AC = 13. "
                       "Find the ratio of AB to the sum of AB and BC.",
            "output": (
                "### Reasoning\n"
                "N0 = 5 (AB), N1 = 8 (BC). First compute sum of AB+BC: Sum N0 N1 S. "
                "Then ratio of AB to sum: Ratio N0 S XY.\n\n"
                "### Program\n"
                "Sum N0 N1 S\n"
                "Ratio N0 S XY\n\n"
                "### Answer\n"
                "ANSWER: 0.384615"
            ),
        },
        # 示例 2: 勾股定理
        {
            "problem": "In a right triangle, the two legs are 3 and 4. Find the hypotenuse.",
            "output": (
                "### Reasoning\n"
                "N0 = 3, N1 = 4. Right triangle → use Pythagorean theorem Gougu: "
                "N0^2 + N1^2 = c^2, so c = sqrt(N0^2 + N1^2) = 5.\n\n"
                "### Program\n"
                "Gougu N0 N1 c\n\n"
                "### Answer\n"
                "ANSWER: 5"
            ),
        },
        # 示例 3: 三角形面积
        {
            "problem": "A triangle has base 10 and height 6. Find its area.",
            "output": (
                "### Reasoning\n"
                "N0 = 10 (base), N1 = 6 (height). Triangle area = base * height / 2 → Tria_BH_Area.\n\n"
                "### Program\n"
                "Tria_BH_Area N0 N1 c\n\n"
                "### Answer\n"
                "ANSWER: 30"
            ),
        },
    ]
    
    examples_text = "Here are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        examples_text += (
            f"Example {i}:\n"
            f"Problem: {ex['problem']}\n"
            f"Output:\n{ex['output']}\n"
            f"{'─' * 60}\n\n"
        )
    
    user_message = (
        f"{examples_text}"
        f"Now solve this problem:\n\n"
        f"Geometry Problem:\n{text}\n\n"
        "Follow the EXACT same output structure as the examples above: "
        "### Reasoning, then ### Program, then ### Answer."
    )
    
    return system_prompt, user_message


def extract_answer_from_response(response: str) -> Optional[float]:
    """
    Extract numerical answer from model response.

    Strategy (by priority):
    1. Try structured ### Answer section first (supports ANSI escape codes)
    2. Fall back to ANSWER: format matching
    3. Fall back to trailing equals sign
    4. Fall back to trailing number
    """
    # Clean ANSI escape codes
    clean_response = _strip_ansi_escape(response)

    # Strategy 1: Extract from ### Answer section
    answer_section_patterns = [
        r'###\s*Answer\s*\n(.*?)(?:\n\s*$|$)',
        r'###\s*Answer\s*[:\-]?\s*(.*?)(?:\n\s*$|$)',
        r'###\s*Answer\s*\n(.*?)$',
    ]
    for pattern in answer_section_patterns:
        matches = re.findall(pattern, clean_response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            section = match.strip()
            if not section:
                continue
            # Try ANSWER: format
            ans_match = re.search(r'ANSWER\s*:\s*([\d]+\.?[\d]*)', section, re.IGNORECASE)
            if ans_match:
                try:
                    return float(ans_match.group(1))
                except ValueError:
                    continue
            # Try direct number extraction
            num_match = re.search(r'([\d]+\.?[\d]+)', section)
            if num_match:
                try:
                    return float(num_match.group(1))
                except ValueError:
                    continue

    # Strategy 2: Global ANSWER: format search
    patterns = [
        r'ANSWER\s*:\s*([\d]+\.?[\d]*)',
        r'answer\s*:\s*([\d]+\.?[\d]*)',
        r'Answer\s*:\s*([\d]+\.?[\d]*)',
        r'=\s*([\d]+\.?[\d]*)\s*$',
        r'([\d]+\.?[\d]*)\s*$',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, clean_response, re.MULTILINE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    return None


def _strip_ansi_escape(text: str) -> str:
    """Remove ANSI escape codes (e.g. \\x1b[0m)."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def _clean_operator_line(line: str) -> Optional[str]:
    """
    Clean a line of operator output:
    - Remove markdown backticks
    - Remove parenthetical comments
    - Remove arrow comments
    - Remove inline backtick wrapping
    Returns cleaned line, or None if empty.
    """
    cleaned = line.strip()
    if not cleaned:
        return None

    # Remove ANSI escape codes
    cleaned = _strip_ansi_escape(cleaned)

    # Remove markdown heading markers
    cleaned = re.sub(r'^#{1,4}\s*', '', cleaned)

    # Remove leading and trailing backticks
    cleaned = re.sub(r'^`+', '', cleaned)
    cleaned = re.sub(r'`+\s*$', '', cleaned)

    # Remove trailing parenthetical comments, e.g. (This sets S = 21)
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned)
    # Remove trailing arrow comments, e.g. -> S = 21
    cleaned = re.sub(r'\s*->\s*.*$', '', cleaned)
    # Remove trailing equals comments, e.g. = 21
    cleaned = re.sub(r'\s*=\s*[\d.]+\s*$', '', cleaned)

    # Remove inline backtick-wrapped operator names, e.g. `Sum` -> Sum
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)

    return cleaned.strip()


def extract_expression_from_response(response: str) -> List[List[str]]:
    """
    Extract operator expression steps from model response.

    Strategy (by priority):
    1. Try structured ### Program section first (supports ANSI escape codes)
    2. Fall back to markdown code blocks
    3. Fall back to heuristic matching: filter by operator name + argument count
    """
    operators = {
        "Sum", "Multiple", "Equal", "Gougu", "Gsin", "Gcos", "Gtan",
        "Cos_Law", "Sin_Law", "Iso_Tri_Ang", "Median", "Geo_Mean",
        "Proportion", "Ratio", "Chord2_Ang", "TanSec_Ang",
        "Tria_BH_Area", "Tria_SAS_Area", "PRK_Perim", "Para_Area",
        "Rect_Area", "Rhom_Area", "Kite_Area", "Trap_Area",
        "Circle_R_Circum", "Circle_D_Circum", "Circle_R_Area",
        "Circle_D_Area", "ArcSeg_Area", "Ngon_Angsum",
        "RNgon_B_Area", "RNgon_L_Area", "RNgon_H_Area", "Get"
    }

    clean_response = _strip_ansi_escape(response)

    # Strategy 1: Extract from structured ### Program section
    program_section_patterns = [
        r'###\s*Program\s*\n(.*?)(?:\n\s*###\s*Answer|\n\s*$)',
        r'###\s*Program\s*\n(.*?)$',
        r'###\s*Program\s*[:\-]?\s*\n(.*?)(?:\n\s*###\s*Answer|\n\s*$)',
    ]

    for pattern in program_section_patterns:
        matches = re.findall(pattern, clean_response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            section = match.strip()
            if not section:
                continue
            expressions = []
            lines = section.split("\n")
            for line in lines:
                cleaned = _clean_operator_line(line)
                if not cleaned:
                    continue
                # Skip ANSWER lines
                if re.match(r'^ANSWER\s*:', cleaned, re.IGNORECASE):
                    continue
                # Skip section headers
                if re.match(r'^###?\s', cleaned):
                    continue
                tokens = cleaned.split()
                if not tokens:
                    continue
                # Check first token is an operator name
                if tokens[0] in operators:
                    expressions.append(tokens)
            if expressions:
                return expressions

    # Strategy 2: Extract from markdown code blocks
    code_block_pattern = re.compile(r'```(?:python|text|plaintext|program)?\s*\n(.*?)\n```', re.DOTALL)
    code_blocks = code_block_pattern.findall(clean_response)
    if code_blocks:
        all_exprs = []
        for block in code_blocks:
            for line in block.strip().split("\n"):
                cleaned = _clean_operator_line(line)
                if not cleaned:
                    continue
                tokens = cleaned.split()
                if tokens and tokens[0] in operators:
                    all_exprs.append(tokens)
        if all_exprs:
            return all_exprs

    # Strategy 3: Heuristic matching — extract from whole response
    # Prefer to search only between ### Program and ### Answer
    program_match = re.search(
        r'###\s*Program\s*\n(.*?)(?:\n\s*###\s*Answer|\n\s*$)',
        clean_response, re.DOTALL | re.IGNORECASE,
    )
    if program_match:
        search_text = program_match.group(1)
    else:
        search_text = clean_response

    expressions = []
    lines = search_text.strip().split("\n")

    for line in lines:
        cleaned = _clean_operator_line(line)
        if not cleaned:
            continue

        # Skip ANSWER lines
        if re.match(r'^ANSWER\s*:', cleaned, re.IGNORECASE):
            continue

        # Skip section headers
        if re.match(r'^###?\s', cleaned):
            continue

        tokens = cleaned.split()
        if not tokens or tokens[0] not in operators:
            continue

        # Relaxed token count limit: operator name + args, max 6 tokens
        if len(tokens) > 6:
            continue

        # Extra check: skip if tokens contain obvious natural language words
        non_operator_words = {
            "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "and", "or", "but", "if", "then", "else", "so", "thus",
            "to", "of", "in", "on", "at", "by", "for", "from", "with",
            "that", "this", "these", "those", "it", "its",
            "i", "a", "an", "not", "no", "yes", "set", "find", "let",
            "step", "theorem", "use", "using", "all", "also", "as",
            "first", "next", "then", "finally", "now", "here", "there",
            "one", "two", "each", "every", "some", "any", "both",
            "which", "what", "when", "how", "why", "where",
        }
        has_natural_language = any(
            t.lower() in non_operator_words for t in tokens[1:]
        )
        if has_natural_language:
            continue

        expressions.append(tokens)

    return expressions


def _analyze_error_reason(
    sample: Dict,
    response: str,
    predicted_answer: Optional[float],
    predicted_expression: List[List[str]],
    ans_correct: bool,
    exp_correct: bool,
) -> str:
    """
    分析错误原因，返回可读的错误描述。
    """
    if ans_correct and exp_correct:
        return "完全正确"

    reasons = []
    ground_truth_answer = float(sample["answer"])
    ground_truth_expression = sample.get("expression", [])

    # 1. 分析答案错误原因
    if not ans_correct:
        if predicted_answer is None:
            reasons.append("模型未输出有效数值答案")
        else:
            diff = abs(predicted_answer - ground_truth_answer)
            if diff < 0.5:
                reasons.append(
                    f"答案偏差较小 (预测={predicted_answer}, 标准={ground_truth_answer}, 偏差={diff:.4f})"
                )
            elif diff < 10:
                reasons.append(
                    f"答案偏差较大 (预测={predicted_answer}, 标准={ground_truth_answer}, 偏差={diff:.2f})"
                )
            else:
                reasons.append(
                    f"答案严重偏离 (预测={predicted_answer}, 标准={ground_truth_answer}, 偏差={diff:.2f})"
                )

    # 2. 分析表达式错误原因
    if not exp_correct:
        if not predicted_expression:
            reasons.append("模型未输出有效的解题表达式")
        else:
            pred_flat = []
            for step in predicted_expression:
                pred_flat.extend(step)

            # 比较长度
            if len(pred_flat) != len(ground_truth_expression):
                reasons.append(
                    f"表达式长度不匹配 (预测={len(pred_flat)} tokens, 标准={len(ground_truth_expression)} tokens)"
                )
            else:
                # 逐 token 比较，找出第一个不同的位置
                mismatch_positions = []
                for i, (pred, gt) in enumerate(zip(pred_flat, ground_truth_expression)):
                    if pred != gt:
                        mismatch_positions.append(f"位置{i}: 预测 '{pred}' != 标准 '{gt}'")
                if mismatch_positions:
                    reasons.append(f"表达式 token 不匹配: {', '.join(mismatch_positions[:5])}")

    # 3. 分析模型输出是否为空或过短
    if not response or len(response.strip()) < 10:
        reasons.append("模型输出为空或过短，可能推理失败")

    return "; ".join(reasons) if reasons else "未知错误"


def evaluate_predictions(
    samples: List[Dict],
    responses: List[str],
    output_dir: str,
) -> Dict:
    """
    评测模型预测结果。
    计算 Answer Accuracy（答案准确率）和 Expression Accuracy（表达式准确率）。
    详细结果包含每个样本的错误原因分析。
    """
    total = len(samples)
    answer_correct = 0
    expression_correct = 0
    detailed_results = []
    error_samples = []  # 只记录错误样本的详细信息

    for sample, response in zip(samples, responses):
        problem_id = sample.get("id", "unknown")
        ground_truth_answer = float(sample["answer"])
        ground_truth_expression = sample.get("expression", [])

        predicted_answer = extract_answer_from_response(response)
        predicted_expression = extract_expression_from_response(response)

        # 判断答案是否正确（容差 5e-3，与 PGPS 原论文一致）
        ans_correct = False
        if predicted_answer is not None:
            ans_correct = abs(predicted_answer - ground_truth_answer) < 5e-3

        # 判断表达式是否完全匹配
        # ground_truth_expression 是列表格式: ["Multiple", "V0", "C2", "N2", "Get", "V0"]
        # predicted_expression 是列表的列表: [["Multiple", "V0", "C2", "N2"], ["Get", "V0"]]
        exp_correct = False
        if predicted_expression:
            # 将预测的表达式展平为单个 token 列表
            pred_flat = []
            for step in predicted_expression:
                pred_flat.extend(step)
            if len(pred_flat) == len(ground_truth_expression):
                exp_correct = all(
                    pred == gt
                    for pred, gt in zip(pred_flat, ground_truth_expression)
                )

        if ans_correct:
            answer_correct += 1
        if exp_correct:
            expression_correct += 1

        # 分析错误原因
        error_reason = _analyze_error_reason(
            sample, response, predicted_answer,
            predicted_expression, ans_correct, exp_correct,
        )

        detail = {
            "id": problem_id,
            "text": sample.get("text", ""),
            "ground_truth_answer": ground_truth_answer,
            "predicted_answer": predicted_answer,
            "answer_correct": ans_correct,
            "expression_correct": exp_correct,
            "error_reason": error_reason,
            "model_response": response,
        }
        detailed_results.append(detail)

        # 只收集错误样本
        if not ans_correct or not exp_correct:
            error_samples.append(detail)

    answer_accuracy = answer_correct / total if total > 0 else 0.0
    expression_accuracy = expression_correct / total if total > 0 else 0.0

    # 统计错误类型分布
    error_type_counts = {}
    for err in error_samples:
        reason = err.get("error_reason", "未知")
        # 提取主要错误类型（取第一个分号前的内容）
        main_type = reason.split(";")[0].strip()
        error_type_counts[main_type] = error_type_counts.get(main_type, 0) + 1

    # 保存详细结果
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": total,
                "answer_correct": answer_correct,
                "answer_accuracy": answer_accuracy,
                "expression_correct": expression_correct,
                "expression_accuracy": expression_accuracy,
                "error_count": len(error_samples),
                "error_type_distribution": error_type_counts,
            },
            "error_samples": error_samples,
            "all_details": detailed_results,
        }, f, ensure_ascii=False, indent=2)
    
    return {
        "total": total,
        "answer_correct": answer_correct,
        "answer_accuracy": answer_accuracy,
        "expression_correct": expression_correct,
        "expression_accuracy": expression_accuracy,
        "error_count": len(error_samples),
        "error_type_distribution": error_type_counts,
        "results_path": results_path,
    }


def run_evaluation(args):
    """
    主评测流程:
    1. 加载数据集
    2. 加载模型
    3. 逐样本推理
    4. 评测计算
    """
    # 设置可见的 GPU 卡
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices
    print(f"CUDA_VISIBLE_DEVICES = {args.visible_cuda_devices}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 1. 加载数据集
    print(f"正在加载数据集: {args.dataset_dir}/{args.dataset}/test.json")
    samples = load_dataset(args.dataset_dir, args.dataset)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"共加载 {len(samples)} 条测试样本")
    
    # 2. 加载模型
    model, processor = load_model_and_processor(
        args.model_path, args.torch_dtype
    )
    
    # 3. 逐样本推理
    responses = []
    # Diagram 图片目录在 dataset_dir 同级目录下
    # 数据集结构: data_pgps9k/PGPS9K/{PGPS9K/test.json, Diagram/*.png}
    # dataset_dir = data_pgps9k/PGPS9K, Diagram 在同级
    diagram_dir = os.path.join(args.dataset_dir, "Diagram")
    
    print(f"\n开始评测，共 {len(samples)} 条样本...")
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        text = sample.get("text", "")
        diagram_filename = sample.get("diagram", "")
        diagram_path = os.path.join(diagram_dir, diagram_filename)
        
        # 构建 prompt
        system_prompt, user_message = build_prompt(text)
        
        # 准备对话消息
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        user_content = []
        # 如果图片存在，加入图片
        if os.path.exists(diagram_path):
            try:
                image = Image.open(diagram_path).convert("RGB")
                user_content.append({"type": "image", "image": image})
            except Exception as e:
                print(f"\n警告: 无法加载图片 {diagram_path}: {e}")
        else:
            print(f"\n警告: 图片不存在 {diagram_path}")
        
        user_content.append({"type": "text", "text": user_message})
        messages.append({"role": "user", "content": user_content})
        
        try:
            # 使用 processor 处理
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0,
                }
                if args.temperature > 0:
                    gen_kwargs["temperature"] = args.temperature
                    gen_kwargs["top_p"] = args.top_p
                generated_ids = model.generate(**inputs, **gen_kwargs)
            
            # 解码输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            responses.append(response)
            
        except Exception as e:
            print(f"\n样本 {sample.get('id', idx)} 推理失败: {e}")
            traceback.print_exc()
            responses.append("")
    
    # 4. 评测
    print("\n正在计算评测指标...")
    results = evaluate_predictions(samples, responses, args.output_dir)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("PGPS 评测结果 (Qwen3.5-9B)")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"总样本数: {results['total']}")
    print(f"Answer Accuracy: {results['answer_accuracy']:.4f} ({results['answer_correct']}/{results['total']})")
    print(f"Expression Accuracy: {results['expression_accuracy']:.4f} ({results['expression_correct']}/{results['total']})")
    print(f"错误样本数: {results['error_count']}")
    print()
    print("错误类型分布:")
    if results.get("error_type_distribution"):
        for error_type, count in sorted(
            results["error_type_distribution"].items(),
            key=lambda x: -x[1]
        ):
            print(f"  - {error_type}: {count} 条")
    print()
    print(f"详细结果已保存至: {results['results_path']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)

