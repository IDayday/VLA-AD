#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.utils.conversation import get_conv_template  # noqa: E402
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image  # noqa: E402

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


@dataclass
class ProbeExample:
    dataset: str
    row_index: int
    pair_index: int
    system: Optional[str]
    history: List[Tuple[str, str]]
    question: str
    answer: str
    images: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether the released ReCogDrive Stage1 VLM looks adapted to the "
            "released Bench2Drive Traj/QA SFT data. The main signal is teacher-forced "
            "answer NLL versus a same-dataset shuffled-answer control."
        )
    )
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "checkpoints/recogdrive/ReCogDrive-VLM-2B")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/project/recogdrive_pretraining"))
    parser.add_argument("--bench2drive-image-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument("--datasets", nargs="+", choices=("traj", "qa"), default=["traj", "qa"])
    parser.add_argument("--max-records", type=int, default=16)
    parser.add_argument("--pairs-per-record", type=int, default=1)
    parser.add_argument("--sample-mode", choices=("head", "reservoir"), default="head")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--mode", choices=("text", "vision"), default="text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-image-patches", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/bench2drive_stage1_probe/report.json")
    parser.add_argument("--use-flash-attn", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def dataset_path(data_root: Path, name: str) -> Path:
    if name == "traj":
        return data_root / "Bench2drive_Traj" / "Bench2drive_Traj.jsonl"
    if name == "qa":
        return data_root / "Bench2drive_QA" / "Bench2drive_QA.jsonl"
    raise ValueError(name)


def iter_jsonl(path: Path, max_records: int, sample_mode: str, rng: random.Random) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if sample_mode == "head":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_records:
                    break
                if line.strip():
                    yield idx, json.loads(line)
        return

    if sample_mode != "reservoir":
        raise ValueError(sample_mode)

    reservoir: List[Tuple[int, str]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            seen += 1
            if len(reservoir) < max_records:
                reservoir.append((idx, line))
            else:
                j = rng.randrange(seen)
                if j < max_records:
                    reservoir[j] = (idx, line)
    for idx, line in sorted(reservoir, key=lambda item: item[0]):
        yield idx, json.loads(line)


def conversation_pairs(row: Dict[str, Any], pairs_per_record: int) -> List[Tuple[Optional[str], List[Tuple[str, str]], str, str]]:
    conversations = list(row.get("conversations") or [])
    system = None
    if conversations and conversations[0].get("from") == "system":
        system = str(conversations.pop(0).get("value", ""))

    pairs: List[Tuple[Optional[str], List[Tuple[str, str]], str, str]] = []
    history: List[Tuple[str, str]] = []
    pending_question: Optional[str] = None
    for turn in conversations:
        speaker = turn.get("from")
        value = str(turn.get("value", ""))
        if speaker == "human":
            pending_question = value
        elif speaker == "gpt" and pending_question is not None:
            pairs.append((system, list(history), pending_question, value))
            history.append((pending_question, value))
            pending_question = None
            if len(pairs) >= pairs_per_record:
                break
    return pairs


def load_examples(args: argparse.Namespace) -> List[ProbeExample]:
    examples: List[ProbeExample] = []
    rng = random.Random(args.seed)
    for dataset in args.datasets:
        path = dataset_path(args.data_root, dataset)
        if not path.is_file():
            raise FileNotFoundError(path)
        for row_index, row in iter_jsonl(path, args.max_records, args.sample_mode, rng):
            images = [str(x) for x in row.get("image") or []]
            for pair_index, (system, history, question, answer) in enumerate(
                conversation_pairs(row, args.pairs_per_record)
            ):
                examples.append(
                    ProbeExample(
                        dataset=dataset,
                        row_index=row_index,
                        pair_index=pair_index,
                        system=system,
                        history=history,
                        question=question,
                        answer=answer,
                        images=images,
                    )
                )
    if not examples:
        raise RuntimeError("No probe examples loaded.")
    return examples


def resolve_image_path(image_root: Path, image_ref: str) -> Path:
    ref = image_ref
    for prefix in ("./Bench2drive/v1/", "Bench2drive/v1/", "./"):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            break
    return image_root / ref


def build_prompt_text(
    *,
    template_name: str,
    system_message: Optional[str],
    history: Sequence[Tuple[str, str]],
    question: str,
    answer: Optional[str],
) -> str:
    template = get_conv_template(template_name)
    if system_message:
        template.system_message = system_message
    for old_question, old_answer in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], answer)
    return template.get_prompt()


def replace_text_image_placeholders(text: str) -> str:
    return text.replace("<image>", "[image]")


def replace_visual_image_placeholders(text: str, num_image_token: int, num_patches_list: Sequence[int]) -> str:
    out = text
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * int(num_patches) + IMG_END_TOKEN
        out = out.replace("<image>", image_tokens, 1)
    return out


def encode_pair(
    tokenizer: Any,
    *,
    prefix_text: str,
    full_text: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
    encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    prefix_len = min(prefix_ids.shape[-1], input_ids.shape[-1])
    labels[:, :prefix_len] = -100
    if (labels != -100).sum().item() == 0:
        raise ValueError("No answer tokens left after tokenization/truncation.")
    return input_ids, attention_mask, labels, prefix_len


def load_visual_inputs(example: ProbeExample, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
    pixel_values: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    for image_ref in example.images[: args.max_images]:
        path = resolve_image_path(args.bench2drive_image_root, image_ref)
        if not path.is_file():
            raise FileNotFoundError(path)
        patches = load_image(str(path), max_num=args.max_image_patches)
        pixel_values.append(patches)
        num_patches_list.append(int(patches.shape[0]))
    if not pixel_values:
        raise ValueError("Vision mode requires at least one image.")
    return torch.cat(pixel_values, dim=0).to(device), num_patches_list


@torch.no_grad()
def answer_loss(
    model: Any,
    tokenizer: Any,
    example: ProbeExample,
    answer: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    prefix_text = build_prompt_text(
        template_name=model.template,
        system_message=example.system,
        history=example.history,
        question=example.question,
        answer=None,
    )
    full_text = build_prompt_text(
        template_name=model.template,
        system_message=example.system,
        history=example.history,
        question=example.question,
        answer=answer,
    )

    pixel_values = None
    image_flags = None
    num_patches_list: List[int] = []
    if args.mode == "vision":
        pixel_values, num_patches_list = load_visual_inputs(example, args, device)
        prefix_text = replace_visual_image_placeholders(prefix_text, model.num_image_token, num_patches_list)
        full_text = replace_visual_image_placeholders(full_text, model.num_image_token, num_patches_list)
    else:
        prefix_text = replace_text_image_placeholders(prefix_text)
        full_text = replace_text_image_placeholders(full_text)

    input_ids, attention_mask, labels, prefix_len = encode_pair(
        tokenizer,
        prefix_text=prefix_text,
        full_text=full_text,
        max_length=args.max_length,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    token_count = int((labels != -100).sum().item())
    if args.mode == "vision":
        image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=device)
        outputs = model(
            pixel_values=pixel_values.to(next(model.parameters()).dtype),
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss.float()
    else:
        outputs = model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss.float()

    return {
        "nll": float(loss.item()),
        "token_count": token_count,
        "prefix_tokens": int(prefix_len),
        "total_tokens": int(input_ids.shape[-1]),
        "num_image_patches": int(sum(num_patches_list)),
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = dtype_from_name(args.dtype)

    examples = load_examples(args)
    shuffled_answers = [ex.answer for ex in examples]
    random.shuffle(shuffled_answers)
    if len(shuffled_answers) > 1 and all(a == b.answer for a, b in zip(shuffled_answers, examples)):
        shuffled_answers = shuffled_answers[1:] + shuffled_answers[:1]

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=args.use_flash_attn,
        local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    rows: List[Dict[str, Any]] = []
    for idx, (example, shuffled_answer) in enumerate(zip(examples, shuffled_answers)):
        real = answer_loss(model, tokenizer, example, example.answer, args, device)
        shuffled = answer_loss(model, tokenizer, example, shuffled_answer, args, device)
        rows.append(
            {
                "index": idx,
                "dataset": example.dataset,
                "row_index": example.row_index,
                "pair_index": example.pair_index,
                "real_nll": real["nll"],
                "shuffled_nll": shuffled["nll"],
                "delta_shuffled_minus_real": shuffled["nll"] - real["nll"],
                "real_token_count": real["token_count"],
                "total_tokens": real["total_tokens"],
                "num_image_patches": real["num_image_patches"],
                "question_preview": example.question[:160],
                "answer_preview": example.answer[:160],
                "shuffled_answer_preview": shuffled_answer[:160],
            }
        )
        print(
            f"[{idx + 1}/{len(examples)}] {example.dataset} row={example.row_index} "
            f"real={real['nll']:.4f} shuffled={shuffled['nll']:.4f} "
            f"delta={shuffled['nll'] - real['nll']:.4f}",
            flush=True,
        )

    real_losses = [row["real_nll"] for row in rows]
    shuffled_losses = [row["shuffled_nll"] for row in rows]
    deltas = [row["delta_shuffled_minus_real"] for row in rows]
    by_dataset: Dict[str, Dict[str, float]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "examples": len(subset),
            "real_nll_mean": mean([row["real_nll"] for row in subset]),
            "shuffled_nll_mean": mean([row["shuffled_nll"] for row in subset]),
            "delta_mean": mean([row["delta_shuffled_minus_real"] for row in subset]),
        }

    report = {
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "mode": args.mode,
        "sample_mode": args.sample_mode,
        "examples": len(rows),
        "summary": {
            "real_nll_mean": mean(real_losses),
            "shuffled_nll_mean": mean(shuffled_losses),
            "delta_mean": mean(deltas),
            "real_lower_than_shuffled_fraction": mean([1.0 if d > 0 else 0.0 for d in deltas]),
        },
        "by_dataset": by_dataset,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
