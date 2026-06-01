from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_MODEL_ROOT = PROJECT_ROOT / "models"

QWEN_MODEL_OPTIONS = {
    "7b": {
        "awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gptq": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "none": "Qwen/Qwen2.5-7B-Instruct",
    },
    "14b": {
        "awq": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "gptq": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "none": "Qwen/Qwen2.5-14B-Instruct",
    },
}

LOCAL_QWEN_MODEL_OPTIONS = {
    "7b": {
        "awq": LOCAL_MODEL_ROOT / "Qwen2.5-7B-Instruct-AWQ",
        "gptq": LOCAL_MODEL_ROOT / "Qwen2.5-7B-Instruct-GPTQ-Int4",
        "none": LOCAL_MODEL_ROOT / "Qwen2.5-7B-Instruct",
    },
    "14b": {
        "awq": LOCAL_MODEL_ROOT / "Qwen2.5-14B-Instruct-AWQ",
        "gptq": LOCAL_MODEL_ROOT / "Qwen2.5-14B-Instruct-GPTQ-Int4",
        "none": LOCAL_MODEL_ROOT / "Qwen2.5-14B-Instruct",
    },
}


@dataclass(frozen=True)
class LLMConfig:
    model_size: str = "7b"
    quantization: str = "none"
    model_name: str | None = None
    local_files_only: bool = True
    max_context_chars: int = 9000
    max_new_tokens: int = 700
    temperature: float = 0.2
    top_p: float = 0.9
    device_map: str = "auto"
    verbose: bool = False


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def resolve_model_name(config: LLMConfig) -> str:
    if config.model_name:
        return config.model_name

    size = config.model_size.lower()
    quantization = config.quantization.lower()
    local_path = LOCAL_QWEN_MODEL_OPTIONS.get(size, {}).get(quantization)
    if local_path is not None and local_path.exists():
        return str(local_path)

    try:
        return QWEN_MODEL_OPTIONS[size][quantization]
    except KeyError as exc:
        sizes = ", ".join(QWEN_MODEL_OPTIONS)
        quantizations = ", ".join(next(iter(QWEN_MODEL_OPTIONS.values())))
        raise ValueError(
            f"Unsupported Qwen option: size={config.model_size!r}, quantization={config.quantization!r}. "
            f"Use size in {{{sizes}}} and quantization in {{{quantizations}}}."
        ) from exc


def format_sources(results: Sequence[Dict[str, Any]], max_chars: int) -> str:
    blocks: List[str] = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        title = safe_text(result.get("title"))
        content = safe_text(result.get("content"))
        if not content:
            continue

        source = "\n".join([
            f"[{index}]",
            f"chunk_id: {safe_text(result.get('chunk_id'))}",
            f"document_id: {safe_text(result.get('document_id'))}",
            f"document_number: {safe_text(result.get('document_number'))}",
            f"legal_type: {safe_text(result.get('legal_type'))}",
            f"title: {title}",
            f"content: {content}",
        ])

        if used_chars + len(source) > max_chars:
            break
        blocks.append(source)
        used_chars += len(source)

    return "\n\n".join(blocks)


def build_messages(query: str, results: Sequence[Dict[str, Any]], max_context_chars: int) -> List[Dict[str, str]]:
    context = format_sources(results, max_context_chars)
    system_prompt = (
        "Bạn là trợ lý pháp lý đất đai Việt Nam. "
        "Chỉ trả lời dựa trên các nguồn được cung cấp. "
        "Nếu nguồn chưa đủ chắc chắn, nói rõ là chưa đủ căn cứ. "
        "Ưu tiên văn bản trung ương và văn bản mới hơn khi có quan hệ sửa đổi/thay thế. "
        "Trích dẫn nguồn bằng chunk_id hoặc số thứ tự nguồn."
    )
    user_prompt = (
        f"Câu hỏi: {query}\n\n"
        f"Nguồn truy xuất:\n{context}\n\n"
        "Hãy trả lời ngắn gọn, có cấu trúc, và nêu căn cứ chính."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class QwenAnswerGenerator:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = resolve_model_name(config)
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        if self.config.verbose:
            print(f"Loading answer model: {self.model_name}")
            print(f"Quantization option: {self.config.quantization}")
            print(f"Local files only: {self.config.local_files_only}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )

        model_kwargs: Dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": True,
            "local_files_only": self.config.local_files_only,
        }

        if self.config.quantization.lower() == "none":
            model_kwargs["torch_dtype"] = "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the selected Qwen model. "
                "For AWQ/GPTQ models, make sure the quantization backend supported by your "
                "transformers install is available. For local/offline use, put the model under "
                f"{LOCAL_MODEL_ROOT} or pass --model-name /path/to/model. "
                f"Selected model: {self.model_name}"
            ) from exc

        self.model.eval()

    def generate(self, query: str, results: Sequence[Dict[str, Any]]) -> str:
        self.load()
        messages = build_messages(query, results, self.config.max_context_chars)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = self.config.temperature > 0
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
