from __future__ import annotations

import argparse
import contextlib
import io
import time
from typing import Any, Dict, Sequence

from Answering.LLMAnswerer import LLMConfig, QwenAnswerGenerator
from Searcher.HybridSearch import hybrid_search


TRIAL_QUERY = "Điều kiện cấp giấy chứng nhận quyền sử dụng đất là gì?"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def print_retrieval_results(results: Sequence[Dict[str, Any]]) -> None:
    print("Sources")
    print("=" * 100)
    for result in results:
        print("-" * 100)
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Content: {clean_text(result.get('content'))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LandBot retrieval and Qwen answer generation.")
    parser.add_argument("--query", default=TRIAL_QUERY, help="Question to answer.")
    parser.add_argument("--model-size", choices=["7b", "14b"], default="7b", help="Qwen2.5 Instruct size.")
    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", "none"],
        default="none",
        help="Model variant. Use none for the installed local full model, or awq/gptq for quantized models.",
    )
    parser.add_argument("--model-name", default=None, help="Override Hugging Face/local model path.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow downloading model files from Hugging Face. Default is local/offline loading.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of final retrieval sources.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Hybrid vector weight.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank smoothing constant.")
    parser.add_argument("--max-context-chars", type=int, default=9000, help="Max source characters sent to LLM.")
    parser.add_argument("--max-new-tokens", type=int, default=700, help="Max answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Generation top-p.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = hybrid_search(
            args.query,
            top_k=args.top_k,
            alpha=args.alpha,
            rrf_k=args.rrf_k,
        )
    loading_time = time.time() - start

    llm_config = LLMConfig(
        model_size=args.model_size,
        quantization=args.quantization,
        model_name=args.model_name,
        local_files_only=not args.online,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generator = QwenAnswerGenerator(llm_config)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        answer = generator.generate(args.query, results)

    print(f"Loading time: {loading_time:.2f} seconds")
    print()
    print_retrieval_results(results)
    print()
    print("LLM answer")
    print("=" * 100)
    print(answer)


if __name__ == "__main__":
    main()
