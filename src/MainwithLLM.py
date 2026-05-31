from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Sequence

from Answering.LLMAnswerer import LLMConfig, QwenAnswerGenerator
from Searcher.HybridSearch import hybrid_search


TRIAL_QUERY = "Điều kiện cấp giấy chứng nhận quyền sử dụng đất là gì?"


def safe_preview(value: Any, limit: int = 260) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_retrieval_results(results: Sequence[Dict[str, Any]]) -> None:
    print()
    print("Retrieved sources")
    print("=" * 100)
    for result in results:
        print("-" * 100)
        print(f"Rank: {result.get('rank')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Fused score: {float(result.get('score', 0.0)):.4f}")
        if "base_fusion_score" in result:
            print(f"Base fusion: {float(result.get('base_fusion_score', 0.0)):.6f}")
        print(f"BM25 raw: {float(result.get('bm25_raw_score', result.get('bm25_score', 0.0))):.4f}")
        print(f"BM25 rank: {result.get('bm25_rank')}")
        print(f"BM25 RRF: {float(result.get('bm25_rrf_score', 0.0)):.6f}")
        print(f"Vector raw: {float(result.get('vector_raw_score', result.get('vector_score', 0.0))):.4f}")
        print(f"Vector rank: {result.get('vector_rank')}")
        print(f"Vector RRF: {float(result.get('vector_rrf_score', 0.0)):.6f}")
        print(f"RRF: {float(result.get('rrf_score', 0.0)):.6f}")
        print(f"Document number: {safe_preview(result.get('document_number'), 120)}")
        print(f"Title: {safe_preview(result.get('title'), 180)}")
        print(f"Content: {safe_preview(result.get('content'))}")


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
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only run retrieval and print sources; do not load/generate with LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Question")
    print("=" * 100)
    print(args.query)
    print()

    start = time.time()
    results = hybrid_search(
        args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
    )
    retrieval_time = time.time() - start

    print(f"Retrieval runtime: {retrieval_time:.2f} seconds")
    print(f"Result count: {len(results)}")
    print_retrieval_results(results)

    if args.no_generate:
        print()
        print("Skipped LLM generation because --no-generate was set.")
        return

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

    start = time.time()
    answer = generator.generate(args.query, results)
    generation_time = time.time() - start

    print()
    print("Generated answer")
    print("=" * 100)
    print(answer)
    print()
    print(f"Generation runtime: {generation_time:.2f} seconds")


if __name__ == "__main__":
    main()
