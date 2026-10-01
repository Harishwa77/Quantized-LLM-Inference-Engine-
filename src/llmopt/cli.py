"""Command line interface for llmopt.

Subcommands:

* ``serve``    -- run the OpenAI-compatible HTTP server.
* ``bench``    -- measure throughput and latency for a workload.
* ``quantize`` -- round-trip a model through weight quantization.
* ``info``     -- print engine and cache configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from llmopt.utils.logging import setup_logging

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top level argument parser."""
    parser = argparse.ArgumentParser(prog="llmopt", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_engine_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--preset", default="tiny", choices=["tiny", "small"])
        p.add_argument("--path", default=None, help="load a saved llmopt model instead")
        p.add_argument("--device", default="auto")
        p.add_argument("--max-model-len", type=int, default=1024)
        p.add_argument("--num-gpu-blocks", type=int, default=256)
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--max-num-seqs", type=int, default=64)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    add_engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-id", default="llmopt-tiny")
    serve.add_argument("--api-key", default=None)
    serve.add_argument("--no-prefix-cache", action="store_true")

    bench = sub.add_parser("bench", help="run a benchmark workload")
    add_engine_args(bench)
    bench.add_argument("--num-requests", type=int, default=32)
    bench.add_argument("--prompt-tokens", type=int, default=64)
    bench.add_argument("--output-tokens", type=int, default=64)
    bench.add_argument("--shared-prefix", type=float, default=0.0)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    quantize = sub.add_parser("quantize", help="quantize a saved model")
    quantize.add_argument("--path", required=True)
    quantize.add_argument("--out", default=None, help="output directory (default: in place)")
    quantize.add_argument("--bits", type=int, default=4, choices=[2, 4, 8])
    quantize.add_argument("--group-size", type=int, default=128)
    quantize.add_argument(
        "--strategy", default="rtn", choices=["rtn", "gptq"], help="gptq needs calibration data"
    )

    sub.add_parser("info", help="print engine configuration")
    return parser


def _build_engine(args: argparse.Namespace):
    from llmopt.engine.llm_engine import LLMEngine

    engine_kwargs: dict[str, Any] = {
        "device": args.device,
        "max_model_len": args.max_model_len,
        "num_gpu_blocks": args.num_gpu_blocks,
        "block_size": args.block_size,
        "max_num_seqs": args.max_num_seqs,
    }
    if getattr(args, "path", None):
        return LLMEngine.from_pretrained(args.path, **engine_kwargs)
    return LLMEngine.from_preset(args.preset, **engine_kwargs)


def _cmd_serve(args: argparse.Namespace) -> int:
    from llmopt.serving.app import ServerConfig, serve

    engine = _build_engine(args)
    if args.no_prefix_cache:
        engine.prefix_cache = None
        engine.scheduler.prefix_cache = None
    serve(
        engine,
        config=ServerConfig(
            host=args.host, port=args.port, model_id=args.model_id, api_key=args.api_key
        ),
    )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from llmopt.benchmark.runner import run_benchmark
    from llmopt.benchmark.workload import WorkloadSpec

    engine = _build_engine(args)
    spec = WorkloadSpec(
        num_requests=args.num_requests,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        shared_prefix=args.shared_prefix,
        seed=args.seed,
    )
    result = run_benchmark(engine, spec, name="llmopt-bench")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.format_table())
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    from llmopt.model.modeling import DecoderOnlyModel
    from llmopt.quantization.quant_linear import quantize_model_

    model = DecoderOnlyModel.from_pretrained(args.path, dtype=None)
    before = sum(p.numel() * p.element_size() for p in model.parameters())
    quantize_model_(
        model, bits=args.bits, group_size=args.group_size, strategy=args.strategy
    )
    after = sum(
        m.qweight.numel() * m.qweight.element_size()
        + m.scales.numel() * m.scales.element_size()
        + m.qzeros.numel() * m.qzeros.element_size()
        for m in model.modules()
        if hasattr(m, "qweight") and getattr(m, "qweight", None) is not None
    )
    out = args.out or args.path
    model.save_pretrained(out)
    print(f"quantized to {args.bits} bits: {before} -> {after} bytes ({out})")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from llmopt.engine.llm_engine import LLMEngine

    engine = LLMEngine.from_preset(args.preset)
    print(json.dumps(engine.config.to_dict(), indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a subcommand and return its exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    handlers = {
        "serve": _cmd_serve,
        "bench": _cmd_bench,
        "quantize": _cmd_quantize,
        "info": _cmd_info,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
