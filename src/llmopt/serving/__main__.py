"""``python -m llmopt.serving`` entry point.

Example::

    python -m llmopt.serving --preset tiny --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from llmopt.config import ServerConfig
from llmopt.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the server."""
    parser = argparse.ArgumentParser(
        prog="llmopt.serving", description="OpenAI-compatible server for llmopt"
    )
    parser.add_argument("--preset", default="tiny", help="built-in model preset")
    parser.add_argument("--path", default=None, help="path to a saved llmopt model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default="llmopt-tiny", help="name reported by /v1/models")
    parser.add_argument("--api-key", default=None, help="require this bearer token")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, float32")
    parser.add_argument("--num-gpu-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--quant-bits", type=int, default=None, choices=[2, 4, 8])
    parser.add_argument("--quant-strategy", default="rtn", choices=["rtn", "gptq"])
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--snapshot", default=None, help="write /info JSON to this path on exit")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Start the server from command line arguments."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    logging.getLogger("llmopt").info("building engine: preset=%s device=%s", args.preset, args.device)

    from llmopt.engine.llm_engine import LLMEngine
    from llmopt.serving.app import serve

    engine_kwargs: dict[str, object] = {
        "device": args.device,
        "max_model_len": args.max_model_len,
        "num_gpu_blocks": args.num_gpu_blocks,
        "block_size": args.block_size,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.path:
        engine = LLMEngine.from_pretrained(args.path, **engine_kwargs)
    else:
        engine = LLMEngine.from_preset(args.preset, **engine_kwargs)
    if args.dtype != "auto":
        engine.config.model.torch_dtype = args.dtype
    if args.no_prefix_cache:
        engine.prefix_cache = None
        engine.scheduler.prefix_cache = None

    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        api_key=args.api_key,
        snapshot_path=args.snapshot,
    )
    # serve() writes config.snapshot_path on clean shutdown.
    serve(engine, config=config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
