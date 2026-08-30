from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .gpu_info import inspect_cuda


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fractal-doctor", description="GPU- und CUDA-Diagnose")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    args = parser.parse_args(argv)

    status = inspect_cuda()
    if args.json:
        print(json.dumps(asdict(status), indent=2, ensure_ascii=False))
    else:
        print(status.report())
    return 0 if status.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
