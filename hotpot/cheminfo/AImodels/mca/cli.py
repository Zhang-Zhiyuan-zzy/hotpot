"""Command-line entry point for SMILES inference."""

from __future__ import annotations

import argparse
import json

from .api import MCAPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict site-resolved MCA values")
    parser.add_argument("smiles", nargs="+")
    parser.add_argument("--model-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--variant", choices=("fp32", "fp16"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    predictor = MCAPredictor(
        model_dir=args.model_dir,
        device=args.device,
        variant=args.variant,
        batch_size=args.batch_size,
    )
    result = predictor.predict(args.smiles)
    print(json.dumps([item.to_dict() for item in result], indent=2))


if __name__ == "__main__":
    main()
