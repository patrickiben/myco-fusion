"""Run the workbench or any of its analyses from the command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Myco-fusion research workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="open the local research app")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--open", action="store_true", dest="open_browser")
    bench = commands.add_parser("bench", help="compare capture scenarios on measured or idealized spectra")
    bench.add_argument("--source", choices=["measured", "idealized"], default="measured")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--noise", type=float)
    bench.add_argument("--out", type=Path)
    validation = commands.add_parser("validate", help="evaluate paired estimated/reference spectra")
    validation.add_argument("--csv", required=True, type=Path)
    validation.add_argument("--out", type=Path)
    validation.add_argument("--uncertainty", choices=["auto", "specimen", "session_cluster"], default="auto")
    calibration = commands.add_parser("calibrate", help="fit on training data, tune on validation, evaluate held-out test data")
    calibration.add_argument("--json", required=True, type=Path, dest="dataset")
    calibration.add_argument("--out", type=Path)
    calibration.add_argument("--model-out", type=Path, help="also save the fitted model for later use with predict")
    example = commands.add_parser("example", help="export a worked example built on measured reflectance and a simulated camera")
    example.add_argument("name", choices=["calibration", "validation", "repeatability"])
    example.add_argument("--out", type=Path)
    prediction = commands.add_parser("predict", help="apply a saved calibration with matching device and processing pipeline")
    prediction.add_argument("--model", required=True, type=Path)
    prediction.add_argument("--capture", required=True, type=Path)
    prediction.add_argument("--out", type=Path)
    repeatability = commands.add_parser("repeatability", help="analyze a phone-only test from a photo table")
    repeatability.add_argument("--csv", required=True, type=Path)
    repeatability.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "serve":
            from .server import serve
            serve(args.port, args.open_browser)
            return
        if args.command == "bench":
            from .bench import run_bench
            result = run_bench(seed=args.seed, noise=args.noise, source=args.source)
        elif args.command == "validate":
            from .validation import evaluate_csv
            result = evaluate_csv(args.csv.read_text(encoding="utf-8"), uncertainty=args.uncertainty)
        elif args.command == "calibrate":
            from .calibration import fit_calibration
            result = fit_calibration(json.loads(args.dataset.read_text(encoding="utf-8")))
            if args.model_out:
                args.model_out.write_text(json.dumps(result["model"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
        elif args.command == "example":
            from . import examples
            if args.name == "calibration":
                result = examples.calibration_dataset()
            else:
                text = examples.validation_csv() if args.name == "validation" else examples.repeatability_csv()
                if args.out:
                    args.out.write_text(text, encoding="utf-8")
                    print(f"Saved {args.name} example to {args.out.resolve()}")
                else:
                    print(text, end="")
                return
        elif args.command == "repeatability":
            from .repeatability import analyze_repeatability
            result = analyze_repeatability(args.csv.read_text(encoding="utf-8"))
        else:
            from .calibration import predict_calibration
            result = predict_calibration(json.loads(args.model.read_text(encoding="utf-8")), json.loads(args.capture.read_text(encoding="utf-8")))
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(output, encoding="utf-8")
            print(f"Saved {result.get('kind', 'dataset')} to {args.out.resolve()}")
        else:
            print(output, end="")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
