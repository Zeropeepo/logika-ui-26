import argparse
import json
from pathlib import Path

from .plan import records, select


def main():
    parser = argparse.ArgumentParser(description="Portable KuMo ablations; no training on import")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create portable, checksummed inputs")
    prep.add_argument("--processed", default="data/processed")
    prep.add_argument("--raw", default="data/raw")
    prep.add_argument("--output", default="data/kumo_inputs")
    plan = sub.add_parser("plan", help="Print configurations without loading models")
    run = sub.add_parser("run", help="Explicitly launch chosen experiments")
    for p in (plan, run):
        p.add_argument("--preset", choices=["screen", "confirm", "all"], default="screen")
        p.add_argument("--names", nargs="+")
    plan.add_argument("--save")
    run.add_argument("--inputs", default="data/kumo_inputs")
    run.add_argument("--output", default="HPC/kumo_results")
    run.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    run.add_argument("--refit", action="store_true", help="Full training pool -> test; no OOF claim")
    run.add_argument("--backend", choices=["kumo", "diagnostic"], default="kumo")
    run.add_argument("--device", default="cuda")
    run.add_argument("--query-batch", type=int, default=512)
    run.add_argument("--validation-limit", type=int, help="Plumbing smoke only, not comparable CV")
    run.add_argument("--predict-test", action="store_true", help="Also predict test for each CV context")
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", default="HPC/kumo_results")
    mix = sub.add_parser("blend", help="Fixed-weight rank blend of full matching KuMo CV runs")
    mix.add_argument("--runs", nargs="+", required=True)
    mix.add_argument("--weights", nargs="+", type=float)
    mix.add_argument("--output", required=True)
    pack = sub.add_parser("bundle", help="Separate source and prepared-input zip archives")
    pack.add_argument("--inputs", default="data/kumo_inputs")
    pack.add_argument("--output", default="transfer")
    args = parser.parse_args()
    if args.command == "plan":
        result = records(args.preset, args.names)
        content = json.dumps(result, indent=2)
        if args.save:
            Path(args.save).write_text(content + "\n")
        print(content)
        print(f"Experiments: {len(result)}; CV folds: {len(result) * 5}. Expert and bag runs require multiple context fits.")
    elif args.command == "prepare":
        from .prepare import prepare
        print(json.dumps(prepare(args.processed, args.raw, args.output), indent=2))
    elif args.command == "run":
        from .runner import run
        if args.validation_limit is not None and args.validation_limit < 2:
            parser.error("validation-limit must be >=2")
        if args.refit and args.validation_limit:
            parser.error("refit cannot use validation-limit")
        for exp in select(args.preset, args.names):
            print(run(exp, args.inputs, args.output, [-1] if args.refit else sorted(set(args.folds)),
                      args.backend, args.device, args.query_batch, args.validation_limit, args.predict_test), flush=True)
    elif args.command == "bundle":
        from .package import package
        print(json.dumps(package(args.inputs, args.output), indent=2))
    elif args.command == "blend":
        from .blend import blend
        print(json.dumps(blend(args.runs, args.output, args.weights), indent=2))
    else:
        from .runner import summarize
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
