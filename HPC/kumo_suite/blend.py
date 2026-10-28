"""Fixed-weight rank blends of complete, matching OOF artifacts only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import digest
from .runner import atomic_json, metrics


def blend(runs, output, weights=None):
    runs, output = [Path(p) for p in runs], Path(output)
    if output.exists():
        raise ValueError("Blend output must not already exist")
    weights = np.ones(len(runs)) if weights is None else np.asarray(weights, dtype=float)
    if len(weights) != len(runs) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Need one finite nonnegative weight per run, with positive sum")
    weights /= weights.sum()
    specs, oofs, tests = [], [], []
    for d in runs:
        spec = json.loads((d / "run.json").read_text())
        if spec["backend"] != "kumo" or spec["validation_limit"] is not None:
            raise ValueError("Blending requires full KuMo CV, not diagnostic or limited validation")
        parts, testparts = [], []
        for fold in range(5):
            complete = json.loads((d / f"fold{fold}_complete.json").read_text())
            for label in ("oof", "test"):
                if digest(d / f"fold{fold}_{label}.parquet") != complete["files"][label]:
                    raise ValueError("Corrupted fold artifact")
            parts.append(pd.read_parquet(d / f"fold{fold}_oof.parquet"))
            testparts.append(pd.read_parquet(d / f"fold{fold}_test.parquet").sort_values("application_id"))
        oofs.append(pd.concat(parts).sort_values("application_id").reset_index(drop=True))
        if not all(len(t) and t.application_id.is_unique for t in testparts):
            raise ValueError("Every fold needs complete test predictions; use --predict-test")
        if any(t.application_id.tolist() != testparts[0].application_id.tolist() for t in testparts):
            raise ValueError("Test IDs differ between folds")
        test = testparts[0].copy()
        test["prediction"] = np.mean([t.prediction.to_numpy() for t in testparts], axis=0)
        tests.append(test.reset_index(drop=True))
        specs.append(spec)
    base = oofs[0]
    pred, testpred = np.zeros(len(base)), np.zeros(len(tests[0]))
    for spec, oof, test, weight in zip(specs, oofs, tests, weights):
        if spec["inputs"] != specs[0]["inputs"] or spec["experiment"]["seed"] != specs[0]["experiment"]["seed"]:
            raise ValueError("Blending requires matching inputs and fold seed")
        if not oof[["application_id", "target", "fold"]].equals(base[["application_id", "target", "fold"]]):
            raise ValueError("OOF alignment mismatch")
        if not test.application_id.equals(tests[0].application_id):
            raise ValueError("Test alignment mismatch")
        for fold in range(5):
            mask = oof.fold.to_numpy() == fold
            pred[mask] += weight * rankdata(oof.prediction.to_numpy()[mask]) / mask.sum()
        testpred += weight * rankdata(test.prediction) / len(test)
    result = {"runs": [str(d) for d in runs], "weights": weights.tolist(),
              "fold_metrics": [metrics(base.target[base.fold == f], pred[base.fold == f]) for f in range(5)],
              "note": "Fixed-weight blend. Selecting members/weights using these scores creates selection optimism; confirm on other split seeds."}
    result["mean_fold_ks"] = float(np.mean([m["ks"] for m in result["fold_metrics"]]))
    output.mkdir(parents=True)
    base["prediction"] = pred
    base.to_parquet(output / "oof.parquet", index=False)
    pd.DataFrame({"application_id": tests[0].application_id,
                  "default_probability": rankdata(testpred) / len(testpred)}).to_csv(output / "submission_rank.csv", index=False)
    atomic_json(output / "blend.json", result)
    return result
