"""CPU tests never download model weights or require CUDA."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp

from HPC.kumo_suite.data import COMPACT, columns, digest, load, transform
from HPC.kumo_suite.plan import Experiment, experiments
from HPC.kumo_suite.runner import metrics, run, sample_context, summarize


class SuiteTests(unittest.TestCase):
    def test_plan(self):
        plan = experiments()
        self.assertGreaterEqual(len(plan), 50)
        self.assertEqual(len(plan), len({json.dumps(vars(x), sort_keys=True) for x in plan.values()}))
        self.assertEqual({x.size for x in plan.values()}, {"small", "medium", "large"})

    def test_metric_orientation_and_ties(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([.1, .1, .8, .5, .4, .5])
        self.assertEqual(metrics(y, p)["ks"], ks_2samp(p[y == 1], p[y == 0]).statistic)
        self.assertEqual(metrics(y, p)["ks"], metrics(y, 1 - p)["ks"])
        self.assertGreater(metrics(y, p)["auc"], .5)
        with self.assertRaises(ValueError):
            metrics(y, p * np.nan)

    def test_context_and_coverage(self):
        y = np.array([0] * 87 + [1] * 13)
        for mode in ("uniform", "stratified", "coverage"):
            idx = sample_context(y, 32, 42, 0, mode)
            self.assertEqual(len(idx), len(set(idx)))
            self.assertEqual(len(idx), 32)
            np.testing.assert_array_equal(idx, sample_context(y, 32, 42, 0, mode))
        a = sample_context(y, 32, 42, 0, "coverage")
        b = sample_context(y, 32, 42, 1, "coverage")
        self.assertEqual(len(set(a) & set(b)), 0)
        self.assertEqual(len(sample_context(y, 4096, 42, 0, "stratified")), 100)

    def test_fold_transform(self):
        train = pd.DataFrame({"x": [1., 2., np.nan], "duplicate": [1., 2., np.nan], "constant": [4] * 3,
                              "cat": ["a", "b", None]})
        q = pd.DataFrame({"x": [100.], "duplicate": [100.], "constant": [9], "cat": ["unseen"]})
        a, b = transform(train, q, "rank")
        self.assertNotIn("duplicate", a)
        self.assertNotIn("constant", a)
        self.assertEqual(b.x.iloc[0], 1)
        self.assertTrue(np.isnan(a.x.iloc[-1]))
        self.assertEqual(b.cat.iloc[0], "unseen")

    def test_resume_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "inputs"
            folder.mkdir()
            rng = np.random.default_rng(0)
            frame = pd.DataFrame({c: rng.normal(size=120) for c in COMPACT})
            frame["application_id"] = [f"A{i}" for i in range(120)]
            frame["customer_id"] = [f"C{i}" for i in range(120)]
            frame["target"] = np.tile([0, 1], 60)
            frame["cov_has_repayment"] = np.tile([1, 0, 0], 40)
            frame["cov_has_prev_apps"] = np.tile([1, 0, 1], 40)
            frame["cov_has_account"] = frame.cov_has_prev_apps
            tr, te = frame.iloc[:100], frame.iloc[100:].drop(columns="target")
            folds = pd.DataFrame({"application_id": tr.application_id,
                                  **{f"fold_seed{s}": np.arange(100) % 5 for s in (42, 2026, 999)}})
            for name, df in (("train", tr), ("test", te), ("folds", folds.sample(frac=1, random_state=2))):
                df.to_parquet(folder / f"{name}.parquet", index=False)
            manifest = {"files": {n: digest(folder / n) for n in ("train.parquet", "test.parquet", "folds.parquet")}}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            _, _, aligned, _ = load(folder)
            self.assertEqual(aligned.application_id.tolist(), tr.application_id.tolist())
            exp = Experiment("test", context=24, bags=2, segment="soft")
            output = Path(tmp) / "results"
            directory = run(exp, folder, output, range(5), "diagnostic", "cpu", 8, predict_test=True)
            artifact = next(directory.glob("fold0_global_bag0.npz"))
            before = artifact.stat().st_mtime_ns
            run(exp, folder, output, [0], "diagnostic", "cpu", 8, predict_test=True)
            self.assertEqual(before, artifact.stat().st_mtime_ns)
            summary = summarize(output)
            self.assertEqual(summary[0]["completed_folds"], 5)
            self.assertEqual(len(pd.read_parquet(directory / "oof.parquet")), 100)
            self.assertEqual(len(pd.read_csv(directory / "submission_rank.csv")), 20)
            run(replace(exp, name="refit"), folder, output, [-1], "diagnostic", "cpu", 8)
            (folder / "test.parquet").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                load(folder)


if __name__ == "__main__":
    unittest.main()
