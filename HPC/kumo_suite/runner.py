"""Resume at (split, fold, segment, context bag); fail loudly on context OOM."""
import contextlib
from dataclasses import asdict
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import log_loss, roc_auc_score

from .data import columns, digest, load, population, transform


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temp, path)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid prediction vector")
    if len(np.unique(y)) != 2:
        return {"rows": len(y), "ks": None, "auc": None, "logloss": None}
    return {"rows": len(y), "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
            "auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1]))}


def sample_context(y, limit, seed, bag, mode):
    n = min(limit, len(y))
    if n < 2 or len(np.unique(y)) != 2:
        raise ValueError("A context needs both classes and at least two rows")
    rng = np.random.default_rng(seed + (bag * 1009 if mode != "coverage" else 0))
    if mode == "uniform":
        idx = rng.choice(len(y), n, replace=False)
        if len(np.unique(y[idx])) < 2:
            raise ValueError("Uniform context has one class; increase context")
        return idx
    positive = max(1, min(n - 1, round(n * np.mean(y))))
    # Availability bounds also matter in tiny expert populations.
    positive = min(np.sum(y == 1), max(positive, n - np.sum(y == 0)))
    idx = []
    for label, count in ((0, n - positive), (1, positive)):
        pool = rng.permutation(np.flatnonzero(y == label))
        if mode == "coverage":
            pool = np.roll(pool, -(bag * count) % len(pool))
        idx.extend(pool[:count])
    return rng.permutation(np.asarray(idx, dtype=int))


class DiagnosticBackend:
    """CPU plumbing check only. Its results must never be called KuMo results."""
    def __init__(self, **kwargs):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        self.classes = (ColumnTransformer, SimpleImputer, LogisticRegression, make_pipeline, OneHotEncoder, StandardScaler)

    def fit(self, x, y, seed, estimators):
        CT, Imputer, LR, pipe, OHE, Scale = self.classes
        nums = x.select_dtypes(include="number").columns.tolist()
        cats = [c for c in x if c not in nums]
        self.model = pipe(CT([("num", pipe(Imputer(strategy="median", keep_empty_features=True), Scale()), nums),
                              ("cat", pipe(Imputer(strategy="constant", fill_value="MISSING"), OHE(handle_unknown="ignore")), cats)]),
                          LR(max_iter=300, C=.1, random_state=seed))
        self.model.fit(x, y)

    def predict(self, x, batch):
        return self.model.predict_proba(x)[:, list(self.model.classes_).index(1)]

    def clear(self):
        self.model = None


class KumoBackend:
    def __init__(self, size, device):
        import torch
        import sdm
        self.torch, self.sdm = torch, sdm
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no automatic CPU or smaller-model fallback")
        self.model = sdm.models.KumoTabular(task="classification", size=size, device=self.device).eval()
        self.min_query_batch = None

    def table(self, frame, target=False):
        overrides = {"target": "categorical"} if target else None
        return self.sdm.TableTensor.from_pandas(frame,
            stypes=self.sdm.infer_stypes(frame, overrides=overrides), device=self.device)

    def autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.device.type == "cuda" else contextlib.nullcontext()

    def fit(self, x, y, seed, estimators):
        self.torch.manual_seed(seed)
        frame = x.reset_index(drop=True).copy()
        frame["target"] = np.asarray(y, dtype=np.int64)
        ctx = self.table(frame, target=True)
        with self.torch.inference_mode(), self.autocast():
            self.model.fit(ctx.drop_columns("target"), ctx["target"], num_estimators=estimators, estimator_batch_size=1)

    def predict(self, x, batch):
        output, start = [], 0
        while start < len(x):
            stop = min(start + batch, len(x))
            try:
                with self.torch.inference_mode(), self.autocast():
                    pred = self.model.predict(self.table(x.iloc[start:stop]))
                    # Class names are defined by SDM. No label-based orientation guessing.
                    scores = pred["1"].numerical.reshape(-1).float().cpu().numpy()
                output.append(scores)
                start = stop
                self.min_query_batch = batch if self.min_query_batch is None else min(self.min_query_batch, batch)
            except self.torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                self.torch.cuda.empty_cache()
                print(f"Query OOM: retry same context with batch={batch}", flush=True)
        return np.concatenate(output) if output else np.empty(0)

    def clear(self):
        self.model.clear()
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def code_fingerprint():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    packages = {}
    for name in ("numpy", "pandas", "polars", "scipy", "scikit-learn", "torch", "structured-data-models"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "packages": packages}
    import torch
    result["cuda_runtime"] = torch.version.cuda
    if torch.cuda.is_available():
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return result


def run(exp, inputs, output, folds, backend="kumo", device="cuda", query_batch=512, validation_limit=None, predict_test=False):
    if query_batch < 1:
        raise ValueError("query-batch must be positive")
    tr, te, split, manifest = load(inputs)
    identity = {"experiment": asdict(exp), "inputs": manifest["files"], "code": code_fingerprint(),
                "environment": environment(), "backend": backend, "device": device, "query_batch": query_batch,
                "validation_limit": validation_limit, "predict_test": predict_test}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(output) / f"{exp.name}__{backend}__{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "w") as lock:
        # Same run writes serially; separate configurations/array tasks can use different directories.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(directory / "run.json", identity)
        cols = columns(tr, exp)
        pops, testpops = population(tr), population(te)
        y = tr.target.to_numpy(dtype=np.int64)
        model = None
        try:
            for fold in folds:
                idx_train = np.arange(len(tr)) if fold == -1 else np.flatnonzero(split[f"fold_seed{exp.seed}"].to_numpy() != fold)
                idx_val = np.empty(0, dtype=int) if fold == -1 else np.flatnonzero(split[f"fold_seed{exp.seed}"].to_numpy() == fold)
                if validation_limit:
                    idx_val = np.random.default_rng(exp.seed + fold).permutation(idx_val)[:validation_limit]
                idx_test = np.arange(len(te)) if predict_test or fold == -1 else np.empty(0, dtype=int)
                query = pd.concat([tr.iloc[idx_val][cols], te.iloc[idx_test][cols]], ignore_index=True)
                xp, xq = transform(tr.iloc[idx_train][cols].reset_index(drop=True), query, exp.transform)
                qp = np.concatenate([pops[idx_val], testpops[idx_test]])
                yp = y[idx_train]
                predictions = {}
                segments = ["global"] if exp.segment == "global" else ["full", "thin", "prev_no_repay"]
                if exp.segment == "soft":
                    segments = ["global"] + segments
                for seg_index, segment in enumerate(segments):
                    candidates = np.arange(len(idx_train)) if segment == "global" else np.flatnonzero(pops[idx_train] == segment)
                    queries = np.arange(len(query)) if segment == "global" else np.flatnonzero(qp == segment)
                    if not len(queries):
                        continue
                    bag_pred = []
                    for bag in range(exp.bags):
                        prefix = directory / f"fold{fold}_{segment}_bag{bag}"
                        metadata, arrays = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
                        if metadata.exists() and arrays.exists() and json.loads(metadata.read_text())["sha256"] == digest(arrays):
                            with np.load(arrays, allow_pickle=False) as saved:
                                if not np.array_equal(saved["query_indices"], queries):
                                    raise ValueError("Resume query alignment mismatch")
                                bag_pred.append(saved["prediction"])
                            print(f"Resume {exp.name} fold={fold} {segment} bag={bag}", flush=True)
                            continue
                        context_seed = exp.seed + (fold + 1) * 100003 + seg_index * 10007
                        chosen = candidates[sample_context(yp[candidates], exp.context, context_seed, bag, exp.sampling)]
                        start = time.monotonic()
                        if model is None:
                            model = KumoBackend(exp.size, device) if backend == "kumo" else DiagnosticBackend()
                        model.fit(xp.iloc[chosen], yp[chosen], context_seed + bag, exp.estimators)
                        pred = model.predict(xq.iloc[queries], query_batch)
                        metrics(np.zeros(len(pred), dtype=int), pred)  # finite/range check even without two classes
                        tmp = arrays.with_name(arrays.name + f".{os.getpid()}.tmp")
                        with open(tmp, "wb") as f:
                            np.savez_compressed(f, prediction=pred, query_indices=queries,
                                context_ids=tr.iloc[idx_train[chosen]].application_id.to_numpy(dtype=str))
                        os.replace(tmp, arrays)
                        info = {"sha256": digest(arrays), "seconds": time.monotonic() - start, "context_rows": len(chosen),
                                "features": list(xp.columns), "positive_rate": float(yp[chosen].mean()),
                                "min_query_batch": getattr(model, "min_query_batch", None),
                                "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                        if backend == "kumo" and model.device.type == "cuda":
                            info["peak_gpu_bytes"] = model.torch.cuda.max_memory_allocated(model.device)
                        atomic_json(metadata, info)
                        bag_pred.append(pred)
                        model.clear()
                        print(f"Done {exp.name} fold={fold} {segment} bag={bag} seconds={info['seconds']:.1f}", flush=True)
                    predictions[segment] = (queries, np.mean(bag_pred, axis=0))
                pred = np.full(len(query), np.nan)
                for segment, (indices, values) in predictions.items():
                    if segment != "global" or exp.segment == "global":
                        pred[indices] = values
                if exp.segment == "soft":
                    pred = .5 * pred + .5 * predictions["global"][1]
                metrics(np.zeros(len(pred), dtype=int), pred)
                val = pd.DataFrame({"application_id": tr.iloc[idx_val].application_id.to_numpy(), "target": y[idx_val],
                                    "fold": fold, "population": pops[idx_val], "prediction": pred[:len(idx_val)]})
                test = pd.DataFrame({"application_id": te.iloc[idx_test].application_id.to_numpy(), "prediction": pred[len(idx_val):]})
                # Readers require the JSON completion marker; intermediate files are never summarized.
                for label, frame in (("oof", val), ("test", test)):
                    temp = directory / f"fold{fold}_{label}.{os.getpid()}.tmp"
                    frame.to_parquet(temp, index=False)
                    os.replace(temp, directory / f"fold{fold}_{label}.parquet")
                result = {"fold": fold, "metrics": metrics(y[idx_val], pred[:len(idx_val)]), "populations": {}}
                for pop in np.unique(pops[idx_val]):
                    mask = pops[idx_val] == pop
                    result["populations"][pop] = metrics(y[idx_val][mask], pred[:len(idx_val)][mask])
                result["files"] = {label: digest(directory / f"fold{fold}_{label}.parquet") for label in ("oof", "test")}
                atomic_json(directory / f"fold{fold}_complete.json", result)
                if fold == -1:
                    pd.DataFrame({"application_id": test.application_id, "default_probability": test.prediction}).to_csv(
                        directory / "refit_probability.csv", index=False)
                    pd.DataFrame({"application_id": test.application_id,
                                  "default_probability": rankdata(test.prediction) / len(test)}).to_csv(
                        directory / "refit_rank.csv", index=False)
        except Exception as error:
            atomic_json(directory / "last_error.json", {"error": repr(error), "time": time.time()})
            raise
        finally:
            if model is not None:
                model.clear()
    return directory


def summarize(output):
    rows = []
    for directory in sorted(Path(output).glob("*__*__*")):
        if not (directory / "run.json").exists():
            continue
        spec = json.loads((directory / "run.json").read_text())
        fold_results = []
        for fold in range(5):
            marker = directory / f"fold{fold}_complete.json"
            if not marker.exists():
                continue
            info = json.loads(marker.read_text())
            for label, expected in info["files"].items():
                if digest(directory / f"fold{fold}_{label}.parquet") != expected:
                    raise ValueError(f"Corrupted completed fold: {directory} {fold}")
            fold_results.append(info)
        row = {"run": directory.name, "backend": spec["backend"], **spec["experiment"],
               "completed_folds": len(fold_results), "limited_validation": spec["validation_limit"] is not None}
        scores = [x["metrics"]["ks"] for x in fold_results if x["metrics"]["ks"] is not None]
        row["mean_fold_ks"] = float(np.mean(scores)) if scores else None
        row["fold_ks"] = [x["metrics"]["ks"] for x in fold_results]
        if len(fold_results) == 5 and spec["validation_limit"] is None:
            oof = pd.concat([pd.read_parquet(directory / f"fold{f}_oof.parquet") for f in range(5)])
            if not oof.application_id.is_unique:
                raise ValueError("OOF IDs overlap")
            row["pooled_oof"] = metrics(oof.target, oof.prediction)
            oof.to_parquet(directory / "oof.parquet", index=False)
            tests = [pd.read_parquet(directory / f"fold{f}_test.parquet") for f in range(5)]
            if all(len(t) for t in tests):
                ids = tests[0].application_id
                aligned = [t.set_index("application_id").loc[ids, "prediction"].to_numpy() for t in tests]
                prob = np.mean(aligned, axis=0)
                pd.DataFrame({"application_id": ids, "default_probability": prob}).to_csv(directory / "submission_probability.csv", index=False)
                pd.DataFrame({"application_id": ids, "default_probability": rankdata(prob) / len(prob)}).to_csv(directory / "submission_rank.csv", index=False)
        rows.append(row)
    atomic_json(Path(output) / "summary.json", rows)
    if rows:
        pd.json_normalize(rows).to_csv(Path(output) / "summary.csv", index=False)
    comparisons = []
    complete = [r for r in rows if r["completed_folds"] == 5 and not r["limited_validation"]]
    for row in complete:
        references = [r for r in complete if r["backend"] == row["backend"] and r["seed"] == row["seed"]
                      and (r["size"], r["view"], r["context"], r["bags"], r["estimators"], r["segment"],
                           r["sampling"], r["transform"], r["drop"]) ==
                          ("large", "compact", 4096, 1, 4, "global", "stratified", "native", "none")]
        for ref in references:
            if ref["run"] == row["run"]:
                continue
            a_dir, b_dir = Path(output) / row["run"], Path(output) / ref["run"]
            a_spec, b_spec = [json.loads((d / "run.json").read_text()) for d in (a_dir, b_dir)]
            if any(a_spec[k] != b_spec[k] for k in ("inputs", "code", "environment")):
                continue
            a, b = [pd.read_parquet(d / "oof.parquet").sort_values("application_id") for d in (a_dir, b_dir)]
            if not a[["application_id", "target", "fold"]].reset_index(drop=True).equals(b[["application_id", "target", "fold"]].reset_index(drop=True)):
                raise ValueError("Paired comparison IDs/labels/folds differ")
            delta = np.asarray(row["fold_ks"]) - np.asarray(ref["fold_ks"])
            comparisons.append({"candidate": row["run"], "reference": ref["run"], "deltas": delta.tolist(),
                                "mean_delta": float(delta.mean()), "wins": int((delta > 0).sum()),
                                "screen_gate": bool(delta.mean() >= .0024 and (delta > 0).sum() >= 4)})
    atomic_json(Path(output) / "paired_comparisons.json", comparisons)
    return rows
