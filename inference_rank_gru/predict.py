"""Inference for the reconstructed 0.56133 rank-GRU ensemble."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import nn

from features import sequences, window_features


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(17, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(66, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, x, lens, stats):
        out, _ = self.gru(x)
        state = out[torch.arange(len(x), device=x.device), (lens-1).clamp(min=0)]
        return self.head(torch.cat([state, stats], dim=1)).flatten()


def check_inputs(test_path, raw_dir, artifacts, verify_raw):
    meta = json.loads((artifacts/'manifest.json').read_text())
    for name, expected in meta['artifacts'].items():
        if digest(artifacts/name) != expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    if digest(test_path) != meta['test_sha256']:
        raise ValueError('Test parquet does not match the test data used for the reported submission')
    if verify_raw:
        for name, expected in meta['raw_sha256'].items():
            if digest(raw_dir/f'{name}.csv') != expected:
                raise ValueError(f'Raw input checksum mismatch: {name}.csv')
    return meta


def predict_tabular(test, artifacts, query_batch):
    import sdm
    schema = joblib.load(artifacts/'feature_schema.joblib')
    columns, categories = schema['columns'], schema['categories']
    missing = set(columns) - set(test.columns)
    if missing:
        raise ValueError(f'Missing tabular inputs: {sorted(missing)}')
    query = test[columns].copy()
    for c in query:
        if c in categories:
            query[c] = query[c].where(query[c].notna(), None).astype(object)
        else:
            query[c] = query[c].replace([np.inf, -np.inf], np.nan)
    lr_x = query.copy()
    for c in categories:
        lr_x[c] = lr_x[c].astype('string').fillna('__missing__').astype(object)
    logistic = joblib.load(artifacts/'logistic.joblib')
    p_lr = logistic.predict_proba(lr_x)[:, 1]
    del lr_x, logistic
    for c, values in categories.items():
        query[c] = query[c].astype(pd.CategoricalDtype(categories=pd.Index(values)))
    booster = lgb.Booster(model_file=str(artifacts/'lightgbm.txt'))
    p_lgb = booster.predict(query, num_threads=4)
    del query, booster

    record = joblib.load(artifacts/'kumo_context.joblib')
    ctx = record['context']
    q = test[record['features']].copy()
    numeric = ctx.select_dtypes(include='number').columns.tolist()
    q[numeric] = q[numeric].replace([np.inf, -np.inf], np.nan)
    for col in q:
        if col not in numeric:
            q[col] = q[col].where(q[col].notna(), None).astype(object)

    def table(frame, target=False):
        frame = frame.copy()
        for c in frame.select_dtypes(include=['object']).columns:
            if pd.api.types.infer_dtype(frame[c], skipna=True) in ('boolean', 'empty'):
                frame[c] = frame[c].astype('string')
        overrides = {'target':'categorical'} if target else None
        return sdm.TableTensor.from_pandas(frame, stypes=sdm.infer_stypes(frame, overrides=overrides), device='cuda')

    torch.manual_seed(record['seed'])
    model = sdm.models.KumoTabular(task='classification', size='medium', device=torch.device('cuda')).eval()
    ctx = ctx.copy()
    ctx['target'] = np.asarray(record['labels'], dtype=np.int64)
    table_ctx = table(ctx, target=True)
    autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    with torch.inference_mode(), autocast():
        model.fit(table_ctx.drop_columns('target'), table_ctx['target'],
                  num_estimators=record['estimators'], estimator_batch_size=1)
    values = []
    offset = 0
    while offset < len(q):
        stop = min(offset+query_batch, len(q))
        try:
            with torch.inference_mode(), autocast():
                pred = model.predict(table(q.iloc[offset:stop]))
                values.append(pred['1'].numerical.reshape(-1).float().cpu().numpy())
            offset = stop
        except torch.cuda.OutOfMemoryError:
            if query_batch <= 1:
                raise
            query_batch = max(1, query_batch//2)
            torch.cuda.empty_cache()
            print(f'KuMo query batch reduced to {query_batch}', flush=True)
    model.clear()
    p_kumo = np.concatenate(values)
    return .4*p_lgb + .4*p_lr + .2*p_kumo


def predict_gru(test, raw_dir, artifacts):
    X, lens, total = sequences(test, raw_dir)
    device = torch.device('cuda')
    ckpt = torch.load(artifacts/'gru.pt', map_location=device, weights_only=True)
    net = SequenceModel().to(device)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    mean, std = ckpt['mean'], ckpt['std']
    result = []
    with torch.inference_mode():
        for start in range(0, len(X), 2048):
            stop = min(start+2048, len(X))
            x = torch.as_tensor(X[start:stop], device=device)
            lengths = torch.as_tensor(lens[start:stop], device=device)
            stats = torch.as_tensor(np.column_stack([np.log1p(total[start:stop])/5,
                lens[start:stop]>0]).astype(np.float32), device=device)
            result.append(torch.sigmoid(net((x-mean)/std, lengths, stats)).cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', type=Path, default=Path('data/kumo_inputs/test.parquet'))
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parent/'artifacts')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('model', 'replay'), default='model',
                   help='model runs reconstructed checkpoints; replay copies the verified original scored CSV')
    p.add_argument('--query-batch', type=int, default=128)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.query_batch < 1 or (args.mode == 'model' and not torch.cuda.is_available()):
        raise ValueError('A CUDA GPU and positive --query-batch are required')
    meta = check_inputs(args.test, args.raw_dir, args.artifacts, args.mode == 'model')
    test = pd.read_parquet(args.test)
    if test.application_id.isna().any() or not test.application_id.is_unique or not test.customer_id.is_unique:
        raise ValueError('Application and customer IDs must be present and unique')
    if args.mode == 'replay':
        original = args.artifacts/'submission_scored.csv'
        saved = pd.read_csv(original, dtype={'application_id':str})
        if (list(saved.columns) != ['application_id', 'default_probability'] or
                not np.array_equal(saved.application_id.to_numpy(), test.application_id.to_numpy(dtype=str)) or
                digest(original) != meta['reference_submission_sha256']):
            raise ValueError('Original submission does not match the expected IDs/schema/checksum')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, args.output)
        print(f'Replayed scored submission: {args.output} sha256={digest(args.output)}', flush=True)
        return
    print('Building point-in-time features', flush=True)
    win = window_features(test, args.raw_dir)
    frame = test.merge(win, on='customer_id', how='left', sort=False, validate='one_to_one', indicator=True)
    if not frame['_merge'].eq('both').all():
        raise ValueError('Missing customers in window features')
    frame = frame.drop(columns='_merge')
    print('Running LightGBM, logistic and KuMo Medium', flush=True)
    base = predict_tabular(frame, args.artifacts, args.query_batch)
    print('Running GRU', flush=True)
    gru = predict_gru(test, args.raw_dir, args.artifacts)
    score = .9*((rankdata(base)-.5)/len(base)) + .1*((rankdata(gru)-.5)/len(gru))
    if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise ValueError('Invalid scores')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'application_id': test.application_id, 'default_probability': score}).to_csv(args.output, index=False)
    print(f'Wrote {args.output}: rows={len(score)}, sha256={digest(args.output)}', flush=True)
    print(f"Reference scored submission sha256={meta['reference_submission_sha256']}", flush=True)


if __name__ == '__main__':
    main()
