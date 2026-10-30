"""Point-in-time inference features matching the scored local candidate."""
from pathlib import Path

import numpy as np
import polars as pl

C = pl.col
MAX_LEN = 64
CATEGORIES = ('groceries', 'fashion', 'electronics', 'home', 'health', 'education', 'personal_care')
METHODS = ('COD', 'e_wallet', 'bank_transfer', 'debit_card', 'credit_card')


def window_features(test, raw_dir):
    raw_dir = Path(raw_dir)
    cut = pl.from_pandas(test[['customer_id', 'decision_datetime']]).rename({'decision_datetime':'cutoff'}).lazy()

    def scan(name):
        return pl.scan_csv(raw_dir / f'{name}.csv', try_parse_dates=True).join(
            cut, on='customer_id', how='inner', validate='m:1')

    def aggregates(frame, tag, age, windows, numeric):
        exprs = []
        for w in windows:
            cond = C(age) <= w if w is not None else (C(age).is_not_null() | C(age).is_null())
            suffix = str(w) if w is not None else 'all'
            exprs.append(cond.sum().alias(f'win_{tag}_n_{suffix}'))
            for col in numeric:
                x = C(col).cast(pl.Float64).filter(cond)
                for stat in ('mean', 'max', 'std'):
                    exprs.append(getattr(x, stat)().alias(f'win_{tag}_{col}_{stat}_{suffix}'))
        return frame.group_by('customer_id').agg(exprs).collect(engine='streaming')

    eco = scan('ecommerce_transactions').filter(C('transaction_datetime') <= C('cutoff'))
    order = eco.group_by('customer_id', 'order_id').agg(
        C('cutoff').first(), C('transaction_datetime').first(), C('order_status').first(),
        C('order_total').first(), C('payment_method').first(), C('quantity').sum().alias('items'),
        C('product_category').n_unique().alias('categories')).with_columns(
        age=(C('cutoff') - C('transaction_datetime')).dt.total_seconds()/86400,
        spend=pl.when(C('order_status') == 'completed').then(C('order_total')).otherwise(None),
        canceled=(C('order_status') == 'cancelled').cast(pl.Float64),
        cod=(C('payment_method') == 'COD').cast(pl.Float64))
    parts = [aggregates(order, 'eco', 'age', [7, 30, 90, 180, 365, None],
                        ['spend', 'items', 'categories', 'canceled', 'cod'])]
    repay = scan('repayment_history').filter(C('due_date').cast(pl.Datetime) <= C('cutoff')).with_columns(
        age=(C('cutoff') - C('due_date').cast(pl.Datetime)).dt.total_seconds()/86400,
        paid_asof=pl.when(C('last_payment_datetime') < C('cutoff')).then(C('amount_paid')).otherwise(0),
        paid_date_asof=pl.when(C('last_payment_datetime') < C('cutoff')).then(C('last_payment_datetime')).otherwise(None))
    repay = repay.with_columns(unpaid=C('amount_due') - C('paid_asof'),
        paid_fraction=C('paid_asof')/C('amount_due'),
        delay=(C('paid_date_asof')-C('due_date').cast(pl.Datetime)).dt.total_seconds()/86400)
    parts.append(aggregates(repay, 'rep', 'age', [30, 90, 180, 365, 730, None],
                            ['amount_due', 'unpaid', 'paid_fraction', 'days_past_due', 'max_days_past_due', 'delay']))
    account = scan('account_behavior').filter(C('month').dt.offset_by('1mo').cast(pl.Datetime) <= C('cutoff')).with_columns(
        age=C('cutoff').dt.year()*12 + C('cutoff').dt.month() - C('month').dt.year()*12 - C('month').dt.month(),
        net=C('inflows')-C('outflows'))
    parts.append(aggregates(account, 'acc', 'age', [1, 3, 6, 12, 24, None],
        ['closing_balance', 'inflows', 'outflows', 'income_like_credits', 'average_balance', 'minimum_balance', 'low_balance_days', 'net']))
    bureau = scan('bureau_observations').filter(C('available_datetime') <= C('cutoff')).with_columns(
        age=(C('cutoff')-C('report_datetime')).dt.total_seconds()/86400)
    parts.append(aggregates(bureau, 'bur', 'age', [30, 90, 180, 365, None],
        ['score', 'active_accounts', 'outstanding_balance', 'revolving_credit_limit',
         'revolving_outstanding_balance', 'utilization', 'inquiries_last_6m', 'delinquent_accounts_last_12m']))
    prev = scan('previous_applications').filter(C('decision_datetime') <= C('cutoff')).with_columns(
        age=(C('cutoff')-C('decision_datetime')).dt.total_seconds()/86400,
        approved=(C('decision') == 'approved').cast(pl.Float64),
        rejected=(C('decision') == 'rejected').cast(pl.Float64),
        haircut=C('approved_amount')/C('requested_amount'))
    parts.append(aggregates(prev, 'prev', 'age', [30, 90, 180, 365, 730, None],
        ['requested_amount', 'approved_amount', 'annual_interest_rate', 'scheduled_monthly_installment', 'approved', 'rejected', 'haircut']))
    result = cut.collect().select('customer_id')
    for part in parts:
        result = result.join(part, on='customer_id', how='left', validate='1:1')
    return result.with_columns(pl.selectors.float().replace([float('inf'), float('-inf')], None)
                               .round_sig_figs(6).cast(pl.Float32)).to_pandas()


def sequences(test, raw_dir):
    apps = pl.DataFrame({'customer_id': test.customer_id.to_numpy(),
                         'cutoff': test.decision_datetime.to_numpy(),
                         'position': np.arange(len(test), dtype=np.int32)}).lazy()
    raw = (pl.scan_csv(Path(raw_dir) / 'ecommerce_transactions.csv', try_parse_dates=True)
           .join(apps, on='customer_id', how='inner', validate='m:1')
           .filter(C('transaction_datetime') <= C('cutoff')))
    aggs = [C('position').first(), C('cutoff').first(), C('transaction_datetime').first(),
            C('order_total').first(), C('order_status').first(), C('payment_method').first(),
            C('quantity').sum().alias('items')]
    aggs.extend(C('item_total').filter(C('product_category') == n).sum().alias('category_'+n) for n in CATEGORIES)
    events = (raw.group_by('customer_id', 'order_id').agg(aggs).sort('position', 'transaction_datetime', 'order_id')
              .with_columns(age=(C('cutoff')-C('transaction_datetime')).dt.total_seconds()/86400,
                            gap=C('transaction_datetime').diff().over('position').dt.total_seconds().fill_null(0)/86400,
                            total=pl.len().over('position'))
              .group_by('position', maintain_order=True).tail(MAX_LEN).sort('position', 'transaction_datetime', 'order_id')
              .with_columns(slot=pl.int_range(pl.len()).over('position')).collect(engine='streaming'))
    pos = events['position'].to_numpy().astype(int)
    slot = events['slot'].to_numpy().astype(int)
    age = events['age'].to_numpy().astype(np.float32)
    gap = events['gap'].to_numpy().astype(np.float32)
    cats = events.select(['category_'+n for n in CATEGORIES]).to_numpy().astype(np.float32)
    denom = np.maximum(cats.sum(axis=1, keepdims=True), 1)
    methods = events['payment_method'].to_numpy()
    arr = np.column_stack([np.log1p(events['order_total'].to_numpy().astype(np.float32)),
        np.log1p(age), np.log1p(gap), events['order_status'].to_numpy() == 'completed',
        np.log1p(events['items'].to_numpy().astype(np.float32)),
        *[(methods == v).astype(np.float32) for v in METHODS], cats/denom]).astype(np.float32)
    X = np.zeros((len(test), MAX_LEN, 17), dtype=np.float32)
    X[pos, slot] = arr
    lens = np.bincount(pos, minlength=len(test)).astype(np.int64)
    total = np.zeros(len(test), dtype=np.float32)
    total[pos] = events['total'].to_numpy().astype(np.float32)
    if (lens == 0).any() or not np.isfinite(X).all():
        raise ValueError('Empty or invalid customer sequence')
    return X, lens, total
