"""Build portable inputs from existing v5 plus new point-in-time raw aggregates.

New aggregates are independently implemented. Pieter's experiment notes informed
the censoring, recency and provider-specific hypotheses; no reference code vendored.
"""
import json
from pathlib import Path

import polars as pl

from .data import digest

C = pl.col


def raw_features(raw, applications):
    cut = applications.select("customer_id", C("decision_datetime").cast(pl.Datetime).alias("cutoff")).lazy()

    def scan(name):
        return pl.scan_csv(Path(raw) / f"{name}.csv", try_parse_dates=True).join(cut, on="customer_id", validate="m:1")

    r = scan("repayment_history").filter(C("due_date").cast(pl.Datetime) <= C("cutoff")).with_columns(
        ago=(C("cutoff") - C("due_date").cast(pl.Datetime)).dt.total_seconds() / 86400,
        known_paid=pl.when(C("last_payment_datetime") < C("cutoff")).then(C("amount_paid")).otherwise(0),
    )
    rep = r.group_by("customer_id").agg(
        **{f"new_rep_eligible_share{d}": (C("max_days_past_due") >= d).filter(C("ago") >= d).mean()
           for d in (30, 60, 90)},
        **{f"new_rep_decay{h}": (C("max_days_past_due") * (-C("ago") / h).exp()).sum() /
           (-C("ago") / h).exp().sum() for h in (30, 90, 365)},
        new_rep_known_paid_ratio=C("known_paid").sum() / C("amount_due").sum(),
    ).collect()
    loan = r.sort("due_date", "installment_id").group_by("customer_id", "loan_id").agg(
        late_share=(C("max_days_past_due") > 0).mean(),
        trend=C("max_days_past_due").tail(3).mean() - C("max_days_past_due").head(3).mean(),
        last=C("max_days_past_due").last(),
    ).group_by("customer_id").agg(
        new_rep_worst_loan_late_share=C("late_share").max(), new_rep_loan_trend_mean=C("trend").mean(),
        new_rep_loan_trend_max=C("trend").max(), new_rep_last_loan_dpd_max=C("last").max(),
    ).collect()
    rep = rep.join(loan, on="customer_id", how="full", coalesce=True, validate="1:1")

    b = scan("bureau_observations").filter(C("available_datetime") <= C("cutoff")).sort("report_datetime", "available_datetime")
    aggs = {}
    for p in ("A", "B", "C"):
        filt = C("bureau_provider") == f"Bureau_{p}"
        for col, short in (("score", "score"), ("outstanding_balance", "debt"), ("inquiries_last_6m", "inq")):
            aggs[f"new_bur_{p}_{short}_delta"] = C(col).filter(filt).last() - C(col).filter(filt).first()
        aggs[f"new_bur_{p}_latest_score"] = C("score").filter(filt).last()
    bur = b.group_by("customer_id").agg(**aggs).collect()

    a = scan("account_behavior").filter(C("month").dt.offset_by("1mo").cast(pl.Datetime) <= C("cutoff")).with_columns(
        age_months=C("cutoff").dt.year() * 12 + C("cutoff").dt.month() - C("month").dt.year() * 12 - C("month").dt.month(),
        net=C("inflows") - C("outflows"),
    ).sort("month")
    acc = a.group_by("customer_id").agg(
        new_acc_negative_share=(C("net") < 0).mean(), new_acc_latest_net=C("net").last(),
        new_acc_net_shift=C("net").filter(C("age_months") <= 3).mean() - C("net").filter(C("age_months") > 3).mean(),
        new_acc_income_shift=C("income_like_credits").filter(C("age_months") <= 3).mean() - C("income_like_credits").filter(C("age_months") > 3).mean(),
        new_acc_latest_closing=C("closing_balance").last(),
        new_acc_lowdays_shift=C("low_balance_days").filter(C("age_months") <= 3).mean() - C("low_balance_days").filter(C("age_months") > 3).mean(),
    ).collect()

    # Item rows collapse to one order before counting, spending and time features.
    e = scan("ecommerce_transactions").filter(C("transaction_datetime") <= C("cutoff")).group_by("customer_id", "order_id").agg(
        C("transaction_datetime").first(), C("cutoff").first(), C("order_status").first(),
        C("order_total").first(), C("payment_method").first(),
    ).with_columns(ago=(C("cutoff") - C("transaction_datetime")).dt.total_seconds() / 86400)
    completed = C("order_status") != "cancelled"
    eco_aggs = {
        "new_eco_weekend_share": (C("transaction_datetime").dt.weekday() >= 6).mean(),
        "new_eco_cancel30": (C("order_status") == "cancelled").filter(C("ago") <= 30).mean(),
    }
    for w in (7, 30, 90, 180):
        within = completed & (C("ago") <= w)
        eco_aggs[f"new_eco_spend{w}"] = C("order_total").filter(within).sum()
        eco_aggs[f"new_eco_orders{w}"] = within.sum()
    eco = e.group_by("customer_id").agg(**eco_aggs).collect()
    result = applications.select("customer_id")
    for df in (rep, bur, acc, eco):
        result = result.join(df, on="customer_id", how="left", validate="1:1")
    return result


def prepare(processed, raw, output):
    processed, output = Path(processed), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Preparation output must be an empty directory (protect existing inputs)")
    tr, te = [pl.read_parquet(processed / f"{s}_features_v5.parquet") for s in ("train", "test")]
    apps = pl.concat([tr.select("customer_id", "decision_datetime"), te.select("customer_id", "decision_datetime")])
    if apps["customer_id"].n_unique() != apps.height:
        raise ValueError("Customer grain is not unique")
    extra = raw_features(raw, apps)
    new_cols = [c for c in extra.columns if c.startswith("new_")]
    for name, df in (("train", tr), ("test", te)):
        df = df.join(extra, on="customer_id", how="left", validate="1:1", maintain_order="left")
        # Deterministic rounding stabilizes ties across parallel aggregation order.
        df = df.with_columns(pl.selectors.float().round_sig_figs(10))
        df.write_parquet(output / f"{name}.parquet")
    folds = pl.read_parquet(processed / "cv_folds_70k.parquet")
    folds.write_parquet(output / "folds.parquet")
    source_files = [processed / f"{s}_features_v5.parquet" for s in ("train", "test")]
    source_files += [processed / "cv_folds_70k.parquet"]
    source_files += [Path(raw) / f"{s}.csv" for s in ("repayment_history", "bureau_observations", "account_behavior", "ecommerce_transactions")]
    manifest = {"schema_version": 1, "new_columns": new_cols, "train_rows": tr.height, "test_rows": te.height,
                "files": {n: digest(output / n) for n in ("train.parquet", "test.parquet", "folds.parquet")},
                "sources": {str(p): digest(p) for p in source_files}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
