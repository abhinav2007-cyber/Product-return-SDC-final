"""
Rebuilds every file the dashboard needs from the raw dataset.

    python train_model.py "Online Retail.csv"

Outputs (in ./data): flagged_records.csv, customer_summary.csv, product_summary.csv,
monthly_returns.csv, country_returns.csv, invoice_data.csv, cancellation_model.pkl,
model_info.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis import customer_segments, invoice_anomalies, train_line_model
from ml_utils import CATEGORICAL, FEATURES, NUMERIC, add_derived

OUT = Path("data")
SEED = 42
# Real products have 5-digit stock codes (e.g. 85123A). POST, D, M, BANK CHARGES... are not products.
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"


def load_raw(path: str):
    df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str})
    stats = {"raw_rows": int(len(df))}
    df = df.drop_duplicates()
    stats["duplicates_removed"] = stats["raw_rows"] - int(len(df))
    # The file uses day/month/year ("01/12/10" = 1 Dec 2010). Parsing it as month/day
    # silently turns day-of-month into "month" and drops every day > 12.
    dates = pd.to_datetime(df["InvoiceDate"], format="%d/%m/%y %H:%M", errors="coerce")
    if dates.isna().mean() > 0.01:
        dates = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
    bad = int(dates.isna().sum())
    print(f"Unparseable dates: {bad}")
    df["InvoiceDate"] = dates
    stats["unparseable_dates"] = bad
    df = df.dropna(subset=["InvoiceDate"]).copy()
    df["is_cancelled"] = df["InvoiceNo"].str.startswith("C").astype(int)
    df["is_product"] = df["StockCode"].str.match(PRODUCT_CODE, na=False)
    df["abs_quantity"] = df["Quantity"].abs()
    df["line_value"] = (df["abs_quantity"] * df["UnitPrice"]).clip(lower=0)
    stats.update(
        rows_kept=int(len(df)),
        missing_customer=int(df["CustomerID"].isna().sum()),
        missing_description=int(df["Description"].isna().sum()),
        non_product_rows=int((~df["is_product"]).sum()),
        zero_or_negative_price=int((df["UnitPrice"] <= 0).sum()),
        cancelled_lines=int(df["is_cancelled"].sum()),
        negative_not_cancelled=int(((df["Quantity"] < 0) & (df["is_cancelled"] == 0)).sum()),
    )
    return df, stats


def build_invoices(df: pd.DataFrame) -> pd.DataFrame:
    inv = df.groupby("InvoiceNo").agg(
        line_count=("StockCode", "size"),
        total_abs_quantity=("abs_quantity", "sum"),
        average_unit_price=("UnitPrice", "mean"),
        unique_products=("StockCode", "nunique"),
        InvoiceDate=("InvoiceDate", "first"),
        country=("Country", "first"),
        is_cancelled=("is_cancelled", "first"),
    ).reset_index()
    inv["month"] = inv["InvoiceDate"].dt.month
    inv["weekday"] = inv["InvoiceDate"].dt.dayofweek
    inv["hour"] = inv["InvoiceDate"].dt.hour
    return add_derived(inv)


def build_reports(df: pd.DataFrame) -> None:
    neg = df["Quantity"] < 0
    reasons = pd.DataFrame({
        "Cancellation invoice": df["is_cancelled"] == 1,
        "Negative quantity": neg,
        "Zero or negative price": df["UnitPrice"] <= 0,
    })
    flagged = df[reasons.any(axis=1)].copy()
    flagged["Reason"] = reasons.loc[flagged.index].apply(lambda r: ", ".join(r.index[r]), axis=1)
    flagged["Return_Value"] = np.where(flagged["Quantity"] < 0,
                                       flagged["Quantity"].abs() * flagged["UnitPrice"], 0.0)
    flagged["CustomerID"] = flagged["CustomerID"].astype("Int64").astype(str).replace("<NA>", "")
    flagged["InvoiceDate"] = flagged["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
    flagged[["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Quantity", "UnitPrice",
             "CustomerID", "Country", "Reason", "Return_Value"]].to_csv(OUT / "flagged_records.csv", index=False)

    returns = df[neg].assign(value=lambda d: d["Quantity"].abs() * d["UnitPrice"])
    (returns.groupby(returns["InvoiceDate"].dt.strftime("%Y-%m"))
        .agg(negative_records=("Quantity", "size"), return_value=("value", "sum"))
        .rename_axis("Month").reset_index().to_csv(OUT / "monthly_returns.csv", index=False))
    (returns.groupby("Country")
        .agg(negative_records=("Quantity", "size"), return_value=("value", "sum"))
        .sort_values("negative_records", ascending=False).reset_index()
        .to_csv(OUT / "country_returns.csv", index=False))

    prod = df[df["is_product"]]
    (prod.groupby("Description").agg(
        total_records=("InvoiceNo", "count"),
        total_quantity=("Quantity", "sum"),
        negative_records=("Quantity", lambda x: int((x < 0).sum())))
        .assign(return_rate=lambda d: (d["negative_records"] / d["total_records"]).round(4))
        .sort_values("negative_records", ascending=False).reset_index()
        .to_csv(OUT / "product_summary.csv", index=False))

    cust = df[df["CustomerID"].notna()].assign(
        value=lambda d: d["Quantity"] * d["UnitPrice"],
        is_neg=lambda d: (d["Quantity"] < 0).astype(int))
    (cust.groupby("CustomerID").agg(
        total_invoices=("InvoiceNo", "nunique"),
        total_lines=("InvoiceNo", "size"),
        negative_records=("is_neg", "sum"),
        total_spending=("value", "sum"))
        .assign(return_rate=lambda d: (d["negative_records"] / d["total_lines"]).round(4))
        .sort_values("negative_records", ascending=False).reset_index()
        .assign(CustomerID=lambda d: d["CustomerID"].astype(int).astype(str))
        .to_csv(OUT / "customer_summary.csv", index=False))


def make_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def scores(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": round(float(thr), 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def train(inv: pd.DataFrame) -> dict:
    X, y = inv[FEATURES], inv["is_cancelled"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    comparison, best_name, best_cv = {}, None, -1
    for name, clf in candidates.items():
        cv_pr = cross_val_score(make_pipeline(clf), X_tr, y_tr, cv=cv, scoring="average_precision").mean()
        comparison[name] = round(float(cv_pr), 4)
        print(f"{name}: CV PR-AUC {cv_pr:.4f}")

    best_name = "Random Forest"
    pipe = make_pipeline(candidates[best_name])
    # Pick the decision threshold on out-of-fold training predictions (never on the test set).
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, oof)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    thr = float(t[int(np.argmax(f1))])

    pipe.fit(X_tr, y_tr)
    test = scores(y_te.to_numpy(), pipe.predict_proba(X_te)[:, 1], thr)
    print(best_name, test)
    joblib.dump({"pipeline": pipe, "threshold": thr, "features": FEATURES}, OUT / "cancellation_model.pkl")
    joblib.dump(pipe, OUT / "best_cancellation_model.pkl")
    joblib.dump(pipe, "best_cancellation_model.pkl")
    return {"name": best_name, "cv_pr_auc": comparison, "test": test,
            "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
            "cancel_rate": round(float(y.mean()), 4),
            "countries": sorted(inv["country"].unique().tolist())}


def main(path: str) -> None:
    OUT.mkdir(exist_ok=True)
    df, cleaning = load_raw(path)
    inv = build_invoices(df)
    inv.assign(InvoiceDate=inv["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")).to_csv(OUT / "invoice_data.csv", index=False)
    build_reports(df)
    info = train(inv)
    info["cleaning"] = cleaning
    info["line_model"] = train_line_model(df, OUT)
    info["segments"] = customer_segments(df, OUT)
    info["anomalies"] = invoice_anomalies(df, OUT)
    info["period"] = {"start": df["InvoiceDate"].min().strftime("%d %b %Y"),
                      "end": df["InvoiceDate"].max().strftime("%d %b %Y")}
    (OUT / "model_info.json").write_text(json.dumps(info, indent=2))
    print("Done. Files written to", OUT.resolve())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Online Retail.csv")
