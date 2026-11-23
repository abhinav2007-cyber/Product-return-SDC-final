"""Line-level 'returned later' model, customer segments and anomaly detection."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
WINDOW_DAYS = 60
LINE_FEATURES = ["log_quantity", "log_unit_price", "basket_size", "hour", "weekday", "is_uk",
                 "product_prior_lines", "product_return_rate", "customer_prior_lines", "customer_return_rate"]
NICE = {"log_quantity": "Quantity bought", "log_unit_price": "Unit price", "basket_size": "Basket size",
        "hour": "Hour of day", "weekday": "Weekday", "is_uk": "UK customer",
        "product_prior_lines": "Product sales history", "product_return_rate": "Product's past return rate",
        "customer_prior_lines": "Customer order history", "customer_return_rate": "Customer's past return rate"}


# ---------------------------------------------------------------- returned-later model
def _history(sales, rets, key, prefix, strength=20):
    """Return rate for each sale using ONLY events that happened before it (smoothed toward the global rate)."""
    ev = pd.concat([
        pd.DataFrame({key: sales[key], "InvoiceDate": sales["InvoiceDate"], "is_ret": 0, "sid": sales["sid"]}),
        pd.DataFrame({key: rets[key], "InvoiceDate": rets["InvoiceDate"], "is_ret": 1, "sid": -1}),
    ])
    ev = ev.sort_values(["InvoiceDate", "is_ret"], kind="stable").reset_index(drop=True)
    g = ev.groupby(key)
    prior_events = g.cumcount()
    prior_rets = g["is_ret"].cumsum() - ev["is_ret"]
    glob = len(rets) / (len(rets) + len(sales))
    ev[f"{prefix}_prior_lines"] = prior_events
    ev[f"{prefix}_return_rate"] = (prior_rets + strength * glob) / (prior_events + strength)
    return ev[ev["is_ret"] == 0].set_index("sid")[[f"{prefix}_prior_lines", f"{prefix}_return_rate"]]


def build_line_dataset(df):
    """One row per sale line. Label = same customer returns the same product within WINDOW_DAYS."""
    d = df[df["CustomerID"].notna()].copy()
    d["CustomerID"] = d["CustomerID"].astype(int)
    sales = d[(d["Quantity"] > 0) & d["is_product"] & (d["UnitPrice"] > 0)].reset_index(drop=True)
    sales["sid"] = sales.index
    rets = d[(d["Quantity"] < 0) & d["is_product"]]

    m = sales[["sid", "CustomerID", "StockCode", "InvoiceDate"]].merge(
        rets[["CustomerID", "StockCode", "InvoiceDate"]].rename(columns={"InvoiceDate": "ret_date"}),
        on=["CustomerID", "StockCode"])
    delta = m["ret_date"] - m["InvoiceDate"]
    hit = m.loc[(delta >= pd.Timedelta(0)) & (delta <= pd.Timedelta(days=WINDOW_DAYS)), "sid"].unique()
    sales["returned_later"] = sales["sid"].isin(hit).astype(int)

    # Labels need a full look-ahead window, so drop sales too close to the end of the data.
    sales = sales[sales["InvoiceDate"] <= d["InvoiceDate"].max() - pd.Timedelta(days=WINDOW_DAYS)].copy()

    sales = sales.join(_history(sales, rets, "StockCode", "product"), on="sid")
    sales = sales.join(_history(sales, rets, "CustomerID", "customer"), on="sid")
    sales["basket_size"] = sales.groupby("InvoiceNo")["sid"].transform("size")
    sales["log_quantity"] = np.log1p(sales["Quantity"])
    sales["log_unit_price"] = np.log1p(sales["UnitPrice"])
    sales["hour"] = sales["InvoiceDate"].dt.hour
    sales["weekday"] = sales["InvoiceDate"].dt.dayofweek
    sales["is_uk"] = (sales["Country"] == "United Kingdom").astype(int)
    return sales


def _pipe(clf):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)])


def _evaluate(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    k = max(1, int(0.1 * len(y)))
    top = y[np.argsort(-proba)[:k]].mean()
    return {"threshold": round(float(thr), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "lift_top10": round(float(top / y.mean()), 2),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def train_line_model(df, out_dir):
    data = build_line_dataset(df).sort_values("InvoiceDate").reset_index(drop=True)
    gap = pd.Timedelta(days=WINDOW_DAYS)
    t_val = data["InvoiceDate"].iloc[int(len(data) * 0.60)]
    t_test = data["InvoiceDate"].iloc[int(len(data) * 0.75)]
    fit = data[data["InvoiceDate"] < t_val - gap]
    val = data[(data["InvoiceDate"] >= t_val) & (data["InvoiceDate"] < t_test)]
    final = data[data["InvoiceDate"] < t_test - gap]
    test = data[data["InvoiceDate"] >= t_test]
    F, T = LINE_FEATURES, "returned_later"
    print(f"Line model rows: fit {len(fit)}, val {len(val)}, final {len(final)}, test {len(test)}; "
          f"base rate test {test[T].mean():.3%}")

    cands = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
             "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)}
    comparison, best, best_pr, best_proba = {}, None, -1, None
    for name, clf in cands.items():
        proba = _pipe(clone(clf)).fit(fit[F], fit[T]).predict_proba(val[F])[:, 1]
        pr = average_precision_score(val[T], proba)
        comparison[name] = round(float(pr), 4)
        if pr > best_pr:
            best, best_pr, best_proba = name, pr, proba
    p, r, t = precision_recall_curve(val[T], best_proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    thr = float(t[int(np.argmax(f1))])

    pipe = _pipe(clone(cands[best])).fit(final[F], final[T])
    proba = pipe.predict_proba(test[F])[:, 1]
    result = _evaluate(test[T].to_numpy(), proba, thr)
    print(best, result)

    sample = test.sample(min(len(test), 20000), random_state=SEED)
    imp = permutation_importance(pipe, sample[F], sample[T], scoring="average_precision",
                                 n_repeats=3, random_state=SEED)
    importances = sorted(({"feature": NICE[f], "importance": round(float(v), 4)}
                          for f, v in zip(F, imp.importances_mean)), key=lambda x: -x["importance"])

    # Compute history dictionaries for serving
    d = df[df["CustomerID"].notna() & df["is_product"]].copy()
    d["CustomerID"] = d["CustomerID"].astype(int)
    sales = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
    rets = d[d["Quantity"] < 0]
    global_rate = float(len(rets) / (len(rets) + len(sales))) if (len(rets) + len(sales)) > 0 else 0.0

    p_sales = sales.groupby("StockCode").agg(lines=("Quantity", "size"), description=("Description", "first"))
    p_rets = rets.groupby("StockCode").size().rename("returns")
    p_df = p_sales.join(p_rets).fillna({"returns": 0})
    p_df["returns"] = p_df["returns"].astype(int)
    products_map = {
        str(k): {"description": str(r["description"]), "lines": int(r["lines"]), "returns": int(r["returns"])}
        for k, r in p_df.iterrows()
    }

    c_sales = sales.groupby("CustomerID").size().rename("lines")
    c_rets = rets.groupby("CustomerID").size().rename("returns")
    c_df = pd.concat([c_sales, c_rets], axis=1).fillna(0).astype(int)
    customers_map = {
        int(k): {"lines": int(r["lines"]), "returns": int(r["returns"])}
        for k, r in c_df.iterrows()
    }

    joblib.dump({
        "pipeline": pipe,
        "threshold": thr,
        "features": F,
        "global_rate": global_rate,
        "products": products_map,
        "customers": customers_map,
    }, out_dir / "line_return_model.pkl")
    return {
        "name": best, "window_days": WINDOW_DAYS, "val_pr_auc": comparison, "test": result,
        "base_rate": round(float(test[T].mean()), 4),
        "baseline_pr_auc": {
            "Product's past return rate alone": round(float(average_precision_score(test[T], test["product_return_rate"])), 4),
            "Customer's past return rate alone": round(float(average_precision_score(test[T], test["customer_return_rate"])), 4)},
        "importances": importances,
        "rows": {"train": int(len(final)), "test": int(len(test)), "positives_test": int(test[T].sum())},
        "test_period_start": t_test.strftime("%d %b %Y"),
        "label": f"A sold line counts as returned if the same customer sends back the same product within {WINDOW_DAYS} days.",
    }


# ---------------------------------------------------------------- customer segments
def customer_segments(df, out_dir):
    d = df[df["CustomerID"].notna() & df["is_product"]].copy()
    d["CustomerID"] = d["CustomerID"].astype(int)
    g = d.groupby("CustomerID")
    neg = d[d["Quantity"] < 0]
    f = pd.DataFrame({
        "invoices": g["InvoiceNo"].nunique(),
        "lines": g.size(),
        "return_line_rate": (d["Quantity"] < 0).groupby(d["CustomerID"]).mean(),
        "gross_sales": d[d["Quantity"] > 0].groupby("CustomerID")["line_value"].sum(),
        "returned_value": neg.groupby("CustomerID")["line_value"].sum(),
        "cancelled_invoice_share": d.drop_duplicates("InvoiceNo").groupby("CustomerID")["is_cancelled"].mean(),
    }).fillna(0)
    f["avg_basket"] = f["lines"] / f["invoices"]
    f["return_value_share"] = (f["returned_value"] / (f["gross_sales"] + 1)).clip(0, 1)

    X = pd.DataFrame({"inv": np.log1p(f["invoices"]), "basket": np.log1p(f["avg_basket"]),
                      "rate": f["return_line_rate"], "canc": f["cancelled_invoice_share"],
                      "share": f["return_value_share"]})
    Z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z), min(len(Z), 3000), replace=False)
    results = []  # (k, silhouette, smallest cluster share)
    for k in range(3, 7):
        labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
        results.append((k, silhouette_score(Z[idx], labels[idx]), np.bincount(labels).min() / len(Z)))
    # Prefer splits that don't isolate a handful of customers; fall back to the best silhouette overall.
    pool = [r for r in results if r[2] >= 0.02] or results
    best_k, best_s, _ = max(pool, key=lambda r: r[1])
    f["cluster"] = KMeans(best_k, n_init=10, random_state=SEED).fit_predict(Z)

    prof = f.groupby("cluster")[["invoices", "avg_basket", "return_line_rate"]].median()
    names, left = {}, set(prof.index)
    for col, label in [("return_line_rate", "Heavy returners"), ("invoices", "Frequent buyers"), ("avg_basket", "Bulk buyers")]:
        if left:
            c = prof.loc[sorted(left), col].idxmax()
            names[c] = label
            left.discard(c)
    for i, c in enumerate(sorted(left), 1):
        names[c] = "Occasional buyers" if len(left) == 1 else f"Occasional buyers {i}"
    f["segment"] = f["cluster"].map(names)

    pc = PCA(2, random_state=SEED).fit_transform(Z)
    f["pc1"], f["pc2"] = pc[:, 0].round(3), pc[:, 1].round(3)
    f.reset_index()[["CustomerID", "segment", "pc1", "pc2", "invoices", "avg_basket", "return_line_rate",
                     "return_value_share"]].round(4).to_csv(out_dir / "customer_segments.csv", index=False)

    total_ret = f["returned_value"].sum()
    summary = f.groupby("segment").agg(
        customers=("invoices", "size"), median_invoices=("invoices", "median"),
        median_basket=("avg_basket", "median"), return_line_rate=("return_line_rate", "mean"),
        returned_value=("returned_value", "sum")).reset_index()
    summary["share_of_returned_value"] = summary["returned_value"] / max(total_ret, 1)
    summary.sort_values("return_line_rate", ascending=False).round(4).to_csv(out_dir / "segment_summary.csv", index=False)
    return {"k": int(best_k), "silhouette": round(float(best_s), 3)}


# ---------------------------------------------------------------- anomalies
def invoice_anomalies(df, out_dir, top=50):
    inv = df.groupby("InvoiceNo").agg(
        InvoiceDate=("InvoiceDate", "first"), Country=("Country", "first"), CustomerID=("CustomerID", "first"),
        cancelled=("is_cancelled", "first"), lines=("StockCode", "size"), units=("abs_quantity", "sum"),
        value=("line_value", "sum"), max_price=("UnitPrice", "max")).reset_index()
    cols = ["lines", "units", "value", "max_price"]
    L = np.log1p(inv[cols].clip(lower=0))
    X = L.assign(cancelled=inv["cancelled"])
    forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED).fit(X)
    inv["score"] = -forest.decision_function(X)
    inv["is_anomaly"] = forest.predict(X) == -1

    z = (L - L.groupby(inv["cancelled"]).transform("mean")) / L.groupby(inv["cancelled"]).transform("std").replace(0, 1)
    worst = z.abs().idxmax(axis=1)
    sign = np.array([z.loc[i, c] for i, c in zip(z.index, worst)]) > 0
    text = {"lines": "line items", "units": "quantity", "value": "value", "max_price": "unit price"}
    inv["Reason"] = [f"Unusually {'high' if s else 'low'} {text[c]} for a {'cancellation' if k else 'sale'}"
                     for c, s, k in zip(worst, sign, inv["cancelled"])]
    inv["is_anomaly"] = inv["is_anomaly"] & sign
    out = inv[inv["is_anomaly"]].nlargest(top, "score").copy()
    out["Type"] = np.where(out["cancelled"] == 1, "Cancellation", "Sale")
    out["CustomerID"] = out["CustomerID"].astype("Int64").astype(str).replace("<NA>", "")
    out["InvoiceDate"] = out["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
    out[["InvoiceNo", "InvoiceDate", "Type", "Country", "CustomerID", "lines", "units", "value", "Reason", "score"]] \
        .round({"value": 2, "score": 4}).to_csv(out_dir / "anomalies.csv", index=False)
    return {"flagged": int(inv["is_anomaly"].sum()), "invoices": int(len(inv))}
