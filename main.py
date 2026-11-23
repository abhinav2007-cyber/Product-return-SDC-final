"""FastAPI application for Product Return and Cancellation Pattern Analysis.

Run:
    python -m uvicorn main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from analysis import LINE_FEATURES
from ml_utils import FEATURES, add_derived

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SORTABLE_RECORDS = {"InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "Country", "Return_Value"}
SORTABLE_PRODUCTS = {"Description", "total_records", "total_quantity", "negative_records", "return_rate"}
SORTABLE_CUSTOMERS = {"CustomerID", "total_invoices", "total_lines", "negative_records", "total_spending", "return_rate"}

VALID_COUNTRIES: Set[str] = set()
info_path = DATA_DIR / "model_info.json"
if info_path.exists():
    try:
        _info_json = json.loads(info_path.read_text(encoding="utf-8"))
        VALID_COUNTRIES = set(_info_json.get("countries", []))
    except Exception:
        pass


def read_data_file(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python train_model.py \"Online Retail.csv\"")
    return pd.read_csv(path, **kwargs)


def load_app_state(target_app: FastAPI):
    """Load CSVs, pickles and info into target_app.state."""
    if not DATA_DIR.exists():
        raise RuntimeError(f"Data directory {DATA_DIR} is missing. Run: python train_model.py \"Online Retail.csv\"")

    global VALID_COUNTRIES
    logger.info("Loading CSV data and ML model artifacts...")
    flagged = read_data_file("flagged_records.csv", dtype={"CustomerID": str, "InvoiceNo": str}).fillna("")
    customers = read_data_file("customer_summary.csv", dtype={"CustomerID": str})
    products = read_data_file("product_summary.csv").fillna("")
    monthly = read_data_file("monthly_returns.csv")
    countries = read_data_file("country_returns.csv")
    invoices = read_data_file("invoice_data.csv")
    segment_summary = read_data_file("segment_summary.csv")
    segment_points = read_data_file("customer_segments.csv", dtype={"CustomerID": str})
    anomalies = read_data_file("anomalies.csv", dtype={"CustomerID": str}).fillna("")

    model_info = json.loads((DATA_DIR / "model_info.json").read_text(encoding="utf-8"))
    VALID_COUNTRIES = set(model_info.get("countries", []))

    cancel_bundle = joblib.load(DATA_DIR / "cancellation_model.pkl")

    line_bundle_path = DATA_DIR / "line_return_model.pkl"
    line_bundle = joblib.load(line_bundle_path) if line_bundle_path.exists() else None

    flagged["_search"] = (
        flagged["InvoiceNo"].astype(str) + " " +
        flagged["Description"].astype(str) + " " +
        flagged["CustomerID"].astype(str) + " " +
        flagged["Country"].astype(str)
    ).str.lower()

    # Feature 2: Pre-score cancellation risk for every flagged record
    try:
        _pipeline = cancel_bundle["pipeline"]
        _threshold = float(cancel_bundle["threshold"])
        _features = cancel_bundle.get("features", FEATURES)
        _scored_df = add_derived(flagged.copy())[_features]
        _probs = _pipeline.predict_proba(_scored_df)[:, 1]
        flagged = flagged.copy()
        flagged["cancel_prob"] = _probs.round(4)
        flagged["cancel_risk"] = pd.cut(
            _probs,
            bins=[-0.001, _threshold * 0.5, _threshold, 1.001],
            labels=["Low", "Medium", "High"]
        ).astype(str)
    except Exception as _pre_score_err:
        logger.warning("Could not pre-score flagged records: %s", _pre_score_err)
        flagged["cancel_prob"] = 0.0
        flagged["cancel_risk"] = "Unknown"

    cust_merged = customers.merge(
        segment_points[["CustomerID", "segment", "pc1", "pc2"]],
        on="CustomerID",
        how="left"
    ).fillna({"segment": "Unassigned", "pc1": 0.0, "pc2": 0.0})

    target_app.state.flagged = flagged
    target_app.state.customers = customers
    target_app.state.products = products
    target_app.state.monthly = monthly
    target_app.state.countries = countries
    target_app.state.invoices = invoices
    target_app.state.segment_summary = segment_summary
    target_app.state.segment_points = segment_points
    target_app.state.anomalies = anomalies
    target_app.state.model_info = model_info
    target_app.state.cust_merged = cust_merged
    target_app.state.cancel_pipeline = cancel_bundle["pipeline"]
    target_app.state.cancel_threshold = float(cancel_bundle["threshold"])
    target_app.state.cancel_features = cancel_bundle.get("features", FEATURES)
    target_app.state.line_bundle = line_bundle


@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    load_app_state(app_inst)
    yield


app = FastAPI(
    title="Product Return & Cancellation Analysis API",
    description="FastAPI service for transaction pattern analysis, invoice cancellation risk screening, and line-level return risk prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

# Eagerly load state so test clients and background imports have ready state
try:
    load_app_state(app)
except Exception as e:
    logger.warning("Initial eager state load deferred to lifespan: %s", e)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"{request.method} {request.url.path}")
        response = await call_next(request)
        return response


app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ----------------------------------------------------------------
# Exception Handlers
# ----------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Map validation errors to {"errors": {"field": "message"}, "detail": [...]}."""
    field_errors: Dict[str, str] = {}
    for err in exc.errors():
        loc = err.get("loc", [])
        field = str(loc[-1]) if loc else "general"
        msg = err.get("msg", "Invalid value")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        field_errors[field] = msg

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"errors": field_errors, "detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ----------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------
class InvoiceFeatures(BaseModel):
    line_count: int = Field(..., ge=1, le=2000, description="Total line items in the invoice (1 to 2000)")
    unique_products: int = Field(..., ge=1, le=2000, description="Number of unique product codes (1 to 2000)")
    total_abs_quantity: float = Field(..., ge=1.0, le=1_000_000.0, description="Sum of absolute item quantities (1 to 1,000,000)")
    average_unit_price: float = Field(..., ge=0.0, le=10_000.0, description="Average unit price across lines (0 to 10,000)")
    month: int = Field(..., ge=1, le=12, description="Month of invoice (1 to 12)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0=Monday to 6=Sunday)")
    hour: int = Field(..., ge=0, le=23, description="Hour of invoice creation (0 to 23)")
    country: str = Field(..., description="Country name, must match training countries")

    @field_validator("unique_products")
    @classmethod
    def check_unique_products(cls, v: int, info: ValidationInfo) -> int:
        line_count = info.data.get("line_count")
        if line_count is not None and v > line_count:
            raise ValueError("unique_products cannot exceed line_count")
        return v

    @field_validator("country")
    @classmethod
    def check_country(cls, v: str) -> str:
        if VALID_COUNTRIES and v not in VALID_COUNTRIES:
            raise ValueError("Choose a country from the training list.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "line_count": 8,
                "unique_products": 8,
                "total_abs_quantity": 60,
                "average_unit_price": 3.5,
                "month": 6,
                "weekday": 2,
                "hour": 12,
                "country": "United Kingdom",
            }
        }
    }


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="1 if probability >= threshold, else 0")
    label: str = Field(..., description="Human-readable prediction label")
    probability: float = Field(..., description="Estimated probability of cancellation")
    risk: str = Field(..., description="Risk category: High, Medium, or Low")
    threshold: float = Field(..., description="Decision threshold calibrated on training data")
    flagged: bool = Field(..., description="True if predicted cancellation (for UI backward compatibility)")


class LineReturnRiskIn(BaseModel):
    stock_code: str = Field(..., description="Product stock code")
    quantity: int = Field(..., gt=0, description="Quantity sold (> 0)")
    unit_price: float = Field(..., gt=0.0, description="Unit price (> 0)")
    basket_size: int = Field(..., ge=1, description="Total items in basket")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0=Monday to 6=Sunday)")
    country: str = Field(..., description="Country of transaction")
    customer_id: Optional[int] = Field(None, description="Optional CustomerID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "stock_code": "85123A",
                "quantity": 6,
                "unit_price": 2.55,
                "basket_size": 12,
                "hour": 14,
                "weekday": 2,
                "country": "United Kingdom",
                "customer_id": 17850,
            }
        }
    }


class LineReturnRiskOut(BaseModel):
    probability_returned_within_60_days: float = Field(..., description="Predicted probability that line is returned within 60 days")
    flagged: bool = Field(..., description="True if probability >= threshold")
    threshold: float = Field(..., description="Classification threshold")
    known_product: bool = Field(..., description="Whether product history was found in training data")
    known_customer: bool = Field(..., description="Whether customer history was found in training data")


# ----------------------------------------------------------------
# Front-end and Health
# ----------------------------------------------------------------
@app.get("/", summary="Dashboard interface")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}


# ----------------------------------------------------------------
# Prediction Endpoints
# ----------------------------------------------------------------
@app.post("/predict", response_model=PredictionOut, summary="Predict invoice cancellation risk")
async def predict(payload: InvoiceFeatures, request: Request):
    pipeline = request.app.state.cancel_pipeline
    threshold = request.app.state.cancel_threshold

    row_dict = payload.model_dump()
    row_df = add_derived(pd.DataFrame([row_dict]))[request.app.state.cancel_features]

    prob = float(pipeline.predict_proba(row_df)[0, 1])
    is_cancelled = int(prob >= threshold)
    risk_level = "High" if prob >= threshold else "Medium" if prob >= (threshold / 2.0) else "Low"
    label = "Likely a cancellation" if is_cancelled else "Probably not a cancellation"

    return PredictionOut(
        prediction=is_cancelled,
        label=label,
        probability=round(prob, 4),
        risk=risk_level,
        threshold=round(threshold, 4),
        flagged=bool(is_cancelled == 1),
    )


@app.post("/api/predict", include_in_schema=False)
async def api_predict_alias(payload: InvoiceFeatures, request: Request):
    return await predict(payload, request)


@app.post(
    "/predict/return-risk",
    response_model=LineReturnRiskOut,
    summary="Predict line-item return risk within 60 days (Defensible Target)",
    description=(
        "Scores whether an individual sold product line will be returned by the same customer within 60 days. "
        "Serving uses full-history smoothed rates, whereas model training strictly used historical events prior to each sale."
    ),
)
async def predict_return_risk(payload: LineReturnRiskIn, request: Request):
    bundle = request.app.state.line_bundle
    if bundle is None:
        raise HTTPException(status_code=500, detail="Line return model not available.")

    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    global_rate = float(bundle.get("global_rate", 0.024))
    products_hist = bundle.get("products", {})
    customers_hist = bundle.get("customers", {})

    log_qty = float(np.log1p(payload.quantity))
    log_price = float(np.log1p(payload.unit_price))
    is_uk = 1 if payload.country.strip().lower() == "united kingdom" else 0

    known_prod = payload.stock_code in products_hist
    if known_prod:
        p_info = products_hist[payload.stock_code]
        p_lines = p_info["lines"]
        p_rets = p_info["returns"]
        p_prior_lines = p_lines
        p_return_rate = (p_rets + 20.0 * global_rate) / (p_lines + 20.0)
    else:
        p_prior_lines = 0
        p_return_rate = global_rate

    known_cust = payload.customer_id is not None and payload.customer_id in customers_hist
    if known_cust:
        c_info = customers_hist[payload.customer_id]
        c_lines = c_info["lines"]
        c_rets = c_info["returns"]
        c_prior_lines = c_lines
        c_return_rate = (c_rets + 20.0 * global_rate) / (c_lines + 20.0)
    else:
        c_prior_lines = 0
        c_return_rate = global_rate

    feat_values = {
        "log_quantity": log_qty,
        "log_unit_price": log_price,
        "basket_size": payload.basket_size,
        "hour": payload.hour,
        "weekday": payload.weekday,
        "is_uk": is_uk,
        "product_prior_lines": p_prior_lines,
        "product_return_rate": p_return_rate,
        "customer_prior_lines": c_prior_lines,
        "customer_return_rate": c_return_rate,
    }

    feat_df = pd.DataFrame([[feat_values[col] for col in LINE_FEATURES]], columns=LINE_FEATURES)
    prob = float(pipeline.predict_proba(feat_df)[0, 1])
    flagged = prob >= threshold

    return LineReturnRiskOut(
        probability_returned_within_60_days=round(prob, 4),
        flagged=flagged,
        threshold=round(threshold, 4),
        known_product=known_prod,
        known_customer=known_cust,
    )


# ----------------------------------------------------------------
# Analytics API Endpoints
# ----------------------------------------------------------------
@app.get("/api/summary", summary="Summary KPIs for overview")
async def get_summary(request: Request):
    invoices = request.app.state.invoices
    flagged = request.app.state.flagged
    info = request.app.state.model_info

    cancelled = int(invoices["is_cancelled"].sum())
    total_inv = len(invoices)
    return {
        "total_invoices": total_inv,
        "cancellations": cancelled,
        "cancellation_rate": round(cancelled / total_inv, 4) if total_inv else 0.0,
        "negative_records": int((flagged["Quantity"] < 0).sum()),
        "flagged_records": len(flagged),
        "return_value": round(float(flagged["Return_Value"].sum()), 2),
        "period": info["period"],
    }


@app.get("/api/charts", summary="Data series for overview charts")
async def get_charts(request: Request, limit: int = Query(10, ge=1, le=50)):
    n = min(limit, 25)
    return {
        "monthly": request.app.state.monthly.to_dict("records"),
        "products": request.app.state.products.head(n).to_dict("records"),
        "customers": request.app.state.customers.head(n).to_dict("records"),
        "countries": request.app.state.countries.head(15).to_dict("records"),
    }


@app.get("/api/model", summary="Model training information and metrics")
async def get_model_info(request: Request):
    return request.app.state.model_info


@app.get("/api/insights", summary="Automated insight sentences for leadership")
async def get_insights(request: Request):
    invoices = request.app.state.invoices
    monthly = request.app.state.monthly
    products = request.app.state.products
    customers = request.app.state.customers

    peak_month = monthly.sort_values("negative_records", ascending=False).iloc[0]["Month"] if len(monthly) else "N/A"
    top_prod = products.iloc[0]["Description"] if len(products) else "N/A"
    top_cust = customers.iloc[0]["CustomerID"] if len(customers) else "N/A"
    cancel_pct = round(float(invoices["is_cancelled"].mean()) * 100, 1) if len(invoices) else 0.0

    return {
        "insights": [
            f"Overall invoice cancellation rate is {cancel_pct}%.",
            f"Returns peaked in {peak_month}.",
            f"Top product associated with returns is '{top_prod}'.",
            f"Customer {top_cust} has the highest number of return-related lines.",
        ],
        "peak_month": peak_month,
        "top_product": top_prod,
        "top_customer": top_cust,
        "cancellation_rate": cancel_pct,
    }


@app.get("/api/finance", summary="Finance and audit risk metrics")
async def get_finance(request: Request):
    flagged = request.app.state.flagged
    invoices = request.app.state.invoices

    return {
        "total_returned_value": round(float(flagged["Return_Value"].sum()), 2),
        "negative_records_count": int((flagged["Quantity"] < 0).sum()),
        "cancellation_invoices_count": int(invoices["is_cancelled"].sum()),
        "zero_or_negative_price_lines": int((flagged["UnitPrice"] <= 0).sum()),
    }


@app.get("/api/segments", summary="Customer clusters and scatter coordinates")
async def get_segments(request: Request):
    segment_points = request.app.state.segment_points
    segment_summary = request.app.state.segment_summary
    info = request.app.state.model_info

    pts = segment_points.sample(min(len(segment_points), 1500), random_state=42)
    return {
        "summary": segment_summary.to_dict("records"),
        "points": pts[["CustomerID", "segment", "pc1", "pc2"]].to_dict("records"),
        **info["segments"],
    }


@app.get("/api/anomalies", summary="Isolation Forest unusual invoices")
async def get_anomalies(request: Request, limit: int = Query(20, ge=1, le=100)):
    anomalies = request.app.state.anomalies
    info = request.app.state.model_info
    return {
        "rows": anomalies.head(limit).to_dict("records"),
        **info["anomalies"],
    }


@app.get("/api/patterns", summary="Temporal cancellation patterns by hour or weekday")
async def get_patterns(request: Request, by: Literal["hour", "weekday"] = "hour"):
    invoices = request.app.state.invoices
    col = "hour" if by == "hour" else "weekday"
    grp = invoices.groupby(col).agg(
        total_invoices=("InvoiceNo", "count"),
        cancellations=("is_cancelled", "sum"),
    ).reset_index()
    grp["cancellation_rate"] = (grp["cancellations"] / grp["total_invoices"].clip(lower=1)).round(4)

    return {
        "by": by,
        "patterns": grp.to_dict("records"),
    }


@app.get("/api/records", summary="Searchable, filterable, paginated flagged records")
async def get_records(
    request: Request,
    q: str = "",
    country: str = "",
    reason: str = "",
    month: int = Query(0, ge=0, le=12),
    sort: str = "InvoiceDate",
    dir: Literal["asc", "desc"] = "desc",
    page: int = Query(1, ge=1),
    per_page: int = Query(15, ge=1, le=100),
):
    df = request.app.state.flagged
    if q.strip():
        df = df[df["_search"].str.contains(q.strip().lower(), regex=False)]
    if country:
        df = df[df["Country"] == country]
    if reason:
        df = df[df["Reason"].str.contains(reason, regex=False)]
    if month:
        df = df[pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce").dt.month == month]

    sort_col = sort if sort in SORTABLE_RECORDS else "InvoiceDate"
    df = df.sort_values(sort_col, ascending=(dir == "asc"))

    clamped_per_page = max(1, min(per_page, 100))
    total = len(df)
    pages = max(1, -(-total // clamped_per_page))
    current_page = max(1, min(page, pages))

    start = (current_page - 1) * clamped_per_page
    end = start + clamped_per_page
    rows = df.iloc[start:end].drop(columns="_search").fillna("").to_dict("records")

    return {
        "total": total,
        "page": current_page,
        "pages": pages,
        "rows": rows,
    }


@app.get("/api/records.csv", summary="Download filtered flagged records as CSV")
async def export_records_csv(
    request: Request,
    q: str = "",
    country: str = "",
    reason: str = "",
    month: int = Query(0, ge=0, le=12),
    sort: str = "InvoiceDate",
    dir: Literal["asc", "desc"] = "desc",
):
    df = request.app.state.flagged
    if q.strip():
        df = df[df["_search"].str.contains(q.strip().lower(), regex=False)]
    if country:
        df = df[df["Country"] == country]
    if reason:
        df = df[df["Reason"].str.contains(reason, regex=False)]
    if month:
        df = df[pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce").dt.month == month]

    sort_col = sort if sort in SORTABLE_RECORDS else "InvoiceDate"
    df = df.sort_values(sort_col, ascending=(dir == "asc"))

    csv_data = df.drop(columns="_search").to_csv(index=False)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=flagged_records.csv"},
    )


@app.get("/api/products.csv", summary="Download full product return statistics as CSV")
async def export_products_csv(request: Request, q: str = "", sort: str = "negative_records"):
    df = request.app.state.products
    if q.strip():
        df = df[df["Description"].astype(str).str.lower().str.contains(q.strip().lower(), regex=False)]
    sort_col = sort if sort in SORTABLE_PRODUCTS else "negative_records"
    df = df.sort_values(sort_col, ascending=False)
    csv_data = df.to_csv(index=False)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=product_summary.csv"},
    )


@app.get("/api/model-comparison", summary="All candidate model CV scores (Feature 5)")
async def get_model_comparison(request: Request):
    info = request.app.state.model_info
    cv_scores = info.get("cv_pr_auc", {})
    selected = info.get("name", "Random Forest")
    rows = [
        {"model": name, "cv_pr_auc": round(score, 4), "selected": name == selected}
        for name, score in cv_scores.items()
    ]
    rows.sort(key=lambda r: r["cv_pr_auc"], reverse=True)
    return {
        "selected": selected,
        "metric": "PR AUC (5-fold cross-validation on training set)",
        "models": rows,
        "why": (
            f"{selected} was chosen because it was explicitly selected for this project "
            "to demonstrate ensemble tree methods. Its PR AUC on cross-validation is "
            f"{cv_scores.get(selected, 0):.4f}. The threshold was then tuned on "
            "out-of-fold predictions to maximise F1."
        ),
    }


@app.get("/api/products", summary="Paginated product return statistics")
async def get_products(
    request: Request,
    q: str = "",
    min_lines: int = 0,
    sort: str = "negative_records",
    page: int = Query(1, ge=1),
    per_page: int = Query(15, ge=1, le=100),
):
    df = request.app.state.products
    if q.strip():
        df = df[df["Description"].astype(str).str.lower().str.contains(q.strip().lower(), regex=False)]
    if min_lines > 0:
        df = df[df["total_records"] >= min_lines]

    sort_col = sort if sort in SORTABLE_PRODUCTS else "negative_records"
    df = df.sort_values(sort_col, ascending=False)

    clamped_per_page = max(1, min(per_page, 100))
    total = len(df)
    pages = max(1, -(-total // clamped_per_page))
    current_page = max(1, min(page, pages))

    start = (current_page - 1) * clamped_per_page
    end = start + clamped_per_page
    rows = df.iloc[start:end].to_dict("records")

    return {"total": total, "page": current_page, "pages": pages, "rows": rows}


@app.get("/api/customers", summary="Paginated customer summary with cluster segments")
async def get_customers(
    request: Request,
    q: str = "",
    segment: str = "",
    sort: str = "negative_records",
    page: int = Query(1, ge=1),
    per_page: int = Query(15, ge=1, le=100),
):
    df = request.app.state.cust_merged
    if q.strip():
        df = df[df["CustomerID"].astype(str).str.contains(q.strip(), regex=False)]
    if segment:
        df = df[df["segment"] == segment]

    sort_col = sort if sort in SORTABLE_CUSTOMERS else "negative_records"
    df = df.sort_values(sort_col, ascending=False)

    clamped_per_page = max(1, min(per_page, 100))
    total = len(df)
    pages = max(1, -(-total // clamped_per_page))
    current_page = max(1, min(page, pages))

    start = (current_page - 1) * clamped_per_page
    end = start + clamped_per_page
    rows = df.iloc[start:end].to_dict("records")

    return {"total": total, "page": current_page, "pages": pages, "rows": rows}


@app.get("/api/customers/{cid}", summary="Customer profile by ID")
async def get_customer_by_id(cid: str, request: Request):
    df = request.app.state.cust_merged
    match = df[df["CustomerID"].astype(str) == str(cid)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    cust_info = match.iloc[0].to_dict()
    flagged = request.app.state.flagged
    c_flagged = flagged[flagged["CustomerID"] == str(cid)].drop(columns="_search").head(10).to_dict("records")

    return {
        "customer": cust_info,
        "recent_flagged_records": c_flagged,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
