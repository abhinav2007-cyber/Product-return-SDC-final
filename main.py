"""FastAPI application for Product Return and Cancellation Pattern Analysis.

Run:
    python -m uvicorn main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
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
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

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

    # Feature 2: Pre-score cancellation risk by scoring invoice-level features and merging onto flagged lines
    try:
        _pipeline = cancel_bundle["pipeline"]
        _threshold = float(cancel_bundle["threshold"])
        _features = cancel_bundle.get("features", FEATURES)
        _inv_for_score = invoices.copy()
        _inv_for_score["InvoiceNo"] = _inv_for_score["InvoiceNo"].astype(str)
        _probs = _pipeline.predict_proba(add_derived(_inv_for_score)[_features])[:, 1]
        _score_map = pd.DataFrame({
            "InvoiceNo": _inv_for_score["InvoiceNo"],
            "cancel_prob": np.round(_probs, 4),
            "cancel_risk": pd.cut(
                _probs,
                bins=[-0.001, _threshold * 0.5, _threshold, 1.001],
                labels=["Low", "Medium", "High"],
            ).astype(str),
        }).drop_duplicates(subset=["InvoiceNo"])
        flagged = flagged.merge(_score_map, on="InvoiceNo", how="left")
        flagged["cancel_prob"] = flagged["cancel_prob"].fillna(0.0)
        flagged["cancel_risk"] = flagged["cancel_risk"].fillna("Unknown")
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
    if not hasattr(target_app.state, "upload_reports"):
        target_app.state.upload_reports = {}


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
# Dataset Context Helper (Default + Uploaded Datasets)
# ----------------------------------------------------------------
def _get_dataset(request: Request, dataset_id: str = "default") -> Dict[str, Any]:
    if not dataset_id or dataset_id == "default":
        st = request.app.state
        return {
            "flagged": st.flagged,
            "customers": st.customers,
            "cust_merged": st.cust_merged,
            "products": st.products,
            "monthly": st.monthly,
            "countries": st.countries,
            "invoices": st.invoices,
            "segment_summary": st.segment_summary,
            "segment_points": st.segment_points,
            "anomalies": st.anomalies,
            "model_info": st.model_info,
            "period": st.model_info["period"],
            "segments_meta": st.model_info["segments"],
            "anomalies_meta": st.model_info["anomalies"],
        }

    rep = request.app.state.upload_reports.get(dataset_id)
    if rep is None:
        bundle_path = UPLOADS_DIR / dataset_id / "bundle.joblib"
        if bundle_path.exists():
            rep = joblib.load(bundle_path)
            request.app.state.upload_reports[dataset_id] = rep
    if rep is None or "dataset" not in rep:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return rep["dataset"]


@app.get("/api/datasets", summary="List available datasets (default + uploaded files)")
async def list_datasets(request: Request):
    datasets = [{
        "id": "default",
        "label": "UCI Online Retail (Default)",
        "invoices": int(len(request.app.state.invoices)),
        "period": request.app.state.model_info["period"],
    }]
    seen = {"default"}
    for rid, rep in request.app.state.upload_reports.items():
        if rid not in seen and "dataset" in rep:
            seen.add(rid)
            datasets.append({
                "id": rid,
                "label": f"{rep['filename']} ({rid})",
                "invoices": int(len(rep["dataset"]["invoices"])),
                "period": rep["dataset"]["period"],
            })
    if UPLOADS_DIR.exists():
        for sub in sorted(UPLOADS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            rid = sub.name
            bpath = sub / "bundle.joblib"
            if rid not in seen and bpath.exists():
                try:
                    rep = joblib.load(bpath)
                    request.app.state.upload_reports[rid] = rep
                    if "dataset" in rep:
                        seen.add(rid)
                        datasets.append({
                            "id": rid,
                            "label": f"{rep['filename']} ({rid})",
                            "invoices": int(len(rep["dataset"]["invoices"])),
                            "period": rep["dataset"]["period"],
                        })
                except Exception:
                    pass
    return {"datasets": datasets}


# ----------------------------------------------------------------
# Analytics API Endpoints
# ----------------------------------------------------------------
@app.get("/api/summary", summary="Summary KPIs for overview")
async def get_summary(request: Request, dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    invoices = ds["invoices"]
    flagged = ds["flagged"]

    cancelled = int(invoices["is_cancelled"].sum())
    total_inv = len(invoices)
    return {
        "total_invoices": total_inv,
        "cancellations": cancelled,
        "cancellation_rate": round(cancelled / total_inv, 4) if total_inv else 0.0,
        "negative_records": int((flagged["Quantity"] < 0).sum()),
        "flagged_records": len(flagged),
        "return_value": round(float(flagged["Return_Value"].sum()), 2),
        "period": ds["period"],
    }


@app.get("/api/charts", summary="Data series for overview charts")
async def get_charts(request: Request, limit: int = Query(10, ge=1, le=50), dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    n = min(limit, 25)
    return {
        "monthly": ds["monthly"].to_dict("records"),
        "products": ds["products"].head(n).to_dict("records"),
        "customers": ds["customers"].head(n).to_dict("records"),
        "countries": ds["countries"].head(15).to_dict("records"),
    }


@app.get("/api/model", summary="Model training information and metrics")
async def get_model_info(request: Request):
    return request.app.state.model_info


@app.get("/api/insights", summary="Automated insight sentences for leadership")
async def get_insights(request: Request, dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    invoices = ds["invoices"]
    monthly = ds["monthly"]
    products = ds["products"]
    customers = ds["customers"]

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
async def get_finance(request: Request, dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    flagged = ds["flagged"]
    invoices = ds["invoices"]

    return {
        "total_returned_value": round(float(flagged["Return_Value"].sum()), 2),
        "negative_records_count": int((flagged["Quantity"] < 0).sum()),
        "cancellation_invoices_count": int(invoices["is_cancelled"].sum()),
        "zero_or_negative_price_lines": int((flagged["UnitPrice"] <= 0).sum()),
    }


@app.get("/api/segments", summary="Customer clusters and scatter coordinates")
async def get_segments(request: Request, dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    segment_points = ds["segment_points"]
    segment_summary = ds["segment_summary"]

    pts = segment_points.sample(min(len(segment_points), 1500), random_state=42) if len(segment_points) else segment_points
    return {
        "summary": segment_summary.to_dict("records"),
        "points": pts[["CustomerID", "segment", "pc1", "pc2"]].to_dict("records") if len(pts) else [],
        **ds["segments_meta"],
    }


@app.get("/api/anomalies", summary="Isolation Forest unusual invoices")
async def get_anomalies(request: Request, limit: int = Query(20, ge=1, le=100), dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    anomalies = ds["anomalies"]
    return {
        "rows": anomalies.head(limit).to_dict("records"),
        **ds["anomalies_meta"],
    }


@app.get("/api/patterns", summary="Temporal cancellation patterns by hour or weekday")
async def get_patterns(request: Request, by: Literal["hour", "weekday"] = "hour", dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    invoices = ds["invoices"]
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
    dataset_id: str = "default",
):
    df = _get_dataset(request, dataset_id)["flagged"]
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
    dataset_id: str = "default",
):
    df = _get_dataset(request, dataset_id)["flagged"]
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
async def export_products_csv(request: Request, q: str = "", sort: str = "negative_records", dataset_id: str = "default"):
    df = _get_dataset(request, dataset_id)["products"]
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
    dataset_id: str = "default",
):
    df = _get_dataset(request, dataset_id)["products"]
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
    dataset_id: str = "default",
):
    df = _get_dataset(request, dataset_id)["cust_merged"]
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
async def get_customer_by_id(cid: str, request: Request, dataset_id: str = "default"):
    ds = _get_dataset(request, dataset_id)
    df = ds["cust_merged"]
    match = df[df["CustomerID"].astype(str) == str(cid)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    cust_info = match.iloc[0].to_dict()
    flagged = ds["flagged"]
    c_flagged = flagged[flagged["CustomerID"].astype(str) == str(cid)].drop(columns="_search").head(10).to_dict("records")

    return {
        "customer": cust_info,
        "recent_flagged_records": c_flagged,
    }


# ----------------------------------------------------------------
# Seller / Business Dataset Upload & Report Generation
# ----------------------------------------------------------------
_UPLOAD_COL_ALIASES = {
    "invoiceno": "InvoiceNo", "invoice_no": "InvoiceNo", "invoice": "InvoiceNo", "order_id": "InvoiceNo", "orderid": "InvoiceNo",
    "stockcode": "StockCode", "stock_code": "StockCode", "sku": "StockCode", "product_id": "StockCode", "item_code": "StockCode",
    "description": "Description", "product_name": "Description", "product": "Description", "item_name": "Description",
    "quantity": "Quantity", "qty": "Quantity", "units": "Quantity",
    "invoicedate": "InvoiceDate", "invoice_date": "InvoiceDate", "date": "InvoiceDate", "order_date": "InvoiceDate",
    "unitprice": "UnitPrice", "unit_price": "UnitPrice", "price": "UnitPrice", "amount": "UnitPrice",
    "customerid": "CustomerID", "customer_id": "CustomerID", "customer": "CustomerID",
    "country": "Country", "region": "Country", "location": "Country",
}


def _normalize_uploaded_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded CSV/Excel columns and validate required fields."""
    rename_map = {}
    for col in raw_df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        compact = key.replace("_", "")
        if key in _UPLOAD_COL_ALIASES:
            rename_map[col] = _UPLOAD_COL_ALIASES[key]
        elif compact in _UPLOAD_COL_ALIASES:
            rename_map[col] = _UPLOAD_COL_ALIASES[compact]

    df = raw_df.rename(columns=rename_map).copy()
    required = ["InvoiceNo", "Quantity", "UnitPrice", "InvoiceDate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required column(s): {', '.join(missing)}. "
                "Your file should include InvoiceNo, Quantity, UnitPrice, InvoiceDate "
                "(plus optional StockCode, Description, CustomerID, Country)."
            ),
        )

    if "StockCode" not in df.columns:
        df["StockCode"] = "00000"
    if "Description" not in df.columns:
        df["Description"] = "Unknown Product"
    if "CustomerID" not in df.columns:
        df["CustomerID"] = ""
    if "Country" not in df.columns:
        df["Country"] = "United Kingdom"

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("Unknown Product").astype(str).str.strip()
    df["Country"] = df["Country"].fillna("United Kingdom").astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"]).copy()

    dates = pd.to_datetime(df["InvoiceDate"], format="%d/%m/%y %H:%M", errors="coerce")
    if dates.isna().mean() > 0.05:
        dates = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
    df["InvoiceDate"] = dates
    df = df.dropna(subset=["InvoiceDate"]).copy()

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid rows remained after parsing dates, quantities, and prices.")
    return df


@app.post("/api/analyze-upload", summary="Upload a CSV/Excel sales file for ML cancellation & return analysis")
async def analyze_uploaded_file(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "uploaded_data.csv"
    ext = Path(filename).suffix.lower()
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        if ext == ".csv":
            raw_df = pd.read_csv(io.BytesIO(raw_bytes), dtype={"InvoiceNo": str, "StockCode": str})
        elif ext in {".xlsx", ".xls"}:
            raw_df = pd.read_excel(io.BytesIO(raw_bytes), dtype={"InvoiceNo": str, "StockCode": str})
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .csv or .xlsx file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    df = _normalize_uploaded_df(raw_df)
    df["is_cancelled"] = df["InvoiceNo"].str.startswith("C").astype(int)
    df["abs_quantity"] = df["Quantity"].abs()
    df["Gross_Value"] = np.where(df["Quantity"] > 0, df["Quantity"] * df["UnitPrice"], 0.0)
    df["Return_Value"] = np.where((df["Quantity"] < 0) | (df["is_cancelled"] == 1), df["abs_quantity"] * df["UnitPrice"], 0.0)

    neg = df["Quantity"] < 0
    reasons = pd.DataFrame({
        "Cancellation invoice": df["is_cancelled"] == 1,
        "Negative quantity": neg,
        "Zero or negative price": df["UnitPrice"] <= 0,
    })
    flagged = df[reasons.any(axis=1)].copy()
    if len(flagged) > 0:
        flagged["Reason"] = reasons.loc[flagged.index].apply(lambda r: ", ".join(r.index[r]), axis=1)
        flagged["InvoiceDateStr"] = flagged["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
    else:
        flagged["Reason"] = pd.Series(dtype=str)
        flagged["InvoiceDateStr"] = pd.Series(dtype=str)

    # Aggregate to invoice level and score with the trained ML pipeline
    inv = df.groupby("InvoiceNo").agg(
        line_count=("StockCode", "size"),
        total_abs_quantity=("abs_quantity", "sum"),
        average_unit_price=("UnitPrice", "mean"),
        unique_products=("StockCode", "nunique"),
        InvoiceDate=("InvoiceDate", "first"),
        country=("Country", "first"),
        actual_is_cancelled=("is_cancelled", "first"),
        gross_value=("Gross_Value", "sum"),
        return_value=("Return_Value", "sum"),
    ).reset_index()
    inv["month"] = inv["InvoiceDate"].dt.month
    inv["weekday"] = inv["InvoiceDate"].dt.dayofweek
    inv["hour"] = inv["InvoiceDate"].dt.hour
    inv = add_derived(inv)

    pipeline = request.app.state.cancel_pipeline
    threshold = float(request.app.state.cancel_threshold)
    features = request.app.state.cancel_features
    probs = pipeline.predict_proba(inv[features])[:, 1]
    inv["cancel_probability"] = np.round(probs, 4)
    inv["predicted_cancellation"] = (probs >= threshold).astype(int)
    inv["risk_level"] = np.where(
        probs >= threshold, "High",
        np.where(probs >= threshold * 0.5, "Medium", "Low"),
    )
    inv["InvoiceDateStr"] = inv["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")

    # Summary KPIs
    gross_total = float(df["Gross_Value"].sum())
    return_total = float(df["Return_Value"].sum())
    summary = {
        "total_rows": int(len(df)),
        "total_invoices": int(len(inv)),
        "flagged_rows": int(len(flagged)),
        "actual_cancellations": int(inv["actual_is_cancelled"].sum()),
        "high_risk_invoices": int((inv["risk_level"] == "High").sum()),
        "medium_risk_invoices": int((inv["risk_level"] == "Medium").sum()),
        "low_risk_invoices": int((inv["risk_level"] == "Low").sum()),
        "gross_sales": round(gross_total, 2),
        "return_value": round(return_total, 2),
        "net_revenue": round(gross_total - return_total, 2),
        "threshold": round(threshold, 4),
    }

    # Breakdowns
    prod_grp = (
        df.assign(is_neg=(df["Quantity"] < 0).astype(int))
        .groupby("Description", as_index=False)
        .agg(
            total_lines=("InvoiceNo", "count"),
            negative_lines=("is_neg", "sum"),
            return_value=("Return_Value", "sum"),
        )
    )
    prod_grp["return_rate"] = (prod_grp["negative_lines"] / prod_grp["total_lines"].clip(lower=1)).round(4)
    prod_grp["return_value"] = prod_grp["return_value"].round(2)
    top_products = prod_grp.sort_values(["negative_lines", "return_value"], ascending=False).head(10).to_dict("records")

    country_grp = (
        df.assign(is_neg=(df["Quantity"] < 0).astype(int))
        .groupby("Country", as_index=False)
        .agg(
            total_lines=("InvoiceNo", "count"),
            negative_lines=("is_neg", "sum"),
            return_value=("Return_Value", "sum"),
        )
    )
    country_grp["return_value"] = country_grp["return_value"].round(2)
    top_countries = country_grp.sort_values(["negative_lines", "total_lines"], ascending=False).head(10).to_dict("records")

    monthly_grp = (
        df.assign(Month=df["InvoiceDate"].dt.strftime("%Y-%m"), is_neg=(df["Quantity"] < 0).astype(int))
        .groupby("Month", as_index=False)
        .agg(total_lines=("InvoiceNo", "count"), negative_lines=("is_neg", "sum"), return_value=("Return_Value", "sum"))
        .sort_values("Month")
    )
    monthly_grp["return_value"] = monthly_grp["return_value"].round(2)

    high_risk_invoices = (
        inv.sort_values("cancel_probability", ascending=False)
        .head(50)[
            [
                "InvoiceNo", "InvoiceDateStr", "country", "line_count",
                "total_abs_quantity", "average_unit_price", "cancel_probability",
                "risk_level", "actual_is_cancelled", "return_value",
            ]
        ]
        .fillna("")
        .to_dict("records")
    )

    report_id = str(uuid.uuid4())[:8]
    reports_store = request.app.state.upload_reports
    if len(reports_store) >= 10:
        oldest_key = next(iter(reports_store))
        reports_store.pop(oldest_key, None)

    invoices_export = inv[[
        "InvoiceNo", "InvoiceDateStr", "country", "line_count", "unique_products",
        "total_abs_quantity", "average_unit_price", "gross_value", "return_value",
        "cancel_probability", "risk_level", "predicted_cancellation", "actual_is_cancelled",
    ]].rename(columns={"InvoiceDateStr": "InvoiceDate"}).sort_values("cancel_probability", ascending=False)

    flagged_export = flagged[[
        "InvoiceNo", "InvoiceDateStr", "StockCode", "Description", "Quantity",
        "UnitPrice", "CustomerID", "Country", "Reason", "Return_Value",
    ]].rename(columns={"InvoiceDateStr": "InvoiceDate"}) if len(flagged) > 0 else pd.DataFrame(
        columns=["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID", "Country", "Reason", "Return_Value"]
    )

    # Build full dashboard dataset bundle so all 5 stakeholder views work dynamically on this upload
    inv_score_map = inv[["InvoiceNo", "cancel_probability", "risk_level"]].rename(
        columns={"cancel_probability": "cancel_prob", "risk_level": "cancel_risk"}
    ).drop_duplicates(subset=["InvoiceNo"])
    ds_flagged = flagged_export.copy()
    ds_flagged["CustomerID"] = ds_flagged["CustomerID"].fillna("").astype(str)
    ds_flagged["InvoiceNo"] = ds_flagged["InvoiceNo"].fillna("").astype(str)
    ds_flagged = ds_flagged.merge(inv_score_map, on="InvoiceNo", how="left")
    ds_flagged["cancel_prob"] = ds_flagged["cancel_prob"].fillna(0.0)
    ds_flagged["cancel_risk"] = ds_flagged["cancel_risk"].fillna("Unknown")
    ds_flagged["_search"] = (
        ds_flagged["InvoiceNo"].astype(str) + " " +
        ds_flagged["Description"].astype(str) + " " +
        ds_flagged["CustomerID"].astype(str) + " " +
        ds_flagged["Country"].astype(str)
    ).str.lower()

    ds_monthly = (
        df.assign(Month=df["InvoiceDate"].dt.strftime("%Y-%m"), is_neg=(df["Quantity"] < 0).astype(int))
        .groupby("Month", as_index=False)
        .agg(negative_records=("is_neg", "sum"), return_value=("Return_Value", "sum"))
        .sort_values("Month")
    )
    ds_monthly["return_value"] = ds_monthly["return_value"].round(2)

    ds_countries = (
        df.assign(is_neg=(df["Quantity"] < 0).astype(int))
        .groupby("Country", as_index=False)
        .agg(negative_records=("is_neg", "sum"), return_value=("Return_Value", "sum"))
        .sort_values("negative_records", ascending=False)
    )
    ds_countries["return_value"] = ds_countries["return_value"].round(2)

    ds_products = (
        df.assign(is_neg=(df["Quantity"] < 0).astype(int))
        .groupby("Description", as_index=False)
        .agg(
            total_records=("InvoiceNo", "count"),
            total_quantity=("Quantity", "sum"),
            negative_records=("is_neg", "sum"),
        )
        .assign(return_rate=lambda d: (d["negative_records"] / d["total_records"].clip(lower=1)).round(4))
        .sort_values("negative_records", ascending=False)
    )

    cust_df = df[df["CustomerID"].astype(str).str.strip() != ""].copy()
    if len(cust_df) > 0:
        ds_customers = (
            cust_df.assign(value=cust_df["Quantity"] * cust_df["UnitPrice"], is_neg=(cust_df["Quantity"] < 0).astype(int))
            .groupby("CustomerID", as_index=False)
            .agg(
                total_invoices=("InvoiceNo", "nunique"),
                total_lines=("InvoiceNo", "size"),
                negative_records=("is_neg", "sum"),
                total_spending=("value", "sum"),
            )
            .assign(return_rate=lambda d: (d["negative_records"] / d["total_lines"].clip(lower=1)).round(4))
            .sort_values("negative_records", ascending=False)
        )
        ds_customers["CustomerID"] = ds_customers["CustomerID"].astype(str)
    else:
        ds_customers = pd.DataFrame(columns=["CustomerID", "total_invoices", "total_lines", "negative_records", "total_spending", "return_rate"])

    ds_cust_merged = ds_customers.copy()
    ds_cust_merged["segment"] = "Uploaded Cohort"
    ds_cust_merged["pc1"] = 0.0
    ds_cust_merged["pc2"] = 0.0
    ds_seg_points = ds_cust_merged[["CustomerID", "segment", "pc1", "pc2"]].copy()
    ds_seg_summary = pd.DataFrame([{
        "segment": "Uploaded Cohort",
        "customers": int(len(ds_customers)),
        "median_invoices": float(ds_customers["total_invoices"].median()) if len(ds_customers) else 0.0,
        "return_rate": float(ds_customers["return_rate"].mean()) if len(ds_customers) else 0.0,
        "share_of_returned_value": 1.0 if len(ds_customers) else 0.0,
    }])

    ds_anomalies = (
        inv.sort_values("total_abs_quantity", ascending=False)
        .head(20)
        .assign(
            type=lambda d: np.where(d["actual_is_cancelled"] == 1, "Cancellation", "Sale"),
            why=lambda d: "High unit volume (" + d["total_abs_quantity"].astype(int).astype(str) + " units)",
        )[["InvoiceNo", "type", "total_abs_quantity", "return_value", "why"]]
        .rename(columns={"total_abs_quantity": "units", "return_value": "value"})
    )

    ds_invoices = inv.rename(columns={"actual_is_cancelled": "is_cancelled"}).copy()
    ds_period = {
        "start": df["InvoiceDate"].min().strftime("%d %b %Y"),
        "end": df["InvoiceDate"].max().strftime("%d %b %Y"),
    }

    report_bundle = {
        "filename": filename,
        "summary": summary,
        "invoices_df": invoices_export,
        "flagged_df": flagged_export,
        "top_products": top_products,
        "top_countries": top_countries,
        "dataset": {
            "flagged": ds_flagged,
            "customers": ds_customers,
            "cust_merged": ds_cust_merged,
            "products": ds_products,
            "monthly": ds_monthly,
            "countries": ds_countries,
            "invoices": ds_invoices,
            "segment_summary": ds_seg_summary,
            "segment_points": ds_seg_points,
            "anomalies": ds_anomalies,
            "model_info": request.app.state.model_info,
            "period": ds_period,
            "segments_meta": {"k": 1, "silhouette": 0.0},
            "anomalies_meta": {"flagged": int(len(ds_anomalies)), "invoices": int(len(ds_invoices))},
        },
    }
    reports_store[report_id] = report_bundle
    try:
        rep_dir = UPLOADS_DIR / report_id
        rep_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(report_bundle, rep_dir / "bundle.joblib")
    except Exception as save_err:
        logger.warning("Could not persist upload report %s: %s", report_id, save_err)

    return {
        "report_id": report_id,
        "filename": filename,
        "summary": summary,
        "top_products": top_products,
        "countries": top_countries,
        "monthly": monthly_grp.to_dict("records"),
        "high_risk_invoices": high_risk_invoices,
        "download_urls": {
            "invoices_csv": f"/api/upload-report/{report_id}/invoices.csv",
            "flagged_csv": f"/api/upload-report/{report_id}/flagged.csv",
            "summary_txt": f"/api/upload-report/{report_id}/summary.txt",
        },
    }


def _load_upload_report(request: Request, report_id: str) -> Dict[str, Any]:
    rep = request.app.state.upload_reports.get(report_id)
    if rep is None:
        bpath = UPLOADS_DIR / report_id / "bundle.joblib"
        if bpath.exists():
            rep = joblib.load(bpath)
            request.app.state.upload_reports[report_id] = rep
    if not rep:
        raise HTTPException(status_code=404, detail="Upload report not found or expired. Please re-upload your file.")
    return rep


@app.get("/api/upload-report/{report_id}/invoices.csv", summary="Download ML-scored invoices CSV for an uploaded file")
async def download_upload_invoices_csv(report_id: str, request: Request):
    rep = _load_upload_report(request, report_id)
    stem = Path(rep["filename"]).stem
    return Response(
        content=rep["invoices_df"].to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={stem}_scored_invoices.csv"},
    )


@app.get("/api/upload-report/{report_id}/flagged.csv", summary="Download flagged return/cancellation lines CSV for an uploaded file")
async def download_upload_flagged_csv(report_id: str, request: Request):
    rep = _load_upload_report(request, report_id)
    stem = Path(rep["filename"]).stem
    return Response(
        content=rep["flagged_df"].to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={stem}_flagged_records.csv"},
    )


@app.get("/api/upload-report/{report_id}/summary.txt", summary="Download executive summary report for an uploaded file")
async def download_upload_summary_txt(report_id: str, request: Request):
    rep = _load_upload_report(request, report_id)
    s = rep["summary"]
    lines = [
        "PRODUCT RETURN & CANCELLATION ANALYSIS REPORT",
        "=" * 52,
        f"Source File            : {rep['filename']}",
        f"Report ID              : {report_id}",
        f"Model Decision Thresh  : {s['threshold'] * 100:.1f}%",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 52,
        f"Total Line Items       : {s['total_rows']:,}",
        f"Total Invoices         : {s['total_invoices']:,}",
        f"Flagged Line Items     : {s['flagged_rows']:,}",
        f"Actual Cancellations   : {s['actual_cancellations']:,}",
        f"High-Risk Invoices     : {s['high_risk_invoices']:,}",
        f"Medium-Risk Invoices   : {s['medium_risk_invoices']:,}",
        f"Low-Risk Invoices      : {s['low_risk_invoices']:,}",
        f"Gross Sales Value      : £{s['gross_sales']:,.2f}",
        f"Returned/Cancelled Val : £{s['return_value']:,.2f}",
        f"Net Revenue            : £{s['net_revenue']:,.2f}",
        "",
        "TOP FLAGGED PRODUCTS",
        "-" * 52,
    ]
    for p in rep["top_products"]:
        lines.append(
            f"- {p['Description']} | Neg Lines: {p['negative_lines']}/{p['total_lines']} "
            f"({p['return_rate'] * 100:.1f}%) | Return Value: £{p['return_value']:,.2f}"
        )
    lines.extend(["", "TOP COUNTRIES", "-" * 52])
    for c in rep["top_countries"]:
        lines.append(
            f"- {c['Country']} | Neg Lines: {c['negative_lines']}/{c['total_lines']} "
            f"| Return Value: £{c['return_value']:,.2f}"
        )
    stem = Path(rep["filename"]).stem
    return Response(
        content="\n".join(lines),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={stem}_analysis_report.txt"},
    )


@app.get("/api/sample-template.csv", summary="Download a sample CSV file to test the dataset uploader")
async def download_sample_template():
    raw_path = BASE_DIR / "Online Retail.csv"
    if raw_path.exists():
        sample_df = pd.read_csv(raw_path, nrows=300, dtype={"InvoiceNo": str, "StockCode": str})
        csv_text = sample_df.to_csv(index=False)
    else:
        csv_text = (
            "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
            "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,01/12/10 08:26,2.55,17850,United Kingdom\n"
            "C536379,D,Discount,-1,01/12/10 09:41,27.50,14527,United Kingdom\n"
            "536381,22633,HAND WARMER UNION JACK,6,01/12/10 09:41,1.85,15311,United Kingdom\n"
            "C536383,35004C,SET OF 3 COLOURED FLYING DUCKS,-1,01/12/10 09:49,4.65,15311,United Kingdom\n"
        )
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_sales_upload.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
