import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_home_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "Returns &amp; Cancellations" in res.text or "Returns & Cancellations" in res.text


def test_api_summary():
    res = client.get("/api/summary")
    assert res.status_code == 200
    data = res.json()
    for key in ["total_invoices", "cancellations", "cancellation_rate", "negative_records", "flagged_records", "return_value", "period"]:
        assert key in data


def test_api_charts():
    res = client.get("/api/charts?limit=10")
    assert res.status_code == 200
    data = res.json()
    for key in ["monthly", "products", "customers", "countries"]:
        assert key in data
        assert isinstance(data[key], list)


def test_api_model():
    res = client.get("/api/model")
    assert res.status_code == 200
    data = res.json()
    assert "name" in data
    assert "test" in data
    assert "countries" in data


def test_api_insights():
    res = client.get("/api/insights")
    assert res.status_code == 200
    data = res.json()
    assert "insights" in data
    assert isinstance(data["insights"], list)


def test_api_finance():
    res = client.get("/api/finance")
    assert res.status_code == 200
    data = res.json()
    assert "total_returned_value" in data or "return_value" in data


def test_api_segments():
    res = client.get("/api/segments")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "points" in data
    assert "k" in data


def test_api_anomalies():
    res = client.get("/api/anomalies?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "rows" in data
    assert "flagged" in data


def test_api_patterns():
    for by in ["hour", "weekday"]:
        res = client.get(f"/api/patterns?by={by}")
        assert res.status_code == 200
        data = res.json()
        assert data["by"] == by
        assert "patterns" in data


def test_api_records():
    res = client.get("/api/records?page=1&per_page=10")
    assert res.status_code == 200
    data = res.json()
    for key in ["total", "page", "pages", "rows"]:
        assert key in data


def test_api_records_csv():
    res = client.get("/api/records.csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers.get("content-type", "")
    assert "InvoiceNo,InvoiceDate" in res.text


def test_api_products():
    res = client.get("/api/products?page=1&per_page=5")
    assert res.status_code == 200
    data = res.json()
    assert "rows" in data
    assert "total" in data


def test_api_customers():
    res = client.get("/api/customers?page=1&per_page=5")
    assert res.status_code == 200
    data = res.json()
    assert "rows" in data
    assert "total" in data


def test_predict_success():
    sample_payload = {
        "line_count": 8,
        "unique_products": 8,
        "total_abs_quantity": 60,
        "average_unit_price": 3.5,
        "month": 6,
        "weekday": 2,
        "hour": 12,
        "country": "United Kingdom",
    }
    res = client.post("/predict", json=sample_payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert "risk" in data
    assert "threshold" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in (0, 1)

    # Test alias
    alias_res = client.post("/api/predict", json=sample_payload)
    assert alias_res.status_code == 200
    assert alias_res.json()["prediction"] == data["prediction"]


def test_predict_invalid_hour():
    payload = {
        "line_count": 8,
        "unique_products": 8,
        "total_abs_quantity": 60,
        "average_unit_price": 3.5,
        "month": 6,
        "weekday": 2,
        "hour": 30,  # Invalid hour
        "country": "United Kingdom",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "errors" in data
    assert "hour" in data["errors"]


def test_predict_unknown_country():
    payload = {
        "line_count": 8,
        "unique_products": 8,
        "total_abs_quantity": 60,
        "average_unit_price": 3.5,
        "month": 6,
        "weekday": 2,
        "hour": 12,
        "country": "Atlantis",  # Unknown country
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "errors" in data
    assert "country" in data["errors"]


def test_predict_unique_products_exceeds_line_count():
    payload = {
        "line_count": 5,
        "unique_products": 10,  # Exceeds line_count
        "total_abs_quantity": 60,
        "average_unit_price": 3.5,
        "month": 6,
        "weekday": 2,
        "hour": 12,
        "country": "United Kingdom",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "errors" in data
    assert "unique_products" in data["errors"]


def test_predict_missing_field():
    payload = {
        "line_count": 8,
        # missing unique_products
        "total_abs_quantity": 60,
        "average_unit_price": 3.5,
        "month": 6,
        "weekday": 2,
        "hour": 12,
        "country": "United Kingdom",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "errors" in data
    assert "unique_products" in data["errors"]


def test_customer_not_found():
    res = client.get("/api/customers/999999999999")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data


def test_predict_return_risk_known_and_unknown():
    # Known product
    known_payload = {
        "stock_code": "85123A",
        "quantity": 6,
        "unit_price": 2.55,
        "basket_size": 12,
        "hour": 14,
        "weekday": 2,
        "country": "United Kingdom",
        "customer_id": 17850,
    }
    res_known = client.post("/predict/return-risk", json=known_payload)
    assert res_known.status_code == 200
    data_k = res_known.json()
    assert "probability_returned_within_60_days" in data_k
    assert 0.0 <= data_k["probability_returned_within_60_days"] <= 1.0
    assert data_k["known_product"] is True

    # Unknown product
    unknown_payload = {
        "stock_code": "UNKNOWN9999",
        "quantity": 1,
        "unit_price": 10.0,
        "basket_size": 1,
        "hour": 10,
        "weekday": 0,
        "country": "France",
    }
    res_unk = client.post("/predict/return-risk", json=unknown_payload)
    assert res_unk.status_code == 200
    data_u = res_unk.json()
    assert "probability_returned_within_60_days" in data_u
    assert 0.0 <= data_u["probability_returned_within_60_days"] <= 1.0
    assert data_u["known_product"] is False


def test_cancel_risk_not_unknown():
    res = client.get("/api/records?page=1&per_page=5")
    assert res.status_code == 200
    rows = res.json()["rows"]
    assert len(rows) > 0
    assert rows[0].get("cancel_risk") not in (None, "", "Unknown", "–")


def test_api_datasets():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    assert "datasets" in data
    assert any(d["id"] == "default" for d in data["datasets"])


def test_dataset_id_summary():
    res_def = client.get("/api/summary")
    res_q = client.get("/api/summary?dataset_id=default")
    assert res_def.status_code == 200
    assert res_q.status_code == 200
    assert res_def.json().keys() == res_q.json().keys()


def test_upload_and_switch_dataset():
    csv_content = (
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "500001,85123A,WHITE HANGING HEART,6,2024-01-10 10:00:00,2.55,17850,United Kingdom\n"
        "C500002,85123A,WHITE HANGING HEART,-2,2024-01-11 11:00:00,2.55,17850,United Kingdom\n"
        "500003,22423,REGENCY CAKESTAND,1,2024-01-12 12:00:00,12.75,17851,France\n"
    )
    res = client.post(
        "/api/analyze-upload",
        files={"file": ("sample_test.csv", csv_content.encode("utf-8"), "text/csv")},
    )
    assert res.status_code == 200
    report_id = res.json()["report_id"]
    res_sum = client.get(f"/api/summary?dataset_id={report_id}")
    assert res_sum.status_code == 200
    assert res_sum.json()["total_invoices"] > 0
