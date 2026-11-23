# Product Return and Cancellation Pattern Analysis - API & Dashboard

FastAPI deployment for the UCI Online Retail transaction monitoring and return/cancellation risk pattern analysis.

---

## 1. Quick Start

### Windows (Detected Environment)
Run the automated batch script:
```cmd
run_local.bat
```
Or manually:
```cmd
python -m uvicorn main:app --reload --port 8000
```

### macOS & Linux
```bash
chmod +x run_local.sh
./run_local.sh
```

Once running:
- **Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 2. Testing the Models & Endpoints

### Automated Test Suite
Run the 21 pytest automated test cases:
```cmd
python -m pytest tests/test_api.py -v
```

### Postman Collection
Import the files located in the `postman/` directory into Postman:
1. `postman/Returns_API.postman_collection.json`
2. `postman/local.postman_environment.json`

Requests included:
- `GET /health` (Verifies status 200 and `"status": "ok"`)
- `GET /api/summary` (Verifies dataset statistics and cancellation rate)
- `POST /predict` (Graded endpoint - tests valid payload)
- `POST /predict` (Negative validation test - expects 422 with field-keyed error)
- `POST /predict/return-risk` (Line-level return risk prediction)

#### Sample JSON Body for `POST /predict`:
```json
{
  "line_count": 8,
  "unique_products": 8,
  "total_abs_quantity": 60,
  "average_unit_price": 3.5,
  "month": 6,
  "weekday": 2,
  "hour": 12,
  "country": "United Kingdom"
}
```

#### Expected `POST /predict` Response:
```json
{
  "prediction": 0,
  "label": "Probably not a cancellation",
  "probability": 0.0429,
  "risk": "Low",
  "threshold": 0.7326,
  "flagged": false
}
```

---

## 3. Five Stakeholder Views (Dashboard)

Navigate between stakeholder perspectives using the top navigation bar or direct URL hashes:

| Stakeholder | Decision | URL Hash |
|---|---|---|
| **Leadership and operations** | Overall cancellation volume & monthly/country trends | `http://127.0.0.1:8000/#overview` |
| **Buying and merchandising** | Most-returned products & return rates catalogue | `http://127.0.0.1:8000/#merchandising` |
| **Customer service and CRM** | Frequent returners & KMeans cluster segments | `http://127.0.0.1:8000/#customers` |
| **Finance, fraud and audit** | Isolation Forest anomalies & flagged records search | `http://127.0.0.1:8000/#finance` |
| **Course reviewers** | Defensible target line model & live invoice classifier | `http://127.0.0.1:8000/#method` |

---

## 4. Public Deployment (Live URL)

- **Public Swagger Documentation:** `https://winnings-flyer-thinning.ngrok-free.dev/docs`
- **Public API Health Check:** `https://winnings-flyer-thinning.ngrok-free.dev/health`
- **Public Predict Endpoint:** `POST https://winnings-flyer-thinning.ngrok-free.dev/predict`
- **Public Web Dashboard:** `https://winnings-flyer-thinning.ngrok-free.dev/`

### Generating / Managing the Tunnel
ngrok has been updated to version 3.39+ on Windows:
```cmd
ngrok http 8000
```

### Cloudflare Alternative (No Account Needed)
If ngrok is unavailable or hits tunnel limits:
```cmd
cloudflared tunnel --url http://localhost:8000
```

### Troubleshooting Notes
- **HTTP 502 Bad Gateway:** Indicates FastAPI / uvicorn is not running or is running on a port other than 8000. Start FastAPI first with `run_local.bat` or `python -m uvicorn main:app --port 8000`.
- **Public URL stops working:** The public URL remains active only while your computer is awake, uvicorn is running, and the tunnel process remains open.
- **ngrok Browser Warning Page:** When testing in Postman or via curl, add the request header:
  `ngrok-skip-browser-warning: true`

---

## 5. Machine Learning Models Summary

1. **Invoice Classifier (Baseline):**
   - Evaluated algorithms: Logistic Regression, Gradient Boosting, Random Forest.
   - Cross-validation PR-AUC: Gradient Boosting (0.7635), Random Forest (0.7587), Logistic Regression (0.4646).
   - Calibrated decision threshold: 0.7326 (maximizing F1 on out-of-fold training data).
2. **Line-Level Return Risk Model (Defensible Target):**
   - Evaluated on sales with a 60-day gap so the model never looks ahead into future data.
   - Uses empirical Bayesian smoothed return rates for product and customer history.
3. **Unsupervised Discoveries:**
   - KMeans customer clustering (k=3, Silhouette=0.393) with PCA projections.
   - Isolation Forest anomaly detection for high-value/atypical transactions.
