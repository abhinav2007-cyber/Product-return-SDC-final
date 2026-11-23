# -------------------------------------------------------------
# upload_to_kaggle.ps1
# Upload the `data/` directory of the project to Kaggle as a dataset
# -------------------------------------------------------------

# ---- 1. Ensure the Kaggle CLI is installed ---------------------------------
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "Kaggle CLI not found – installing..."
    python -m pip install --upgrade kaggle
}

# ---- 2. Verify the Kaggle API token ---------------------------------------
# Place the token file at %USERPROFILE%\.kaggle\kaggle.json  (or set env vars)
$kaggleTokenPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleTokenPath)) {
    Write-Error "Kaggle token not found at $kaggleTokenPath.\nFollow the steps in the README to download and place your token."
    exit 1
}

# ---- 3. Prepare the dataset metadata ---------------------------------------
$datasetSlug   = "product-return-cancellation-data"
$datasetTitle  = "Product Return & Cancellation – ML Model & Artifacts"
$datasetDesc   = @"
Dataset containing the cleaned `Online Retail.csv`, all CSV exports generated
by `train_model.py`, the trained model pickles (`cancellation_model.pkl`,
`line_return_model.pkl`), and `model_info.json`.

Used for the *Product Return and Cancellation Pattern Analysis‑III* course
project (FastAPI dashboard, ngrok public URL, Postman collection, etc.).
"@

# Write a temporary metadata.json that Kaggle expects
$metaPath = "metadata.json"
@"
{
  "title": "$datasetTitle",
  "id": "$($env:KAGGLE_USERNAME)/$datasetSlug",
  "subtitle": "$datasetDesc",
  "resources": [
    {
      "path": "data/",
      "description": "All CSV exports and model artefacts produced by the project."
    }
  ],
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
"@ | Set-Content -Encoding UTF8 $metaPath

# ---- 4. Create / update the Kaggle dataset ---------------------------------
# The first run creates the dataset; subsequent runs will update it.
# Use zip mode to upload the whole folder in one request.
kaggle datasets create -p . -r $metaPath -u $env:KAGGLE_USERNAME `
    --dir-mode zip --skip-existing

# ---- 5. Clean up ------------------------------------------------------------
Remove-Item $metaPath -Force

Write-Host "`n✅  Dataset upload finished. Open https://www.kaggle.com/$($env:KAGGLE_USERNAME)/$datasetSlug to view it."
