from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import os

app = Flask()

# Load CSV files
flagged = pd.read_csv("flagged_records.csv")
customers = pd.read_csv("customer_summary.csv")
products = pd.read_csv("product_summary.csv")
monthly = pd.read_csv("monthly_returns.csv")
countries = pd.read_csv("country_returns.csv")
invoices = pd.read_csv("invoice_data.csv")

# Load trained ML model
model = joblib.load("cancellation_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/dashboard")
def dashboard():

    total_invoices = len(invoices)

    cancellations = int(invoices["is_cancelled"].sum())

    negative_records = int(
        (flagged["Quantity"] < 0).sum()
    )

    return jsonify({

        "total_invoices": total_invoices,

        "cancellations": cancellations,

        "negative_records": negative_records,

        "flagged_records": len(flagged),

        "top_products": products.head(10).to_dict("records"),

        "top_customers": customers.head(10).to_dict("records"),

        "monthly": monthly.to_dict("records"),

        "countries": countries.to_dict("records"),

        "flagged": flagged.head(100).fillna("").to_dict("records")

    })


if __name__ == "__main__":
    app.run(debug=True)