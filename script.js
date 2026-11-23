let flaggedData = [];

async function loadDashboard() {

    const response = await fetch("/api/dashboard");
    const data = await response.json();

    document.getElementById("total").innerText =
        data.total_invoices;

    document.getElementById("cancel").innerText =
        data.cancellations;

    document.getElementById("negative").innerText =
        data.negative_records;

    document.getElementById("flagged").innerText =
        data.flagged_records;

    flaggedData = data.flagged;

    // Monthly chart
    new Chart(document.getElementById("monthlyChart"), {
        type: "line",

        data: {
            labels: data.monthly.map(x => x.Month ?? x.month ?? x["Unnamed: 0"]),

            datasets: [{
                label: "Negative Quantity Records",
                data: data.monthly.map(x => x[0] ?? x["0"]),
                borderWidth: 2
            }]
        }
    });

    // Product chart
    new Chart(document.getElementById("productChart"), {
        type: "bar",

        data: {
            labels: data.top_products.map(x => x.Description),
            datasets: [{
                label: "Negative Records",
                data: data.top_products.map(x => x.negative_records)
            }]
        },

        options: {
            indexAxis: "y",
            responsive: true
        }
    });

    // Country chart
    new Chart(document.getElementById("countryChart"), {
        type: "bar",

        data: {
            labels: data.countries.map(x => x[0]),
            datasets: [{
                label: "Negative Records",
                data: data.countries.map(x => x[1])
            }]
        }
    });

    // Customer chart
    new Chart(document.getElementById("customerChart"), {
        type: "bar",

        data: {
            labels: data.top_customers.map(x => x.CustomerID),
            datasets: [{
                label: "Negative Records",
                data: data.top_customers.map(x => x.negative_records)
            }]
        }
    });

    displayRecords(flaggedData);
}

function displayRecords(records) {

    const tbody = document.getElementById("records");

    tbody.innerHTML = "";

    records.forEach(record => {

        const row = document.createElement("tr");

        const values = [
            record.InvoiceNo,
            record.Description,
            record.Quantity,
            record.UnitPrice,
            record.Country
        ];

        values.forEach(value => {
            const cell = document.createElement("td");
            cell.textContent = value ?? "";
            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });
}

document.getElementById("search").addEventListener(
    "input",
    function() {

        const value = this.value.toLowerCase();

        const filtered = flaggedData.filter(record =>
            Object.values(record).some(item =>
                String(item ?? "").toLowerCase().includes(value)
            )
        );

        displayRecords(filtered);
    }
);

loadDashboard().catch(error => {
    console.error("Dashboard loading failed:", error);
});