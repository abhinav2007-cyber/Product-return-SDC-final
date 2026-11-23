const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-GB");
const money = new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP", maximumFractionDigits: 0 });
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

async function getJSON(url, options) {
    const res = await fetch(url, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(body.error || `Request failed (${res.status})`), { body });
    return body;
}

function showError(message) {
    const box = $("error");
    box.textContent = message;
    box.hidden = false;
}

const trunc = (s, n = 30) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));
const monthLabel = (ym) => {
    const [y, m] = ym.split("-");
    return `${MONTHS[Number(m) - 1]} ${y}`;
};

/* ---------- View Router (#overview, #merchandising, #customers, #finance, #method) ---------- */
const VIEWS = ["overview", "merchandising", "customers", "finance", "method"];
function updateView() {
    const hash = (window.location.hash || "#overview").replace("#", "").toLowerCase();
    const active = VIEWS.includes(hash) ? hash : "overview";
    VIEWS.forEach((v) => {
        const el = $("view-" + v);
        if (el) el.hidden = (v !== active);
        const tab = document.querySelector(`.nav-tab[href="#${v}"]`);
        if (tab) tab.classList.toggle("active", v === active);
    });
    // Trigger Chart.js recalculation when visible
    setTimeout(() => {
        Object.values(charts).forEach((c) => {
            if (c && typeof c.resize === "function") c.resize();
        });
    }, 50);
}
window.addEventListener("hashchange", updateView);

/* ---------- Charts ---------- */
const charts = {};
function draw(id, config) {
    if (charts[id]) charts[id].destroy();
    Chart.defaults.font.family = '"IBM Plex Sans", system-ui, sans-serif';
    Chart.defaults.color = css("--muted");
    config.options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, ...(config.options?.plugins || {}) },
        ...config.options,
    };
    charts[id] = new Chart($(id), config);
}

const gridScale = (extra = {}) => ({ grid: { color: css("--chart-grid") }, beginAtZero: true, ...extra });

function barChart(id, labels, values, label, { horizontal = false, fullLabels = labels, onClick = null } = {}) {
    draw(id, {
        type: "bar",
        data: { labels, datasets: [{ label, data: values, backgroundColor: css("--accent"), borderRadius: 3 }] },
        options: {
            indexAxis: horizontal ? "y" : "x",
            onClick: (e, elements) => {
                if (elements.length > 0 && onClick) {
                    onClick(labels[elements[0].index]);
                }
            },
            scales: horizontal ? { x: gridScale(), y: { grid: { display: false } } }
                               : { y: gridScale(), x: { grid: { display: false } } },
            plugins: { tooltip: { callbacks: { title: (items) => fullLabels[items[0].dataIndex] } } },
        },
    });
}

let countryRows = [];
function drawCountries() {
    const rows = $("noUK").checked ? countryRows.filter((r) => r.Country !== "United Kingdom") : countryRows;
    barChart("countryChart", rows.map((r) => r.Country), rows.map((r) => r.negative_records), "Negative-quantity lines");
}

function renderCharts(d) {
    if (d.monthly && d.monthly.length > 0) {
        const peak = d.monthly.reduce((a, b) => (b.negative_records > a.negative_records ? b : a));
        $("n-monthly").textContent = `Peak: ${monthLabel(peak.Month)} with ${nf.format(peak.negative_records)} negative-quantity lines.`;
        draw("monthlyChart", {
            type: "line",
            data: {
                labels: d.monthly.map((r) => monthLabel(r.Month)),
                datasets: [{
                    label: "Negative-quantity lines", data: d.monthly.map((r) => r.negative_records),
                    borderColor: css("--accent"), backgroundColor: css("--accent") + "22", fill: true, tension: 0.3, pointRadius: 3,
                }],
            },
            options: { scales: { y: gridScale(), x: { grid: { display: false } } } },
        });
    }

    const p = d.products;
    if ($("n-products")) {
        $("n-products").textContent = p.length ? `${trunc(p[0].Description, 40)} leads with ${p[0].negative_records} (${(p[0].return_rate * 100).toFixed(1)}% of its lines).` : "";
    }
    barChart("productChart", p.map((r) => trunc(r.Description)), p.map((r) => r.negative_records), "Negative-quantity lines",
             { horizontal: true, fullLabels: p.map((r) => r.Description) });

    countryRows = d.countries;
    drawCountries();

    const c = d.customers;
    if ($("n-customers")) {
        $("n-customers").textContent = c.length ? `Customer ${c[0].CustomerID} has ${c[0].negative_records} negative-quantity lines. Click a customer bar to filter their records.` : "";
    }
    barChart("customerChart", c.map((r) => r.CustomerID), c.map((r) => r.negative_records), "Negative-quantity lines", {
        onClick: (cid) => {
            selectCustomer(cid);
        }
    });
}

function selectCustomer(cid) {
    window.location.hash = "#finance";
    const sInput = $("search");
    if (sInput) {
        sInput.value = cid;
        resetAndLoad();
    }
}

/* ---------- KPIs and model ---------- */
function renderSummary(s) {
    $("k-total").textContent = nf.format(s.total_invoices);
    $("k-cancel").textContent = nf.format(s.cancellations);
    $("k-rate").textContent = `${(s.cancellation_rate * 100).toFixed(1)}% of invoices`;
    $("k-negative").textContent = nf.format(s.negative_records);
    $("k-flagged").textContent = nf.format(s.flagged_records);
    $("k-value").textContent = money.format(s.return_value);
    $("period").textContent = `Data: ${s.period.start} to ${s.period.end}`;
}

function renderModel(m) {
    const t = m.test, pct = (v) => (v * 100).toFixed(1) + "%";
    const box = $("model");
    box.replaceChildren();
    const h = document.createElement("h3");
    h.textContent = `Baseline: ${m.name}, held-out data`;
    const dl = document.createElement("dl");
    [["Precision", pct(t.precision)], ["Recall", pct(t.recall)], ["F1", pct(t.f1)],
     ["ROC AUC", t.roc_auc.toFixed(3)], ["PR AUC", t.pr_auc.toFixed(3)]].forEach(([k, v]) => {
        const dt = document.createElement("dt"), dd = document.createElement("dd");
        dt.textContent = k; dd.textContent = v;
        dl.append(dt, dd);
    });
    const note = document.createElement("p");
    note.textContent = `Of ${nf.format(m.test_rows)} test invoices, it caught ${t.confusion.tp} of ${t.confusion.tp + t.confusion.fn} cancellations and raised ${t.confusion.fp} false alarms.`;
    box.append(h, dl, note);
}

function fillSelect(select, options) {
    select.replaceChildren(...options.map(([value, text]) => {
        const o = document.createElement("option");
        o.value = value; o.textContent = text;
        return o;
    }));
}

/* ---------- Analysis sections ---------- */
const pct1 = (v) => (v * 100).toFixed(1) + "%";
function addRow(tbody, cells, numeric = []) {
    const tr = tbody.insertRow();
    cells.forEach((v, i) => { const td = tr.insertCell(); td.textContent = v; if (numeric.includes(i)) td.className = "num"; });
}
function facts(el, pairs) {
    el.replaceChildren();
    pairs.forEach(([k, v]) => {
        const dt = document.createElement("dt"), dd = document.createElement("dd");
        dt.textContent = k; dd.textContent = v;
        el.append(dt, dd);
    });
}

function renderLineModel(m) {
    const t = m.test, base = Object.entries(m.baseline_pr_auc).sort((a, b) => b[1] - a[1])[0];
    $("lm-label").textContent = m.label + ` Tested on sales from ${m.test_period_start} onward, so the model never sees the future.`;
    facts($("lm-facts"), [
        ["Chosen model", m.name],
        ["Share of sales that come back", pct1(m.base_rate)],
        ["Model PR AUC", t.pr_auc.toFixed(3)],
        [`Best single-rule baseline (${base[0].toLowerCase()})`, base[1].toFixed(3)],
        ["Top 10% riskiest lines", `${t.lift_top10}× more likely to return`],
        ["Precision / recall at threshold", `${pct1(t.precision)} / ${pct1(t.recall)}`],
        ["Test lines / returned", `${nf.format(m.rows.test)} / ${nf.format(m.rows.positives_test)}`],
    ]);
    $("lm-caveat").textContent = "Limits: only customers with an ID are included, and returns after the dataset ends are not visible.";
    const imp = m.importances.slice(0, 7);
    draw("importanceChart", {
        type: "bar",
        data: { labels: imp.map((r) => r.feature), datasets: [{ label: "Drop in PR AUC when shuffled", data: imp.map((r) => r.importance), backgroundColor: css("--accent"), borderRadius: 3 }] },
        options: { indexAxis: "y", scales: { x: gridScale(), y: { grid: { display: false } } } },
    });
}

function renderSegments(s) {
    const names = [...new Set(s.summary.map((r) => r.segment))];
    const palette = ["#0f766e", "#b45309", "#6d5bd0", "#b42318", "#2f7d4f", "#0369a1"];
    $("n-seg").textContent = `${s.k} groups found by clustering on activity and return behaviour (silhouette ${s.silhouette}). Labels are assigned from each group's profile.`;
    draw("segmentChart", {
        type: "scatter",
        data: { datasets: names.map((n, i) => ({
            label: n, backgroundColor: palette[i % palette.length] + "99", pointRadius: 3,
            data: s.points.filter((p) => p.segment === n).map((p) => ({ x: p.pc1, y: p.pc2 })),
        })) },
        options: {
            plugins: { legend: { display: true, position: "bottom" } },
            scales: { x: { grid: { color: css("--chart-grid") }, title: { display: true, text: "Component 1" } },
                      y: { grid: { color: css("--chart-grid") }, title: { display: true, text: "Component 2" } } },
        },
    });
    const body = $("segTable");
    body.replaceChildren();
    s.summary.forEach((r) => addRow(body, [r.segment, nf.format(r.customers), r.median_invoices, pct1(r.return_line_rate), pct1(r.share_of_returned_value)], [1, 2, 3, 4]));
}

function renderAnomalies(a) {
    $("n-anom").textContent = `${nf.format(a.flagged)} of ${nf.format(a.invoices)} invoices stand out from similar invoices (Isolation Forest). Largest shown.`;
    const body = $("anomTable");
    body.replaceChildren();
    a.rows.slice(0, 10).forEach((r) => addRow(body, [r.InvoiceNo, r.Type, nf.format(r.units), money.format(r.value), r.Reason], [2, 3]));
    if (!a.rows.length) addRow(body, ["No unusual invoices found."]);
}

function renderCleaning(c) {
    facts($("cleaning"), [
        ["Rows in raw file", nf.format(c.raw_rows)],
        ["Exact duplicates removed", nf.format(c.duplicates_removed)],
        ["Rows with unreadable dates", nf.format(c.unparseable_dates)],
        ["Rows without a customer ID", nf.format(c.missing_customer)],
        ["Non-product rows (postage, fees, adjustments)", nf.format(c.non_product_rows)],
        ["Zero or negative price rows", nf.format(c.zero_or_negative_price)],
        ["Negative quantity without a cancellation invoice", nf.format(c.negative_not_cancelled)],
    ]);
}

/* ---------- Products Table (Merchandising View) ---------- */
async function loadProducts(q = "") {
    try {
        const data = await getJSON(`/api/products?q=${encodeURIComponent(q)}&per_page=10`);
        const tbody = $("productTableBody");
        if (!tbody) return;
        tbody.replaceChildren();
        // Feature 4: Update product CSV download link with current query
        const pLink = $("productExportLink");
        if (pLink) pLink.href = `/api/products.csv?q=${encodeURIComponent(q)}`;
        if (!data.rows.length) {
            const tr = tbody.insertRow();
            const td = tr.insertCell();
            td.colSpan = 6;
            td.textContent = "No products found.";
            td.style.textAlign = "center";
            return;
        }
        // Find max return_rate for insight threshold (Feature 3)
        const maxRate = data.rows.reduce((m, r) => Math.max(m, r.return_rate), 0);
        data.rows.forEach((r) => {
            const tr = tbody.insertRow();
            const tdDesc = tr.insertCell();
            tdDesc.textContent = r.Description;
            const tdTot = tr.insertCell();
            tdTot.textContent = nf.format(r.total_records);
            tdTot.className = "num";
            const tdQty = tr.insertCell();
            tdQty.textContent = nf.format(r.total_quantity);
            tdQty.className = "num";
            const tdNeg = tr.insertCell();
            tdNeg.textContent = nf.format(r.negative_records);
            tdNeg.className = "num neg";
            const tdRate = tr.insertCell();
            tdRate.textContent = pct1(r.return_rate);
            tdRate.className = "num";
            // Feature 3: Insight badge
            const tdInsight = tr.insertCell();
            if (r.return_rate >= 0.4) {
                const badge = document.createElement("span");
                badge.className = "insight-badge insight-high";
                badge.textContent = "⚠ High return rate";
                tdInsight.appendChild(badge);
            } else if (r.return_rate >= 0.2) {
                const badge = document.createElement("span");
                badge.className = "insight-badge insight-mid";
                badge.textContent = "↑ Elevated";
                tdInsight.appendChild(badge);
            } else {
                tdInsight.textContent = "–";
                tdInsight.style.color = "var(--muted)";
            }
        });
    } catch (err) {
        console.error("Failed loading products", err);
    }
}

/* ---------- Predictor ---------- */
function renderResult(r) {
    const box = $("result");
    box.replaceChildren();
    const pct = document.createElement("div");
    pct.className = `pct risk-${r.risk}`;
    pct.textContent = (r.probability * 100).toFixed(1) + "%";
    const verdict = document.createElement("div");
    verdict.className = `verdict risk-${r.risk}`;
    verdict.textContent = `${r.risk} risk: ${r.flagged ? "likely a cancellation" : "probably not a cancellation"}`;
    const bar = document.createElement("div");
    bar.className = `bar risk-${r.risk}`;
    const fill = document.createElement("i");
    fill.style.width = Math.max(2, r.probability * 100) + "%";
    const mark = document.createElement("b");
    mark.style.left = r.threshold * 100 + "%";
    mark.title = "Decision threshold";
    bar.append(fill, mark);
    const hint = document.createElement("p");
    hint.className = "note";
    hint.textContent = `The black marker is the decision threshold (${(r.threshold * 100).toFixed(0)}%).`;
    box.append(pct, verdict, bar, hint);
}

function setupPredictor(countries) {
    const form = $("predictForm");
    fillSelect(form.month, MONTHS.map((m, i) => [i + 1, m]));
    fillSelect(form.weekday, DAYS.map((d, i) => [i, d]));
    fillSelect(form.country, countries.map((c) => [c, c]));
    form.country.value = countries.includes("United Kingdom") ? "United Kingdom" : countries[0];
    form.month.value = 6;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        form.querySelectorAll(".field-error").forEach((n) => n.remove());
        form.querySelectorAll(".invalid").forEach((n) => n.classList.remove("invalid"));
        const btn = form.querySelector("button");
        btn.disabled = true;
        try {
            const payload = Object.fromEntries(new FormData(form));
            // Cast numeric fields
            payload.line_count = Number(payload.line_count);
            payload.unique_products = Number(payload.unique_products);
            payload.total_abs_quantity = Number(payload.total_abs_quantity);
            payload.average_unit_price = Number(payload.average_unit_price);
            payload.month = Number(payload.month);
            payload.weekday = Number(payload.weekday);
            payload.hour = Number(payload.hour);

            renderResult(await getJSON("/predict", {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
            }));
        } catch (err) {
            Object.entries(err.body?.errors || {}).forEach(([name, msg]) => {
                const field = form.elements[name];
                if (field) {
                    field.classList.add("invalid");
                    const span = document.createElement("span");
                    span.className = "field-error";
                    span.textContent = msg;
                    field.parentElement.append(span);
                }
            });
            if (!err.body?.errors) showError(err.message);
        } finally {
            btn.disabled = false;
        }
    });
}

/* ---------- Records table ---------- */
const state = { page: 1, sort: "InvoiceDate", dir: "desc" };

function queryString(extra = {}) {
    const params = new URLSearchParams({ q: $("search").value, country: $("countryFilter").value, reason: $("reasonFilter").value, month: $("monthFilter").value, ...extra });
    return params.toString();
}

async function loadRecords() {
    try {
        const data = await getJSON(`/api/records?${queryString({ page: state.page, sort: state.sort, dir: state.dir })}`);
        state.page = data.page;
        const tbody = $("records");
        tbody.replaceChildren();
        if (!data.rows.length) {
            const tr = tbody.insertRow();
            tr.className = "empty-row";
            const td = tr.insertCell();
            td.colSpan = 7;
            td.textContent = "No records match these filters. Clear the search or choose All countries.";
        }
        data.rows.forEach((r) => {
            const tr = tbody.insertRow();
            [r.InvoiceNo, r.InvoiceDate, r.Description, r.Quantity, Number(r.UnitPrice).toFixed(2), r.Country, r.Reason].forEach((v, i) => {
                const td = tr.insertCell();
                if (i === 0 && r.CustomerID) {
                    const span = document.createElement("span");
                    span.textContent = v;
                    td.appendChild(span);
                } else {
                    td.textContent = v;
                }
                if (i === 3 || i === 4) td.className = "num" + (i === 3 && r.Quantity < 0 ? " neg" : "");
            });
            // Feature 2: Cancellation risk badge
            const tdRisk = tr.insertCell();
            const risk = r.cancel_risk || "–";
            tdRisk.className = "num";
            if (risk !== "–") {
                const badge = document.createElement("span");
                badge.className = `risk-badge risk-badge-${risk}`;
                badge.textContent = `${risk} (${r.cancel_prob !== undefined ? (r.cancel_prob * 100).toFixed(0) + "%" : "–"})`;
                tdRisk.appendChild(badge);
            } else {
                tdRisk.textContent = "–";
            }
        });
        $("pageInfo").textContent = `${nf.format(data.total)} records · page ${data.page} of ${data.pages}`;
        $("prev").disabled = data.page <= 1;
        $("next").disabled = data.page >= data.pages;
        $("exportLink").href = `/api/records.csv?${queryString({ sort: state.sort, dir: state.dir })}`;
    } catch (err) {
        showError(err.message);
    }
}

function resetAndLoad() { state.page = 1; loadRecords(); }

function setupTable(countries) {
    countries.forEach((c) => $("countryFilter").append(new Option(c, c)));
    let timer;
    $("search").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(resetAndLoad, 250); });
    $("countryFilter").addEventListener("change", resetAndLoad);
    $("reasonFilter").addEventListener("change", resetAndLoad);
    // Feature 1: Month filter
    const mf = $("monthFilter");
    if (mf) mf.addEventListener("change", resetAndLoad);
    $("prev").addEventListener("click", () => { state.page--; loadRecords(); });
    $("next").addEventListener("click", () => { state.page++; loadRecords(); });
    document.querySelectorAll("th[data-sort]").forEach((th) => th.addEventListener("click", () => {
        const key = th.dataset.sort;
        state.dir = state.sort === key && state.dir === "desc" ? "asc" : "desc";
        state.sort = key;
        document.querySelectorAll("th[data-sort]").forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", state.dir === "asc" ? "ascending" : "descending");
        resetAndLoad();
    }));
    const defaultSortTh = document.querySelector('th[data-sort="InvoiceDate"]');
    if (defaultSortTh) defaultSortTh.setAttribute("aria-sort", "descending");

    const pSearch = $("productSearch");
    if (pSearch) {
        let pTimer;
        pSearch.addEventListener("input", () => {
            clearTimeout(pTimer);
            pTimer = setTimeout(() => loadProducts(pSearch.value), 250);
        });
    }
}

/* ---------- Feature 5: Model Comparison ---------- */
function renderModelComparison(mc) {
    const whyEl = $("mc-why");
    if (whyEl) whyEl.textContent = mc.why || "";

    const tbody = $("modelCompTable");
    if (tbody) {
        tbody.replaceChildren();
        (mc.models || []).forEach((r) => {
            const tr = tbody.insertRow();
            const tdName = tr.insertCell();
            tdName.textContent = r.model;
            if (r.selected) { tdName.style.fontWeight = "600"; }
            const tdScore = tr.insertCell();
            tdScore.textContent = r.cv_pr_auc.toFixed(4);
            tdScore.className = "num";
            const tdSel = tr.insertCell();
            tdSel.textContent = r.selected ? "✓ Selected" : "";
            tdSel.style.color = r.selected ? "var(--ok)" : "var(--muted)";
            tdSel.style.fontWeight = r.selected ? "600" : "400";
        });
    }

    const names = (mc.models || []).map((r) => r.model);
    const scores = (mc.models || []).map((r) => r.cv_pr_auc);
    const bgColors = (mc.models || []).map((r) => r.selected ? css("--accent") : css("--accent") + "66");
    if ($("modelCompChart")) {
        draw("modelCompChart", {
            type: "bar",
            data: { labels: names, datasets: [{ label: "CV PR AUC", data: scores, backgroundColor: bgColors, borderRadius: 4 }] },
            options: {
                scales: { y: gridScale({ min: 0, max: 1 }), x: { grid: { display: false } } },
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: (item) => `PR AUC: ${item.raw.toFixed(4)}` } } },
            },
        });
    }
}

/* ---------- Start ---------- */
async function init() {
    $("noUK").addEventListener("change", drawCountries);
    try {
        const [summary, charts, model, segs, anoms, modelComp] = await Promise.all([
            getJSON("/api/summary"), getJSON("/api/charts"), getJSON("/api/model"),
            getJSON("/api/segments"), getJSON("/api/anomalies"), getJSON("/api/model-comparison")]);
        renderSummary(summary);
        renderCharts(charts);
        renderModel(model);
        renderLineModel(model.line_model);
        renderSegments(segs);
        renderAnomalies(anoms);
        renderCleaning(model.cleaning);
        renderModelComparison(modelComp);
        setupPredictor(model.countries);
        setupTable(model.countries);
        loadRecords();
        loadProducts();
        updateView();
    } catch (err) {
        showError(`Could not load the dashboard: ${err.message}`);
    }
}
init();
