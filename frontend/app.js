const $ = (id) => document.getElementById(id);
const charts = {};

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}
function num(x, d = 3) {
  return Number(x).toFixed(d);
}
function kpi(label, value) {
  return `<div class="kpi"><span>${label}</span><strong>${value}</strong></div>`;
}
function table(rows, cols) {
  if (!rows || !rows.length) return "<p class='muted'>No rows.</p>";
  const keys = cols || Object.keys(rows[0]);
  const head = keys.map((k) => `<th>${k}</th>`).join("");
  const body = rows
    .map((r) => "<tr>" + keys.map((k) => `<td>${fmt(r[k])}</td>`).join("") + "</tr>")
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function fmt(v) {
  if (typeof v === "number") {
    if (v > 0 && v < 1) return num(v, 3);
    return Number.isInteger(v) ? v : num(v, 2);
  }
  return v ?? "";
}
function barChart(canvasId, rows, title) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || typeof Chart === "undefined") return;
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: rows.map((r) => r.group),
      datasets: [
        {
          label: title,
          data: rows.map((r) => r.return_rate * 100),
          backgroundColor: "#1f6f8b",
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: "Return rate (%)" } } },
      maintainAspectRatio: false,
    },
  });
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${url} failed (${res.status}): ${text.slice(0, 200)}`);
  }
  return res.json();
}

function showError(id, err) {
  const el = $(id);
  if (!el) return;
  el.innerHTML = `<div class="card"><h2>Could not load this section</h2><p class="note">${err.message}</p><p>Run <code>python backend/app.py</code> from the project folder and open <code>http://127.0.0.1:5000</code>.</p></div>`;
}

async function loadOverview() {
  const d = await fetchJson("/api/overview");
  $("overview").innerHTML = `
    <div class="card">
      <h2>${d.title}</h2>
      <p><strong>Business problem.</strong> ${d.business_problem}</p>
      <p><strong>Objective.</strong> ${d.objective}</p>
      <p><strong>Prediction point.</strong> ${d.prediction_point}</p>
      <div class="kpis">
        ${kpi("Orders", d.dataset.n_orders.toLocaleString())}
        ${kpi("Customers", d.dataset.n_customers.toLocaleString())}
        ${kpi("Return rate", pct(d.overall_return_rate))}
        ${kpi("Date range", d.dataset.date_min + " → " + d.dataset.date_max)}
        ${kpi("Final model", d.selected_model)}
        ${kpi("Locked threshold", num(d.threshold, 2))}
      </div>
    </div>
    <div class="card">
      <h3>Dataset</h3>
      <p>${d.dataset.dataset_name}. Grain: ${d.dataset.modeling_grain}.
      Countries in the labeled table: ${d.dataset.countries.join(", ")}.
      India is ${d.dataset.india_in_modeling_data ? "" : "not "}present — the app does not invent Indian rows.</p>
      <p class="note">${d.dataset.global_superstore_inspection.reason}</p>
    </div>
    <div class="card">
      <h3>Methodology</h3>
      <ul class="plain">${d.methodology.map((x) => `<li>${x}</li>`).join("")}</ul>
      <p class="muted">Train ${d.split.n_train.toLocaleString()} · Validation ${d.split.n_val.toLocaleString()} · Test ${d.split.n_test.toLocaleString()} (2021 held out).</p>
    </div>
  `;
}

async function loadEda() {
  const d = await fetchJson("/api/eda");
  const q = d.quality;
  const e = d.eda;
  $("eda").innerHTML = `
    <div class="card">
      <h2>Data quality</h2>
      <div class="kpis">
        ${kpi("Missing values (max)", Math.max(...Object.values(q.missing_values)))}
        ${kpi("Duplicate order IDs after clean", q.duplicate_order_ids_after_clean)}
        ${kpi("Duplicate rows dropped", q.duplicate_order_ids_dropped)}
        ${kpi("Invalid discounts", q.invalid_discount)}
        ${kpi("Invalid quantities", q.invalid_quantity)}
        ${kpi("Returns", q.n_returns + " / " + q.n_orders)}
      </div>
      <p class="note">${q.leakage_notes.join(" ")}</p>
    </div>
    <div class="card">
      <h2>Exploratory analysis</h2>
      <p>Each chart is an observed return rate. None of these plots establish causation.</p>
      <div class="charts">
        <div><p><strong>By customer segment</strong></p><div class="chart-box"><canvas id="cSeg"></canvas></div><p class="muted">${e.interpretations.segment}</p></div>
        <div><p><strong>By discount range</strong></p><div class="chart-box"><canvas id="cDisc"></canvas></div><p class="muted">${e.interpretations.discount}</p></div>
        <div><p><strong>By order value</strong></p><div class="chart-box"><canvas id="cVal"></canvas></div><p class="muted">${e.interpretations.order_value}</p></div>
        <div><p><strong>By prior-return history</strong></p><div class="chart-box"><canvas id="cHist"></canvas></div><p class="muted">${e.interpretations.history}</p></div>
      </div>
    </div>
    <div class="card">
      <h3>Leakage audit</h3>
      ${table(d.leakage_audit)}
    </div>
    <div class="card">
      <h3>SQL analysis (SQLite)</h3>
      <p class="muted">Queries live in <code>sql/analysis.sql</code>. Results below are executed on the local orders table.</p>
      ${d.sql
        .map(
          (s) => `<h4>${s.title}</h4>${table(s.rows)}`
        )
        .join("")}
    </div>
  `;
  barChart("cSeg", e.return_rate_by_segment, "Segment");
  barChart("cDisc", e.return_rate_by_discount, "Discount");
  barChart("cVal", e.return_rate_by_order_value, "Value");
  barChart("cHist", e.return_rate_by_prior_returns, "History");
}

async function loadStats() {
  const t = await fetchJson("/api/statistics");
  $("stats").innerHTML = `
    <div class="card">
      <h2>Statistical analysis</h2>
      <p><strong>Research question.</strong> ${t.research_question}</p>
      <p><strong>Null.</strong> ${t.null_hypothesis}</p>
      <p><strong>Alternative.</strong> ${t.alternative_hypothesis}</p>
      <p><strong>Test.</strong> ${t.test_name}</p>
      <div class="kpis">
        ${kpi("Z-statistic", num(t.z_statistic, 3))}
        ${kpi("p-value", t.p_value < 0.0001 ? t.p_value.toExponential(2) : num(t.p_value, 4))}
        ${kpi("Difference pA − pB", pct(t.diff_proportions))}
        ${kpi("95% CI", pct(t.ci_95[0]) + " to " + pct(t.ci_95[1]))}
        ${kpi("Risk ratio", num(t.risk_ratio, 2))}
        ${kpi("Reject H0 (α=0.05)", t["significant_at_0.05"] ? "Yes" : "No")}
      </div>
      <p><strong>Group A</strong> (n=${t.n1.toLocaleString()}, returns=${t.x1.toLocaleString()}): ${t.group_a_definition}. Rate ${pct(t.p1)}.</p>
      <p><strong>Group B</strong> (n=${t.n2.toLocaleString()}, returns=${t.x2.toLocaleString()}): ${t.group_b_definition}. Rate ${pct(t.p2)}.</p>
      <p>${t.interpretation}</p>
      <p class="note">${t.causation_disclaimer}</p>
    </div>
  `;
}

function lineChart(id, x, y, title, xLabel, yLabel, diag = false) {
  const ctx = document.getElementById(id);
  if (!ctx || typeof Chart === "undefined") return;
  if (charts[id]) charts[id].destroy();
  const datasets = [{ label: title, data: x.map((xi, i) => ({ x: xi, y: y[i] })), borderColor: "#0f4c6e", pointRadius: 0, tension: 0.1 }];
  if (diag) datasets.push({ label: "Reference", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#999", borderDash: [5, 5], pointRadius: 0 });
  charts[id] = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      parsing: false,
      maintainAspectRatio: false,
      plugins: { legend: { display: diag } },
      scales: {
        x: { type: "linear", title: { display: true, text: xLabel }, min: 0, max: 1 },
        y: { title: { display: true, text: yLabel }, min: 0, max: 1 },
      },
    },
  });
}

async function loadModel() {
  const d = await fetchJson("/api/model");
  const t = d.test;
  const cm = t.confusion_matrix;
  $("model").innerHTML = `
    <div class="card">
      <h2>Model comparison (validation, 2020)</h2>
      <p>Models were compared using PR-AUC at ranking time. Precision/recall/F1 below use a 0.5 cutoff only as a common reference. The operating threshold was tuned later, and only for the selected model.</p>
      ${table(d.validation_comparison)}
      <p><strong>Selected model:</strong> ${d.selection.selected_model} because it had the highest validation PR-AUC (${num(d.selection.validation_pr_auc, 3)}).</p>
      <p class="muted">${d.cv.reason} Mean train-fold PR-AUC: ${d.cv.mean_pr_auc ? num(d.cv.mean_pr_auc, 3) : "n/a"}.</p>
    </div>
    <div class="card">
      <h3>Threshold analysis (validation only)</h3>
      <p>${d.selection.threshold_rule} Locked threshold: <strong>${num(d.selection.locked_threshold, 2)}</strong>.</p>
      <div class="chart-box"><canvas id="cThr"></canvas></div>
    </div>
    <div class="card">
      <h3>Final test evaluation (2021, untouched)</h3>
      <div class="kpis">
        ${kpi("ROC-AUC", num(t.roc_auc))}
        ${kpi("PR-AUC", num(t.pr_auc))}
        ${kpi("Precision", num(t.precision))}
        ${kpi("Recall", num(t.recall))}
        ${kpi("F1", num(t.f1))}
        ${kpi("Accuracy", num(t.accuracy))}
        ${kpi("Brier", num(t.brier_score))}
      </div>
      <p>Confusion matrix at the locked threshold: TN ${cm.tn}, FP ${cm.fp}, FN ${cm.fn}, TP ${cm.tp}.</p>
      <div class="charts">
        <div><p><strong>ROC (test)</strong></p><div class="chart-box"><canvas id="cRoc"></canvas></div></div>
        <div><p><strong>Precision-recall (test)</strong></p><div class="chart-box"><canvas id="cPr"></canvas></div></div>
        <div><p><strong>Calibration (test)</strong></p><div class="chart-box"><canvas id="cCal"></canvas></div></div>
        <div><p><strong>Confusion matrix</strong></p><img class="fig" alt="Confusion matrix" src="/figures/confusion_test.png" /></div>
      </div>
      <p class="note">Brier score and the calibration plot describe whether probabilities are roughly reliable. They are not a claim of perfect calibration.</p>
    </div>
  `;
  const thr = d.threshold_tuning;
  const ctx = document.getElementById("cThr");
  if (charts.cThr) charts.cThr.destroy();
  charts.cThr = new Chart(ctx, {
    type: "line",
    data: {
      labels: thr.map((r) => r.threshold),
      datasets: [
        { label: "Precision", data: thr.map((r) => r.precision), borderColor: "#0f4c6e" },
        { label: "Recall", data: thr.map((r) => r.recall), borderColor: "#9b2c2c" },
        { label: "F1", data: thr.map((r) => r.f1), borderColor: "#1f6f4a" },
      ],
    },
    options: { maintainAspectRatio: false },
  });
  const testc = d.curves.test;
  lineChart("cRoc", testc.roc.fpr, testc.roc.tpr, "ROC", "FPR", "TPR", true);
  lineChart("cPr", testc.pr.recall, testc.pr.precision, "PR", "Recall", "Precision");
  const cal = testc.calibration;
  lineChart("cCal", cal.prob_pred, cal.prob_true, "Calibration", "Predicted", "Observed", true);
}

async function loadExplain() {
  const d = await fetchJson("/api/explainability");
  const g = d.global;
  const ex = d.example;
  $("explain").innerHTML = `
    <div class="card">
      <h2>Explainability</h2>
      <p>${g.plain_language}</p>
      <p>Computed on a ${g.sample_size}-row ${g.sample_source} using ${g.model_type}.</p>
      <img class="fig" src="/figures/shap_global.png" alt="Global SHAP importance" />
      <img class="fig" src="/figures/shap_summary.png" alt="SHAP summary" />
    </div>
    <div class="card">
      <h3>Example individual explanation</h3>
      <p>Order <code>${ex.order_id}</code> for customer <code>${ex.customer_id}</code> on ${ex.order_date}. Actual returned = ${ex.actual_returned}.</p>
      <p>${ex.plain_language}</p>
      ${ex.contributions
        .map(
          (c) => `<div class="shap-row"><span>${c.feature}</span>
            <div class="bar ${c.direction === "increase" ? "up" : "down"}" style="width:${Math.min(100, Math.abs(c.shap) * 400)}%"></div>
            <span>${c.shap >= 0 ? "+" : ""}${num(c.shap, 3)}</span></div>`
        )
        .join("")}
    </div>
    <div class="card">
      <h3>Business insights</h3>
      ${d.insights.items
        .map(
          (i) => `<div class="qa"><strong>${i.title}</strong><p>${i.finding}</p><p class="note">${i.association_not_causation}</p></div>`
        )
        .join("")}
      <p><strong>Recommendation.</strong> ${d.insights.recommendation}</p>
    </div>
  `;
}

async function loadInterview() {
  const d = await fetchJson("/api/interview");
  $("interview").innerHTML = `
    <div class="card">
      <h2>Interview guide</h2>
      ${d.answers.map((x) => `<div class="qa"><strong>${x.q}</strong><p>${x.a}</p></div>`).join("")}
    </div>
  `;
}

let options = null;

async function setupStudio() {
  options = await fetchJson("/api/options");
  $("orderDate").value = options.date_max;
  $("orderDate").min = options.date_min;
  fillSelect($("region"), options.regions);
  fillSelect($("segment"), options.segments);
  fillSelect($("priority"), options.priorities);
  await refreshCustomers();
  $("custSearch").addEventListener("input", refreshCustomers);
  $("customerSelect").addEventListener("change", loadHistory);
  $("orderDate").addEventListener("change", loadHistory);
  $("predictBtn").addEventListener("click", runPredict);
}

function fillSelect(el, values) {
  el.innerHTML = values.map((v) => `<option value="${v}">${v}</option>`).join("");
}

async function refreshCustomers() {
  const q = $("custSearch").value;
  const d = await fetchJson("/api/customers?q=" + encodeURIComponent(q));
  const sel = $("customerSelect");
  const prev = sel.value;
  sel.innerHTML = d.customers
    .map(
      (c) =>
        `<option value="${c.customer_id}">${c.customer_id} · ${c.n_orders} orders · ${c.n_returns} returns</option>`
    )
    .join("");
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  if (sel.value) loadHistory();
}

async function loadHistory() {
  const customer_id = $("customerSelect").value;
  const date = $("orderDate").value;
  if (!customer_id || !date) return;
  const h = await fetchJson(`/api/customer-history?customer_id=${encodeURIComponent(customer_id)}&date=${date}`);
  $("custMeta").textContent = `${h.customer_id} · ${h.segment} · ${h.region} · ${h.country}`;
  $("region").value = h.region;
  $("segment").value = h.segment;
  const days = h.is_first_order ? "First order" : h.days_since_previous_order;
  $("histMetrics").innerHTML = [
    ["Previous orders", h.previous_order_count],
    ["Previous returns", h.previous_return_count],
    ["Historical return rate", pct(h.historical_return_rate)],
    ["Avg previous order value", "$" + num(h.historical_average_order_value, 2)],
    ["Days since previous order", days],
  ]
    .map(([lab, val]) => `<div class="metric"><div class="lab">${lab}</div><div class="val">${val}</div></div>`)
    .join("");
  const rows = h.history_table || [];
  if (!rows.length) {
    $("histTable").innerHTML = "<tr><td>No prior orders before this date.</td></tr>";
    return;
  }
  const keys = Object.keys(rows[0]);
  $("histTable").innerHTML =
    "<thead><tr>" +
    keys.map((k) => `<th>${k}</th>`).join("") +
    "</tr></thead><tbody>" +
    rows.map((r) => "<tr>" + keys.map((k) => `<td>${r[k]}</td>`).join("") + "</tr>").join("") +
    "</tbody>";
}

async function runPredict() {
  const payload = {
    customer_id: $("customerSelect").value,
    order_date: $("orderDate").value,
    region: $("region").value,
    segment: $("segment").value,
    order_priority: $("priority").value,
    order_value: $("orderValue").value,
    quantity: $("quantity").value,
    discount: $("discount").value,
    max_discount: $("maxDiscount").value,
    number_of_products: $("nProducts").value,
    number_of_categories: $("nCategories").value,
  };
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const d = await res.json();
  if (d.error) {
    $("predResult").innerHTML = `<p class="note">${d.error}</p>`;
    return;
  }
  const high = d.classification.toLowerCase().includes("high");
  $("predResult").innerHTML = `
    <div class="card" style="margin-top:1rem">
      <div class="prob ${high ? "high" : "low"}">${pct(d.probability)}</div>
      <p><strong>${d.classification}</strong></p>
      <p class="muted">Model threshold: ${pct(d.threshold)}. History used: ${d.history.previous_order_count} prior orders, ${d.history.previous_return_count} prior returns, rate ${pct(d.history.historical_return_rate)}.</p>
      <h3>Key contributing features (SHAP)</h3>
      <p class="note">${d.shap_note}</p>
      ${d.shap
        .map(
          (s) => `<div class="shap-row"><span>${s.feature}</span>
            <div class="bar ${s.direction === "increase" ? "up" : "down"}" style="width:${Math.min(100, Math.abs(s.contribution) * 250)}%"></div>
            <span>${s.contribution >= 0 ? "+" : ""}${num(s.contribution, 3)}</span></div>`
        )
        .join("")}
    </div>
  `;
}

async function safe(fn, id) {
  try {
    await fn();
  } catch (err) {
    console.error(err);
    showError(id, err);
  }
}

safe(loadOverview, "overview");
safe(loadEda, "eda");
safe(loadStats, "stats");
safe(loadModel, "model");
safe(loadExplain, "explain");
safe(loadInterview, "interview");
safe(setupStudio, "custMeta");
