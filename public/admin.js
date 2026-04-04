const tbody = document.getElementById("orders-tbody");
const btnRefresh = document.getElementById("btn-refresh");
const btnScore = document.getElementById("btn-score");
const btnScoreAll = document.getElementById("btn-score-all");
const selectAll = document.getElementById("select-all");
const statusEl = document.getElementById("admin-status");
const errEl = document.getElementById("admin-error");
const searchInput = document.getElementById("order-search");
const pageSizeSelect = document.getElementById("page-size");
const adminMeta = document.getElementById("admin-meta");
const paginationEl = document.getElementById("pagination");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const paginationInfo = document.getElementById("pagination-info");

/** @type {object[]} */
let allOrders = [];
let currentPage = 1;

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function setError(msg) {
  errEl.hidden = !msg;
  errEl.textContent = msg || "";
}

function fmtMoney(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return "$" + Number(n).toFixed(2);
}

function fmtProb(n) {
  if (n == null) return "—";
  return Number(n).toFixed(4);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function getSearchTerms() {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) return null;
  return q.split(/\s+/).filter(Boolean);
}

function orderMatchesSearch(o, terms) {
  if (!terms || terms.length === 0) return true;
  const cust = o.customers || {};
  const hay = [
    String(o.order_id),
    String(o.order_datetime || ""),
    String(cust.full_name || ""),
    String(cust.email || ""),
    fmtMoney(o.order_total),
    String(o.order_total ?? ""),
    String(o.payment_method || ""),
    String(o.shipping_state || ""),
    String(o.ip_country || ""),
    String(o.is_fraud ?? ""),
    String(o.ml_predicted_fraud ?? ""),
    String(o.ml_fraud_probability ?? ""),
    o.admin_reviewed ? "yes reviewed" : "no",
  ]
    .join(" ")
    .toLowerCase();
  return terms.every((t) => hay.includes(t));
}

function getFilteredOrders() {
  const terms = getSearchTerms();
  return allOrders.filter((o) => orderMatchesSearch(o, terms));
}

function getPageSize() {
  const n = parseInt(pageSizeSelect.value, 10);
  return Number.isFinite(n) && n > 0 ? n : 25;
}

function selectedIds() {
  const boxes = tbody.querySelectorAll('input[type="checkbox"][data-order-id]:checked');
  return Array.from(boxes).map((el) => parseInt(el.getAttribute("data-order-id"), 10));
}

function rowHtml(o) {
  const cust = o.customers || {};
  const name = cust.full_name || cust.email || "—";
  return `<tr data-order-id="${o.order_id}">
    <td><input type="checkbox" data-order-id="${o.order_id}" aria-label="Select order ${o.order_id}" /></td>
    <td class="mono">${o.order_id}</td>
    <td class="mono td-nowrap">${escapeHtml(String(o.order_datetime || "").slice(0, 19))}</td>
    <td>${escapeHtml(name)}</td>
    <td class="mono">${fmtMoney(o.order_total)}</td>
    <td class="mono">${fmtProb(o.ml_fraud_probability)}</td>
    <td>${o.ml_predicted_fraud == null ? "—" : o.ml_predicted_fraud}</td>
    <td>${o.is_fraud}</td>
    <td>${o.admin_reviewed ? "yes" : "no"}</td>
    <td class="td-actions">
      <button type="button" class="btn btn-ghost btn-tiny" data-action="legit" data-id="${o.order_id}">Legit</button>
      <button type="button" class="btn btn-ghost btn-tiny btn-tiny--danger" data-action="fraud" data-id="${o.order_id}">Fraud</button>
    </td>
  </tr>`;
}

function bindRowActions() {
  tbody.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = parseInt(btn.getAttribute("data-id"), 10);
      const isFraud = btn.getAttribute("data-action") === "fraud" ? 1 : 0;
      saveLabel(id, isFraud);
    });
  });
}

function renderPage() {
  const filtered = getFilteredOrders();
  const pageSize = getPageSize();
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  if (currentPage > totalPages) currentPage = totalPages;
  if (currentPage < 1) currentPage = 1;

  const start = (currentPage - 1) * pageSize;
  const pageRows = filtered.slice(start, start + pageSize);

  selectAll.checked = false;

  if (allOrders.length === 0) {
    paginationEl.hidden = true;
    adminMeta.textContent = "";
    tbody.innerHTML = `<tr><td colspan="10" class="td-muted">No orders loaded.</td></tr>`;
    return;
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="td-muted">No orders match your search.</td></tr>`;
    adminMeta.textContent = `0 of ${allOrders.length} orders match.`;
    paginationEl.hidden = true;
    return;
  }

  adminMeta.textContent = `Showing ${start + 1}–${start + pageRows.length} of ${filtered.length} matching (${allOrders.length} total).`;
  tbody.innerHTML = pageRows.map(rowHtml).join("");
  bindRowActions();

  paginationEl.hidden = totalPages <= 1;
  paginationInfo.textContent = `Page ${currentPage} / ${totalPages}`;
  btnPrev.disabled = currentPage <= 1;
  btnNext.disabled = currentPage >= totalPages;
}

async function loadOrders() {
  setError("");
  tbody.innerHTML = `<tr><td colspan="10" class="td-muted">Loading…</td></tr>`;
  adminMeta.textContent = "";
  paginationEl.hidden = true;

  const res = await fetch("/api/orders");
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    tbody.innerHTML = `<tr><td colspan="10" class="td-muted">Failed to load</td></tr>`;
    setError(body.error || `HTTP ${res.status}`);
    allOrders = [];
    return;
  }
  allOrders = body.orders || [];
  currentPage = 1;
  if (allOrders.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="td-muted">No orders</td></tr>`;
    return;
  }
  renderPage();
}

async function saveLabel(orderId, isFraud) {
  setError("");
  setStatus("Saving label…");
  try {
    const res = await fetch("/api/orders/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_id: orderId, is_fraud: isFraud }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      setError(body.error || `HTTP ${res.status}`);
      setStatus("");
      return;
    }
    setStatus(`Order ${orderId} saved as ${isFraud ? "fraud" : "legitimate"}.`);
    await loadOrders();
  } catch (e) {
    setError(e instanceof Error ? e.message : "Network error");
    setStatus("");
  }
}

async function runScore(all) {
  setError("");
  const ids = all ? null : selectedIds();
  if (!all && (!ids || ids.length === 0)) {
    setError("Select at least one order on this page, or use “Score all”.");
    return;
  }
  setStatus(all ? "Scoring all (up to 400)…" : `Scoring ${ids.length} order(s)…`);
  btnScore.disabled = true;
  btnScoreAll.disabled = true;
  try {
    const res = await fetch("/api/orders/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(all ? { all: true } : { order_ids: ids }),
    });
    const ct = res.headers.get("Content-Type") || "";
    let body = {};
    if (ct.includes("application/json")) {
      body = await res.json().catch(() => ({}));
    } else {
      const text = await res.text().catch(() => "");
      if (!res.ok) setError(text.slice(0, 500) || `HTTP ${res.status}`);
    }
    if (!res.ok) {
      if (Object.keys(body).length) setError(body.error || `HTTP ${res.status}`);
      setStatus("");
      return;
    }
    const ok = body.results?.filter((r) => r.ok).length ?? 0;
    const bad = body.results?.filter((r) => !r.ok).length ?? 0;
    setStatus(`Scored ${ok} OK, ${bad} failed. Threshold ${body.threshold}.`);
    await loadOrders();
  } catch (e) {
    setError(e instanceof Error ? e.message : "Network error");
    setStatus("");
  } finally {
    btnScore.disabled = false;
    btnScoreAll.disabled = false;
  }
}

selectAll.addEventListener("change", () => {
  tbody.querySelectorAll('input[type="checkbox"][data-order-id]').forEach((cb) => {
    cb.checked = selectAll.checked;
  });
});

searchInput.addEventListener("input", () => {
  currentPage = 1;
  renderPage();
});

pageSizeSelect.addEventListener("change", () => {
  currentPage = 1;
  renderPage();
});

btnPrev.addEventListener("click", () => {
  currentPage -= 1;
  renderPage();
});

btnNext.addEventListener("click", () => {
  currentPage += 1;
  renderPage();
});

btnRefresh.addEventListener("click", () => loadOrders());
btnScore.addEventListener("click", () => runScore(false));
btnScoreAll.addEventListener("click", () => runScore(true));

loadOrders();
