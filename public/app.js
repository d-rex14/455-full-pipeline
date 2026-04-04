const form = document.getElementById("predict-form");
const orderInput = document.getElementById("order-id");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");

const emptyEl = document.getElementById("result-empty");
const errorEl = document.getElementById("result-error");
const cardEl = document.getElementById("result-card");
const statusEl = document.getElementById("result-status");
const metricOrder = document.getElementById("metric-order");
const metricProb = document.getElementById("metric-prob");
const metricThreshold = document.getElementById("metric-threshold");
const meterFill = document.getElementById("meter-fill");

function setLoading(loading) {
  submitBtn.disabled = loading;
  btnSpinner.hidden = !loading;
  btnLabel.textContent = loading ? "Scoring…" : "Run prediction";
}

function showEmpty() {
  emptyEl.hidden = false;
  errorEl.hidden = true;
  cardEl.hidden = true;
  errorEl.textContent = "";
}

function showError(message) {
  emptyEl.hidden = true;
  errorEl.hidden = false;
  cardEl.hidden = true;
  errorEl.textContent = message;
}

function showResult(data) {
  emptyEl.hidden = true;
  errorEl.hidden = true;
  cardEl.hidden = false;

  const fraud = Boolean(data.is_fraud);
  statusEl.textContent = fraud ? "Flagged as fraud" : "Likely legitimate";
  statusEl.className = "result-status " + (fraud ? "result-status--fraud" : "result-status--clear");

  metricOrder.textContent = String(data.order_id);
  metricProb.textContent = typeof data.probability === "number" ? data.probability.toFixed(4) : "—";
  metricThreshold.textContent =
    typeof data.threshold === "number" ? data.threshold.toFixed(4) : String(data.threshold ?? "—");

  const p = typeof data.probability === "number" ? Math.min(1, Math.max(0, data.probability)) : 0;
  meterFill.style.width = `${(p * 100).toFixed(1)}%`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const raw = orderInput.value.trim();
  const orderId = parseInt(raw, 10);
  if (!Number.isFinite(orderId) || orderId < 1) {
    showError("Enter a positive integer order ID.");
    return;
  }

  setLoading(true);
  showEmpty();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_id: orderId }),
    });
    const body = await res.json().catch(() => ({}));

    if (!res.ok) {
      const msg = body.error || `Request failed (${res.status})`;
      showError(msg);
      return;
    }

    showResult(body);
  } catch (err) {
    showError(err instanceof Error ? err.message : "Network error");
  } finally {
    setLoading(false);
  }
});
