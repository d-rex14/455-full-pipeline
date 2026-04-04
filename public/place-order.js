const form = document.getElementById("place-form");
const modeEl = document.getElementById("customer-mode");
const blockExisting = document.getElementById("block-existing");
const blockNew = document.getElementById("block-new");
const customerSelect = document.getElementById("customer-id");
const linesWrap = document.getElementById("lines-wrap");
const addLineBtn = document.getElementById("add-line");
const errEl = document.getElementById("form-error");
const okEl = document.getElementById("form-success");
const submitBtn = document.getElementById("submit-order");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");

let products = [];
let customers = [];

function setLoading(loading) {
  submitBtn.disabled = loading;
  btnSpinner.hidden = !loading;
  btnLabel.textContent = loading ? "Submitting…" : "Submit order";
}

function showError(msg) {
  errEl.hidden = !msg;
  errEl.textContent = msg || "";
  okEl.hidden = true;
}

function showSuccess(html) {
  okEl.hidden = false;
  okEl.innerHTML = html;
  errEl.hidden = true;
}

function syncCustomerBlocks() {
  const mode = modeEl.value;
  blockExisting.hidden = mode !== "existing";
  blockNew.hidden = mode !== "new";
}

function productOptionsHtml(selectedId) {
  return products
    .map(
      (p) =>
        `<option value="${p.product_id}" data-price="${p.price}" ${String(p.product_id) === String(selectedId) ? "selected" : ""}>${escapeHtml(p.product_name)} — $${Number(p.price).toFixed(2)}</option>`
    )
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function addLineRow() {
  const defaultPid = products[0]?.product_id ?? "";
  const row = document.createElement("div");
  row.className = "line-row";
  row.innerHTML = `
    <label class="field">
      <span class="field-label">Product</span>
      <select class="line-product">${productOptionsHtml(defaultPid)}</select>
    </label>
    <label class="field">
      <span class="field-label">Qty</span>
      <input type="number" class="line-qty" min="1" step="1" value="1" />
    </label>
    <button type="button" class="btn btn-ghost line-remove" aria-label="Remove line">Remove</button>
  `;
  row.querySelector(".line-remove").addEventListener("click", () => {
    if (linesWrap.querySelectorAll(".line-row").length > 1) row.remove();
  });
  linesWrap.appendChild(row);
}

async function loadCatalog() {
  const [pr, cr] = await Promise.all([
    fetch("/api/products").then((r) => r.json()),
    fetch("/api/customers").then((r) => r.json()),
  ]);
  if (pr.error) throw new Error(pr.error);
  if (cr.error) throw new Error(cr.error);
  products = pr.products || [];
  customers = cr.customers || [];
  customerSelect.innerHTML = customers
    .map((c) => `<option value="${c.customer_id}">${escapeHtml(c.email)} — ${escapeHtml(c.full_name)}</option>`)
    .join("");
  linesWrap.innerHTML = "";
  if (products.length === 0) {
    showError("No products in database. Seed Supabase first.");
    return;
  }
  addLineRow();
}

modeEl.addEventListener("change", syncCustomerBlocks);
addLineBtn.addEventListener("click", () => addLineRow());

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  okEl.hidden = true;

  const lines = [];
  for (const row of linesWrap.querySelectorAll(".line-row")) {
    const sel = row.querySelector(".line-product");
    const qty = parseInt(row.querySelector(".line-qty").value, 10);
    const pid = parseInt(sel.value, 10);
    if (!Number.isFinite(pid) || !Number.isFinite(qty) || qty < 1) {
      showError("Each line needs a product and quantity ≥ 1.");
      return;
    }
    lines.push({ product_id: pid, quantity: qty });
  }

  const payload = {
    customer_mode: modeEl.value,
    payment_method: document.getElementById("payment-method").value,
    device_type: document.getElementById("device-type").value,
    ip_country: document.getElementById("ip-country").value.trim() || "US",
    shipping_state: document.getElementById("shipping-state").value.trim() || "TX",
    billing_zip: document.getElementById("billing-zip").value.trim(),
    shipping_zip: document.getElementById("shipping-zip").value.trim(),
    shipping_fee: parseFloat(document.getElementById("shipping-fee").value),
    tax_rate: parseFloat(document.getElementById("tax-rate").value),
    promo_used: document.getElementById("promo-used").checked ? 1 : 0,
    lines,
  };

  if (modeEl.value === "existing") {
    payload.customer_id = parseInt(customerSelect.value, 10);
  } else {
    payload.full_name = document.getElementById("full-name").value.trim();
    payload.email = document.getElementById("email").value.trim();
    payload.gender = document.getElementById("gender").value;
    const bd = document.getElementById("birthdate").value;
    payload.birthdate = bd || "1990-01-01";
    payload.city = document.getElementById("city").value.trim() || null;
    payload.state = document.getElementById("state").value.trim() || null;
    if (!payload.full_name || !payload.email) {
      showError("Full name and email are required for a new customer.");
      return;
    }
  }

  setLoading(true);
  try {
    const res = await fetch("/api/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      showError(body.error || `Request failed (${res.status})`);
      return;
    }
    showSuccess(
      `<strong>Order created.</strong> order_id <code>${body.order_id}</code>, total <code>$${Number(body.order_total).toFixed(2)}</code>. <a href="/admin">Open admin</a> to score and label.`
    );
    form.reset();
    document.getElementById("birthdate").value = "1990-01-15";
    syncCustomerBlocks();
    await loadCatalog();
  } catch (err) {
    showError(err instanceof Error ? err.message : "Network error");
  } finally {
    setLoading(false);
  }
});

loadCatalog()
  .then(() => syncCustomerBlocks())
  .catch((err) => showError(err instanceof Error ? err.message : String(err)));
