// Helper
const $ = (id) => document.getElementById(id);

const form = $("fraud-form");
const resultStatus = $("result-status");
const resultScore = $("result-score");
const resultNote = $("result-note");
const resultSource = $("result-source");
const historyBody = $("history-body");
const resetBtn = $("reset-btn");

// In-browser heuristic fallback (if backend / model fails)
function calculateDummyRisk(payload) {
  let risk = 0;

  const amount = Number(payload.tx_amount || 0);
  const hour = Number(payload.hour_of_day || 0);
  const accountAge = Number(payload.account_age_days || 0);
  const velocity = Number(payload.tx_velocity || 0);
  const failed = Number(payload.num_failed_tx || 0);
  const credit = Number(payload.credit_score || 0);
  const vpn = Number(payload.vpn_detected || 0);

  // High-value + night + new account
  if (amount > 5000 && hour >= 0 && hour <= 6 && accountAge < 30) {
    risk += 0.45;
  }

  // High velocity + VPN
  if (velocity > 10 && vpn === 1) {
    risk += 0.25;
  }

  // Many failed tx
  if (failed >= 3) {
    risk += 0.15;
  }

  // Low credit + high amount
  if (credit > 0 && credit < 550 && amount > 3000) {
    risk += 0.25;
  }

  // Base scaling by amount
  risk += Math.min(amount / 20000, 0.15);

  // clamp 0–1
  risk = Math.max(0, Math.min(1, risk));
  return risk;
}

// Call Flask backend
async function predictWithBackend(payload) {
  try {
    const res = await fetch("http://127.0.0.1:5000/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error("Network error: " + res.status);
    }

    const data = await res.json();
    const riskScore = Number(data.fraud_probability ?? 0);
    const source = data.source || "ml_model";
    const thr = data.threshold ?? 0.5;

    return { riskScore, source, threshold: thr };
  } catch (err) {
    console.error("Prediction error, using fallback heuristic:", err);
    const riskScore = calculateDummyRisk(payload);
    return { riskScore, source: "heuristic_fallback", threshold: 0.5 };
  }
}

// Update result panel
function updateResultUI(riskScore, sourceText) {
  const percent = (riskScore * 100).toFixed(1) + "%";
  resultScore.textContent = percent;
  resultSource.textContent = `Source: ${sourceText}`;

  resultStatus.classList.remove("ok", "warn", "danger");

  if (riskScore < 0.3) {
    resultStatus.textContent = "Low Risk — Likely Legitimate";
    resultStatus.classList.add("ok");
    resultNote.textContent =
      "Transaction looks normal based on current rules/model. In production, this would likely pass automatically.";
  } else if (riskScore < 0.7) {
    resultStatus.textContent = "Medium Risk — Manual Review Suggested";
    resultStatus.classList.add("warn");
    resultNote.textContent =
      "Some suspicious patterns detected. In a real system, this would be routed for analyst review.";
  } else {
    resultStatus.textContent = "High Risk — Potential Fraud";
    resultStatus.classList.add("danger");
    resultNote.textContent =
      "Multiple high-risk signals triggered. In production, this would likely be blocked or strongly challenged.";
  }
}

// Add row to session history
function pushToHistory(payload, riskScore) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  const row = document.createElement("tr");

  const riskLabel =
    riskScore < 0.3 ? "low" : riskScore < 0.7 ? "med" : "high";

  const country = (payload.ip_country || "-").toUpperCase();
  const amount = Number(payload.tx_amount || 0).toFixed(2);
  const vpnText = payload.vpn_detected === "1" ? "Yes" : "No";

  row.innerHTML = `
    <td>${timeStr}</td>
    <td>₹${amount}</td>
    <td>${country}</td>
    <td>${vpnText}</td>
    <td>
      <span class="badge ${riskLabel}">
        ${(riskScore * 100).toFixed(1)}%
      </span>
    </td>
    <td>${riskLabel === "high" ? "Flagged" : "Clear"}</td>
  `;

  historyBody.prepend(row);
}

// Form submit handler
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    tx_amount: form.tx_amount.value,
    hour_of_day: form.hour_of_day.value,
    day_of_week: form.day_of_week.value,
    month: form.month.value,
    account_age_days: form.account_age_days.value,
    tx_velocity: form.tx_velocity.value,
    num_failed_tx: form.num_failed_tx.value,
    credit_score: form.credit_score.value,
    device_type: form.device_type.value,
    browser: form.browser.value,
    ip_country: form.ip_country.value,
    vpn_detected: form.vpn_detected.value,
    desc_summary: form.desc_summary.value,
  };

  resultStatus.textContent = "Calculating...";
  resultStatus.classList.remove("ok", "warn", "danger");
  resultScore.textContent = "…";
  resultNote.textContent =
    "Sending payload to backend API. If model is not available, heuristic rules are used.";
  resultSource.textContent = "Source: pending";

  const { riskScore, source } = await predictWithBackend(payload);

  updateResultUI(riskScore, source);
  pushToHistory(payload, riskScore);
});

// Reset behaviour
resetBtn.addEventListener("click", () => {
  resultStatus.textContent = "No prediction yet.";
  resultStatus.classList.remove("ok", "warn", "danger");
  resultScore.textContent = "–";
  resultNote.textContent =
    "Submit a transaction to see the predicted fraud risk.";
  resultSource.textContent = "Source: –";
});
