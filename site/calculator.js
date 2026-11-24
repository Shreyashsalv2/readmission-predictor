// Scores the exported logistic-regression model (model.json) in the browser.
// Same arithmetic as readmit/export.py:score; tests check both against scikit-learn.
function score(model, record) {
  let z = model.intercept;
  for (const f of model.numeric) z += f.coef * (Number(record[f.name]) - f.mean) / f.scale;
  for (const f of model.categorical) {
    const i = f.categories.indexOf(String(record[f.name]));
    if (i >= 0) z += f.coefs[i];                 // unseen value -> no term, like handle_unknown="ignore"
  }
  return 1 / (1 + Math.exp(-z));
}

const LABELS = {
  age_years: "Age (years)", time_in_hospital: "Days in hospital", num_medications: "Medications given",
  number_diagnoses: "Number of diagnoses", number_inpatient: "Inpatient visits, past year",
  number_emergency: "Emergency visits, past year", number_outpatient: "Outpatient visits, past year",
  discharge_group: "Discharged to", admission_source_group: "Admitted from", diag_1_group: "Main diagnosis",
  A1Cresult: "HbA1c result", insulin: "Insulin during stay", diabetesMed: "On diabetes medication",
};
const DEFAULTS = { age_years: 65, time_in_hospital: 4, num_medications: 15, number_diagnoses: 8,
  number_inpatient: 0, number_emergency: 0, number_outpatient: 0 };

function buildCalculator(model, root) {
  const form = document.createElement("form");
  form.className = "calc";
  for (const f of model.numeric) {
    const id = "f_" + f.name;
    form.insertAdjacentHTML("beforeend",
      `<label for="${id}">${LABELS[f.name] || f.name}</label>` +
      `<input id="${id}" name="${f.name}" type="number" min="0" step="1" value="${DEFAULTS[f.name] ?? 0}">`);
  }
  for (const f of model.categorical) {
    const id = "f_" + f.name;
    const opts = f.categories.map(c => `<option>${c}</option>`).join("");
    form.insertAdjacentHTML("beforeend",
      `<label for="${id}">${LABELS[f.name] || f.name}</label><select id="${id}" name="${f.name}">${opts}</select>`);
  }
  const out = document.createElement("div");
  out.className = "calc-out";
  root.append(form, out);
  const pick = (name, v) => { const el = form.elements[name]; if ([...el.options].some(o => o.value === v)) el.value = v; };
  pick("discharge_group", "Home"); pick("admission_source_group", "Emergency room");
  pick("diag_1_group", "Circulatory"); pick("A1Cresult", "Not measured"); pick("insulin", "No"); pick("diabetesMed", "Yes");
  const update = () => {
    const rec = Object.fromEntries(new FormData(form).entries());
    const p = score(model, rec);
    const base = model.meta.base_rate;
    out.innerHTML = `<div class="risk">${(100 * p).toFixed(1)}%</div>` +
      `<div class="risk-sub">estimated chance of readmission within 30 days ` +
      `(average patient: ${(100 * base).toFixed(1)}%, so ${(p / base).toFixed(1)}× average)</div>`;
  };
  form.addEventListener("input", update);
  update();
}

if (typeof module !== "undefined") module.exports = { score };
