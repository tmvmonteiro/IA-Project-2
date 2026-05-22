const form = document.querySelector("#predictionForm");
const prediction = document.querySelector("#prediction");
const statusText = document.querySelector("#status");
const modelMeta = document.querySelector("#modelMeta");
const numericFields = document.querySelector("#numericFields");
const predictButton = document.querySelector("#predictButton");

let options = null;

const numericLabels = {
  TotalArea: "Total area (m2)",
  LivingArea: "Living area (m2)",
  ConstructionYear: "Construction year",
  TotalRooms: "Total rooms",
  NumberOfBedrooms: "Bedrooms",
  NumberOfBathrooms: "Bathrooms",
};

function fillSelect(id, values) {
  const select = document.querySelector(`#${id}`);
  select.innerHTML = "";

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function createNumericInput(feature, settings) {
  const label = document.createElement("label");
  const caption = document.createElement("span");
  caption.textContent = numericLabels[feature] || feature;

  const input = document.createElement("input");
  input.name = feature;
  input.id = feature;
  input.type = "number";
  input.step = Number.isInteger(settings.median) ? "1" : "0.1";
  input.min = Math.max(0, Math.floor(settings.min));
  input.value = Math.round(settings.median);

  const hint = document.createElement("small");
  hint.textContent = `Observed range ${Math.floor(settings.min)}-${Math.ceil(settings.max)}; common range ${Math.floor(settings.recommendedMin)}-${Math.ceil(settings.recommendedMax)}. Larger values are allowed.`;

  label.appendChild(caption);
  label.appendChild(input);
  label.appendChild(hint);
  numericFields.appendChild(label);
}

function updateParishes() {
  const municipality = document.querySelector("#Municipality").value;
  const parishes = options.options.parishesByMunicipality[municipality] || [];
  fillSelect("Parish", parishes);
}

async function loadOptions() {
  const response = await fetch("/api/options");
  if (!response.ok) {
    throw new Error("Could not load model options.");
  }

  options = await response.json();
  modelMeta.textContent = `Trained on ${options.trainingRows.toLocaleString()} filtered Porto properties.`;

  fillSelect("Municipality", options.options.categorical.Municipality);
  updateParishes();
  fillSelect("Type", options.options.categorical.Type);
  fillSelect("EnergyCertificate", options.options.categorical.EnergyCertificate);
  fillSelect("ConservationStatus", options.options.categorical.ConservationStatus);
  fillSelect("Parking", options.options.categorical.Parking);
  fillSelect("Elevator", options.options.categorical.Elevator);
  fillSelect("Floor", options.options.categorical.Floor);

  options.numericalFeatures.forEach((feature) => {
    createNumericInput(feature, options.options.numerical[feature]);
  });
}

function formPayload() {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

async function predictValue(event) {
  event.preventDefault();
  predictButton.disabled = true;
  statusText.classList.remove("error");
  statusText.textContent = "Estimating value...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });

    if (!response.ok) {
      throw new Error("Prediction request failed.");
    }

    const result = await response.json();
    prediction.textContent = result.formattedPrediction;
    statusText.textContent = `Estimate for ${result.received.Parish}, ${result.received.Municipality}.`;
  } catch (error) {
    statusText.classList.add("error");
    statusText.textContent = error.message;
  } finally {
    predictButton.disabled = false;
  }
}

document.querySelector("#Municipality").addEventListener("change", updateParishes);
form.addEventListener("submit", predictValue);

loadOptions().catch((error) => {
  statusText.classList.add("error");
  statusText.textContent = error.message;
});
