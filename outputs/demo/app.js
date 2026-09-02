"use strict";

const buttons = Array.from(document.querySelectorAll("[data-frame]"));
const replayImage = document.getElementById("replay-image");
const replayCaption = document.getElementById("replay-caption");
const detailsTitle = document.getElementById("frame-details-title");
const depthFacts = document.getElementById("depth-facts");

function formatDepth(value) {
  if (value === 0) return "0 µm";
  return `${value > 0 ? "+" : "−"}${Math.abs(value)} µm`;
}

function formatPq(value) {
  return typeof value === "number" ? value.toFixed(3) : "not scored";
}

function addFact(system) {
  const row = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = system.label;
  const after = typeof system.after_pq === "number" ? formatPq(system.after_pq) : "not scored";
  description.textContent = `${formatPq(system.before_pq)} → ${after} PQ · residual ${
    system.residual_depth_um >= 0 ? "+" : "−"
  }${Math.abs(system.residual_depth_um).toFixed(2)} µm`;
  row.append(term, description);
  return row;
}

function showFrame(data, index) {
  const frame = data.frames[index];
  buttons.forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.frame) === index));
  });
  replayImage.src = frame.path;
  replayImage.alt = frame.alt;
  replayCaption.textContent = frame.caption;
  detailsTitle.textContent = formatDepth(frame.depth_um);
  depthFacts.replaceChildren(...frame.systems.map(addFact));
}

fetch("site-data.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Evidence data returned ${response.status}`);
    return response.json();
  })
  .then((data) => {
    buttons.forEach((button) => {
      button.addEventListener("click", () => showFrame(data, Number(button.dataset.frame)));
    });
  })
  .catch(() => {
    replayCaption.textContent =
      "The interactive index could not load; the frozen representative frame remains visible.";
    replayCaption.setAttribute("role", "alert");
  });
