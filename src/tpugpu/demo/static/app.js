const runButton = document.getElementById("runButton");
const labelInput = document.getElementById("labelInput");
const stepsInput = document.getElementById("stepsInput");
const leftConnector = document.getElementById("lineUsEu");
const rightConnector = document.getElementById("lineEuAsia");
const expertLeft = document.getElementById("nodeTpu");
const expertRight = document.getElementById("nodeGpu");
const canvas = document.getElementById("stateCanvas");
const ctx = canvas.getContext("2d");
const machineCardObjects = Array.from(document.querySelectorAll(".machine-card-fo"));

let currentSource = null;

function sizeMachineCards() {
  for (const foreignObject of machineCardObjects) {
    const card = foreignObject.querySelector(".machine-card");
    if (!card) {
      continue;
    }

    const width = Math.ceil(card.scrollWidth);
    const height = Math.ceil(card.scrollHeight);
    const anchorX = Number(foreignObject.dataset.anchorX || 0);
    const gap = Number(foreignObject.dataset.gap || 0);
    const anchor = foreignObject.dataset.anchor || "start";

    foreignObject.setAttribute("width", String(width));
    foreignObject.setAttribute("height", String(height));

    let x = anchorX + gap;
    if (anchor === "end") {
      x = anchorX - gap - width;
    } else if (anchor === "middle") {
      x = anchorX - width / 2;
    }

    foreignObject.setAttribute("x", String(Math.round(x)));
  }
}

function drawFrame(flatPixels) {
  const imageData = ctx.createImageData(32, 32);
  for (let i = 0; i < flatPixels.length; i += 1) {
    const value = flatPixels[i];
    const offset = i * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = 32;
  offscreen.height = 32;
  offscreen.getContext("2d").putImageData(imageData, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

function clearGlow() {
  leftConnector.classList.remove("active");
  rightConnector.classList.remove("active");
  expertLeft.classList.remove("active");
  expertRight.classList.remove("active");
}

function restartPulse(element) {
  element.classList.remove("active");
  void element.getBoundingClientRect();
  element.classList.add("active");
}

function flashExpert(expertId) {
  clearGlow();
  if (expertId === 0) {
    restartPulse(leftConnector);
    restartPulse(expertLeft);
  } else if (expertId === 1) {
    restartPulse(rightConnector);
    restartPulse(expertRight);
  }

  window.setTimeout(clearGlow, 180);
}

function startDemo() {
  if (currentSource) {
    currentSource.close();
  }

  const seed = Math.floor(Math.random() * 2147483647);
  const params = new URLSearchParams({
    label: labelInput.value,
    steps: stepsInput.value,
    strategy: "alternating",
    seed: String(seed),
  });
  const source = new EventSource(`/api/demo/stream?${params.toString()}`);
  currentSource = source;
  runButton.disabled = true;
  runButton.textContent = "Generating...";

  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    drawFrame(payload.frame);

    if (payload.type === "start") {
      clearGlow();
      return;
    }

    if (payload.selected_expert !== null) {
      flashExpert(payload.selected_expert);
    }

    if (payload.type === "done") {
      runButton.disabled = false;
      runButton.textContent = "Generate";
      source.close();
    }
  };

  source.onerror = () => {
    runButton.disabled = false;
    runButton.textContent = "Generate";
    source.close();
  };
}

runButton.addEventListener("click", startDemo);
window.addEventListener("load", sizeMachineCards);
window.addEventListener("resize", sizeMachineCards);
sizeMachineCards();
drawFrame(new Array(32 * 32).fill(0));
