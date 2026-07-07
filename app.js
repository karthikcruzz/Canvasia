// Build API URLs that work both locally and behind workspace proxies like:
// /dev-workspaces/<workspace>/proxy/8000
function getBasePath() {
  let path = window.location.pathname;

  if (path.endsWith("/index.html")) {
    path = path.slice(0, -"/index.html".length);
  }

  if (path.endsWith("/")) {
    path = path.slice(0, -1);
  }

  return path;
}

const BASE_PATH = getBasePath();
const BACKEND_PORT = "8001";

function getApiBasePath() {
  if (BASE_PATH.includes(`/proxy/${BACKEND_PORT}`)) {
    return BASE_PATH;
  }

  if (window.location.port === BACKEND_PORT) {
    return BASE_PATH;
  }

  if (window.location.protocol.startsWith("http")) {
    return `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`;
  }

  return BASE_PATH;
}

const API_BASE_PATH = getApiBasePath();

function withBase(route) {
  const cleanRoute = route.startsWith("/") ? route : `/${route}`;
  return `${API_BASE_PATH}${cleanRoute}` || cleanRoute;
}

const API = {
  state: withBase("/api/state"),
  start: withBase("/api/start"),
  reset: withBase("/api/reset"),
  turn: withBase("/api/turn"),
  decide: withBase("/api/decide"),
  generate: withBase("/api/generate"),
  edit: withBase("/api/edit-image"),
  sketch: withBase("/api/sketch"),
};

function resourceUrl(path) {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_PATH}${cleanPath}` || cleanPath;
}

const ui = {
  startButton: document.querySelector("#startButton"),
  resetButton: document.querySelector("#resetButton"),
  starterButtons: [...document.querySelectorAll(".starter-button")],
  chatContainer: document.querySelector("#chatContainer"),
  conversationInput: document.querySelector("#conversationInput"),
  generateButton: document.querySelector("#generateButton"),
  editImageButton: document.querySelector("#editImageButton"),
  sketchButton: document.querySelector("#sketchButton"),
  aestheticScore: document.querySelector("#aestheticScore"),
  objectDetails: document.querySelector("#objectDetails"),
  styleValue: document.querySelector("#styleValue"),
  mediumValue: document.querySelector("#mediumValue"),
  paletteValue: document.querySelector("#paletteValue"),
  layoutValue: document.querySelector("#layoutValue"),
  livePrompt: document.querySelector("#livePrompt"),
  finalOutput: document.querySelector("#finalOutput"),
  generatedImage: document.querySelector("#generatedImage"),
};

let selectedStarterValue = null;
let busy = false;

let state = {
  chatStarted: false,
  generatedImage: null,
  finalPrompt: null,
  stage: "Not started",
  conversationHistory: [],
  summary: {},
  aestheticScore: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function boldMarkdownToHtml(value) {
  return escapeHtml(value).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
}

function pending(value) {
  if (value === null || value === undefined) return "Pending...";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "Pending...";
  if (typeof value === "object") return Object.keys(value).length ? JSON.stringify(value) : "Pending...";
  return String(value).trim() || "Pending...";
}

function normalizeMessage(message) {
  const role = String(message.role || message.sender || message.type || "bot").toLowerCase();
  const content = message.content || message.message || message.text || "";
  const isUser = ["user", "human", "artist", "me"].includes(role);

  return {
    role: isUser ? "user" : "bot",
    label: isUser ? "You" : "Canvasia",
    html: boldMarkdownToHtml(content),
  };
}

function buildSummaryFromBackendState(backendState) {
  return {
    objects: backendState.finalized_objects || backendState.objects || backendState.selected_objects || {},
    style: backendState.style,
    medium: backendState.medium,
    colorPalette: backendState.color_palette || backendState.colorPalette,
    layout: backendState.layout,
    livePrompt: backendState.live_prompt || "",
  };
}

function normalizeApiState(payload) {
  const backendState = payload.backendState || payload.state || {};

  return {
    chatStarted: Boolean(payload.chatStarted ?? payload.chat_started),
    generatedImage: payload.generatedImage || payload.generated_image || null,
    finalPrompt: payload.finalPrompt || payload.final_prompt || null,
    starter: backendState.starter || payload.starter || null,
    stage: backendState.stage || payload.stage || state.stage || "Not started",
    conversationHistory: payload.conversationHistory || payload.conversation_history || [],
    summary: payload.summary || buildSummaryFromBackendState(backendState),
    aestheticScore:
      payload.aestheticScore ??
      payload.aesthetic_score ??
      backendState.aesthetic_score ??
      backendState.image_aesthetic_score ??
      null,
  };
}

function contributionRows(objects) {
  if (Array.isArray(objects?.order) && objects.order.length) {
    const counts = { human: 0, canvasia: 0 };
    return objects.order.map((item) => {
      const source = item.source === "human" ? "human" : "canvasia";
      counts[source] += 1;
      return {
        label: source === "human" ? `Your idea ${counts[source]}` : `Canvasia idea ${counts[source]}`,
        value: item.value,
      };
    });
  }

  const human = objects?.human || objects?.human_objects || objects?.user || [];
  const bot = objects?.bot || objects?.bot_objects || objects?.ai || [];

  if (Array.isArray(objects) && objects.length) {
    return objects.map((value, index) => ({
      label: index % 2 === 0 ? `Your idea ${Math.floor(index / 2) + 1}` : `Canvasia idea ${Math.floor(index / 2) + 1}`,
      value,
    }));
  }

  const humanList = Array.isArray(human) ? human : human ? [human] : [];
  const botList = Array.isArray(bot) ? bot : bot ? [bot] : [];
  const rows = [];
  const count = Math.max(humanList.length, botList.length);

  for (let index = 0; index < count; index += 1) {
    if (index < humanList.length) rows.push({ label: `Your idea ${index + 1}`, value: humanList[index] });
    if (index < botList.length) rows.push({ label: `Canvasia idea ${index + 1}`, value: botList[index] });
  }

  return rows;
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";

  if (!contentType.includes("application/json")) {
    const text = await response.text();
    const preview = text.slice(0, 120).replace(/\s+/g, " ");

    throw new Error(
      `Expected JSON but received HTML/text from ${url}. ` +
      `This usually means the request missed the proxy base path. ` +
      `Response preview: ${preview}`
    );
  }

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error || "The backend request failed.");
  }

  return normalizeApiState(payload);
}

function setBusy(nextBusy) {
  busy = nextBusy;
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = nextBusy;
  });
  syncButtons();
}

function syncButtons() {
  ui.starterButtons.forEach((button) => {
    const active = button.dataset.starter === selectedStarterValue;
    button.classList.toggle("active", active);
    button.disabled = busy;
  });

  ui.startButton.disabled = busy || !selectedStarterValue;
  ui.resetButton.disabled = busy;
  ui.generateButton.disabled = busy || !state.chatStarted;
  ui.sketchButton.disabled = busy || !state.chatStarted;
  ui.editImageButton.disabled = busy || !state.generatedImage;
}

function renderMessages() {
  const messages = state.conversationHistory.map(normalizeMessage);

  if (!messages.length) {
    ui.chatContainer.innerHTML = "";
    return;
  }

  ui.chatContainer.innerHTML = messages.slice().reverse().map((message) => `
    <div class="message-row ${message.role}">
      <div class="chat-bubble ${message.role}">
        <span class="chat-author">${message.label}</span>
        ${message.html}
      </div>
    </div>
  `).join("");
}

function renderConversationInput() {
  const disabled = state.chatStarted ? "" : "disabled";
  const canCanvasiaDecide = state.chatStarted && ["Style", "Medium", "Color", "Layout"].includes(state.stage);
  const decideMarkup = canCanvasiaDecide
    ? '<button class="suggestion-button" id="decideButton" type="button">Canvasia decides</button>'
    : "";

  ui.conversationInput.innerHTML = `
    ${decideMarkup}
    <form class="chat-form" id="chatForm">
      <input class="chat-input" id="chatInput" autocomplete="off" placeholder="Enter your chat message..." ${disabled} />
      <button class="send-button" type="submit" aria-label="Send message" ${disabled}>Send</button>
    </form>`;

  document.querySelector("#chatForm").addEventListener("submit", submitTurn);
  const decideButton = document.querySelector("#decideButton");
  if (decideButton) {
    decideButton.addEventListener("click", canvasiaDecides);
  }
  if (state.chatStarted) {
    document.querySelector("#chatInput").focus();
  }
}

function summaryLine(label, value) {
  return `
    <div class="summary-line">
      <div class="summary-key">${escapeHtml(label)}</div>
      <div class="summary-value">${escapeHtml(pending(value))}</div>
    </div>`;
}

function renderSummary() {
  const summary = state.summary || {};
  const rows = contributionRows(summary.objects);

  ui.objectDetails.innerHTML = rows.length
    ? rows.map((row) => summaryLine(row.label, row.value)).join("")
    : summaryLine("Selections", "Pending...");

  ui.styleValue.textContent = pending(summary.style);
  ui.mediumValue.textContent = pending(summary.medium);
  ui.paletteValue.textContent = pending(summary.colorPalette);
  ui.layoutValue.textContent = pending(summary.layout);

  ui.livePrompt.textContent = summary.livePrompt || "Pending...";
}

function renderScore() {
  if (state.aestheticScore === null || state.aestheticScore === undefined || state.aestheticScore === "") {
    ui.aestheticScore.textContent = "Pending";
    return;
  }

  const numeric = Number(state.aestheticScore);
  ui.aestheticScore.textContent = Number.isFinite(numeric) ? `${numeric.toFixed(1)} / 10` : String(state.aestheticScore);
}

function renderFinalOutput() {
  if (!state.generatedImage) {
    ui.finalOutput.hidden = true;
    return;
  }

  ui.finalOutput.hidden = false;
  ui.generatedImage.src = `${resourceUrl(state.generatedImage)}?t=${Date.now()}`;
}

function render() {
  if (!selectedStarterValue && state.starter) {
    selectedStarterValue = state.starter;
  }
  renderMessages();
  renderConversationInput();
  renderSummary();
  renderScore();
  renderFinalOutput();
  syncButtons();
}

async function refreshState() {
  state = await apiRequest(API.state);
  render();
}

async function startConversation(starter = selectedStarterValue) {
  if (!starter) return;
  selectedStarterValue = starter;

  try {
    setBusy(true);
    state = await apiRequest(API.start, {
      method: "POST",
      body: JSON.stringify({ starter }),
    });
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    setBusy(false);
  }
}

function selectStarter(starter) {
  selectedStarterValue = starter;
  syncButtons();
}

async function resetConversation() {
  try {
    setBusy(true);
    selectedStarterValue = null;
    state = await apiRequest(API.reset, {
      method: "POST",
      body: "{}",
    });
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function submitTurn(event) {
  event.preventDefault();
  const input = document.querySelector("#chatInput");
  const message = input.value.trim();
  if (!message) return;

  try {
    setBusy(true);
    state = await apiRequest(API.turn, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function generatePainting() {
  try {
    setBusy(true);
    ui.generateButton.innerHTML = '<span class="spinner"></span>Generating';
    state = await apiRequest(API.generate, {
      method: "POST",
      body: "{}",
    });
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    ui.generateButton.textContent = "Generate Image";
    setBusy(false);
  }
}

async function canvasiaDecides() {
  try {
    setBusy(true);
    state = await apiRequest(API.decide, {
      method: "POST",
      body: "{}",
    });
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function callOptionalAction(url, label) {
  try {
    setBusy(true);
    state = await apiRequest(url, {
      method: "POST",
      body: JSON.stringify({ image: state.generatedImage, prompt: state.finalPrompt }),
    });
    render();
  } catch (error) {
    window.alert(`${label} needs a matching backend endpoint. ${error.message}`);
  } finally {
    setBusy(false);
  }
}

ui.starterButtons.forEach((button) => {
  button.addEventListener("click", () => selectStarter(button.dataset.starter));
});
ui.startButton.addEventListener("click", () => startConversation(selectedStarterValue));
ui.resetButton.addEventListener("click", resetConversation);
ui.generateButton.addEventListener("click", generatePainting);
ui.editImageButton.addEventListener("click", () => callOptionalAction(API.edit, "Edit Generated Image"));
ui.sketchButton.addEventListener("click", () => callOptionalAction(API.sketch, "Sketch"));

refreshState()
  .catch(() => {
    render();
  });
