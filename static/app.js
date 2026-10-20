const $ = (id) => document.getElementById(id);
const API_BASE = String(window.KG_API_BASE || "").replace(/\/$/, "");
const api = (path) => `${API_BASE}${path}`;

const uploadForm = $("upload-form");
const askForm = $("ask-form");
const fileInput = $("pdf-file");
const uploadZone = $("upload-zone");
const uploadButton = $("upload-button");
const askButton = $("ask-button");
const questionInput = $("question-input");
const uploadStatus = $("upload-status");
const askStatus = $("ask-status");
const conversation = $("conversation");
const pipeline = $("pipeline");

const NOT_FOUND = "It is not in the uploaded document. Please check the text.";

let uploadInProgress = false;
let graphReady = false;
let pipelineTimer = null;

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function extractErrorMessage(payload, fallbackMessage) {
  if (!payload || typeof payload !== "object") return fallbackMessage;
  if (typeof payload.error === "string" && payload.error.trim()) return payload.error;
  if (payload.detail && typeof payload.detail === "object" && payload.detail.error) return payload.detail.error;
  if (typeof payload.detail === "string" && payload.detail.trim()) return payload.detail;
  return fallbackMessage;
}

function setStatus(element, text, kind = "muted") {
  element.textContent = text;
  element.classList.remove("muted", "error");
  if (kind) element.classList.add(kind);
}

function relationText(relation) {
  return String(relation || "").replace(/_/g, " ").toLowerCase();
}

/* ------------------------------------------------------------------ health */

async function loadHealth() {
  try {
    const response = await fetch(api("/health"));
    const data = await response.json();
    $("service-status").classList.add("hidden");
    if (data.active_document && data.active_document.document_id) {
      graphReady = true;
      setStatus(uploadStatus, `Active document: ${data.active_document.file_name}. Upload another PDF to replace it.`);
      setStatus(askStatus, "Ready. Ask about the active document.");
      askButton.disabled = false;
      loadGraph();
    }
  } catch (error) {
    $("service-status").classList.remove("hidden");
  }
}

/* ------------------------------------------------------------------ upload */

function setPipeline(activeIndex) {
  const steps = [...pipeline.querySelectorAll("li")];
  steps.forEach((step, index) => {
    step.classList.toggle("done", index < activeIndex);
    step.classList.toggle("active", index === activeIndex);
  });
}

function startPipeline() {
  pipeline.classList.remove("hidden");
  const startedAt = Date.now();
  setPipeline(0);
  clearInterval(pipelineTimer);
  pipelineTimer = setInterval(() => {
    const seconds = (Date.now() - startedAt) / 1000;
    // The upload is one request, so the steps are paced by elapsed time.
    setPipeline(seconds < 1.5 ? 0 : seconds < 3 ? 1 : 2);
    setStatus(uploadStatus, `Building the graph… ${Math.floor(seconds)}s`);
  }, 400);
}

function stopPipeline(success) {
  clearInterval(pipelineTimer);
  if (success) setPipeline(4);
  else pipeline.classList.add("hidden");
}

function renderSampleTriples(samples) {
  const box = $("sample-triples");
  box.innerHTML = (samples || []).slice(0, 6).map((triple) => {
    const [subject, relation, object] = Array.isArray(triple)
      ? triple
      : [triple.subject, triple.relation, triple.object];
    return `<div class="triple"><span class="ent">${escapeHtml(subject)}</span>` +
      `<span class="rel">${escapeHtml(relation)}</span><span class="ent">${escapeHtml(object)}</span></div>`;
  }).join("");
}

function chooseFile(file) {
  $("upload-title").textContent = file ? file.name : "Drop a PDF or click to choose";
  $("upload-subtitle").textContent = file
    ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to build`
    : "Text is chunked, entities and relations are extracted with spaCy, and everything is stored as a graph.";
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((name) => uploadZone.addEventListener(name, (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => uploadZone.addEventListener(name, (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragging");
}));
uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file && file.name.toLowerCase().endsWith(".pdf")) {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    chooseFile(file);
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    setStatus(uploadStatus, "Choose a PDF first.", "error");
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);
  $("upload-json-input").textContent = prettyJson({ file_name: file.name, size_bytes: file.size });
  $("upload-json-content").textContent = prettyJson({ status: "processing" });

  uploadInProgress = true;
  uploadButton.disabled = true;
  askButton.disabled = true;
  $("upload-summary").classList.add("hidden");
  startPipeline();

  try {
    const response = await fetch(api("/upload_pdf"), { method: "POST", body: formData });
    const data = await response.json();
    $("upload-json-content").textContent = prettyJson(data);

    if (!response.ok || data.error || (data.detail && data.detail.error)) {
      stopPipeline(false);
      setStatus(uploadStatus, extractErrorMessage(data, "The PDF could not be processed."), "error");
      return;
    }

    stopPipeline(true);
    graphReady = true;
    const debug = data.debug || {};
    setStatus(uploadStatus, `Graph ready · ${data.file_name || file.name}`);
    $("upload-message").textContent = data.summary || data.message || "";
    $("upload-chunks").textContent = data.chunks_processed ?? 0;
    $("upload-triples").textContent = data.triples_added ?? 0;
    $("upload-mentions").textContent = debug.entity_mentions ?? "—";
    renderSampleTriples(data.sample_triples);
    $("upload-summary").classList.remove("hidden");
    setStatus(askStatus, "Ready. Ask about the document.");
    conversation.querySelectorAll(".turn").forEach((turn) => turn.remove());
    $("empty-state").classList.remove("hidden");
    loadGraph();
  } catch (error) {
    stopPipeline(false);
    setStatus(uploadStatus, "The upload request failed. Is the server running?", "error");
    $("upload-json-content").textContent = prettyJson({ error: String(error) });
  } finally {
    uploadInProgress = false;
    uploadButton.disabled = false;
    askButton.disabled = !graphReady;
  }
});

/* --------------------------------------------------------------------- ask */

function modeBadge(mode) {
  if (mode && mode.startsWith("llm:")) return `<span class="badge llm">written by ${escapeHtml(mode.slice(4))}</span>`;
  if (mode === "graph-chain") return `<span class="badge chain">multi-hop chain</span>`;
  if (mode === "passage") return `<span class="badge">from the document</span>`;
  return `<span class="badge">from the graph</span>`;
}

function renderEvidence(evidence) {
  if (!evidence) return "";
  const edges = evidence.edges || [];
  const passages = evidence.passages || [];
  if (!edges.length && !passages.length) return "";
  const maxHop = edges.reduce((max, edge) => Math.max(max, edge.distance || 0), 0);
  const edgeHtml = edges.slice(0, 8).map((edge) =>
    `<div class="triple"><span class="hop-tag ${edge.distance >= 2 ? "h2" : ""}">hop ${edge.distance}</span>` +
    `<span class="ent">${escapeHtml(edge.from)}</span><span class="rel">${escapeHtml(relationText(edge.relation))}</span>` +
    `<span class="ent">${escapeHtml(edge.to)}</span></div>`).join("");
  const passageHtml = passages.slice(0, 3).map((passage) =>
    `<div class="passage">${escapeHtml(String(passage.text || "").slice(0, 320))}` +
    `${passage.page ? ` <span class="page">· p.${escapeHtml(passage.page)}</span>` : ""}</div>`).join("");
  const anchors = (evidence.anchors || []).join(", ");
  return `<details class="evidence"><summary>How this was answered${maxHop >= 2 ? ` · ${maxHop} hops` : ""}</summary>
    <div class="evidence-body">
      ${anchors ? `<div><h4>Started from</h4>${escapeHtml(anchors)}</div>` : ""}
      ${edgeHtml ? `<div><h4>Graph facts</h4>${edgeHtml}</div>` : ""}
      ${passageHtml ? `<div><h4>Source passages</h4>${passageHtml}</div>` : ""}
    </div></details>`;
}

async function ask(question) {
  if (!question) return;
  if (uploadInProgress) {
    setStatus(askStatus, "The PDF is still processing.", "error");
    return;
  }
  if (!graphReady) {
    setStatus(askStatus, "Upload a PDF first.", "error");
    return;
  }
  $("empty-state").classList.add("hidden");
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `<div class="bubble-q">${escapeHtml(question)}</div>` +
    `<div class="bubble-a pending"><p>Searching the graph…</p></div>`;
  conversation.appendChild(turn);
  conversation.scrollTop = conversation.scrollHeight;
  askButton.disabled = true;
  setStatus(askStatus, "Searching the knowledge graph…");

  const requestBody = { question };
  $("ask-json-input").textContent = prettyJson(requestBody);
  const startedAt = performance.now();
  try {
    const response = await fetch(api("/ask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    const data = await response.json();
    $("ask-json-content").textContent = prettyJson(data);
    const answerBox = turn.querySelector(".bubble-a");
    answerBox.classList.remove("pending");
    if (!response.ok) {
      answerBox.innerHTML = `<p>${escapeHtml(extractErrorMessage(data, "The question could not be processed."))}</p>`;
      setStatus(askStatus, "Something went wrong.", "error");
      return;
    }
    const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    answerBox.innerHTML = `<p>${escapeHtml(data.answer || NOT_FOUND)}</p>` +
      `<div class="answer-meta">${modeBadge(data.answer_mode)}<span class="badge">${seconds}s</span></div>` +
      renderEvidence(data.graph_evidence);
    setStatus(askStatus, "Answer ready.");
    highlightEvidence(data.graph_evidence);
  } catch (error) {
    turn.querySelector(".bubble-a").innerHTML = `<p>The request failed. Is the server running?</p>`;
    setStatus(askStatus, "Request failed.", "error");
  } finally {
    askButton.disabled = !graphReady;
    conversation.scrollTop = conversation.scrollHeight;
  }
}

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  questionInput.value = "";
  ask(question);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askForm.requestSubmit();
  }
});

$("suggestions").addEventListener("click", (event) => {
  if (event.target.matches(".chip")) {
    questionInput.value = event.target.textContent;
    questionInput.focus();
  }
});

$("upload-json-toggle").addEventListener("click", () => $("upload-json-panel").classList.toggle("hidden"));
$("ask-json-toggle").addEventListener("click", () => $("ask-json-panel").classList.toggle("hidden"));

/* ------------------------------------------------------------------- graph */

const svg = $("graph-canvas");
const SVG_NS = "http://www.w3.org/2000/svg";
const graphState = { nodes: [], edges: [], byId: new Map(), view: { x: 0, y: 0, k: 1 }, frame: null };

function svgEl(name, attrs = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

async function loadGraph() {
  try {
    const response = await fetch(api("/graph?limit=180"));
    const data = await response.json();
    buildGraph(data);
    updateSuggestions(data.nodes || []);
  } catch (error) {
    $("graph-empty").textContent = "Could not load the graph.";
  }
}

function updateSuggestions(nodes) {
  // Suggest questions about the document's best-connected, readable entities.
  const readable = nodes
    .filter((node) => /^[A-Za-z][\w .'-]{2,30}$/.test(node.label) && !/^\$/.test(node.label))
    .sort((a, b) => (b.degree || 0) - (a.degree || 0))
    .slice(0, 2)
    .map((node) => node.label);
  const questions = ["What is this PDF about?"];
  if (readable[0]) questions.push(`What is ${readable[0]}?`);
  if (readable[1]) questions.push(`How is ${readable[0]} related to ${readable[1]}?`);
  $("suggestions").innerHTML = questions
    .map((question) => `<button type="button" class="chip">${escapeHtml(question)}</button>`)
    .join("");
}

function buildGraph(data) {
  cancelAnimationFrame(graphState.frame);
  svg.innerHTML = "";
  // Seed on a sunflower spiral so no two nodes start on top of each other.
  const nodes = (data.nodes || []).map((node, index) => {
    const radius = 14 * Math.sqrt(index + 1);
    const angle = index * 2.39996;
    return { ...node, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, vx: 0, vy: 0 };
  });
  graphState.byId = new Map(nodes.map((node) => [node.id, node]));
  const edges = (data.edges || []).filter((edge) => graphState.byId.has(edge.source) && graphState.byId.has(edge.target));
  graphState.nodes = nodes;
  graphState.edges = edges;
  $("graph-empty").classList.toggle("hidden", nodes.length > 0);
  $("graph-meta").textContent = nodes.length
    ? `showing ${nodes.length} of ${data.total_entities ?? nodes.length} entities · ${data.total_relations ?? edges.length} relations`
    : "";
  if (!nodes.length) return;

  const root = svgEl("g", { id: "graph-root" });
  const edgeLayer = svgEl("g");
  const nodeLayer = svgEl("g");
  root.append(edgeLayer, nodeLayer);
  svg.appendChild(root);

  edges.forEach((edge) => {
    edge.el = svgEl("line");
    edgeLayer.appendChild(edge.el);
  });
  nodes.forEach((node) => {
    const radius = Math.min(14, 4 + Math.sqrt(node.degree || 1) * 2);
    node.r = radius;
    const degree = node.degree || 1;
    const tier = degree >= 12 ? 4 : degree >= 6 ? 3 : degree >= 3 ? 2 : 1;
    node.el = svgEl("circle", { r: radius, class: `tier-${tier}` });
    node.textEl = svgEl("text", { dy: -radius - 3, "text-anchor": "middle", class: tier >= 3 ? "hub" : "" });
    node.textEl.textContent = String(node.label).slice(0, 28);
    node.textEl.style.display = (node.degree || 0) >= 3 ? "" : "none";
    node.el.addEventListener("mouseenter", (event) => showTooltip(node, event));
    node.el.addEventListener("mouseleave", hideTooltip);
    node.el.addEventListener("click", () => {
      questionInput.value = `What is ${node.label}?`;
      questionInput.focus();
    });
    nodeLayer.append(node.el, node.textEl);
  });
  resetView();
  simulate();
}

function simulate() {
  const { nodes, edges } = graphState;
  let alpha = 1;
  const tick = () => {
    alpha *= 0.985;
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        const distance2 = Math.max(dx * dx + dy * dy, 25);
        if (distance2 > 90000) continue;
        const force = (600 / distance2) * alpha;
        dx *= force;
        dy *= force;
        a.vx += dx; a.vy += dy;
        b.vx -= dx; b.vy -= dy;
      }
    }
    edges.forEach((edge) => {
      const a = graphState.byId.get(edge.source);
      const b = graphState.byId.get(edge.target);
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = ((distance - 70) / distance) * 0.04 * alpha;
      a.vx += dx * force; a.vy += dy * force;
      b.vx -= dx * force; b.vy -= dy * force;
    });
    nodes.forEach((node) => {
      node.vx -= node.x * 0.006 * alpha;
      node.vy -= node.y * 0.006 * alpha;
      const speed = Math.hypot(node.vx, node.vy);
      if (speed > 12) {
        node.vx *= 12 / speed;
        node.vy *= 12 / speed;
      }
      node.x += node.vx;
      node.y += node.vy;
      node.vx *= 0.6;
      node.vy *= 0.6;
    });
    draw();
    if (alpha > 0.02) graphState.frame = requestAnimationFrame(tick);
    else fitView();
  };
  graphState.frame = requestAnimationFrame(tick);
}

function draw() {
  graphState.edges.forEach((edge) => {
    const a = graphState.byId.get(edge.source);
    const b = graphState.byId.get(edge.target);
    edge.el.setAttribute("x1", a.x); edge.el.setAttribute("y1", a.y);
    edge.el.setAttribute("x2", b.x); edge.el.setAttribute("y2", b.y);
  });
  graphState.nodes.forEach((node) => {
    node.el.setAttribute("cx", node.x); node.el.setAttribute("cy", node.y);
    node.textEl.setAttribute("x", node.x); node.textEl.setAttribute("y", node.y);
  });
}

function applyView() {
  const root = $("graph-root");
  const { x, y, k } = graphState.view;
  if (root) root.setAttribute("transform", `translate(${x},${y}) scale(${k})`);
}

function fitView() {
  const { nodes } = graphState;
  if (!nodes.length) return;
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const box = svg.getBoundingClientRect();
  const k = Math.max(0.15, Math.min(2, Math.min(box.width / (maxX - minX + 80), box.height / (maxY - minY + 80))));
  graphState.view = {
    k,
    x: box.width / 2 - ((minX + maxX) / 2) * k,
    y: box.height / 2 - ((minY + maxY) / 2) * k,
  };
  applyView();
}

function resetView() {
  const box = svg.getBoundingClientRect();
  graphState.view = { x: box.width / 2, y: box.height / 2, k: 1 };
  applyView();
}

function canonical(name) {
  return String(name || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function highlightEvidence(evidence) {
  if (!graphState.nodes.length) return;
  const distances = new Map();
  (evidence?.nodes || []).forEach((node) => distances.set(canonical(node.name), node.distance));
  const pathPairs = new Set((evidence?.edges || []).map((edge) => `${canonical(edge.from)}|${canonical(edge.to)}`));
  const any = distances.size > 0;
  graphState.nodes.forEach((node) => {
    const distance = distances.get(node.id);
    node.el.classList.remove("anchor", "hop1", "hop2", "dim");
    node.textEl.classList.remove("dim");
    if (distance === 0) node.el.classList.add("anchor");
    else if (distance === 1) node.el.classList.add("hop1");
    else if (distance >= 2) node.el.classList.add("hop2");
    else if (any) { node.el.classList.add("dim"); node.textEl.classList.add("dim"); }
    node.textEl.style.display = distance !== undefined || (!any && node.degree >= 3) ? "" : "none";
  });
  graphState.edges.forEach((edge) => {
    const onPath = pathPairs.has(`${edge.source}|${edge.target}`) || pathPairs.has(`${edge.target}|${edge.source}`);
    edge.el.classList.toggle("hl", onPath);
    edge.el.classList.toggle("dim", any && !onPath);
  });
}

function showTooltip(node, event) {
  const tooltip = $("graph-tooltip");
  const relations = graphState.edges
    .filter((edge) => edge.source === node.id || edge.target === node.id)
    .slice(0, 5)
    .map((edge) => {
      const other = graphState.byId.get(edge.source === node.id ? edge.target : edge.source);
      return edge.source === node.id
        ? `${relationText(edge.relation)} → ${escapeHtml(other.label)}`
        : `← ${relationText(edge.relation)} ${escapeHtml(other.label)}`;
    });
  tooltip.innerHTML = `<strong>${escapeHtml(node.label)}</strong><br>${relations.join("<br>")}`;
  const box = svg.getBoundingClientRect();
  tooltip.style.left = `${Math.min(event.clientX - box.left + 12, box.width - 290)}px`;
  tooltip.style.top = `${event.clientY - box.top + 12}px`;
  tooltip.classList.remove("hidden");
}

function hideTooltip() {
  $("graph-tooltip").classList.add("hidden");
}

let pan = null;
svg.addEventListener("pointerdown", (event) => {
  if (event.target.tagName === "circle") return;
  pan = { x: event.clientX, y: event.clientY, view: { ...graphState.view } };
  svg.classList.add("panning");
  svg.setPointerCapture(event.pointerId);
});
svg.addEventListener("pointermove", (event) => {
  if (!pan) return;
  graphState.view.x = pan.view.x + event.clientX - pan.x;
  graphState.view.y = pan.view.y + event.clientY - pan.y;
  applyView();
});
svg.addEventListener("pointerup", () => {
  pan = null;
  svg.classList.remove("panning");
});
svg.addEventListener("wheel", (event) => {
  event.preventDefault();
  const box = svg.getBoundingClientRect();
  const mx = event.clientX - box.left;
  const my = event.clientY - box.top;
  const { x, y, k } = graphState.view;
  const next = Math.max(0.15, Math.min(3, k * (event.deltaY < 0 ? 1.12 : 0.89)));
  graphState.view = { k: next, x: mx - ((mx - x) / k) * next, y: my - ((my - y) / k) * next };
  applyView();
}, { passive: false });

$("graph-filter").addEventListener("input", (event) => {
  const query = event.target.value.trim().toLowerCase();
  graphState.nodes.forEach((node) => {
    const match = query && String(node.label).toLowerCase().includes(query);
    node.el.classList.toggle("dim", Boolean(query) && !match);
    node.textEl.classList.toggle("dim", Boolean(query) && !match);
    node.textEl.style.display = match || (!query && node.degree >= 3) ? "" : "none";
  });
});

$("graph-refresh").addEventListener("click", loadGraph);

askButton.disabled = true;
loadHealth();
