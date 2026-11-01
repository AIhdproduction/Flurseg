import { apiBytes, apiJson, apiRequest, saveMask } from "./api.js";
import { bindHelp } from "./help.js";
import { applyHistory, createHistory, pushHistory } from "./history.js";
import { createLayers, layerDefinitions } from "./layers.js";
import { renderQueue, populateFilter } from "./queue.js";
import { createRenderer } from "./render.js";
import { createKeymap, handleShortcut } from "./shortcuts.js";
import {
  acceptProposalPixels,
  countProposalPixels,
  createState,
  markDirty as markDirtyState,
  markDirtyChunk,
  parseProposalWindow,
  PROPOSAL_THRESHOLD,
  proposalAcceptedStatus,
  proposalDiscardedStatus,
} from "./state.js";
import {
  acceptRegion,
  buildSuperpixelIndex,
  fillPolygon,
  floodFill,
  interpolateStroke,
  paintBrush,
  paintSuperpixel,
} from "./tools.js";
import { fitImage, imagePoint, zoomAt } from "./viewport.js";

const state = createState();
const canvas = document.querySelector("#editor");
const statusNode = document.querySelector("#status");
const toastNode = document.querySelector("#toast");
const colors = {};
let renderer;
let helpDialog;

function status(message) {
  statusNode.textContent = message;
}

function toast(message) {
  toastNode.textContent = message;
  toastNode.classList.add("visible");
  window.setTimeout(() => toastNode.classList.remove("visible"), 2500);
}

function requestRender() {
  if (renderer) renderer.request();
}

function markDirty(target) {
  markDirtyState(state, target, renderer);
}

function finishEdit(changes) {
  if (!pushHistory(state.history, changes)) return;
  markDirty(changes);
  state.unsavedChanges = true;
  scheduleSave();
}

function scheduleSave() {
  window.clearTimeout(state.draftTimer);
  state.draftTimer = window.setTimeout(() => {
    state.draftTimer = null;
    saveDraft();
  }, 2000);
  status("Unsaved changes");
}

async function saveDraft(keepalive = false) {
  if (!state.unsavedChanges || !state.mask || !state.meta) return;
  window.clearTimeout(state.draftTimer);
  state.draftTimer = null;
  try {
    await saveMask(state.meta.id, state.mask, keepalive);
    state.unsavedChanges = false;
    status("Saved");
  } catch (error) {
    toast(error.message);
    status("Unsaved changes");
  }
}

function selectedPoint(event) {
  const point = imagePoint(event, canvas, state);
  return {
    x: Math.max(0, Math.min(state.meta.width - 1, point.x)),
    y: Math.max(0, Math.min(state.meta.height - 1, point.y)),
  };
}

function selectClass(classId) {
  state.activeClass = classId;
  document.querySelectorAll(".class-button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.id) === classId);
  });
  const selected = state.config.classes.find((definition) => definition.id === classId);
  status(`Active class: ${selected.name}`);
}

function selectTool(tool) {
  state.tool = tool;
  if (tool === "eraser") selectClass(255);
  document.querySelectorAll("#tools button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === tool);
  });
  status(`Tool: ${tool}`);
}

function paintStroke(from, to, edit) {
  const radius = Math.max(1, state.brush / state.scale);
  for (const point of interpolateStroke(from, to, Math.max(1, radius / 2))) {
    paintBrush(
      state.mask,
      state.meta.width,
      state.meta.height,
      point,
      radius,
      state.activeClass,
      edit.changes,
      edit.changed,
    );
  }
}

function paintSelectedSuperpixel(point, edit) {
  if (!state.superpixels || !state.superpixelIndex) {
    paintStroke(point, point, edit);
    return;
  }
  const pixel = point.y * state.meta.width + point.x;
  paintSuperpixel(
    state.mask,
    state.superpixelIndex,
    state.superpixels[pixel],
    state.activeClass,
    edit.changes,
    edit.changed,
  );
}

function updateGranularityUI(enabled, activeLevel) {
  const container = document.querySelector("#superpixel-control");
  if (!container) return;
  const fineBtn = document.querySelector("#sp-fine");
  const coarseBtn = document.querySelector("#sp-coarse");
  if (fineBtn) {
    fineBtn.disabled = !enabled;
    fineBtn.classList.toggle("active", activeLevel === "fine");
    fineBtn.setAttribute("aria-pressed", String(activeLevel === "fine"));
  }
  if (coarseBtn) {
    coarseBtn.disabled = !enabled;
    coarseBtn.classList.toggle("active", activeLevel === "coarse");
    coarseBtn.setAttribute("aria-pressed", String(activeLevel === "coarse"));
  }
  container.setAttribute("aria-disabled", String(!enabled));
}

function buildIndexForLevel(level, labels, imageId) {
  const buckets = new Map();
  let offset = 0;
  const schedule = window.requestIdleCallback || ((cb) => window.setTimeout(() => {
    cb({ timeRemaining: () => 15, didTimeout: true });
  }, 1));

  const step = (deadline) => {
    if (state.meta?.id !== imageId) return;
    while (offset < labels.length && (deadline.timeRemaining() > 2 || deadline.didTimeout)) {
      const end = Math.min(labels.length, offset + 100000);
      for (; offset < end; offset += 1) {
        const label = labels[offset];
        let bucket = buckets.get(label);
        if (!bucket) {
          bucket = [];
          buckets.set(label, bucket);
        }
        bucket.push(offset);
      }
    }
    if (offset < labels.length) {
      schedule(step, { timeout: 50 });
      return;
    }
    const index = new Map(
      [...buckets].map(([label, pixels]) => [label, Int32Array.from(pixels)]),
    );
    state.superpixelIndices[level] = index;
    if (state.superpixelLevel === level && state.meta?.id === imageId) {
      state.superpixelIndex = index;
      status(`Superpixel index (${level}) ready`);
      requestRender();
    }
  };
  schedule(step, { timeout: 50 });
}

async function setSuperpixelGranularity(level) {
  if (!state.meta) return;
  if (state.superpixelLevel === level && state.superpixels) return;
  const coarseBtn = document.querySelector("#sp-coarse");
  if (level === "coarse" && coarseBtn?.disabled) return;

  state.superpixelLevel = level;
  const toggleEnabled = !coarseBtn?.disabled;
  updateGranularityUI(toggleEnabled, level);
  status(`Superpixels: ${level}`);

  const imageId = state.meta.id;
  if (state.superpixelBuffers[level]) {
    state.superpixels = state.superpixelBuffers[level];
    if (state.superpixelIndices[level]) {
      state.superpixelIndex = state.superpixelIndices[level];
      requestRender();
    } else {
      buildIndexForLevel(level, state.superpixels, imageId);
    }
    return;
  }

  try {
    const bytes = await apiBytes(`/images/${imageId}/superpixels?level=${level}`);
    if (state.meta?.id !== imageId) return;
    state.superpixelBuffers[level] = new Uint32Array(bytes.buffer);
    state.superpixels = state.superpixelBuffers[level];
    buildIndexForLevel(level, state.superpixels, imageId);
    requestRender();
  } catch {
    status(`Failed to load ${level} superpixels`);
  }
}

function toggleSuperpixelLevel() {
  const coarseBtn = document.querySelector("#sp-coarse");
  if (coarseBtn?.disabled) return;
  const target = state.superpixelLevel === "coarse" ? "fine" : "coarse";
  setSuperpixelGranularity(target);
}

async function loadImage(index) {
  await saveDraft();
  const item = state.items[index];
  if (!item) return;
  state.index = index;
  state.meta = await apiJson(`/images/${item.id}`);
  state.mask = await apiBytes(`/images/${item.id}/mask`);
  state.proposal = await apiBytes(`/images/${item.id}/proposal-classes`);
  state.superpixelBuffers = { coarse: null, fine: null };
  state.superpixelIndices = { coarse: null, fine: null };
  state.superpixels = null;
  state.superpixelIndex = null;
  state.history = createHistory();
  state.boxes = [];
  state.polygon = [];
  state.proposalMap = null;
  state.proposalWindow = null;
  window.clearTimeout(state.draftTimer);
  state.draftTimer = null;
  state.maskChunks.clear();
  state.proposalChunks.clear();
  state.renderDirtyChunks.clear();
  state.unsavedChanges = false;
  state.viewportAdjusted = false;
  state.started = performance.now();
  fitImage(canvas, state);
  requestRender();
  state.superpixelLevel = "coarse";
  try {
    const bytes = await apiBytes(`/images/${item.id}/superpixels?level=coarse`);
    state.superpixelBuffers.coarse = new Uint32Array(bytes.buffer);
    state.superpixels = state.superpixelBuffers.coarse;
    updateGranularityUI(true, "coarse");
    buildIndexForLevel("coarse", state.superpixels, item.id);
  } catch {
    state.superpixelLevel = "fine";
    updateGranularityUI(false, "fine");
    try {
      const bytes = await apiBytes(`/images/${item.id}/superpixels?level=fine`);
      state.superpixelBuffers.fine = new Uint32Array(bytes.buffer);
      state.superpixels = state.superpixelBuffers.fine;
      buildIndexForLevel("fine", state.superpixels, item.id);
    } catch {
      status("Superpixels unavailable; brush fallback enabled");
      requestRender();
    }
  }
  renderQueue(document.querySelector("#queue-list"), state.items, state.index, loadImage);
  status(`Loaded ${item.id} · Saved`);
  requestRender();
}

async function loadQueue(pool = state.currentPool) {
  state.currentPool = pool;
  const search = new URLSearchParams();
  if (!pool) {
    for (const name of ["source", "session", "reason"]) {
      const value = document.querySelector(`#${name}-filter`).value;
      if (value) search.set(name, value);
    }
  }
  state.items = await apiJson(pool ? "/pool" : `/queue?${search}`);
  populateFilter(document.querySelector("#source-filter"), state.items.map((item) => item.source));
  populateFilter(document.querySelector("#session-filter"), state.items.map((item) => item.session));
  state.index = -1;
  renderQueue(document.querySelector("#queue-list"), state.items, -1, loadImage);
}

async function refresh() {
  const stats = await apiJson("/stats");
  document.querySelector("#progress").textContent = `${stats.done} done / ${stats.open} open`;
  document.querySelector("#error-rate").textContent =
    `Auto-accept error: ${(stats.auto_accept_error_rate.total * 100).toFixed(1)}%`;
  await loadQueue();
}

async function action(name) {
  if (!state.meta) return;
  if (name === "reject" && !window.confirm("Reject this image?")) return;
  await saveDraft();
  const seconds = state.visible ? (performance.now() - state.started) / 1000 : 0;
  try {
    const result = await apiJson(`/images/${state.meta.id}/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_seconds: seconds }),
    });
    toast(`${name}: ${result.status}`);
    await refresh();
    if (name !== "skip" && state.items.length) await loadImage(Math.min(state.index, state.items.length - 1));
  } catch (error) {
    toast(error.message);
  }
}

async function findSimilar() {
  if (!state.boxes.length) {
    toast("Draw one or more boxes first.");
    return;
  }
  if (!state.unsavedChanges) {
    window.clearTimeout(state.draftTimer);
    state.draftTimer = null;
  }
  const button = document.querySelector('[data-tool="similar"]');
  button.disabled = true;
  status("Finding similar regions");
  try {
    const response = await apiRequest(`/images/${state.meta.id}/find-similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_id: state.activeClass, boxes: state.boxes }),
    });
    const proposalMap = new Uint8Array(await response.arrayBuffer());
    const expectedLength = state.meta.width * state.meta.height;
    if (proposalMap.length !== expectedLength) {
      throw new Error(`Find similar returned ${proposalMap.length} bytes, expected ${expectedLength}`);
    }
    state.proposalMap = proposalMap;
    state.proposalWindow = parseProposalWindow(response.headers.get("X-Window"), state.meta.width, state.meta.height);
    state.proposalChunks.clear();
    state.boxes = [];
    state.layers.proposal = true;
    const found = countProposalPixels(state.proposalMap, state.proposalWindow, state.meta.width);
    status(`Proposal: ${found} px found - Y accept, N discard`);
    requestRender();
  } catch (error) {
    const message = error.message.includes("unavailable")
      ? "Find similar needs CUDA and models/sam3.pt"
      : error.message;
    state.proposalMap = null;
    state.proposalWindow = null;
    state.proposalChunks.clear();
    status(`Find similar failed: ${message}`);
    toast(message);
    requestRender();
  } finally {
    button.disabled = false;
  }
}

function acceptProposal() {
  if (!state.proposalMap) return;
  const changes = acceptProposalPixels(state.mask, state.proposalMap, state.activeClass);
  state.proposalMap = null;
  state.proposalWindow = null;
  state.proposalChunks.clear();
  finishEdit(changes);
  status(proposalAcceptedStatus(changes.length));
  requestRender();
}

function fitCurrentImage() {
  if (!state.meta) return;
  fitImage(canvas, state);
  state.viewportAdjusted = false;
  requestRender();
}

function bindCanvas() {
  let edit = null;
  let lastPoint = null;
  let pan = false;
  let boxStart = null;
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", (event) => {
    if (!state.meta) return;
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch {
      // Ignore in headless or unsupported environments
    }
    const point = selectedPoint(event);
    lastPoint = { ...point, clientX: event.clientX, clientY: event.clientY };
    if (event.button === 1 || (event.button === 0 && (state.space || state.tool === "pan"))) {
      pan = true;
      state.viewportAdjusted = true;
      return;
    }
    if (state.tool === "polygon") {
      state.polygon.push(point);
      return;
    }
    if (state.tool === "similar") {
      boxStart = point;
      return;
    }
    if (state.tool === "accept-region") {
      finishEdit(acceptRegion(state.mask, state.proposal, state.meta.width, state.meta.height,
        point.y * state.meta.width + point.x));
      return;
    }
    if (state.tool === "flood") {
      finishEdit(floodFill(state.mask, state.meta.width, state.meta.height,
        point.y * state.meta.width + point.x, state.activeClass));
      return;
    }
    edit = { changed: new Set(), changes: [] };
    if (state.tool === "superpixel") paintSelectedSuperpixel(point, edit);
    else paintStroke(point, point, edit);
    if (edit.changes.length > 0) markDirty(edit.changes);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.meta || !lastPoint) return;
    const point = selectedPoint(event);
    if (pan) {
      state.viewportAdjusted = true;
      state.pan.x += event.clientX - lastPoint.clientX;
      state.pan.y += event.clientY - lastPoint.clientY;
      requestRender();
    } else if (edit && event.buttons === 1) {
      const prevCount = edit.changes.length;
      if (state.tool === "superpixel") paintSelectedSuperpixel(point, edit);
      else paintStroke(lastPoint, point, edit);
      if (edit.changes.length > prevCount) {
        markDirty(edit.changes.slice(prevCount));
      }
    }
    lastPoint = { ...point, clientX: event.clientX, clientY: event.clientY };
  });
  const endStroke = (event) => {
    if (!lastPoint && !edit && !boxStart && !pan) return;
    if (boxStart && state.meta) {
      const point = selectedPoint(event);
      state.boxes.push([boxStart.x, boxStart.y, point.x, point.y]);
      boxStart = null;
      requestRender();
      toast("Box added. Press Enter to search.");
    }
    if (edit) {
      finishEdit(edit.changes);
      edit = null;
    }
    pan = false;
    lastPoint = null;
    try {
      if (event?.pointerId && canvas.hasPointerCapture?.(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    } catch {
      // Ignore
    }
  };
  canvas.addEventListener("pointerup", endStroke);
  window.addEventListener("pointerup", endStroke);
  window.addEventListener("pointercancel", endStroke);
  canvas.addEventListener("wheel", (event) => {
    if (!state.meta) return;
    event.preventDefault();
    zoomAt(event, canvas, state);
    state.viewportAdjusted = true;
    requestRender();
  }, { passive: false });
  canvas.addEventListener("dblclick", () => finishPolygon());
}

function finishPolygon() {
  if (state.polygon.length < 3) return;
  finishEdit(fillPolygon(
    state.mask,
    state.meta.width,
    state.meta.height,
    state.polygon,
    state.activeClass,
  ));
  state.polygon = [];
  requestRender();
}

function bindControls() {
  document.querySelector("#brush-size").addEventListener("input", (event) => {
    state.brush = Number(event.target.value);
    document.querySelector("#brush-output").textContent = `${state.brush} px`;
  });
  document.querySelector("#mask-opacity").addEventListener("input", (event) => {
    state.opacity = Number(event.target.value) / 100;
    requestRender();
  });
  document.querySelector("#tools").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tool]");
    if (button) selectTool(button.dataset.tool);
  });
  for (const name of ["accept", "reject", "skip"]) {
    document.querySelector(`#${name}`).addEventListener("click", () => globals.action(name));
  }
  document.querySelector("#next").addEventListener("click", () => globals.next());
  document.querySelector("#previous").addEventListener("click", () => globals.previous());
  for (const name of ["source", "session", "reason"]) {
    document.querySelector(`#${name}-filter`).addEventListener("change", () => loadQueue(false));
  }
  for (const [tab, pool] of [["queue-tab", false], ["pool-tab", true]]) {
    document.querySelector(`#${tab}`).addEventListener("click", () => {
      document.querySelector("#queue-tab").setAttribute("aria-selected", String(!pool));
      document.querySelector("#pool-tab").setAttribute("aria-selected", String(pool));
      loadQueue(pool);
    });
  }
  document.querySelector("#sp-fine")
    ?.addEventListener("click", () => setSuperpixelGranularity("fine"));
  document.querySelector("#sp-coarse")
    ?.addEventListener("click", () => setSuperpixelGranularity("coarse"));
}

const globals = {
  acceptProposal,
  action,
  cycleOpacity: () => {
    state.opacity = state.opacity >= 0.8 ? 0.25 : state.opacity + 0.25;
    document.querySelector("#mask-opacity").value = String(state.opacity * 100);
    requestRender();
  },
  decreaseBrush: () => {
    state.brush = Math.max(1, state.brush - 4);
  },
  discardProposal: () => {
    state.proposalMap = null;
    state.proposalWindow = null;
    state.proposalChunks.clear();
    status(proposalDiscardedStatus());
    requestRender();
  },
  enter: () => {
    if (state.tool === "polygon") finishPolygon();
    else if (state.tool === "similar") findSimilar();
    else action("accept");
  },
  escape: () => {
    if (state.polygon.length || state.boxes.length) {
      state.polygon = [];
      state.boxes = [];
      requestRender();
    } else {
      helpDialog.close();
    }
  },
  increaseBrush: () => {
    state.brush += 4;
  },
  fit: fitCurrentImage,
  next: () => loadImage(Math.min(state.index + 1, state.items.length - 1)),
  openHelp: () => helpDialog.showModal(),
  previous: () => loadImage(Math.max(0, state.index - 1)),
  redo: () => {
    if (!state.mask || !state.history) return;
    const restored = applyHistory(state.mask, state.history, true);
    if (!restored || restored.length === 0) return;
    markDirty(restored);
    state.unsavedChanges = true;
    scheduleSave();
    requestRender();
  },
  reject: () => action("reject"),
  selectClass,
  selectTool,
  skip: () => action("skip"),
  toggleLayer: (index) => {
    const [id] = layerDefinitions[index];
    state.layers[id] = !state.layers[id];
    requestRender();
  },
  toggleSuperpixelLevel,
  undo: () => {
    if (!state.mask || !state.history) return;
    const restored = applyHistory(state.mask, state.history);
    if (!restored || restored.length === 0) return;
    markDirty(restored);
    state.unsavedChanges = true;
    scheduleSave();
    requestRender();
  },
};

async function init() {
  state.config = await apiJson("/config");
  for (const definition of state.config.classes) {
    colors[definition.id] = [
      Number.parseInt(definition.color.slice(1, 3), 16),
      Number.parseInt(definition.color.slice(3, 5), 16),
      Number.parseInt(definition.color.slice(5, 7), 16),
    ];
  }
  colors[255] = [155, 155, 155];
  renderer = createRenderer(canvas, state, colors);
  helpDialog = bindHelp(state.config);
  const palette = document.querySelector("#palette");
  for (const definition of state.config.classes) {
    const button = document.createElement("button");
    const swatch = document.createElement("span");
    const name = document.createElement("span");
    const hotkey = document.createElement("span");
    button.className = "class-button";
    button.dataset.id = String(definition.id);
    button.setAttribute("aria-keyshortcuts", definition.hotkey);
    swatch.className = definition.id === 255 ? "swatch ignore-swatch" : "swatch";
    if (definition.id !== 255) swatch.style.background = definition.color;
    name.textContent = definition.name;
    hotkey.className = "hotkey";
    hotkey.textContent = definition.hotkey;
    button.append(swatch, name, hotkey);
    button.addEventListener("click", () => selectClass(definition.id));
    palette.append(button);
  }
  selectClass(0);
  state.layers = createLayers(document.querySelector("#layers"), requestRender);
  bindControls();
  bindCanvas();
  const keymap = createKeymap(globals);
  window.addEventListener("keydown", (event) => handleShortcut(event, keymap));
  window.addEventListener("keydown", (event) => {
    if (event.code === "Space") state.space = true;
  });
  window.addEventListener("keyup", (event) => {
    if (event.code === "Space") state.space = false;
  });
  document.addEventListener("visibilitychange", () => {
    state.visible = !document.hidden;
  });
  window.addEventListener("beforeunload", () => saveDraft(true));
  new ResizeObserver(() => {
    renderer.resize();
    if (!state.viewportAdjusted) fitCurrentImage();
  }).observe(canvas);
  await refresh();
}

init().catch((error) => {
  status(error.message);
  toast(error.message);
});
