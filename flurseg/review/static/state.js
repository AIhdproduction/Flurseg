/** Confidence threshold shared by proposal rendering and acceptance. */
export const PROPOSAL_THRESHOLD = 128;

/** Apply qualifying proposal pixels and return one compact undo entry. */
export function acceptProposalPixels(mask, proposal, activeClass) {
  const changes = [];
  for (let index = 0; index < mask.length; index += 1) {
    if (proposal[index] >= PROPOSAL_THRESHOLD && mask[index] !== activeClass) {
      changes.push([index, mask[index]]);
      mask[index] = activeClass;
    }
  }
  return changes;
}

/** Parse a valid full-image proposal window from the API response header. */
export function parseProposalWindow(header, width, height) {
  if (!header) return null;
  const window = header.split(",").map(Number);
  if (window.length !== 4 || window.some((value) => !Number.isInteger(value))) return null;
  const [x0, y0, x1, y1] = window;
  if (x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0 || x1 > width || y1 > height) return null;
  return window;
}

/** Count thresholded proposal pixels, optionally restricted to its search window. */
export function countProposalPixels(proposal, window = null, width = 0) {
  if (!proposal) return 0;
  if (!window || !width) {
    let count = 0;
    for (const value of proposal) count += value >= PROPOSAL_THRESHOLD;
    return count;
  }
  const [x0, y0, x1, y1] = window;
  let count = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      count += proposal[y * width + x] >= PROPOSAL_THRESHOLD;
    }
  }
  return count;
}

/** Return the status text after accepting a proposal. */
export function proposalAcceptedStatus(pixelCount) {
  return `Proposal accepted: ${pixelCount} px`;
}

/** Return the status text after discarding a proposal. */
export function proposalDiscardedStatus() {
  return "Proposal discarded";
}

/** Create the mutable application state for one review session. */
export function createState() {
  return {
    activeClass: 0,
    boxes: [],
    brush: 24,
    colors: {},
    config: null,
    currentPool: false,
    draftTimer: null,
    history: null,
    index: -1,
    items: [],
    layers: null,
    mask: null,
    maskChunks: new Map(),
    meta: null,
    opacity: 0.55,
    pan: { x: 0, y: 0 },
    polygon: [],
    proposal: null,
    proposalChunks: new Map(),
    proposalMap: null,
    proposalWindow: null,
    renderDirtyChunks: new Set(),
    scale: 1,
    started: typeof performance !== "undefined" ? performance.now() : Date.now(),
    superpixelBuffers: { coarse: null, fine: null },
    superpixelIndex: null,
    superpixelIndices: { coarse: null, fine: null },
    superpixelLevel: "coarse",
    superpixels: null,
    tool: "superpixel",
    unsavedChanges: false,
    visible: typeof document !== "undefined" ? !document.hidden : true,
    viewportAdjusted: false,
  };
}

/** Mark a pixel index's actual mask chunk as needing a redraw. */
export function markDirtyChunk(state, index) {
  const width = state.meta?.width || state.width || 1024;
  const x = index % width;
  const y = Math.floor(index / width);
  state.renderDirtyChunks.add(`${Math.floor(x / 1024)}:${Math.floor(y / 1024)}`);
}

/**
 * Mark touched chunk keys as dirty from either pixel indices or a bounding box.
 *
 * @param {object} state - Application state containing meta and renderDirtyChunks.
 * @param {Array<number|Array<number>>|Set<number>|number|object} target - Indices or bbox.
 * @param {object} [renderer] - Optional renderer instance with a request() method.
 * @returns {Set<string>} The set of dirty chunk keys.
 */
export function markDirty(state, target, renderer = null) {
  if (!state || !state.renderDirtyChunks) return new Set();
  const width = state.meta?.width || state.width || 1024;
  const height = state.meta?.height || state.height || 1024;

  let minX;
  let minY;
  let maxX;
  let maxY;

  if (Array.isArray(target) && target.length === 4
      && typeof target[0] === "number" && typeof target[1] === "number") {
    minX = Math.min(target[0], target[2]);
    minY = Math.min(target[1], target[3]);
    maxX = Math.max(target[0], target[2]);
    maxY = Math.max(target[1], target[3]);
  } else if (target && typeof target === "object" && !Array.isArray(target) && !(target instanceof Set)) {
    if ("left" in target || "minX" in target || "x" in target) {
      minX = target.left ?? target.minX ?? target.x;
      minY = target.top ?? target.minY ?? target.y;
      maxX = target.right ?? target.maxX ?? (minX + (target.width ?? 0));
      maxY = target.bottom ?? target.maxY ?? (minY + (target.height ?? 0));
    }
  }

  if (minX !== undefined && minY !== undefined && maxX !== undefined && maxY !== undefined) {
    const minChunkX = Math.max(0, Math.floor(Math.min(minX, maxX) / 1024));
    const maxChunkX = Math.min(Math.floor((width - 1) / 1024), Math.floor(Math.max(minX, maxX) / 1024));
    const minChunkY = Math.max(0, Math.floor(Math.min(minY, maxY) / 1024));
    const maxChunkY = Math.min(Math.floor((height - 1) / 1024), Math.floor(Math.max(minY, maxY) / 1024));
    for (let cy = minChunkY; cy <= maxChunkY; cy += 1) {
      for (let cx = minChunkX; cx <= maxChunkX; cx += 1) {
        state.renderDirtyChunks.add(`${cx}:${cy}`);
      }
    }
  } else if (typeof target === "number") {
    markDirtyChunk(state, target);
  } else if (target && (Array.isArray(target) || target instanceof Set)) {
    for (const item of target) {
      const index = Array.isArray(item) ? item[0] : item;
      if (typeof index === "number") {
        markDirtyChunk(state, index);
      }
    }
  }

  if (renderer && typeof renderer.request === "function") {
    renderer.request();
  }
  return state.renderDirtyChunks;
}
