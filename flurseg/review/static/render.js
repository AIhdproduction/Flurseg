import { PROPOSAL_THRESHOLD } from "./state.js";

/**
 * Compute integer device-pixel draw rectangle where adjacent tiles share edges exactly.
 *
 * @param {{x: number, y: number}|number} pan - Current pan offset object or x offset.
 * @param {number} tileX - Tile column index.
 * @param {number} [tileY=0] - Tile row index.
 * @param {number} [sizeX=256] - Scaled size of a full tile along x.
 * @param {number} [sizeY] - Scaled size of a full tile along y.
 * @param {number} [fractionX=1] - Width fraction for partial edge tiles.
 * @param {number} [fractionY=1] - Height fraction for partial edge tiles.
 * @returns {{x: number, y: number, width: number, height: number, right: number, bottom: number}}
 */
export function computeDrawRect(
  pan,
  tileX,
  tileY = 0,
  sizeX = 256,
  sizeY = sizeX,
  fractionX = 1,
  fractionY = 1,
) {
  let px;
  let py;
  let tx;
  let ty;
  let sx;
  let sy;
  let fx;
  let fy;

  if (typeof pan === "number") {
    px = pan;
    py = tileX;
    tx = tileY;
    ty = sizeX;
    sx = sizeY;
    sy = fractionX ?? sizeY;
    fx = fractionY ?? 1;
    fy = arguments[7] ?? 1;
  } else {
    px = pan.x;
    py = pan.y;
    tx = tileX;
    ty = tileY;
    sx = sizeX;
    sy = sizeY ?? sizeX;
    fx = fractionX ?? 1;
    fy = fractionY ?? 1;
  }

  const x0 = Math.round(px + tx * sx);
  const x1 = Math.round(px + (tx + fx) * sx);
  const y0 = Math.round(py + ty * sy);
  const y1 = Math.round(py + (ty + fy) * sy);

  return {
    x: x0,
    y: y0,
    width: x1 - x0,
    height: y1 - y0,
    right: x1,
    bottom: y1,
  };
}

/** Create a requestAnimationFrame renderer for image tiles and editable mask chunks. */
export function createRenderer(canvas, state, colors) {
  const context = canvas.getContext("2d");
  const cache = new Map();
  let needed = true;
  let currentImageId = null;

  const request = () => {
    needed = true;
  };

  const clearCache = () => {
    cache.clear();
    state.maskChunks?.clear();
    state.proposalChunks?.clear();
    state.renderDirtyChunks?.clear();
  };

  const resize = () => {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    context.imageSmoothingQuality = "high";
    request();
  };

  const cachedImage = (url) => {
    if (cache.has(url)) return cache.get(url);
    const image = new Image();
    image.src = url;
    image.addEventListener("load", request);
    cache.set(url, image);
    if (cache.size > 160) cache.delete(cache.keys().next().value);
    return image;
  };

  const getActiveColors = () => {
    if (state.colors && Object.keys(state.colors).length > 0) return state.colors;
    if (colors && Object.keys(colors).length > 0) return colors;
    if (state.config && Array.isArray(state.config.classes)) {
      const parsed = {};
      for (const definition of state.config.classes) {
        if (definition.color && definition.color.length >= 7) {
          parsed[definition.id] = [
            Number.parseInt(definition.color.slice(1, 3), 16),
            Number.parseInt(definition.color.slice(3, 5), 16),
            Number.parseInt(definition.color.slice(5, 7), 16),
          ];
        }
      }
      parsed[255] = [155, 155, 155];
      state.colors = parsed;
      return parsed;
    }
    return null;
  };

  const chunk = (buffer, chunks, chunkX, chunkY, editable) => {
    const key = `${chunkX}:${chunkY}`;
    if (chunks.has(key) && (!editable || !state.renderDirtyChunks.has(key))) return chunks.get(key);
    const activeColors = getActiveColors();
    if (!activeColors) return null;
    const left = chunkX * 1024;
    const top = chunkY * 1024;
    const width = Math.min(1024, state.meta.width - left);
    const height = Math.min(1024, state.meta.height - top);
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const image = offscreen.getContext("2d");
    const pixels = image.createImageData(width, height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const value = buffer[(top + y) * state.meta.width + left + x];
        const rgba = (y * width + x) * 4;
        const color = value === 255 && (left + x + top + y) % 10 < 2
          ? [155, 155, 155]
          : activeColors[value] || [96, 165, 250];
        pixels.data[rgba] = color[0];
        pixels.data[rgba + 1] = color[1];
        pixels.data[rgba + 2] = color[2];
        pixels.data[rgba + 3] = value === 255 ? 110 : 255;
      }
    }
    image.putImageData(pixels, 0, 0);
    chunks.set(key, offscreen);
    state.renderDirtyChunks.delete(key);
    return offscreen;
  };

  const drawTiles = (layer, alpha = 1) => {
    if (!state.layers?.[layer] || !state.meta) return;
    const level = Math.min(state.meta.levels - 1, Math.max(0, Math.floor(Math.log2(1 / state.scale))));
    const multiplier = state.scale * 2 ** level;
    const width = Math.ceil(state.meta.width / 2 ** level);
    const height = Math.ceil(state.meta.height / 2 ** level);
    const tileSize = 256 * multiplier;
    const firstX = Math.max(0, Math.floor(-state.pan.x / tileSize));
    const firstY = Math.max(0, Math.floor(-state.pan.y / tileSize));
    const lastX = Math.ceil((canvas.clientWidth - state.pan.x) / tileSize);
    const lastY = Math.ceil((canvas.clientHeight - state.pan.y) / tileSize);
    context.globalAlpha = alpha;
    for (let y = firstY; y <= lastY && y * 256 < height; y += 1) {
      for (let x = firstX; x <= lastX && x * 256 < width; x += 1) {
        const url = `/api/images/${state.meta.id}/tiles/${layer}/${level}/${x}/${y}`;
        const image = cachedImage(url);
        if (image.complete && image.naturalWidth) {
          const rect = computeDrawRect(
            state.pan,
            x,
            y,
            tileSize,
            tileSize,
            image.naturalWidth / 256,
            image.naturalHeight / 256,
          );
          context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
        }
      }
    }
    context.globalAlpha = 1;
  };

  const drawMask = (buffer, alpha, chunks, editable = false) => {
    if (!buffer || (editable && state.layers && !state.layers.mask)) return;
    const chunkSize = 1024 * state.scale;
    const firstX = Math.max(0, Math.floor(-state.pan.x / chunkSize));
    const firstY = Math.max(0, Math.floor(-state.pan.y / chunkSize));
    const lastX = Math.ceil((canvas.clientWidth - state.pan.x) / chunkSize);
    const lastY = Math.ceil((canvas.clientHeight - state.pan.y) / chunkSize);
    context.globalAlpha = alpha;
    for (let y = firstY; y <= lastY && y * 1024 < state.meta.height; y += 1) {
      for (let x = firstX; x <= lastX && x * 1024 < state.meta.width; x += 1) {
        const offscreen = chunk(buffer, chunks, x, y, editable);
        if (offscreen) {
          const rect = computeDrawRect(
            state.pan,
            x,
            y,
            chunkSize,
            chunkSize,
            offscreen.width / 1024,
            offscreen.height / 1024,
          );
          context.drawImage(offscreen, rect.x, rect.y, rect.width, rect.height);
        }
      }
    }
    context.globalAlpha = 1;
  };

  const proposalChunk = (chunkX, chunkY) => {
    const key = `${chunkX}:${chunkY}`;
    if (state.proposalChunks.has(key)) return state.proposalChunks.get(key);
    const left = chunkX * 1024;
    const top = chunkY * 1024;
    const width = Math.min(1024, state.meta.width - left);
    const height = Math.min(1024, state.meta.height - top);
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const image = offscreen.getContext("2d");
    const pixels = image.createImageData(width, height);
    const color = getActiveColors()?.[state.activeClass] || [96, 165, 250];
    const isProposal = (x, y) => x >= 0 && y >= 0 && x < state.meta.width && y < state.meta.height
      && state.proposalMap[y * state.meta.width + x] >= PROPOSAL_THRESHOLD;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (!isProposal(left + x, top + y)) continue;
        const rgba = (y * width + x) * 4;
        const edge = !isProposal(left + x - 1, top + y) || !isProposal(left + x + 1, top + y)
          || !isProposal(left + x, top + y - 1) || !isProposal(left + x, top + y + 1);
        pixels.data[rgba] = color[0];
        pixels.data[rgba + 1] = color[1];
        pixels.data[rgba + 2] = color[2];
        pixels.data[rgba + 3] = edge ? 255 : 140;
      }
    }
    image.putImageData(pixels, 0, 0);
    state.proposalChunks.set(key, offscreen);
    return offscreen;
  };

  const drawProposal = () => {
    if (!state.proposalMap || !state.layers?.proposal) return;
    const chunkSize = 1024 * state.scale;
    const firstX = Math.max(0, Math.floor(-state.pan.x / chunkSize));
    const firstY = Math.max(0, Math.floor(-state.pan.y / chunkSize));
    const lastX = Math.ceil((canvas.clientWidth - state.pan.x) / chunkSize);
    const lastY = Math.ceil((canvas.clientHeight - state.pan.y) / chunkSize);
    context.globalAlpha = 0.55;
    for (let y = firstY; y <= lastY && y * 1024 < state.meta.height; y += 1) {
      for (let x = firstX; x <= lastX && x * 1024 < state.meta.width; x += 1) {
        const offscreen = proposalChunk(x, y);
        const rect = computeDrawRect(state.pan, x, y, chunkSize, chunkSize,
          offscreen.width / 1024, offscreen.height / 1024);
        context.drawImage(offscreen, rect.x, rect.y, rect.width, rect.height);
      }
    }
    context.globalAlpha = 1;
  };

  const frame = () => {
    requestAnimationFrame(frame);
    if (!needed || !state.meta) return;
    if (state.meta.id !== currentImageId) {
      currentImageId = state.meta.id;
      cache.clear();
      state.maskChunks?.clear();
      state.proposalChunks?.clear();
      state.renderDirtyChunks?.clear();
    }
    needed = false;
    context.imageSmoothingQuality = "high";
    context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    drawTiles("image");
    drawTiles("uncertainty", 0.45);
    drawTiles("auto_change", 0.5);
    drawTiles("raw_mask", 0.38);
    drawMask(state.mask, state.opacity, state.maskChunks, true);
    drawProposal();
    if (state.proposalMap && state.proposalWindow) {
      const [x0, y0, x1, y1] = state.proposalWindow;
      context.save();
      context.strokeStyle = "#FBBF24";
      context.lineWidth = 1.5;
      context.setLineDash([6, 4]);
      context.strokeRect(state.pan.x + x0 * state.scale, state.pan.y + y0 * state.scale,
        (x1 - x0) * state.scale, (y1 - y0) * state.scale);
      context.restore();
    }
    context.strokeStyle = "#FBBF24";
    context.lineWidth = 2;
    for (const box of state.boxes) {
      context.strokeRect(state.pan.x + box[0] * state.scale, state.pan.y + box[1] * state.scale,
        (box[2] - box[0]) * state.scale, (box[3] - box[1]) * state.scale);
    }
  };

  resize();
  requestAnimationFrame(frame);
  return { request, resize, clearCache };
}
