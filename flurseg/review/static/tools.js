/** Return the indices in a 4-connected region and update them to a new value. */
export function floodFill(mask, width, height, start, value) {
  const previous = mask[start];
  if (previous === value || start < 0 || start >= mask.length) return [];
  const changes = [];
  const stack = [start];
  const seen = new Uint8Array(mask.length);
  while (stack.length) {
    const index = stack.pop();
    if (seen[index] || mask[index] !== previous) continue;
    seen[index] = 1;
    changes.push([index, mask[index]]);
    mask[index] = value;
    const x = index % width;
    const y = Math.floor(index / width);
    if (x > 0) stack.push(index - 1);
    if (x + 1 < width) stack.push(index + 1);
    if (y > 0) stack.push(index - width);
    if (y + 1 < height) stack.push(index + width);
  }
  return changes;
}

/** Check whether a point lies inside a polygon using the even-odd rule. */
export function polygonContains(points, x, y) {
  let inside = false;
  for (let current = 0, previous = points.length - 1; current < points.length; previous = current++) {
    const a = points[current];
    const b = points[previous];
    const crosses = (a.y > y) !== (b.y > y);
    const intersection = ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x;
    if (crosses && x < intersection) inside = !inside;
  }
  return inside;
}

/** Fill a polygon and return one compact change list. */
export function fillPolygon(mask, width, height, points, value) {
  if (points.length < 3) return [];
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.max(0, Math.floor(Math.min(...xs)));
  const right = Math.min(width - 1, Math.ceil(Math.max(...xs)));
  const top = Math.max(0, Math.floor(Math.min(...ys)));
  const bottom = Math.min(height - 1, Math.ceil(Math.max(...ys)));
  const changes = [];
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const index = y * width + x;
      if (polygonContains(points, x + 0.5, y + 0.5) && mask[index] !== value) {
        changes.push([index, mask[index]]);
        mask[index] = value;
      }
    }
  }
  return changes;
}

/** Return points along a segment so fast brush strokes cannot leave gaps. */
export function interpolateStroke(from, to, spacing = 1) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(1, spacing)));
  return Array.from({ length: steps + 1 }, (_, step) => ({
    x: from.x + (dx * step) / steps,
    y: from.y + (dy * step) / steps,
  }));
}

/** Paint a circular brush dab and append its changed pixels to an edit. */
export function paintBrush(mask, width, height, point, radius, value, changes, changed) {
  const left = Math.max(0, Math.floor(point.x - radius));
  const right = Math.min(width - 1, Math.ceil(point.x + radius));
  const top = Math.max(0, Math.floor(point.y - radius));
  const bottom = Math.min(height - 1, Math.ceil(point.y + radius));
  const limit = radius * radius;
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const pixel = y * width + x;
      const distance = (x - point.x) ** 2 + (y - point.y) ** 2;
      if (distance <= limit && !changed.has(pixel) && mask[pixel] !== value) {
        changes.push([pixel, mask[pixel]]);
        changed.add(pixel);
        mask[pixel] = value;
      }
    }
  }
}

/**
 * Paint an interpolated brush stroke between two points.
 *
 * @param {Uint8Array} mask - Segmentation mask buffer.
 * @param {number} width - Image width.
 * @param {number} height - Image height.
 * @param {{x: number, y: number}} from - Start point in image coordinates.
 * @param {{x: number, y: number}} to - End point in image coordinates.
 * @param {number} radius - Brush radius in image pixels.
 * @param {number} value - Class label to paint.
 * @param {Array<[number, number]>} changes - Accumulator for pixel change records.
 * @param {Set<number>} changed - Set of already changed pixel indices.
 * @returns {Array<[number, number]>} The updated changes array.
 */
export function paintStroke(mask, width, height, from, to, radius, value, changes, changed) {
  for (const point of interpolateStroke(from, to, Math.max(1, radius / 2))) {
    paintBrush(mask, width, height, point, radius, value, changes, changed);
  }
  return changes;
}

export { markDirty } from "./state.js";

/** Build a lookup from superpixel id to typed pixel-index arrays. */
export function buildSuperpixelIndex(labels) {
  const buckets = new Map();
  labels.forEach((label, index) => {
    const pixels = buckets.get(label) || [];
    pixels.push(index);
    buckets.set(label, pixels);
  });
  return new Map([...buckets].map(([label, pixels]) => [label, Int32Array.from(pixels)]));
}

/** Update one indexed superpixel and append the changes to a single edit. */
export function paintSuperpixel(mask, index, label, value, changes, changed) {
  const pixels = index.get(label) || new Int32Array();
  for (const pixel of pixels) {
    if (!changed.has(pixel) && mask[pixel] !== value) {
      changes.push([pixel, mask[pixel]]);
      changed.add(pixel);
      mask[pixel] = value;
    }
  }
}

/** Fill only the clicked 255 component with non-255 proposal classes. */
export function acceptRegion(mask, proposal, width, height, start) {
  if (mask[start] !== 255) return [];
  const changes = [];
  const stack = [start];
  const seen = new Uint8Array(mask.length);
  while (stack.length) {
    const pixel = stack.pop();
    if (seen[pixel] || mask[pixel] !== 255) continue;
    seen[pixel] = 1;
    if (proposal[pixel] !== 255) {
      changes.push([pixel, mask[pixel]]);
      mask[pixel] = proposal[pixel];
    }
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    if (x > 0) stack.push(pixel - 1);
    if (x + 1 < width) stack.push(pixel + 1);
    if (y > 0) stack.push(pixel - width);
    if (y + 1 < height) stack.push(pixel + width);
  }
  return changes;
}
