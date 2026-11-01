/**
 * Convert a browser pointer event into a full-resolution image coordinate.
 *
 * @param {PointerEvent|MouseEvent|WheelEvent} event - Pointer event.
 * @param {HTMLCanvasElement} canvas - Editor canvas element.
 * @param {object} state - Mutable application state.
 * @returns {{x: number, y: number}} Clamped or unrounded image coordinates.
 */
export function imagePoint(event, canvas, state) {
  const box = canvas.getBoundingClientRect();
  return {
    x: Math.floor((event.clientX - box.left - state.pan.x) / state.scale),
    y: Math.floor((event.clientY - box.top - state.pan.y) / state.scale),
  };
}

/**
 * Zoom centered beneath the pointer with normalized wheel delta.
 *
 * @param {WheelEvent} event - Wheel event with deltaY and deltaMode.
 * @param {HTMLCanvasElement} canvas - Editor canvas element.
 * @param {object} state - Mutable application state.
 */
export function zoomAt(event, canvas, state) {
  if (!state.meta) return;
  const before = imagePoint(event, canvas, state);
  const notches = event.deltaMode === 1
    ? event.deltaY / 3
    : event.deltaMode === 2
      ? event.deltaY * 10
      : event.deltaY / 100;
  const factor = 1.25 ** Math.max(-5, Math.min(5, -notches));
  const canvasWidth = canvas.clientWidth || canvas.width || 800;
  const canvasHeight = canvas.clientHeight || canvas.height || 600;
  const fitScale = Math.min(canvasWidth / state.meta.width, canvasHeight / state.meta.height);
  const minScale = fitScale / 4;
  const maxScale = 16;
  state.scale = Math.max(minScale, Math.min(maxScale, state.scale * factor));
  const box = canvas.getBoundingClientRect();
  state.pan.x = Math.round(event.clientX - box.left - before.x * state.scale);
  state.pan.y = Math.round(event.clientY - box.top - before.y * state.scale);
}

/** Calculate a centered fit-to-view transform in CSS canvas pixels. */
export function fitTransform(imageWidth, imageHeight, canvasWidth, canvasHeight) {
  if (imageWidth <= 0 || imageHeight <= 0 || canvasWidth <= 0 || canvasHeight <= 0) {
    return null;
  }
  const scale = Math.min(canvasWidth / imageWidth, canvasHeight / imageHeight);
  return {
    scale,
    pan: {
      x: Math.round((canvasWidth - imageWidth * scale) / 2),
      y: Math.round((canvasHeight - imageHeight * scale) / 2),
    },
  };
}

/**
 * Center an image in the available canvas at a fit-to-view zoom.
 *
 * @param {HTMLCanvasElement} canvas - Editor canvas element.
 * @param {object} state - Mutable application state.
 */
export function fitImage(canvas, state) {
  if (!state.meta) return;
  const transform = fitTransform(
    state.meta.width,
    state.meta.height,
    canvas.clientWidth,
    canvas.clientHeight,
  );
  if (!transform) return;
  state.scale = transform.scale;
  state.pan = transform.pan;
}
