/** Fetch an API response and raise its client-facing error when it fails. */
export async function apiRequest(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (response.ok) {
    return response;
  }

  const fallback = { detail: response.statusText };
  const body = await response.json().catch(() => fallback);
  const error = new Error(body.detail || response.statusText);
  error.status = response.status;
  throw error;
}

/** Fetch an API resource as a Uint8Array. */
export async function apiBytes(path) {
  const response = await apiRequest(path);
  return new Uint8Array(await response.arrayBuffer());
}

/** Fetch an API resource as JSON. */
export async function apiJson(path, options = {}) {
  const response = await apiRequest(path, options);
  return response.json();
}

/** Send an uncompressed mask buffer to the draft endpoint. */
export async function saveMask(imageId, mask, keepalive = false) {
  return apiRequest(`/images/${imageId}/mask`, {
    method: "PUT",
    body: mask,
    keepalive,
  });
}
