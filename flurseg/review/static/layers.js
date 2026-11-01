/** Definitions shared by the layer controls and keyboard keymap. */
export const layerDefinitions = [
  ["image", "Image", "Shift+1", true],
  ["mask", "Mask", "Shift+2", true],
  ["uncertainty", "Uncertainty", "Shift+3", false],
  ["auto_change", "Auto changes", "Shift+4", false],
  ["raw_mask", "Raw SAM mask", "Shift+5", false],
  ["proposal", "Proposal", "Shift+6", false],
];

/** Create accessible layer checkbox controls and return their shared state. */
export function createLayers(target, onChange) {
  const state = Object.fromEntries(layerDefinitions.map(([id, , , visible]) => [id, visible]));
  for (const [id, name, hotkey, visible] of layerDefinitions) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    const title = document.createElement("span");
    const key = document.createElement("span");
    input.type = "checkbox";
    input.checked = visible;
    input.setAttribute("aria-keyshortcuts", hotkey);
    title.textContent = name;
    key.className = "hotkey";
    key.textContent = hotkey;
    input.addEventListener("change", () => {
      state[id] = input.checked;
      onChange(state);
    });
    label.append(input, title, key);
    target.append(label);
  }
  return state;
}
