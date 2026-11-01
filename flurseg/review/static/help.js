import { layerDefinitions } from "./layers.js";

/**
 * Format a class shortcut action element with colour swatch and class name.
 *
 * @param {object} classDef - Class definition containing name, color, and id.
 * @returns {HTMLElement} Formatted action container.
 */
function createClassAction(classDef) {
  const container = document.createElement("span");
  container.className = "shortcut-action class-shortcut";
  const swatch = document.createElement("span");
  swatch.className = classDef.id === 255 ? "swatch ignore-swatch" : "swatch";
  if (classDef.id !== 255 && classDef.color) {
    swatch.style.background = classDef.color;
  }
  const name = document.createElement("span");
  name.textContent = classDef.name;
  container.append(swatch, name);
  return container;
}

/**
 * Bind the accessible help dialog and populate it with server-provided instructions.
 *
 * @param {object} config - Configuration object from the server.
 * @returns {HTMLDialogElement} Bound dialog instance.
 */
export function bindHelp(config) {
  const dialog = document.querySelector("#help-dialog");
  const content = document.querySelector("#help-content");

  const classMap = new Map();
  for (const definition of config.classes || []) {
    if (definition.hotkey) {
      classMap.set(definition.hotkey.toLowerCase(), definition);
    }
    classMap.set(String(definition.id), definition);
  }

  const layerMap = new Map();
  for (const [, name, hotkey] of layerDefinitions) {
    layerMap.set(hotkey.toLowerCase(), name);
  }

  const shortcutsList = [...(config.shortcuts || [])];
  if (!shortcutsList.some((item) => item.keys.toLowerCase() === "shift+p")) {
    const pIdx = shortcutsList.findIndex((item) => item.keys.toLowerCase() === "p");
    const spShortcut = {
      keys: "Shift+P",
      action: "Toggle superpixel granularity (fine/coarse)",
    };
    if (pIdx !== -1) {
      shortcutsList.splice(pIdx + 1, 0, spShortcut);
    } else {
      shortcutsList.push(spShortcut);
    }
  }

  const shortcutsContainer = document.createElement("div");
  shortcutsContainer.className = "shortcut-list";

  for (const shortcut of shortcutsList) {
    const key = document.createElement("kbd");
    key.textContent = shortcut.keys;

    const lowerKey = shortcut.keys.toLowerCase();
    let action;

    if (classMap.has(lowerKey)) {
      action = createClassAction(classMap.get(lowerKey));
    } else if (layerMap.has(lowerKey)) {
      action = document.createElement("span");
      action.textContent = layerMap.get(lowerKey);
    } else {
      action = document.createElement("span");
      action.textContent = shortcut.action;
    }

    shortcutsContainer.append(key, action);
  }

  const heading = document.createElement("h3");
  heading.textContent = "Labeling rules";

  const rulesList = document.createElement("ol");
  for (const rule of config.labeling_rules || []) {
    const item = document.createElement("li");
    item.textContent = rule;
    rulesList.append(item);
  }

  content.replaceChildren(shortcutsContainer, heading, rulesList);
  document.querySelector("#help-button").addEventListener("click", () => dialog.showModal());
  document.querySelector("#close-help").addEventListener("click", () => dialog.close());
  return dialog;
}
