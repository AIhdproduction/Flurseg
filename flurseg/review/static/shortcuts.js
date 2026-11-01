/** Return whether a keyboard event originates in editable or dialog UI. */
export function ignoresShortcut(event) {
  const target = event.target;
  if (event.key === "Escape") return false;
  return target instanceof Element && (target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || target.closest("dialog[open]"));
}

/** Create the one keymap table used to handle all review keyboard shortcuts. */
export function createKeymap(actions) {
  return [
    { key: "0", run: () => actions.selectClass(0) },
    { key: "1", run: () => actions.selectClass(1) },
    { key: "2", run: () => actions.selectClass(2) },
    { key: "3", run: () => actions.selectClass(3) },
    { key: "4", run: () => actions.selectClass(4) },
    { key: "5", run: () => actions.selectClass(5) },
    { key: "6", run: () => actions.selectClass(6) },
    { key: "7", run: () => actions.selectClass(7) },
    { key: "8", run: () => actions.selectClass(8) },
    { key: "9", run: () => actions.selectClass(9) },
    { key: "-", run: () => actions.selectClass(10) },
    { key: "i", run: () => actions.selectClass(255) },
    { key: "[", run: actions.decreaseBrush },
    { key: "]", run: actions.increaseBrush },
    { key: "e", run: () => actions.selectTool("eraser") },
    { key: "p", run: () => actions.selectTool("superpixel") },
    { key: "p", shift: true, run: actions.toggleSuperpixelLevel },
    { key: "f", run: () => actions.selectTool("flood") },
    { key: "g", run: () => actions.selectTool("polygon") },
    { key: "a", run: () => actions.selectTool("accept-region") },
    { key: "b", run: () => actions.selectTool("similar") },
    { key: "y", run: actions.acceptProposal },
    { key: "n", run: actions.discardProposal },
    { key: "o", run: actions.cycleOpacity },
    { key: "ArrowLeft", run: actions.previous },
    { key: "ArrowRight", run: actions.next },
    { key: "Home", run: actions.fit },
    { key: "r", run: actions.reject },
    { key: "s", run: actions.skip },
    { key: "?", run: actions.openHelp },
    { key: "F1", run: actions.openHelp },
    { key: "Escape", run: actions.escape },
    { key: "Enter", run: actions.enter },
    { key: "z", control: true, run: actions.undo },
    { key: "y", control: true, run: actions.redo },
    { key: "z", control: true, shift: true, run: actions.redo },
    ...[1, 2, 3, 4, 5, 6].map((number) => ({
      key: String(number),
      shift: true,
      run: () => actions.toggleLayer(number - 1),
    })),
  ];
}

/** Find and execute a matching shortcut table entry. */
export function handleShortcut(event, keymap) {
  if (ignoresShortcut(event)) return false;
  const key = (event.key || "").toLowerCase();
  const ctrl = Boolean(event.ctrlKey || event.metaKey);
  const shift = Boolean(event.shiftKey);
  const entry = keymap.find((item) => {
    const itemKey = (item.key || "").toLowerCase();
    const keyMatch = itemKey === key
      || (Boolean(item.shift) && event.code === `Digit${item.key}`)
      || (Boolean(item.shift) && event.code === `Key${item.key.toUpperCase()}`);
    return keyMatch
      && Boolean(item.control) === ctrl
      && Boolean(item.shift) === shift;
  });
  if (!entry) return false;
  event.preventDefault();
  entry.run();
  return true;
}
