/** Create undo and redo stacks containing compact pixel change entries. */
export function createHistory() {
  return { redo: [], undo: [] };
}

/** Store one finished edit as an undo operation. */
export function pushHistory(history, changes) {
  if (changes.length === 0) {
    return false;
  }

  history.undo.push(changes);
  history.redo.length = 0;
  return true;
}

/** Apply one undo or redo operation and return changed pixel indices. */
export function applyHistory(mask, history, redo = false) {
  const source = redo ? history.redo : history.undo;
  const target = redo ? history.undo : history.redo;
  const entry = source.pop();
  if (!entry) {
    return [];
  }

  const inverse = [];
  for (const [index, previous] of entry) {
    inverse.push([index, mask[index]]);
    mask[index] = previous;
  }
  target.push(inverse);
  return entry.map(([index]) => index);
}
