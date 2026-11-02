/** Replace select options with unique queue values while retaining the first option. */
export function populateFilter(select, values) {
  const first = select.options[0];
  const unique = [...new Set(values.filter(Boolean))].sort();
  select.replaceChildren(first, ...unique.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    return option;
  }));
}

/** Render queue buttons without interpolating untrusted data into HTML. */
export function renderQueue(container, items, selectedIndex, onSelect) {
  if (items.length === 0) {
    const message = document.createElement("p");
    message.className = "empty-state";
    message.textContent = "No images match the current queue filters.";
    container.replaceChildren(message);
    return;
  }

  const nodes = items.map((item, index) => {
    const button = document.createElement("button");
    const name = document.createElement("strong");
    const detail = document.createElement("small");
    const row = document.createElement("div");
    const uncertText = document.createElement("span");
    const bar = document.createElement("div");
    const value = document.createElement("span");
    const staleDraft = document.createElement("small");
    const percent = Math.round((item.uncertain_fraction || 0) * 100);

    button.className = "queue-item";
    button.classList.toggle("selected", index === selectedIndex);
    button.type = "button";
    name.textContent = item.id;
    detail.textContent = `${item.source} / ${item.session}`;

    row.className = "uncertainty-row";
    uncertText.className = "uncertainty-text";
    uncertText.textContent = `${percent}% uncertain`;

    bar.className = "uncertainty-bar";
    value.style.width = `${percent}%`;
    value.setAttribute("aria-label", `${percent}% uncertainty`);
    bar.append(value);
    row.append(uncertText, bar);

    button.append(name, detail, row);
    if (item.draft_stale) {
      staleDraft.className = "draft-stale";
      staleDraft.textContent = "outdated draft";
      button.append(staleDraft);
    }
    button.addEventListener("click", () => onSelect(index));
    return button;
  });

  container.replaceChildren(...nodes);
  const selected = container.querySelector(".selected");
  selected?.scrollIntoView({ block: "nearest" });
}
