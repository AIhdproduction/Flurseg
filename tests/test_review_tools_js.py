"""Node-backed tests for the pure review editor JavaScript modules."""

from __future__ import annotations

import shutil
import subprocess
from textwrap import dedent

import pytest

NODE = shutil.which("node")


def _run(script: str) -> None:
    """Run a small ESM harness when Node is available on PATH."""
    if NODE is None:
        pytest.skip("node is not available on PATH")
    subprocess.run(
        [NODE, "--input-type=module", "-e", dedent(script)],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )


def test_flood_fill_uses_four_connectivity() -> None:
    _run(
        """
        import { floodFill } from "./flurseg/review/static/tools.js";
        const mask = new Uint8Array([1, 0, 1, 1]);
        floodFill(mask, 2, 2, 0, 9);
        if (mask.join(",") !== "9,0,9,9") throw new Error("unexpected fill");
        """,
    )


def test_shortcuts_ignore_non_element_targets() -> None:
    _run(
        """
        globalThis.Element = class Element {};
        globalThis.HTMLInputElement = class HTMLInputElement extends Element {};
        globalThis.HTMLSelectElement = class HTMLSelectElement extends Element {};
        globalThis.HTMLTextAreaElement = class HTMLTextAreaElement extends Element {};
        import { ignoresShortcut } from "./flurseg/review/static/shortcuts.js";
        if (ignoresShortcut({ key: "a", target: globalThis })) {
          throw new Error("window target should not ignore shortcuts");
        }
        """,
    )


def test_polygon_fill_handles_a_triangle() -> None:
    _run(
        """
        import { fillPolygon } from "./flurseg/review/static/tools.js";
        const mask = new Uint8Array(25);
        fillPolygon(mask, 5, 5, [{x: 0, y: 0}, {x: 4, y: 0}, {x: 0, y: 4}], 2);
        if (mask[0] !== 2 || mask[24] !== 0) throw new Error("unexpected polygon");
        """,
    )


def test_superpixel_index_uses_typed_arrays() -> None:
    _run(
        """
        import { buildSuperpixelIndex } from "./flurseg/review/static/tools.js";
        const index = buildSuperpixelIndex(new Uint32Array([2, 1, 2, 1]));
        if (!(index.get(2) instanceof Int32Array) || index.get(2).length !== 2) {
          throw new Error("missing typed superpixel index");
        }
        """,
    )


def test_brush_stroke_interpolation_has_no_gaps() -> None:
    _run(
        """
        import { interpolateStroke } from "./flurseg/review/static/tools.js";
        const points = interpolateStroke({x: 0, y: 0}, {x: 10, y: 0}, 1);
        if (points.length !== 11 || points.some((point, index) => point.x !== index)) {
          throw new Error("stroke has a gap");
        }
        """,
    )


def test_undo_redo_restores_exact_bytes() -> None:
    _run(
        """
        import { applyHistory, createHistory, pushHistory } from "./flurseg/review/static/history.js";
        const mask = new Uint8Array([1, 2, 3]);
        const history = createHistory();
        pushHistory(history, [[1, 2]]);
        mask[1] = 9;
        applyHistory(mask, history);
        applyHistory(mask, history, true);
        if (mask.join(",") !== "1,9,3") throw new Error("undo/redo mismatch");
        """,
    )


def test_accept_region_respects_ignore_boundaries() -> None:
    _run(
        """
        import { acceptRegion } from "./flurseg/review/static/tools.js";
        const mask = new Uint8Array([255, 255, 1, 255]);
        const proposal = new Uint8Array([3, 255, 9, 4]);
        acceptRegion(mask, proposal, 2, 2, 0);
        if (mask.join(",") !== "3,255,1,4") throw new Error("ignore boundary changed");
        """,
    )


def test_proposal_acceptance_uses_threshold_and_one_undo_entry() -> None:
    _run(
        """
        import { acceptProposalPixels } from "./flurseg/review/static/state.js";
        import { applyHistory, createHistory, pushHistory } from "./flurseg/review/static/history.js";
        const mask = new Uint8Array([0, 1, 2, 3]);
        const proposal = new Uint8Array([127, 128, 230, 0]);
        const history = createHistory();
        const changes = acceptProposalPixels(mask, proposal, 7);
        if (mask.join(",") !== "0,7,7,3") throw new Error("threshold changed wrong pixels");
        if (!pushHistory(history, changes) || history.undo.length !== 1) {
          throw new Error("proposal was not one undo entry");
        }
        applyHistory(mask, history);
        if (mask.join(",") !== "0,1,2,3") throw new Error("proposal undo failed");
        """,
    )


def test_fit_transform_uses_current_canvas_size_and_centers_image() -> None:
    _run(
        """
        import { fitTransform } from "./flurseg/review/static/viewport.js";
        const fit = fitTransform(400, 200, 300, 300);
        if (!fit || fit.scale !== 0.75 || fit.pan.x !== 0 || fit.pan.y !== 75) {
          throw new Error("fit transform is not centered in current canvas size");
        }
        if (fitTransform(400, 200, 0, 300) !== null) {
          throw new Error("zero-sized canvas should not produce a transform");
        }
        """,
    )


def test_proposal_status_texts() -> None:
    _run(
        """
        import {
          proposalAcceptedStatus,
          proposalDiscardedStatus,
        } from "./flurseg/review/static/state.js";
        if (proposalAcceptedStatus(42) !== "Proposal accepted: 42 px") {
          throw new Error("accepted proposal status changed");
        }
        if (proposalDiscardedStatus() !== "Proposal discarded") {
          throw new Error("discarded proposal status changed");
        }
        """,
    )


def test_proposal_window_and_count_limit_rendered_proposal_to_valid_response_data() -> None:
    _run(
        """
        import { countProposalPixels, parseProposalWindow } from "./flurseg/review/static/state.js";
        const proposal = new Uint8Array(24);
        proposal[1 * 6 + 2] = 128;
        proposal[1 * 6 + 5] = 255;
        const window = parseProposalWindow("2,1,5,3", 6, 4);
        if (!window || countProposalPixels(proposal, window, 6) !== 1) {
          throw new Error("proposal count was not limited to the valid window");
        }
        if (parseProposalWindow("2,1,7,3", 6, 4) !== null) {
          throw new Error("out-of-bounds window was accepted");
        }
        """,
    )


def test_stroke_undo_redo_roundtrip() -> None:
    _run(
        """
        import { paintStroke } from "./flurseg/review/static/tools.js";
        import { applyHistory, createHistory, pushHistory } from "./flurseg/review/static/history.js";

        const width = 16;
        const height = 16;
        const original = new Uint8Array(width * height);
        const mask = new Uint8Array(original);
        const history = createHistory();

        // 1. Begin stroke: initialize tracking state
        const edit = { changed: new Set(), changes: [] };

        // 2. Paint 3 points along a stroke
        const p1 = { x: 2, y: 2 };
        const p2 = { x: 5, y: 2 };
        const p3 = { x: 8, y: 2 };
        paintStroke(mask, width, height, p1, p1, 2, 1, edit.changes, edit.changed);
        paintStroke(mask, width, height, p1, p2, 2, 1, edit.changes, edit.changed);
        paintStroke(mask, width, height, p2, p3, 2, 1, edit.changes, edit.changed);

        const painted = new Uint8Array(mask);
        if (edit.changes.length === 0 || mask.every((v) => v === 0)) {
          throw new Error("stroke did not modify mask");
        }

        // 3. End stroke: commit to history
        pushHistory(history, edit.changes);

        // 4. Undo: mask bytes equal the original
        applyHistory(mask, history);
        if (mask.some((v, i) => v !== original[i])) {
          throw new Error("undo failed to restore original mask bytes");
        }

        // 5. Redo: mask bytes equal the painted state
        applyHistory(mask, history, true);
        if (mask.some((v, i) => v !== painted[i])) {
          throw new Error("redo failed to restore painted state bytes");
        }
        """,
    )
