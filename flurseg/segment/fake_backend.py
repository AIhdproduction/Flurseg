"""Deterministic NumPy-only backend for tests and CPU-only workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

_STANDARD_PROMPTS = (
    "person", "pole", "street lamp", "traffic sign", "fence", "wall", "water", "swimming pool",
    "bench", "power line", "hydrant", "bare soil", "dirt", "gravel", "sand", "mud",
    "dirt road", "excavation pit", "soil pile", "asphalt road", "sidewalk", "parking lot",
    "concrete pavement", "paving stones", "grass", "lawn", "meadow", "shrub", "hedge",
    "crop field", "flower bed", "tree", "tree canopy", "forest", "roof", "building",
    "house", "facade", "garage", "solar panel", "chimney", "car", "truck", "van",
    "bus", "trailer", "motorcycle", "excavator", "crane", "construction machine",
    "scaffolding", "site container", "shipping container", "building materials", "formwork",
    "curb", "curbstone", "kerb", "road marking", "lane marking", "crosswalk", "painted line",
    "parking space marking", "manhole cover", "storm drain", "drain grate", "gully grate",
    "valve cover", "cable pit cover", "utility cover", "vegetation", "asphalt", "road",
    "marking", "line", "vehicle", "license plate",
)


class FakeBackend:
    """Prompt-sensitive colour heuristics with no GPU or torch dependency."""

    name = "fake"

    def __init__(self, *, equivariant: bool = True, disagreement: float = 0.0) -> None:
        if not 0 <= disagreement <= 1:
            raise ValueError("disagreement must be between 0 and 1")
        self.equivariant = equivariant
        self.disagreement = disagreement
        self._noise_cache: dict[tuple[int, int], np.ndarray] = {}
        self._disagreement_cache: dict[tuple[int, int], np.ndarray] = {}
        self._prompt_recipes: dict[str, str] = {
            prompt: self._resolve_recipe(prompt) for prompt in _STANDARD_PROMPTS
        }

    @staticmethod
    def _resolve_recipe(prompt: str) -> str:
        lower = prompt.lower()
        if any(word in lower for word in ("grass", "tree", "forest", "vegetation", "hedge", "shrub", "crop", "lawn")):
            return "vegetation"
        if any(word in lower for word in ("asphalt", "road", "sidewalk", "pavement", "parking", "concrete")):
            return "paved"
        if any(word in lower for word in ("marking", "line", "crosswalk")):
            return "marking"
        if any(word in lower for word in ("roof", "building", "house", "facade")):
            return "roof"
        if any(word in lower for word in ("person", "vehicle", "car", "truck", "license")):
            return "person_vehicle"
        return "fallback"

    def _get_recipe(self, prompt: str) -> str:
        recipe = self._prompt_recipes.get(prompt)
        if recipe is None:
            recipe = self._resolve_recipe(prompt)
            self._prompt_recipes[prompt] = recipe
        return recipe

    def _get_cached_noise(self, shape: tuple[int, int]) -> np.ndarray:
        noise = self._noise_cache.get(shape)
        if noise is None:
            rng = np.random.default_rng(seed=42)
            noise = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
            self._noise_cache[shape] = noise
        return noise

    def _get_cached_disagreement(self, shape: tuple[int, int]) -> np.ndarray:
        noise = self._disagreement_cache.get(shape)
        if noise is None:
            rows, columns = np.indices(shape)
            noise = (((rows * 17 + columns * 31) % 101) / 100.0 - 0.5).astype(np.float32)
            self._disagreement_cache[shape] = noise
        return noise

    @staticmethod
    def _noise(image: np.ndarray, prompt: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "little")
        pixels = image.astype(np.uint32)
        values = pixels[..., 0] * 73856093 ^ pixels[..., 1] * 19349663 ^ pixels[..., 2] * 83492791 ^ seed
        return (values % 1009).astype(np.float32) / 1008.0

    def _score(self, image: np.ndarray, prompt: str) -> np.ndarray:
        return self.score_tile(image, [prompt])[0]

    def score_tile(self, tile: np.ndarray, prompts: Sequence[str]) -> np.ndarray:
        """Return deterministic per-prompt probabilities for an RGB tile."""
        if tile.ndim != 3 or tile.shape[2] != 3 or tile.dtype != np.uint8:
            raise ValueError("tile must be an HxWx3 uint8 RGB array")

        height, width = tile.shape[:2]
        if not prompts:
            return np.empty((0, height, width), dtype=np.float32)

        # Compute shared per-tile features once in float32
        pixels = tile.astype(np.float32) / 255.0
        red = pixels[..., 0]
        green = pixels[..., 1]
        blue = pixels[..., 2]

        brightness = pixels.mean(axis=2)
        exg = 2.0 * green - red - blue
        greyness = 1.0 - np.abs(red - green) - np.abs(green - blue)
        redness = red - 0.5 * (green + blue)
        red_minus_green = red - green
        line_response = (brightness - 0.65) * 3.0

        # Vectorised recipe evaluation
        recipe_maps = {
            "vegetation": np.clip((exg + 0.3) / 1.3, 0.0, 1.0),
            "paved": np.clip(greyness - brightness * 0.35, 0.0, 1.0),
            "marking": np.clip(line_response, 0.0, 1.0),
            "roof": np.clip((redness + 0.25) / 0.8, 0.0, 1.0),
            "person_vehicle": np.clip((red_minus_green + 0.4) / 1.2, 0.0, 1.0),
        }

        shape = (height, width)
        scores = np.zeros((len(prompts), height, width), dtype=np.float32)

        base_noise: np.ndarray | None = None
        if not self.equivariant:
            base_noise = self._get_cached_noise(shape)

        for idx, prompt in enumerate(prompts):
            recipe = self._get_recipe(prompt)
            if recipe == "fallback":
                if not self.equivariant and base_noise is not None:
                    prompt_hash = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "little")
                    shift_y = prompt_hash % height
                    shift_x = (prompt_hash >> 16) % width
                    scores[idx] = np.roll(base_noise, (shift_y, shift_x), axis=(0, 1)) * 0.2
            else:
                scores[idx] = recipe_maps[recipe]

        if not self.equivariant and self.disagreement > 0:
            orientation_noise = self._get_cached_disagreement(shape)
            scores += orientation_noise * self.disagreement
            np.clip(scores, 0.0, 1.0, out=scores)

        return scores

    def score_boxes(
        self, image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], prompt: str
    ) -> np.ndarray:
        """Score colour similarity to the mean colour of same-image boxes."""
        samples = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes if x1 > x0 and y1 > y0]
        if not samples:
            return np.zeros(image.shape[:2], dtype=np.float32)
        target = np.concatenate([sample.reshape(-1, 3) for sample in samples]).mean(axis=0)
        distance = np.linalg.norm(image.astype(np.float32) - target, axis=2) / (255 * np.sqrt(3))
        return np.clip(1 - distance, 0, 1).astype(np.float32)

    def close(self) -> None:
        """Match the real backend lifecycle without owning resources."""
        self._noise_cache.clear()
        self._disagreement_cache.clear()
