# Visual Exemplars

Flurseg supports visual prompting through reference exemplars to assist class prediction.

## Overview

SAM 3 natively accepts visual prompts (bounding boxes) within the current image context (used in the Review UI as "Find similar"). However, SAM 3 does not offer native cross-image exemplar prompting.

Flurseg provides two distinct exemplar mechanisms:
1. **In-Image Visual Prompts ("Find Similar"):** Used interactively in the review application. A user draws a bounding box on an object in the current image to retrieve visually similar objects across the same scene. This leverages official SAM 3 geometric prompts.
2. **Cross-Image Mosaic Exemplars (Experimental):** Places cropped reference patches along an edge strip of the 1008 px inference tile, feeds them to SAM 3 with box prompts, and crops out predictions that fall inside the reference strip.

## Experimental Cross-Image Mosaic Format

When cross-image mosaic assistance is enabled (`exemplars.mosaic.enabled: true` in `config/pipeline.yaml`), exemplar files are loaded from subdirectories organized by class name:

```
config/exemplars/
├── README.md
├── tree/
│   ├── sample_01.jpg
│   └── sample_01.json
├── utility_cover/
│   ├── cover_01.jpg
│   └── cover_01.json
└── vehicle/
    ├── car_01.jpg
    └── car_01.json
```

### File Structure and Annotations

Each exemplar consists of an image file and a corresponding sidecar JSON file sharing the exact same basename:
- Image: `<image>.jpg` (or `.png`)
- Sidecar JSON: `<image>.json`

The JSON sidecar contains bounding box coordinates in `[x0, y0, x1, y1]` format (pixel coordinates relative to the exemplar image):

```json
[
  [12, 18, 95, 104],
  [140, 55, 230, 160]
]
```

### Configuration and Behavior

- **Disabled by default:** `exemplars.mosaic.enabled` is `false` by default.
- **Canvas Reservation:** When active, `exemplars.mosaic.strip_fraction` (default: `0.2`) reserves a portion of the 1008 px canvas for exemplar crops.
- **Maximum Crops:** `exemplars.mosaic.max_per_class` (default: `4`) sets the maximum number of crops stitched into a tile.
- **Privacy and Git:** All images and JSON annotations in subdirectories under `config/exemplars/*/*` are ignored by Git and remain strictly local.
