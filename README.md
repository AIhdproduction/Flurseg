# Flurseg

Flurseg is an offline semantic segmentation dataset curation pipeline that distills Meta's Segment Anything Model 3 (SAM 3) teacher into high-quality training data for a lightweight custom student model (EoMT + DINOv3). Designed specifically for aerial drone imagery, handheld ground photos, and high-resolution orthophotography, Flurseg combines automated multi-scale zero-shot mask generation, domain-specific geometric and spectral refinement rules, multi-factor uncertainty estimation, and an efficient human-in-the-loop review interface. Training data, source images, and checkpoints remain strictly on the user's machine and are never uploaded or committed.

### How It Works

The end-to-end dataset distillation process flows through the following stages:

1. **Teacher Inference:** SAM 3 evaluates text prompts across overlapping tiles and two resolution scales with test-time augmentation (TTA), outputting dense class probabilities.
2. **Refinement:** Six rule-based filters clean edge boundaries, enforce physical dimensions, eliminate small islands, and verify color spectra.
3. **Uncertainty Estimation:** Multi-component pixel scoring combines prediction confidence, margin, TTA variance, and refinement delta.
4. **Automated Routing:** Low-uncertainty images pass automatically into the verified training pool; ambiguous scenes and deterministic spot checks route to human review.
5. **Human Review:** An offline web application with superpixel assistance, keyboard workflows, and visual search allows rapid annotator corrections.
6. **Training Pool:** Accepted and verified masks populate the deduplicated image pool.
7. **Dataset Export:** Pool images are partitioned into session-stratified splits (train/val/test), sliced into 1024 px tiles, and exported with full frequency metadata.
8. **Student Model:** The resulting dataset serves to train an Encoder-only Mask Transformer (EoMT; Kerssies et al., CVPR 2025) with DINOv3 ViT-L/16 backbone (via `train.bat`).

For complete technical and architectural specifications, see [docs/architecture.md](docs/architecture.md).

---

## Requirements

- **Operating System:** Windows 11 (64-bit)
- **GPU:** NVIDIA GPU with CUDA support (developed and tested on NVIDIA GeForce RTX 4090 24 GB)
- **Python:** Python 3.13 (64-bit)
- **Disk Storage:** Storage scales linearly with dataset size; allow tens of MB per image for intermediate work artifacts (scores, pyramids, diagnostic maps).

---

## Installation

1. **Clone the repository:**
   ```bat
   git clone https://github.com/AIhdproduction/Flurseg.git
   cd Flurseg
   ```

2. **Run Environment Setup:**
   Execute `setup.bat` from the repository root. This script verifies Python 3.13, creates an isolated virtual environment (`venv`), installs pinned dependencies (PyTorch with CUDA 12.6, Ultralytics, Rasterio, OpenCV, Scikit-Image, FastAPI), and validates package imports.
   ```bat
   setup.bat
   ```

3. **Obtain SAM 3 Checkpoint:**
   SAM 3 model weights are NOT distributed with this repository due to Meta license terms.
   - Request access to the SAM 3 weights on Hugging Face: [`facebook/sam3`](https://huggingface.co/facebook/sam3).
   - Download the official `sam3.pt` checkpoint file.
   - Place the downloaded weights file into the local models directory at:
     ```
     models/sam3.pt
     ```

4. **Verify Installation:**
   Run the test and environment validation script:
   ```bat
   selftest.bat
   ```

---

## Quick Start with Public Sample Data

Flurseg includes an automated fetcher for public sample imagery (Swiss orthophotos, OpenDroneMap surveys, and Wikimedia Commons street photography):

1. **Preview available samples (dry run):**
   ```bat
   fetch_samples.bat --dry-run
   ```
   This displays the target file list, resolution, license, and download size without transferring files.

2. **Download samples:**
   ```bat
   fetch_samples.bat
   ```
   The script prompts for confirmation before downloading, populates `input/` with sample sessions, generates `session.yaml` descriptors, and records content hashes in `config/samples.lock.yaml`.

3. **Run the workflow:**
   Follow the standard pipeline steps: `prelabel.bat` -> `review.bat` -> `build_dataset.bat` -> `report.bat`.

---

## Workflow with Own Data

### 1. Folder Layout and Metadata

Place raw, unedited survey imagery into `input/` grouped by data source and session:

```
input/
├── drone/
│   └── site_alpha_2026/
│       ├── DJI_0001.JPG
│       ├── DJI_0002.JPG
│       └── session.yaml        (optional)
├── phone/
│   └── street_walk_01/
│       ├── IMG_20260401_101.jpg
│       └── session.yaml        (optional)
└── ortho/
    └── canton_aerial_tiles/
        ├── tile_2680_1250.tif
        └── session.yaml        (optional)
```

Each session directory may optionally contain a `session.yaml` file defining flight conditions and survey properties:

```yaml
camera: "DJI FC3411"
date: 2026-05-12
area_type: urban  # One of: construction_site, street, residential, urban, agriculture, forest_water
gsd_m: 0.02       # Ground sampling distance in metres
test: false       # Set to true to hold out session for final evaluation
notes: "Industrial park mapping under clear sky."
```

### 2. Processing Steps

- **Prelabeling:**
  ```bat
  prelabel.bat
  ```
  Ingests candidates, anonymizes phone images, computes SAM 3 dense scores, runs refinement, calculates uncertainty, routes images, and archives originals to `archive/`.

- **Human Review:**
  ```bat
  review.bat
  ```
  Launches the offline review application on `http://127.0.0.1:8765`. Annotators inspect proposed masks, apply superpixel and manual edits, and accept or reject images.

- **Dataset Construction:**
  ```bat
  build_dataset.bat
  ```
  Validates pool masks, splits sessions into train/val/test splits, slices imagery into 1024x1024 px tiles, and generates `train/dataset/dataset.json`.

- **Quality Reporting:**
  ```bat
  report.bat
  ```
  Generates a self-contained HTML report with embedded SVG charts in `reports/latest.html` documenting status distributions, spot-check error rates, class frequencies, and prompt precisions.

- **Training:**
  ```bat
  train.bat
  ```
  Trains the EoMT + DINOv3 student model on the built dataset (`train/dataset/dataset.json`). When finished, displays the run folder and best mIoU, and opens `train/runs/<timestamp>/report.html` automatically.
  - Resume from the latest checkpoint:
    ```bat
    train.bat --resume
    ```
  - Dry run for quick verification (executes up to 20 optimizer steps without full training):
    ```bat
    train.bat --dry-run
    ```
  - Additional training options via the CLI:
    ```bat
    venv\Scripts\python -m flurseg train [--config config/train.yaml] [--resume] [--steps <int>] [--dry-run]
    ```

- **Use the Model:**
  Predict semantic masks on new imagery using a trained run via the predict CLI (`flurseg/cli.py`):
  ```bat
  venv\Scripts\python -m flurseg predict --run <run_path> --image <image_path> --out <output_dir>
  ```
  Arguments:
  - `--run`: Path to the training run directory (e.g. `train/runs/<timestamp>`, which automatically resolves `best/`) or directly to a checkpoint folder (`best` or `last`).
  - `--image`: Path to the input image file to predict.
  - `--out`: Destination directory where the predicted class index mask and transparent color overlay are saved.

- **Reprocessing after Configuration Changes:**
  If you modify parameters in `config/pipeline.yaml` (such as refinement rules, priority, or uncertainty thresholds), re-run fusion and routing without recomputing SAM 3 inference:
  ```bat
  venv\Scripts\python -m flurseg reprocess
  ```

---

## Folder Structure

```
flurseg/
├── .gitignore
├── LICENSE
├── NOTICE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.bat
├── selftest.bat
├── fetch_samples.bat
├── prelabel.bat
├── review.bat
├── build_dataset.bat
├── report.bat
├── train.bat
├── config/
│   ├── classes.yaml
│   ├── pipeline.yaml
│   ├── samples.yaml
│   ├── samples.lock.yaml          # Sample lock metadata (local only, git-ignored)
│   ├── train.yaml
│   └── exemplars/
│       ├── README.md
│       └── <class_name>/          # Reference exemplars (local only, git-ignored)
├── docs/
│   └── architecture.md
├── flurseg/                       # Core package source code
├── tests/                         # Test suite
├── flurseg.db                     # Manifest database (local only, git-ignored)
├── models/                        # SAM 3 weights sam3.pt (local only, git-ignored)
├── input/                         # Raw survey input (local only, git-ignored)
├── archive/                       # Ingested original archives (local only, git-ignored)
├── work/                          # Per-image intermediate artifacts (local only, git-ignored)
├── review/                        # Active review queue & drafts (local only, git-ignored)
├── train/                         # Training data artifacts (local only, git-ignored)
│   ├── pool/                      # Verified images and masks
│   │   ├── images/
│   │   └── masks/
│   └── dataset/                   # Exported 1024 px tiles & dataset.json
│       ├── train/
│       ├── val/
│       ├── test/
│       └── dataset.json
├── reports/                       # Quality HTML reports (local only, git-ignored)
└── logs/                          # System logs (local only, git-ignored)
```

---

## Classes and Labeling Rules

### Class Catalogue

Defined in `config/classes.yaml`:

| ID | Class Name | Hex Swatch | Includes | Excludes | Cyclone Code |
|---|---|---|---|---|---|
| `0` | `other` | `#9E9E9E` | everything not covered by another class | (none) | 64 |
| `1` | `ground` | `#A0522D` | unsealed terrain, gravel, dirt roads, excavation pits, soil piles | paving, lawn | 2 |
| `2` | `paved_surface` | `#5C6BC0` | sealed surfaces: asphalt, concrete, paving stones, slabs | gravel, curb, covers | 71 |
| `3` | `low_vegetation` | `#9CCC65` | plants below about 3 m including hedges | tree canopy | 3 |
| `4` | `tree` | `#1B5E20` | trees and forest including trunk | hedges | 5 |
| `5` | `building` | `#E53935` | permanent buildings with roof and roof structures | containers, scaffolding | 6 |
| `6` | `vehicle` | `#FFEB3B` | road vehicles including trucks on construction sites | excavators, cranes | 91 |
| `7` | `construction` | `#FF9800` | construction machines, scaffolding, containers, building material | soil piles, excavation pits | 151 |
| `8` | `curb` | `#00BCD4` | top and front face of curbstones | gutters made of paving stones | 74 |
| `9` | `road_marking` | `#FFFFFF` | paint on the pavement | covers, curbs | 82 |
| `10` | `utility_cover` | `#E040FB` | manhole covers, gullies, valve caps, cable pits, flush hydrant covers | standing hydrants | 75 |
| `255` | `ignore` | `#000000` | uncertain or not checked, never trained | (none) | not exported |

### Priority Order

When multiple classes claim the same pixel, priority is resolved strictly from top to bottom:
1. `vehicle`
2. `construction`
3. `other`
4. `building`
5. `tree`
6. `utility_cover`
7. `curb`
8. `road_marking`
9. `low_vegetation`
10. `paved_surface`
11. `ground`

Priority resolution applies only within `fusion.priority_margin` (default `0.15`) of the best score among classes meeting `fusion.presence_threshold` (default `0.50`, or per-class in `presence_threshold_per_class`). A higher-priority class cannot override a more confident class if the difference exceeds this margin.

### Priority Overrides

- **`tree` over `vehicle`:** When tree canopy overhangs a vehicle below, `tree` overrides `vehicle` because the overhead aerial canopy is the visible surface from above. Configured under `fusion.priority_overrides`, overrides additionally require that the winner reaches `min_winner_score` (default `0.50`) and falls within `fusion.priority_margin` of the loser (`score_winner >= score_loser - priority_margin`).
- **Priority Conflicts:** When competing classes have strong evidence but cannot be resolved because their difference exceeds `fusion.priority_margin`, the conflict cannot be overridden and is flagged as uncertain, converting the area to `ignore` (ID 255) when it meets `uncertainty.min_region_px`.

### The 5 Labeling Rules

1. **Shadows:** Shadows belong to the surface below.
2. **Vehicles under Canopy:** A car under a tree is `tree` because the canopy is visible from above.
3. **Surfaces:** Gravel is always `ground`. Sealed surfaces are `paved_surface` and unsealed surfaces are `ground`.
4. **Earthworks & Construction:** Soil piles and excavation pits are `ground`. Construction is only what disappears after construction.
5. **Doubt:** When in doubt use `ignore` and never guess.

---

## Review UI and Keyboard Shortcuts

The review application (`review.bat`) runs an interactive canvas featuring deep-zoom pyramid rendering, dual-level SLIC superpixels (fine and coarse), and instant mask toggling.

### Complete Keyboard Shortcuts

| Shortcut | Action | Description |
|---|---|---|
| `0` | Select Class 0 | Select `other` |
| `1` | Select Class 1 | Select `ground` |
| `2` | Select Class 2 | Select `paved_surface` |
| `3` | Select Class 3 | Select `low_vegetation` |
| `4` | Select Class 4 | Select `tree` |
| `5` | Select Class 5 | Select `building` |
| `6` | Select Class 6 | Select `vehicle` |
| `7` | Select Class 7 | Select `construction` |
| `8` | Select Class 8 | Select `curb` |
| `9` | Select Class 9 | Select `road_marking` |
| `-` | Select Class 10 | Select `utility_cover` |
| `I` | Select Class 255 | Select `ignore` |
| `[` | Decrease Brush | Decrease brush radius |
| `]` | Increase Brush | Increase brush radius |
| `E` | Ignore Eraser | Paint directly with `ignore` class |
| `P` | Superpixel Tool | Click to fill SLIC superpixel segments |
| `Shift+P` | Toggle Superpixels | Toggle between fine and coarse superpixel segments |
| `F` | Flood Fill | Fill contiguous identical class region |
| `G` | Polygon Tool | Draw vector polygon boundary to assign class |
| `A` | Accept Region | Accept/lock verified proposal region |
| `B` | Find Similar | Box-prompted visual search using SAM 3 (requires CUDA & checkpoint) |
| `Y` | Accept Proposal | Revert/accept proposed mask as active draft |
| `N` | Discard Proposal | Discard unsaved edits / clear active draft |
| `O` | Cycle Opacity | Cycle mask overlay opacity (0% to 100%) |
| `ArrowLeft` | Previous Image | Navigate to previous image in review queue |
| `ArrowRight` | Next Image | Navigate to next image in review queue |
| `Home` | Fit Image | Reset zoom and fit image to window |
| `R` | Reject Image | Mark current image as rejected from pool |
| `S` | Skip Image | Skip current image and advance queue |
| `?` or `F1` | Open Help | Display shortcut reference and labeling rules |
| `Escape` | Cancel / Escape | Deselect active tool or dismiss dialog |
| `Enter` | Accept / Finish | Finalize reviewed image into pool, or complete polygon |
| `Ctrl+Z` | Undo | Undo last drawing operation |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo | Redo last undone operation |
| `Shift+1` | Toggle Layer 1 | Toggle Raw Proposal overlay |
| `Shift+2` | Toggle Layer 2 | Toggle Refinement Auto-Change overlay |
| `Shift+3` | Toggle Layer 3 | Toggle Uncertainty Heatmap overlay |
| `Shift+4` | Toggle Layer 4 | Toggle TTA Disagreement overlay |
| `Shift+5` | Toggle Layer 5 | Toggle Prompt Winner diagnostic overlay |
| `Shift+6` | Toggle Layer 6 | Toggle Superpixel Boundaries overlay |

---

## Configuration Reference

Pipeline configuration settings are specified in `config/pipeline.yaml`, and student model training parameters are specified in `config/train.yaml`. Below are the complete parameter dictionaries:

### Pipeline Configuration (`config/pipeline.yaml`)

### `sam3`
| Key | Default | Meaning |
|---|---|---|
| `checkpoint_path` | `models/sam3.pt` | Relative path to the manually supplied checkpoint. |
| `imgsz` | `1008` | Native SAM 3 tile size. |
| `precision` | `auto` | Precision mode (`auto`, `fp32`, `fp16`, or `bf16`). |
| `conf` | `0.25` | Predictor confidence threshold. |
| `query_threshold` | `0.25` | Minimum query confidence included in dense scores. |
| `query_fusion_chunk_size` | `128` | Maximum query heads combined in one dense-fusion operation. |
| `prompt_chunk_size` | `16` | Maximum prompts evaluated together on one tile. |
| `max_vram_gb` | `22` | VRAM budget kept below the 24 GB device capacity. |

### `tiling`
| Key | Default | Meaning |
|---|---|---|
| `tile_size` | `1008` | Pixel size of each inference tile. |
| `min_overlap` | `0.25` | Minimum fractional overlap between adjacent tiles. |

### `scales`
| Factor | Classes | Meaning |
|---|---|---|
| `1.0` | `[curb, road_marking, utility_cover, vehicle, other]` | Full resolution retains small objects. |
| `0.5` | `[ground, paved_surface, low_vegetation, tree, building, construction]` | Half resolution improves large-area context. |

### `tta`
| Key | Default | Meaning |
|---|---|---|
| `variants` | `[identity, rot90, hflip, vflip]` | Supported transform names for test-time augmentation. |

### `exemplars`
| Key | Default | Meaning |
|---|---|---|
| `mosaic.enabled` | `false` | Experimental cross-image mosaic workaround. |
| `mosaic.strip_fraction` | `0.2` | Fraction of the canvas reserved for exemplar crops. |
| `mosaic.max_per_class` | `4` | Maximum exemplar crops for one class. |
| `find_similar.enabled` | `true` | Enable same-image box prompted assistance when available. |

### `anonymization`
| Key | Default | Meaning |
|---|---|---|
| `enabled_for` | `[phone]` | Sources that should be anonymized. |
| `prompts` | `[person, license plate]` | SAM 3 prompts for regions to blur. |
| `blur_kernel` | `31` | Odd Gaussian blur kernel width in pixels. |
| `blur_sigma` | `0.0` | Gaussian sigma; zero lets OpenCV derive it automatically. |
| `min_score` | `0.5` | Minimum score required before a region is blurred. |

### `ingest`
| Key | Default | Meaning |
|---|---|---|
| `jpeg_quality` | `95` | High quality with no chroma subsampling for canonical image storage. |

### `ortho`
| Key | Default | Meaning |
|---|---|---|
| `tile_size` | `2048` | Pixel size of extracted orthophoto tiles. |
| `max_nodata_fraction` | `0.5` | Tiles with more nodata are dropped. |
| `black_threshold` | `5` | Pixel values at or below this are considered black nodata. |
| `gsd_m` | `0.10` | Fixed SWISSIMAGE ground sampling distance in metres. |

### `fusion`
| Key | Default | Meaning |
|---|---|---|
| `priority_overrides` | `[{winner: tree, loser: vehicle, min_winner_score: 0.5}]` | Canopy overrides that also respect `priority_margin`. |
| `presence_threshold` | `0.5` | A class must reach this score before normal priority can select it. |
| `presence_threshold_per_class` | `{}` | Optional per-class presence thresholds overriding default. |
| `priority_margin` | `0.15` | Priority can select a present class only when its score is within this margin of the most confident class. |
| `no_prompt_score` | `0.15` | Scores below this mean no prompt supplied evidence. |

### `refinement.edge_snap` (Rule 1)
| Key | Default | Meaning |
|---|---|---|
| `rule_id` | `1` | Stable auto-change map rule identifier. |
| `enabled` | `true` | Align class boundaries to SLIC superpixels and image edges. |
| `slic_n_segments_per_mp` | `1000` | SLIC segments per megapixel. |
| `compactness` | `10.0` | SLIC spatial compactness parameter. |
| `protected_classes` | `[curb, road_marking]` | Thin classes preserved during edge snapping. |
| `canny_low_threshold` | `50` | Lower Canny threshold used to protect strong image edges. |
| `canny_high_threshold` | `150` | Upper Canny threshold used to protect strong image edges. |
| `max_edge_fraction` | `0.35` | Superpixels with more edge pixels than this remain unchanged. |

### `refinement.morphology` (Rule 2)
| Key | Default | Meaning |
|---|---|---|
| `rule_id` | `2` | Stable auto-change map rule identifier. |
| `enabled` | `true` | Remove small islands and fill small holes. |
| `min_island_m2` | `{ground: 0.02, paved_surface: 0.02, low_vegetation: 0.02, tree: 0.05, building: 0.05, vehicle: 0.5, construction: 0.1, curb: 0.01, road_marking: 0.01, utility_cover: 0.01, other: 0.01}` | Minimum retained island areas by class in square metres. |
| `max_hole_m2` | `0.05` | Largest hole filled in square metres. |

### `refinement.shape` (Rule 3)
| Key | Default | Meaning |
|---|---|---|
| `rule_id` | `3` | Stable auto-change map rule identifier. |
| `enabled` | `true` | Apply metric size and compactness checks. |
| `utility_cover_diameter_m` | `[0.1, 1.5]` | Allowed utility-cover diameter range in metres. |
| `utility_cover_min_compactness` | `0.45` | Minimum utility-cover compactness score. |
| `vehicle_area_m2` | `[1.5, 40.0]` | Allowed vehicle area range in square metres. |

### `refinement.context` (Rule 4)
| Key | Default | Meaning |
|---|---|---|
| `rule_id` | `4` | Stable auto-change map rule identifier. |
| `enabled` | `true` | Mark context violations as ignore. |
| `vehicle_ring_width_px` | `3` | Width of neighbourhood ring evaluated around vehicle component. |
| `vehicle_on_building_ratio` | `0.6` | Building share of ring that invalidates a vehicle. |
| `utility_cover_ring_width_px` | `5` | Width of the ground-plane neighbourhood ring evaluated around a utility-cover component. |
| `utility_cover_ground_ratio` | `0.5` | Minimum non-ignore ground-plane share required to retain a utility-cover component. |
| `curb_max_distance_px` | `3` | Maximum distance from paved surface border for a curb. |

### `refinement.color` (Rule 5)
| Key | Default | Meaning |
|---|---|---|
| `rule_id` | `5` | Stable auto-change map rule identifier. |
| `enabled` | `true` | Use colour evidence for vegetation and markings. |
| `exg_threshold` | `0.08` | Normalised chromatic ExG threshold (2g-r-b, range -1 to 2). |
| `marking_brightness_percentile` | `90.0` | Brightness percentile used for road markings. |

### `refinement.ortho` (Rule 6)
| Key | Default | Meaning |
|---|---|---|
| `rule_id` | `6` | Stable auto-change map rule identifier. |
| `enabled` | `true` | Apply orthophoto-specific minimum dimensions. |
| `min_small_object_m` | `0.4` | Minimum curb or utility-cover size in metres. |
| `min_marking_width_m` | `0.3` | Minimum road-marking width in metres. |

### `uncertainty`
| Key | Default | Meaning |
|---|---|---|
| `weights.top_score` | `0.4` | Weight for low winning probability. |
| `weights.margin` | `0.3` | Weight for small best-versus-second margin. |
| `weights.priority_conflict` | `0.3` | Weight for priority selecting a lower-scoring class within the allowed margin. |
| `weights.tta` | `0.2` | Weight for transform disagreement. |
| `weights.rule_change` | `0.1` | Weight for refinement changes. |
| `threshold` | `0.5` | Pixels above this composite uncertainty become ignore. |

### `routing`
| Key | Default | Meaning |
|---|---|---|
| `auto_accept_max_uncertain_fraction` | `0.03` | Maximum uncertainty for automatic acceptance. |
| `auto_accept_max_ignore_fraction` | `0.10` | Images with more ignore pixels always go to review. |
| `spot_check_fraction` | `0.10` | Deterministic share sent to review for quality checks. |
| `seed` | `42` | Seed used for reproducible spot checks. |

### `review`
| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Loopback-only host for the review server. |
| `port` | `8765` | Local review server port. |
| `open_browser` | `true` | Open default web browser after starting review server. |
| `coarse_superpixels_per_mp` | `150` | Coarse SLIC segments per megapixel for review UI. |

### `dataset`
| Key | Default | Meaning |
|---|---|---|
| `tile_size` | `1024` | Pixel size of training tiles. |
| `max_ignore_fraction` | `0.9` | Tiles with more ignore pixels are dropped. |
| `split.train` | `0.70` | Fraction of sessions assigned to training. |
| `split.val` | `0.15` | Fraction of sessions assigned to validation. |
| `split.test` | `0.15` | Fraction of sessions assigned to testing. |
| `seed` | `42` | Seed used for reproducible session splitting. |

### `report`
| Key | Default | Meaning |
|---|---|---|
| `spot_check_error_limit` | `0.02` | Error rate that triggers a report warning. |

---

### Training Configuration (`config/train.yaml`)

#### `model`
| Key | Default | Meaning |
|---|---|---|
| `pretrained` | `tue-mps/eomt-dinov3-ade-semantic-large-512` | Hugging Face checkpoint loaded only for a real training run. |
| `tiny_random` | `false` | Tests set this to true to construct a small random model without downloads. |
| `image_mean` | `[0.485, 0.456, 0.406]` | ImageNet RGB channel mean values for image normalization. |
| `image_std` | `[0.229, 0.224, 0.225]` | ImageNet RGB channel standard deviation values for image normalization. |

#### `dataset_split`
| Key | Default | Meaning |
|---|---|---|
| `overrides` | `{}` | Optional stable session overrides, for example: `{drone/odm_zoo: test}`. |

#### `data`
| Key | Default | Meaning |
|---|---|---|
| `crop_size` | `768` | Random crop resolution in pixels for paired training tiles. |
| `scale_min` | `0.5` | Minimum scaling factor for random multi-scale zoom augmentation. |
| `scale_max` | `2.0` | Maximum scaling factor for random multi-scale zoom augmentation. |
| `color_jitter` | `0.2` | Maximum brightness and saturation jitter factor applied to training images. |

#### `training`
| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | Random seed for data loader shuffling, augmentation, and model initialization. |
| `batch_size` | `2` | Physical per-GPU micro-batch size. |
| `gradient_accumulation` | `8` | Number of gradient accumulation steps before an optimizer step (effective batch size = 16). |
| `epochs_target` | `60` | Target training epoch count used to compute total training steps. |
| `total_steps` | `auto` | Total optimizer steps; `auto` derives `round(epochs_target * train_tiles / effective_batch)`, clamped between 2,000 and 40,000. |
| `warmup_steps` | `auto` | Linear warmup steps; `auto` uses 10% of total steps, capped at 1,000 steps. |
| `eval_every` | `auto` | Step interval between validation evaluations; `auto` evaluates roughly every 2 epochs (minimum 200 steps). |
| `patience` | `8` | Early stopping tolerance in consecutive evaluations without validation mIoU improvement. |
| `head_lr` | `0.0001` | Learning rate for task-specific mask and classification query heads (1e-4). |
| `backbone_lr` | `0.00001` | Base learning rate for the pretrained DINOv3 ViT-L/16 backbone (1e-5). |
| `layer_decay` | `0.8` | Layer-wise learning rate decay factor applied progressively to earlier backbone layers. |
| `weight_decay` | `0.05` | Weight decay rate for AdamW optimizer on non-bias/norm parameters. |
| `max_grad_norm` | `1.0` | Maximum gradient norm for gradient clipping. |
| `polynomial_power` | `0.9` | Exponent for polynomial learning rate decay schedule after warmup. |

#### `loss`
| Key | Default | Meaning |
|---|---|---|
| `mask_weight` | `5.0` | Weight for binary cross-entropy mask matching and loss. |
| `dice_weight` | `5.0` | Weight for Dice loss on predicted class masks. |
| `class_weight` | `2.0` | Weight for class query classification logits. |
| `no_object_weight` | `0.1` | Weight for unmatched "no object" null queries. |

#### `inference`
| Key | Default | Meaning |
|---|---|---|
| `window_size` | `768` | Sliding window crop size in pixels (must equal `data.crop_size`). |
| `stride` | `384` | Stride step size in pixels between adjacent inference windows (default 50% overlap). |

---

## Exemplars and Visual Prompting

- **In-Image Visual Prompts ("Find Similar"):**
  Supported natively in the Review UI. When CUDA and `models/sam3.pt` are present, pressing `B` allows the annotator to draw a bounding box around an object (e.g., an unusual utility cover or rare vehicle). The backend uses SAM 3's geometric box prompt interface to detect visually similar instances across the active image.
- **Experimental Cross-Image Mosaic:**
  SAM 3 has **no native cross-image exemplar support**. Flurseg implements an experimental workaround (`exemplars.mosaic.enabled: false`, off by default): exemplar crops from `config/exemplars/<class_name>/` are composited into a border strip on the 1008 px canvas, box-prompted, and predictions falling inside the exemplar strip are discarded. For folder layout instructions, see [config/exemplars/README.md](config/exemplars/README.md).

---

## Privacy and Data Governance

- **Air-Gapped Operation:** All raw survey files, intermediate tiles, proposed masks, review records, checkpoints, and final training datasets remain exclusively on the user's local disk.
- **Git Ignore Protection:** All runtime folders and mutable data stores are ignored by Git (`models/`, `input/`, `archive/`, `work/`, `review/`, `train/`, `reports/`, `logs/`, `flurseg.db`).
- **Automated Phone Anonymization:** Phone photographs undergo automatic face and license plate blurring (`flurseg.ingest.anonymize`) during ingestion using SAM 3 zero-shot detection before intermediate artifacts are saved.

---

## Sample Data Licences

Public sample datasets specified in `config/samples.yaml` and documented in `NOTICE.md` carry the following licenses and attribution terms:
- **swisstopo:** SWISSIMAGE orthophotos are provided under the Open Government Data (OGD) terms of the Swiss Confederation. Attribution requirement: **© swisstopo**.
- **OpenDroneMap (ODM):** Aerial drone imagery from `OpenDroneMap/odm_data_aukerman` and `OpenDroneMap/odm_data_zoo` is released under the **Creative Commons CC0 1.0 Universal Public Domain Dedication**.
- **Wikimedia Commons:** Street photographs are retrieved under open licenses (CC-BY, CC-BY-SA, or Public Domain). Per-file licenses and authors are recorded in each session's `session.yaml` notes and locked in `config/samples.lock.yaml`.

---

## Training

### Architecture and Pretrained Base Checkpoint

The student segmentation model is an **Encoder-only Mask Transformer** (EoMT; Kerssies et al., CVPR 2025) leveraging Meta's **DINOv3 ViT-L/16** vision backbone.
- **Base Checkpoint:** Initialized from [`tue-mps/eomt-dinov3-ade-semantic-large-512`](https://huggingface.co/tue-mps/eomt-dinov3-ade-semantic-large-512) on Hugging Face. The model weights (~1.3 GB) download automatically on the first training run.
- **Hardware Performance (Measured on RTX 4090 24 GB):**
  - **Optimizer Step Time:** ~3.2 s per optimizer step (micro-batch size 2 with 8 gradient accumulation steps = effective batch size of 16 tiles).
  - **Memory:** ~17 GB VRAM with bfloat16 mixed precision and gradient checkpointing. The engine catches CUDA out-of-memory errors dynamically, halves the micro-batch size to 1, doubles accumulation to 16, and clears the cache to recover without loss of progress.

### Configuration Defaults (`config/train.yaml`) and Rationale

The default hyperparameters in `config/train.yaml` are designed for fine-tuning large vision transformers on high-resolution orthophotography and drone imagery:
- `model.pretrained` (`tue-mps/eomt-dinov3-ade-semantic-large-512`): Provides strong general-purpose semantic representations learned on ADE20K.
- `model.image_mean` & `model.image_std` (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`): Standard ImageNet RGB normalisation matching the backbone pretraining.
- `data.crop_size` (`768`): Balances contextual receptive field for large aerial structures with quadratic attention memory scaling.
- `data.scale_min` (`0.5`) & `data.scale_max` (`2.0`): Multi-scale random zoom augmentation accommodates ground sampling distance (GSD) variations across aerial platforms.
- `data.color_jitter` (`0.2`): Mild photometric perturbation ensures robustness against varying lighting and seasonal conditions.
- `training.batch_size` (`2`) & `training.gradient_accumulation` (`8`): Reaches an effective batch size of 16 for stable gradient estimates while fitting within 24 GB VRAM.
- `training.epochs_target` (`60`): Convergence target providing sufficient optimization passes over pool images without overfitting.
- `training.total_steps` (`auto`): Dynamically resolves to `round(epochs_target * train_tiles / 16)`, clamped between 2,000 and 40,000 steps.
- `training.warmup_steps` (`auto`): Linear warmup across the first 10% of steps (capped at 1,000 steps) preventing gradient destabilization of pretrained weights.
- `training.eval_every` (`auto`): Evaluates validation metrics every 2 epochs (minimum 200 steps) to track generalization without incurring excessive validation overhead.
- `training.patience` (`8`): Early stopping threshold; terminates training when validation mIoU fails to improve for 8 consecutive evaluations.
- `training.head_lr` (`1e-4`): Accelerated learning rate for randomly initialized mask queries and classification projection heads.
- `training.backbone_lr` (`1e-5`): Conservative learning rate for the pretrained DINOv3 backbone to prevent catastrophic forgetting.
- `training.layer_decay` (`0.8`): Progressive layer-wise learning rate decay dampening updates to earlier transformer blocks (`backbone_lr * layer_decay^(num_layers - 1 - depth)`).
- `training.weight_decay` (`0.05`): AdamW weight decay applied to non-bias and non-normalization parameters.
- `training.max_grad_norm` (`1.0`): Gradient norm clipping to ensure numerical stability during attention mask annealing.
- `training.polynomial_power` (`0.9`): Polynomial decay schedule smoothly decreasing learning rate to zero following warmup.
- `loss.mask_weight` (`5.0`), `loss.dice_weight` (`5.0`), `loss.class_weight` (`2.0`), `loss.no_object_weight` (`0.1`): Hungarian matching and loss weights prioritizing boundary sharpness and mask overlap over classification logits.
- `inference.window_size` (`768`) & `inference.stride` (`384`): Sliding-window evaluation with 50% Hann-window blending; window size must match `data.crop_size` because the patch grid is fixed.

### Ignore-Aware Loss

Pixels labeled as `ignore` (ID 255) are strictly excluded from Hungarian matching and loss computation. Upstream Hugging Face Transformers `EomtLoss` samples points across the entire image canvas, inadvertently treating unannotated or ambiguous pixels as negative background targets. Flurseg's custom implementation (`flurseg/train/loss.py`) samples candidate points only from valid pixels (`mask != 255`). Ignored areas neither penalize false positives nor contribute to binary cross-entropy or Dice loss.

### Outputs in `train/runs/<timestamp>/`

Each training execution writes complete artifacts into an isolated directory:
- `best/`: Checkpoint folder with the highest validation mIoU (contains `model.safetensors`, `config.json`, `preprocessor_config.json`, and `training_state.pt`).
- `last/`: Most recent checkpoint state for resuming interrupted training with `--resume`.
- `metrics.csv`: Step-by-step log of total loss, mask loss, Dice loss, classification loss, learning rate, and validation mIoU.
- `report.html`: Interactive, self-contained HTML report with embedded SVG loss and mIoU trendlines, per-class IoU summary, and confusion matrix.
- `MODEL_CARD.md`: Comprehensive model card documenting base checkpoint, dataset version, class catalogues, tile/session provenance, evaluation metrics, intended use, limitations, and license terms.
- `train.log`: Plain-text console log recording training steps, hardware adjustments, and validation evaluations.
- `config.yaml` & `dataset.json`: Snapshot copies of the exact configuration and dataset manifest used during the run.

### Growing the Dataset

Flurseg is designed for ongoing dataset expansion as new survey sessions are collected:
- **Split Registry (`train/split_registry.json`):** Persisted outside the disposable dataset folder to track session assignments and assignment timestamps across rebuilds.
- **Split Stability:** Existing sessions strictly retain their previously assigned split (`train`, `val`, or `test`) across dataset rebuilds (`build_dataset.bat`), preventing validation leakage.
- **Deficit-Based Allocation:** Newly added sessions are greedily assigned to the split with the largest image deficit relative to target ratios (70% train, 15% val, 15% test), favoring splits where the session's data source is not yet represented.
- **Test Sessions:** Sessions flagged with `test: true` in their `session.yaml` descriptor are always allocated to the `test` split.
- **Manual Overrides:** Explicit session assignments can be locked in `config/train.yaml` under `dataset_split.overrides` (e.g. `{ "drone/site_alpha": "test" }`).
- **`dataset_version`:** A monotonically incremented version counter in `train/split_registry.json` tracks dataset revisions across rebuilds and is embedded in `dataset.json` and `MODEL_CARD.md`.
- **`val_empty` Fallback:** If a small dataset has no sessions allocated to validation (`val_empty: true` in `dataset.json`), the training engine automatically creates a deterministic 10% holdout of training tiles (selected via SHA-256 hashes of tile IDs) for validation evaluation.

### Dataset Format Details

Exported under `train/dataset/` into `train/`, `val/`, and `test/` splits:
- **Images:** Non-overlapping 1024x1024 px JPEG tiles (`images/<tile_name>.jpg`).
- **Masks:** Single-channel 8-bit PNG masks (`masks/<tile_name>.png`) with pixel values 0 through 10 representing trainable classes, and 255 representing `ignore`.
- **Manifest (`dataset.json`):** Comprehensive catalog documenting tile dimensions, split partition mappings, tile counts per source, and exact class pixel frequencies.

### Model Licence

Fine-tuned model weights derive from Meta's DINOv3 ViT-L/16 backbone; the **DINOv3 License** applies to all derived weights. A copy of the DINOv3 License must accompany any published or redistributed weights. Consult the DINOv3 License regarding commercial and prohibited use cases before redistribution.

---

## Development & Testing

- **Self-Test Suite:**
  ```bat
  selftest.bat
  ```
- **Run Pytest Directly:**
  ```bat
  venv\Scripts\python -m pytest
  ```
- **Code Linting and Formatting:**
  ```bat
  venv\Scripts\python -m ruff check
  ```

---

## License

This software is licensed under the **GNU Affero General Public License v3.0 or later** (GNU AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full license text.

For third-party software and dataset notices, see [NOTICE.md](NOTICE.md).
