# Architecture

Flurseg is an offline pipeline designed to prepare high-precision semantic segmentation datasets from drone imagery, handheld phone photos, and Swiss aerial orthophotos. Segment Anything Model 3 (SAM 3) acts as an automated teacher model generating initial proposals. Rule-based geometric and spectral refinement clean edge artifacts, while multi-factor uncertainty estimation routes ambiguous imagery to human annotators. Reviewed and auto-accepted masks populate a verified training pool, which is partitioned and tiled into training datasets for a student model (EoMT + DINOv3) destined for Hugging Face.

```mermaid
flowchart LR
    A[Local images] --> B[SAM 3 teacher]
    B --> C[Refinement]
    C --> D[Uncertainty]
    D --> E{Routing}
    E -->|Confident| F[Training pool]
    E -->|Uncertain| G[Human review]
    G --> F
    F --> H[Dataset]
    H --> I["Student training\nEoMT + DINOv3 (train.bat)"]
    I --> J[Trained model]
    J --> K[predict]
```

## Teacher (SAM 3 Inference)

- **Package Modules:** `flurseg.segment.sam3_backend`, `flurseg.segment.heads`, `flurseg.segment.runner`, `flurseg.segment.tiling`, `flurseg.segment.scales`, `flurseg.segment.tta`
- **Dense Head Fusion:** Rather than relying on standard bounding-box or point-prompt outputs, Flurseg subclasses `SAM3SemanticPredictor` to access raw internal decoder heads (`semantic_seg`, `pred_masks`, `pred_logits`, `presence_logit_dec`). Dense per-prompt probabilities are fused on GPU via:
  $$\text{Score} = \max\left(\sigma(\text{semantic\_seg}), \max_{\text{query}}(\sigma(\text{pred\_masks}) \odot \sigma(\text{pred\_logits}))\right) \odot \sigma(\text{presence\_logit\_dec})$$
- **Tiling & Blending:** Inference tiles run natively at 1008x1008 px with a minimum overlap of 25% (`min_overlap: 0.25`). Recombined tile borders are smoothed using 2D linear-ramp feathering weights.
- **Dual-Scale Strategy:** Imagery is processed across two dedicated scale passes to balance detail and spatial context:
  - Scale factor 1.0 (full resolution) for small, detail-critical objects: `curb`, `road_marking`, `utility_cover`, `vehicle`, `other`.
  - Scale factor 0.5 (half resolution) for expansive contextual regions: `ground`, `paved_surface`, `low_vegetation`, `tree`, `building`, `construction`.
- **Test-Time Augmentation (TTA):** Every tile is evaluated across four spatial transformations (`identity`, `rot90`, `hflip`, `vflip`), inversely transformed, and averaged to suppress orientation artifacts and compute pixel disagreement.
- **Prompt Aggregation & Memory:** Each semantic class maps to multiple text prompts defined in `config/classes.yaml`. Dense scores take the maximum probability across associated prompts. Prompts are evaluated in bounded chunks (`prompt_chunk_size: 16`, `query_fusion_chunk_size: 128`) maintaining an $O(\text{classes})$ GPU memory footprint within the 22 GB allocation ceiling (`max_vram_gb: 22`).

## Fusion

- **Package Module:** `flurseg.fusion`
- **Resolution Alignment:** Class score maps from the dual-scale passes are upsampled to original canvas resolution via bilinear interpolation.
- **Priority with Margin:** Ambiguous pixels are resolved deterministically following the global priority hierarchy in `config/classes.yaml` (`vehicle` > `construction` > `other` > `building` > `tree` > `utility_cover` > `curb` > `road_marking` > `low_vegetation` > `paved_surface` > `ground`) only when the present class is within `priority_margin` (default: 0.15) of the most confident class. Otherwise, the most confident class wins.
- **Presence Thresholding:** A class must exceed the baseline `presence_threshold` (default: 0.50) or its class-specific threshold in `presence_threshold_per_class` before it is eligible to claim pixels.
- **Priority Overrides:** Contextual overrides retain their minimum-score conditions and must also be within `priority_margin` of the loser (e.g., `tree` overrides `vehicle` when tree score $\ge 0.50$ to preserve canopy views from overhead perspectives).
- **Diagnostics:** The fusion pass computes the winning class mask, the assigned-class probability (`top_score`), the best other probability (`second_score`), the clipped assigned-versus-other margin, and `conflict = max(best_score - top_score, 0)`. A non-zero conflict means priority selected a lower-scoring class. Unassigned pixels meeting `no_prompt_score` (0.15) take argmax across classes, while remaining pixels fall back to `ignore` (255).

## Refinement Rules

- **Package Modules:** `flurseg.refine.registry`, `flurseg.refine.edge_snap`, `flurseg.refine.morphology`, `flurseg.refine.shape_rules`, `flurseg.refine.context_rules`, `flurseg.refine.color_rules`, `flurseg.refine.ortho_rules`
- Rules keep their stable identifiers but execute in this quality-preserving order: 1 (Edge Snap), 5 (Color), 4 (Context), 3 (Shape), 6 (Ortho), then 2 (Morphology). `auto_change.png` still records the stable identifier of the final rule that changed a pixel:
  - **Rule 1 (Edge Snap):** Snaps class boundaries to SLIC superpixel segments while protecting Canny image gradients and preserving thin classes (`curb`, `road_marking`).
  - **Rule 2 (Morphology):** Final cleanup for every image. It merges islands below metric (`min_island_m2`) or pixel (`min_component_px`) thresholds into their surrounding majority class and fills enclosed small components using the enclosing class threshold.
  - **Rule 3 (Shape):** Enforces geometric physical bounds on specific object categories, such as manhole cover diameter (0.10 m to 1.50 m), circular compactness ($\ge 0.45$), and vehicle footprint areas (1.5 m² to 40.0 m²).
  - **Rule 4 (Context):** Reclassifies topological impossibilities to `ignore` (255) per connected component, keeping complete marking and curb components when their configurable paved-surface context is valid.
  - **Rule 5 (Color):** Changes only `ground`, `paved_surface`, and `low_vegetation` on fine SLIC superpixels. Ground or paving becomes low vegetation only with bright-enough colour evidence (mean HSV V at least 70, normalised $\text{ExG} = 2g-r-b \ge 0.15$), a mean fused low-vegetation score of at least 0.25, and a connected converted region of at least 3,000 px. Low vegetation can return to ground only when equally bright and $\text{ExG} < -0.02$; smaller conversion groups are restored to their original class. This prevents dark chromatic noise from creating blocky vegetation speckles.
  - **Rule 6 (Ortho):** Enforces minimum physical dimension thresholds specific to nadir orthophotography (minimum 0.40 m for covers/curbs, 0.30 m width for road markings).

## Uncertainty Scoring

- **Package Module:** `flurseg.uncertainty`
- **Composite Score:** Per-pixel uncertainty is calculated as a normalized linear combination of five diagnostic indicators:
  $$\text{Uncertainty} = w_{\text{top}} \cdot (1 - \text{top\_score}) + w_{\text{margin}} \cdot (1 - \text{margin}) + w_{\text{conflict}} \cdot \text{priority\_conflict} + w_{\text{tta}} \cdot \text{TTA\_disagreement} + w_{\text{rule}} \cdot \mathbf{1}_{\{\text{auto\_change} \ne 0\}}$$
- **Default Weights:** Top score weight 0.40, margin weight 0.30, priority-conflict weight 0.30, TTA disagreement weight 0.20, refinement change weight 0.10.
- **Ignore Masking:** Connected uncertain regions at or above `uncertainty.min_region_px` whose composite uncertainty exceeds `uncertainty.threshold` (0.50) are assigned to `ignore` (255) in `proposed_mask.png`; smaller uncertain specks retain their refined class. A non-zero priority conflict at or above `priority_margin` is always routed to review when its region meets that minimum.

## Routing

- **Package Module:** `flurseg.routing`
- **Automated Routing Criteria:**
  - **Uncertainty Ceiling:** Images with an uncertain pixel ratio $\le 0.03$ (`auto_accept_max_uncertain_fraction`) pass towards automatic acceptance.
  - **Ignore Guard:** Images with more than 10% `ignore` pixels (`auto_accept_max_ignore_fraction: 0.10`) are diverted to human review regardless of low average uncertainty.
  - **Test Session Quarantine:** Any session designated with `test: true` in `session.yaml` bypasses automated acceptance entirely and routes all images into review.
  - **Deterministic Spot Checks:** A reproducible pseudo-random sample (`spot_check_fraction: 0.10`, seeded by SHA-256 and `routing.seed: 42`) of otherwise passing images is routed to human review for continuous validation.
- **Materialization:** Routing atomically provisions work directories and moves images and masks into either `train/pool/` (auto-accepted) or `review/<image_id>/` (needs review).

## Human Review

- **Package Modules:** `flurseg.review.app`, `flurseg.review.service`, `flurseg.review.api`, `flurseg.review.pyramid`
- **Local Application:** Self-contained FastAPI web server (`review.bat`) bound strictly to loopback (`127.0.0.1:8765`) serving interactive deep-zoom image pyramids.
- **Superpixel Integration:** Two levels of SLIC segments (fine and coarse at 150 segments per megapixel) permit single-click regional corrections and rapid boundary filling.
- **Atomic Operations:** In-progress annotator modifications are recorded via atomic file updates (`atomic_save_jpg`, `atomic_save_png`), preventing incomplete or corrupt files on disk.
- **Reprocessing:** Reprocess rewrites every derived mask, diagnostic, and `overlay.jpg` artifact. Pyramid layers track source modification time and size, rebuild into a new versioned directory when a source changes, then atomically switch the active version.
- **Quality Metrics Logged:** Review sessions record elapsed time (`review_seconds`), per-class pixel divergence between teacher proposal and human final (`review_changed_pixels`), and action outcomes (`accept`, `reject`, `skip`) in SQLite manifest records (`flurseg.db`).

## Dataset Construction

- **Package Modules:** `flurseg.dataset.build`, `flurseg.dataset.split`, `flurseg.dataset.tiles`, `flurseg.dataset.validate`
- **Integrity Validation:** Validates all pool masks for dimension agreement, valid class IDs (0–10, 255), and absence of file truncation prior to dataset extraction.
- **Session-Level Splitting:** Splits are partitioned strictly by session (default: 70% train, 15% validation, 15% test) to prevent spatial autocorrelation and data leakage across splits.
- **Extraction & Tiling:** Slices pool imagery and verified masks into non-overlapping 1024x1024 px tiles. Tiles exceeding `max_ignore_fraction: 0.90` are discarded.
- **Manifest Export:** Generates `dataset.json` containing class catalogues, session split memberships, tile counts, and precise per-class pixel counts and relative frequencies.

## Dataset Versioning

- **Package Modules:** `flurseg.dataset.split`, `flurseg.dataset.build`
- **Split Registry:** Prior session assignments are persisted in `train/split_registry.json` outside the disposable `train/dataset/` directory.
- **Stable Assignments:** When new imagery is added and the dataset is rebuilt, existing `(source, session)` pairs retain their previously assigned splits (`train`, `val`, `test`), preserving split stability across incremental dataset builds. Optional manual split overrides can be configured in `config/train.yaml` under `dataset_split.overrides`.
- **Dataset Versioning:** A monotonically incrementing `dataset_version` counter in `split_registry.json` is mirrored into `dataset.json` and training run reports each time sessions are partitioned or repartitioned.
- **Validation Holdout (val_empty):** When the training pool contains too few sessions to allocate an independent validation split (`val_empty: true`), the training engine deterministically holds out 10% of training tiles (selected via SHA-256 hash of tile IDs) for validation during training so metric evaluation and early stopping remain fully functional.

## Student Training

- **Package Modules:** `flurseg.train.engine`, `flurseg.train.data`, `flurseg.train.loss`, `flurseg.train.predict`, `flurseg.train.config`
- **Entry Points:** Executed via `train.bat` or `python -m flurseg train` (supporting `--resume`, `--steps`, `--dry-run`, and `--config`), with `python -m flurseg predict` for sliding-window inference on arbitrary images.
- **Model Architecture:** Encoder-only Mask Transformer (EoMT) pairing a DINOv3 ViT-L/16 backbone from pretrained checkpoint `tue-mps/eomt-dinov3-ade-semantic-large-512`. The classification head is replaced for 11 trainable semantic classes (IDs 0–10 from `config/classes.yaml`, excluding ignore class 255). The input `image_size` is fixed to the crop size (768 px) because the EoMT patch grid is fixed.
- **Data Pipeline:**
  - Extracts 768x768 px crops from 1024x1024 px dataset tiles, padding smaller images with black pixels and ignored mask labels.
  - Random scale augmentation between 0.5 and 2.0.
  - Horizontal flips (50% probability).
  - Quarter rotations (`rot90`) and vertical flips applied only for nadir perspectives (`drone` and `ortho` sources); phone imagery retains upright orientation.
  - Colour jitter (brightness and saturation within ±20%).
  - Normalisation using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
- **Loss Computation:**
  - Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`) between predicted mask queries and ground-truth masks.
  - Weighted combination of mask binary cross-entropy (BCE, weight 5.0), dice loss (weight 5.0), and query classification cross-entropy (CE, weight 2.0, with null/no-object weight 0.1).
  - Ignore pixels (index 255) are strictly excluded from point sampling, matching cost, and mask loss calculations, avoiding the negative-target bias of standard full-image point sampling in `EomtLoss`.
- **Optimisation:**
  - AdamW optimizer (`betas=(0.9, 0.999)`, `weight_decay: 0.05`, excluding 1D/norm/bias/position/register/query parameters from decay).
  - Layer-wise learning rate decay (`layer_decay: 0.8`) across transformer backbone layers with `backbone_lr: 1e-5` and `head_lr: 1e-4`.
  - Automatic linear warmup (10% of total steps, capped at 1,000 steps) followed by polynomial decay (`polynomial_power: 0.9`).
  - Mixed precision training with `bfloat16` (`torch.autocast`) on CUDA.
  - EoMT mask annealing: block-wise staggered linear mask-disable schedule via `attn_mask_probs`.
  - Gradient accumulation (batch size 2, accumulation 8, effective batch size 16) with total steps bounded between 2,000 and 40,000 based on 60 target epochs.
  - Gradient checkpointing enabled when supported, and automatic CUDA OOM recovery (halving batch size, doubling accumulation).
- **Evaluation & Inference:**
  - Sliding-window evaluation and prediction with 2D Hann-window blending (`window_size: 768`, `stride: 384`).
  - Semantic mean Intersection over Union (mIoU) and pixel accuracy evaluated with ignore index 255 excluded from the confusion matrix.
  - Checkpointing stores `last/` (every evaluation) and `best/` (improved validation mIoU) containing model weights, preprocessor configuration, optimizer and scheduler states, CSV metric logs, and Python/NumPy/Torch/CUDA RNG states.
  - Early stopping triggers when validation mIoU does not improve for `patience: 8` evaluations.
  - Resumption via `--resume` restores model weights, optimizer/scheduler states, step counters, metric rows, and RNG states.
  - `flurseg predict` writes class-index masks and blended transparent colour overlays.
- **Model Card & Deployment:** Each completed run writes `report.html` and a standalone `MODEL_CARD.md` documenting base checkpoint provenance, dataset version, per-class IoU, tile counts, and DINOv3 licence terms for downstream Hugging Face release.

## Quality Reporting

- **Package Modules:** `flurseg.report.render`, `flurseg.report.metrics`
- **Continuous Quality Control:** Aggregates database records to measure the spot-check error rate against human corrections. If the divergence exceeds `spot_check_error_limit: 0.02` (2%), visual warnings are highlighted in the report.
- **Precision Diagnostics:** Compares SAM 3 prompt-winner pixels against human ground truth to determine empirical precision for individual prompts and classes.
- **Presence Threshold Recommendations:** Calculates the minimum prediction confidence necessary for each class to achieve 90% precision against human corrections.
- **Self-Contained Artifacts:** Renders standalone HTML documents (`reports/report_<timestamp>.html` and `reports/latest.html`) containing inline SVG charts with zero external CDN dependencies.

## Reprocess Workflow

- **Package Module:** `flurseg.pipeline` (`reprocess` entry point)
- **Zero-Inference Adjustments:** During initial prelabeling, raw model probability outputs are preserved as compressed NumPy arrays (`scores.npz`).
- **Rapid Iteration:** Changes to fusion priority, presence thresholds, refinement parameters, or uncertainty weights can be reapplied instantaneously across the entire dataset via `flurseg reprocess` without running the GPU-intensive SAM 3 model again.

## Data Privacy & Security

- **Air-Gapped Local Operation:** Training imagery, drone surveys, high-resolution orthophotos, segmentation masks, intermediate cache files, and SQLite databases remain entirely on the local system and are git-ignored.
- **Automatic Anonymization:** Handheld mobile phone photos undergo automated facial and license plate obfuscation (`flurseg.ingest.anonymize`) via SAM 3 text prompts before entering the processing pipeline.
