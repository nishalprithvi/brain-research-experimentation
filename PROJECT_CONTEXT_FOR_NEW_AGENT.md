# Brain Graph Diffusion + GCL Project Context (3-Class Focus)

## 1) Main Goal
Build a robust and publishable **3-class brain graph classifier** (CN, MCI, AD) under severe class imbalance and limited real data.

Core objective:
- Improve **Macro-F1** and minority recalls (especially **MCI** and **AD**) without manipulating evaluation.
- Keep test evaluation strictly on **real graphs only**.
- Use synthetic data as training augmentation only, with quality controls.

---

## 2) Problem Setting and Why This Pipeline Exists
Real ADNI brain graph data is small and imbalanced, and this hurts minority-class learning. The project uses a multi-phase workflow to:
1. Learn latent structure from real brain graphs.
2. Generate class-conditional synthetic graphs (AD/MCI) with guidance.
3. Filter synthetic graphs for realism and uniqueness.
4. Use real + filtered synthetic data in contrastive pretraining.
5. Fine-tune a StandardGCN classifier for 3-class prediction.

The design intent is to improve minority recall and macro metrics while preserving reproducibility and scientific defensibility.

---

## 3) Methodology (Phase-by-Phase)

## Phase 1: Generative Training
Entrypoint:
- `python main_3class.py train ...`

Main components:
- **Graph VAE**: maps connectivity graphs to latent representation.
- **Latent Diffusion Model**: learns latent denoising/sampling process.
- **Teacher model for guidance**: latent classifier used later for class-conditional generation.

Key artifacts:
- `vae_3class.pth`
- `diffusion_3class.pth`
- `gcn_3class.pth` (or selected teacher artifact)
- Phase-1 QA outputs in `results_phase1_quality/...`

What this phase must guarantee:
- latent space is usable,
- diffusion denoising is stable,
- teacher guidance signal is not collapsed.

---

## Phase 2: Guided Synthetic Generation
Entrypoint:
- `python main_3class.py guide ...`

Main components:
- Reverse diffusion with classifier guidance to generate class-targeted synthetic graphs.
- Current 3-class focus uses synthetic AD and synthetic MCI generation.

Key artifacts:
- `results_guidance_3class/synthetic_ad.bin`
- `results_guidance_3class/synthetic_mci.bin`
- Phase-2 QA/diagnostics under `results_guidance_3class/...`

What this phase must guarantee:
- generated graphs follow class condition,
- diversity is sufficient,
- near-duplicates are controlled,
- unrealistic samples are rejected or downweighted.

---

## Phase 3: Synthetic Filtering
Entrypoint:
- `python main_3class.py filter --threshold_min ... --threshold_max ...`

Main components:
- Correlation-based filtering (realism + uniqueness window).
- Keep only synthetic samples that are neither too dissimilar nor near-duplicate.

Key artifacts:
- `results_guidance_3class/filtered_synthetic_ad.bin`
- `results_guidance_3class/filtered_synthetic_mci.bin`
- Phase-3 QA outputs (including embedding checks).

What this phase must guarantee:
- retained synthetic pool is plausible,
- minority sample retention is not over-pruned,
- filtering behavior is reproducible.

---

## Phase 4: Contrastive Pretraining
Entrypoint:
- `python main_3class.py pretrain ...`

Main components:
- GraphCL-style pretraining with StandardGCN encoder.
- Real + (optionally capped) synthetic graphs.
- Quality logging for representation behavior.

Key artifact:
- `gcn_pretrained_3class.pth` (or run-specific checkpoint)
- Phase-4 QA logs in configured quality directory.

What this phase must guarantee:
- no representation collapse,
- useful pretraining signal for downstream 3-class fine-tune,
- stable behavior across runs.

---

## Phase 5: 3-Class Fine-Tuning
Entrypoint:
- `python main_3class.py finetune ...`

Main components:
- Fine-tune StandardGCN classifier head (and optionally encoder unfreeze) on real + capped filtered synthetic.
- Evaluate with class-wise metrics and macro metrics.

Output metrics of interest:
- Accuracy
- Macro-F1
- AUC-ROC (OVR)
- Recall per class (CN/AD/MCI)

What this phase must guarantee:
- improvement over weak baseline behavior,
- better minority recall without collapsing CN/MCI,
- reproducible band, not one-off spikes.

---

## 4) Quality Assessment Framework Added So Far

## Phase-1 quality checks
- VAE reconstruction quality: MSE/MAE/SSIM-like summaries and class-wise errors.
- Latent quality: separability and drift indicators.
- Diffusion quality: denoising loss behavior by timestep buckets.
- Teacher quality: macro-F1, class recalls, confusion behavior, calibration-style signals.

Interpretation pattern seen:
- VAE/diffusion often stable,
- teacher collapse was the main Phase-1 risk and required iterative tuning.

## Phase-2 quality checks
- Teacher confidence distribution on synthetic outputs.
- Spectral plausibility checks (class-aware).
- Duplicate / near-duplicate checks (real-neighbor and intra-synthetic).
- Diversity stats and guidance trajectory diagnostics.
- Optional enforceable quality gates + fallback/min-keep controls.

Interpretation pattern seen:
- hard gating can over-prune and hurt downstream class balance if thresholds are too strict.

## Phase-3 quality checks
- Retention-rate diagnostics by class.
- Embedding-space real-vs-synthetic checks (including t-SNE artifact and kNN purity).
- Reproducibility check for fixed thresholds.

Interpretation pattern seen:
- filtering can be reproducible but still leave alignment gaps in embedding space.

## Phase-4 quality checks
- Contrastive loss trends.
- Positive/negative similarity trends.
- Embedding spread/collapse indicators.

## Phase-5 evaluation checks
- Macro-F1/AUC + class recalls under fixed synthetic caps and class-weight modes.
- Repeat runs to estimate stability band, not single run claims.

---

## 5) Current Status Snapshot (from logged runs)

Important context:
- Multiple experimental windows exist; metrics vary by synthetic caps and weighting strategy.
- A prior reference run from the project showed around Macro-F1 ~0.46 and AUC ~0.78 band.
- Later phase-wise tuning runs reached higher macro scores in some settings (around mid-0.5 band), but reproducibility and AD/MCI balance remained a tuning challenge.

Current practical takeaway:
- Pipeline is operational and phase-wise instrumented with QA.
- Stability improved versus earlier uncontrolled runs.
- Remaining challenge is consistent joint optimization of:
  - Macro-F1,
  - AD recall,
  - MCI recall,
  - and reproducibility band.

---

## 6) Constraints That Must Be Respected
These are enforced project constraints and should remain unchanged unless explicitly approved:
- Do **not** use Focal Loss.
- Do **not** introduce Supervised Contrastive Learning.
- Do **not** modify Youden-threshold logic.
- Do **not** switch to GAT; keep **StandardGCN**.
- Keep older options/functionality available; add features as configurable extensions.
- Synthetic data must not contaminate real test evaluation.

---

## 7) Reproducibility and Seeding Policy
- Default seed in 3-class CLI is set to `100`.
- Seed fixing is used to reduce variance during tuning comparisons.
- Even with fixed seed, some variation is expected from end-to-end stochastic training; repeated runs are required for stable claims.

---

## 8) Recommended Next Actions (High Impact, Low Risk)
1. Keep Phase-1/2/3 artifacts fixed while running controlled Phase-5 micro-sweeps.
2. Use selection rule with AUC guardrail + Macro-F1 ranking + AD/MCI recall tie-break.
3. Add repeat validation for top configs (minimum 2 repeats) before declaring winner.
4. Continue embedding QA tracking, but do not over-optimize t-SNE visually at the cost of end metrics.
5. Record every run’s config + metrics in a single run registry for faster supervisor reporting.

---

## 9) Fast Start Commands (3-Class)
Examples (adjust args as needed):

```bash
# Phase 1
python main_3class.py --seed 100 train

# Phase 2
python main_3class.py --seed 100 guide --scale_ad 2.0 --scale_mci 2.0

# Phase 3
python main_3class.py --seed 100 filter --threshold_min 0.45 --threshold_max 0.99

# Phase 4
python main_3class.py --seed 100 pretrain --epochs 80 --pretrain_syn_ad_cap 100 --pretrain_syn_mci_cap 161

# Phase 5
python main_3class.py --seed 100 finetune --epochs 30 --unfreeze --max_syn_ad 120 --max_syn_mci 161 --loss_class_weight_mode inverse
```

---

## 10) One-Line Abstract (for handoff)
This project develops a phase-wise, quality-gated synthetic-augmentation pipeline for 3-class ADNI brain graph classification, combining latent diffusion generation with contrastive pretraining and StandardGCN fine-tuning to improve minority-class performance under strict reproducibility and research-validity constraints.
