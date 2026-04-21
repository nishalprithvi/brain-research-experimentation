# ADNI 3-Class Pipeline: Full Context, Status, and Redesign Handoff

Last updated: 2026-04-21
Working repo path: `/home/msai/prithvi005/brain_research/experimentation`
Primary branch in use: `featureBranch`

---

## 1. Project Goal and Problem Setting

### 1.1 Main objective
Build a robust **3-class brain-graph classifier** for ADNI subjects:
- Class 0: CN (Cognitively Normal)
- Class 1: AD (Alzheimer's Disease)
- Class 2: MCI (Mild Cognitive Impairment)

The project uses **synthetic graph augmentation** to address class imbalance and improve minority-class recall, especially AD and MCI.

### 1.2 Core difficulty
- Severe class imbalance in real data (AD minority)
- Synthetic quality must be high enough to help, not hurt
- Pipeline has multi-stage stochastic behavior, so reproducibility is non-trivial

---

## 2. Constraints (Hard Rules to Respect)

These were repeatedly enforced across iterations:
- Do **not** use Focal Loss
- Do **not** use Supervised Contrastive Learning
- Do **not** modify Youden threshold logic
- Do **not** use GAT; keep **StandardGCN** path
- Preserve prior functionality; add options/configs instead of replacing old behavior
- Keep synthetic samples out of final real-test evaluation contamination

---

## 3. Data and Baseline Snapshot

### 3.1 Real dataset snapshot (baseline run)
From baseline run logs:
- Total valid labeled graphs: 1254 (after dropping invalid labels)
- Full distribution: CN 819, AD 65, MCI 370
- Split (baseline run): train 1003, test 251

### 3.2 Baseline (real-only, StandardGCN, no synthetic, no contrastive)
Reference run: `job_logs/baseline_3class_18960.out`
- Accuracy: **0.7331**
- Macro-F1: **0.4704**
- AUC-OVR: **0.7253**
- Recall CN/AD/MCI: **0.8598 / 0.0000 / 0.5811**

This is the key real-only baseline anchor.

---

## 4. Pipeline Overview (End-to-End)

Entrypoint CLI: `main_3class.py`

Phase mapping:
1. `train` -> Phase 1 (VAE + Diffusion + Teacher)
2. `guide` -> Phase 2 (Guided synthetic generation + quality checks)
3. `filter` -> Phase 3 (Correlation-window filtering)
4. `pretrain` -> Phase 4 (Contrastive pretraining)
5. `finetune` -> Phase 5 (Final supervised StandardGCN)

Canonical run scripts (active/useful):
- `run_3class_pipeline.sh`
- `run_frozen_pipeline_3class.sh`
- `run_baseline_3class.sh`
- `run_phase1_quality_3class.sh`
- `run_phase2_quality_3class.sh`

---

## 5. Phase-by-Phase Methodology and Implementation

## Phase 1: Generative Training (VAE + Latent Diffusion + Teacher)

### 5.1 What Phase 1 does
- Trains VAE to encode/decode brain graph matrices into latent space
- Trains diffusion model in latent space
- Trains teacher classifier in latent space to provide guidance in Phase 2

Primary file: `src/train_3class.py`

### 5.2 Teacher model options
- `latent_densegcn`
- `latent_mlp`

Teacher controls introduced:
- class weighting modes (`none`, `inverse`, `sqrt_inverse`, `effective`)
- optional balanced sampler
- anti-collapse regularization
- early stopping on macro-F1 style quality signal

### 5.3 Dense vs sparse teacher trials (Phase-1 internal quality, not E2E)
Kept comparison runs:
- `phase1_20260408_024515_seed_100` (MLP+dense)
- `phase1_20260408_024656_seed_100` (MLP+sparse)
- `phase1_20260408_024916_seed_100` (GCN+dense)
- `phase1_20260408_025055_seed_100` (GCN+sparse)

Observed best macro-F1 per setup (phase-1 teacher quality summary):
- MLP + dense: **0.5049**
- MLP + sparse: **0.4196**
- GCN + dense: **0.1518**
- GCN + sparse: **0.1518**

Interpretation in this project:
- On compressed latent (16x16) input, MLP was more stable than latent DenseGCN
- Sparse latent variant did not help in this test window

### 5.4 Phase-1 quality metrics implemented
- Teacher: macro-F1, class recalls, ECE/Brier/entropy, confusion behavior
- VAE: val MSE/MAE, class-wise MSE, latent drift/stability metrics
- Diffusion: bucket-wise denoising MSE and noise correlation

### 5.5 Key Phase-1 observations
- VAE and diffusion were generally stable
- Major risk was teacher collapse in some settings
- Teacher quality strongly controls Phase-2 synthetic quality downstream

---

## Phase 2: Guided Synthetic Generation

Primary file: `src/guided_sampling_3class.py`

### 6.1 What Phase 2 does
- Uses `vae_3class.pth`, `diffusion_3class.pth`, `gcn_3class.pth`
- Generates class-conditional synthetic AD and MCI samples via guided reverse diffusion
- Saves synthetic graph binaries:
  - `results_guidance_3class/synthetic_ad.bin`
  - `results_guidance_3class/synthetic_mci.bin`

### 6.2 Generation logic (core)
For each reverse timestep:
- UNet predicts noise
- Teacher predicts logits from estimated latent
- CE loss to target class is differentiated wrt latent
- Guidance term adjusts denoising trajectory (`guidance_scale`)

### 6.3 Quality checks implemented in Phase 2
Quality report function computes per class:
1. Teacher confidence distribution and pass rate
2. Real-duplicate correlation risk (syn vs real)
3. Intra-synthetic duplicate risk
4. Spectral plausibility (top-k eigen profile distance)
5. Diversity stats (pairwise distances)
6. Topology checks (efficiency, clustering, path length, connectedness)
7. Guidance trajectory monotonicity + plots
8. Gate summary (`pass_all`)

### 6.4 Spectral plausibility details
Implementation behavior:
- sanitize adjacency matrix (symmetrize, nonnegative, zero diagonal)
- compute eigenvalues with symmetric eigensolver
- sort descending, keep top-k (default k=10)
- class-mean real spectrum computed from real class pool
- synthetic distance: L2 distance to class mean spectrum
- threshold (`tau`) = fixed input if provided, else recommended from real-class distribution percentile

### 6.5 Current retained Phase-2 quality snapshot
Retained snapshot folder:
- `results_guidance_3class/phase2_quality/20260408_023757`

Reported signals in that strict snapshot:
- AD mean target probability ~0.1517
- MCI mean target probability ~0.4418
- pass-all-gates collapsed to 0 in strict settings

Interpretation:
- Quality checks are informative and working
- Gate strictness + guidance quality can over-prune minority synthetics

### 6.6 Phase-2 artifacts currently kept (active)
- `synthetic_ad.bin`
- `synthetic_mci.bin`
- `filtered_synthetic_ad.bin`
- `filtered_synthetic_mci.bin`

Note: duplicate backup binaries `*_seed_100.bin` were removed to save space.

---

## Phase 3: Synthetic Filtering + Embedding QA

Primary file: `src/filter_synthetic_3class.py`

### 7.1 What Phase 3 does
- Applies class-conditional similarity window filtering with (`threshold_min`, `threshold_max`)
- Removes synthetic samples that are too dissimilar (unrealistic) or too similar (near-clones)

### 7.2 Quality checks used around Phase 3
- t-SNE real-vs-synthetic embedding visualization
- kNN purity in embedding space
- centroid distance (real vs synthetic) per class

Kept Phase-3 quality artifacts:
- `results_guidance_3class/phase3_quality/phase3_tsne_real_vs_syn.png`
- `results_guidance_3class/phase3_quality/phase3_embedding_metrics.txt`

Current kept metrics in the file:
- kNN purity (k=10): **0.4542**
- centroid distance real-vs-syn [AD]: **43.3938**
- centroid distance real-vs-syn [MCI]: **43.0581**

Interpretation:
- Embedding QA is functioning and reproducible
- Real/synthetic alignment gap remains significant

---

## Phase 4: Contrastive Pretraining

Primary file: `src/train_contrastive_3class.py`

### 8.1 What is done
- SimCLR-style graph representation pretraining before final finetune
- Options to include or skip synthetic data in pretrain
- Synthetic caps (AD/MCI) for controlled pretrain composition

### 8.2 Added quality indicators
- contrastive loss trend
- positive/negative similarity trend
- embedding spread/collapse indicators

### 8.3 Observed behavior
- Representation pretraining works, but downstream gains depend heavily on synthetic quality from phases 1-3

---

## Phase 5: Final Fine-tuning and Evaluation

Primary file: `src/finetune_3class.py`

### 9.1 What Phase 5 does
- StandardGCN supervised training on real + selected synthetic
- evaluates accuracy, macro-F1, AUC-OVR, class recalls

### 9.2 Introduced controls and upgrades
- differential LR between encoder/head
- LayerNorm integration after pooling locations
- class-weighted CE paths
- EMA checkpointing (stability)
- dropout in classifier head
- optional edge masking
- model-selection guardrails (including AUC thresholds)
- synthetic selection controls and caps

### 9.3 Current practical behavior
- Can reach better-than-baseline zones in some runs
- Reproducibility/variance is still a key challenge
- AD/MCI tradeoff remains sensitive to synthetic retention and phase coupling

---

## 10. Metrics Landscape (Targets vs Current)

### 10.1 Internal target ladder (used in status reporting)
From current planning docs:
- Phase-1 targets (intermediate): Acc >= 0.74, Macro-F1 >= 0.58, AUC >= 0.81, AD Recall >= 0.70, MCI Recall >= 0.40
- Phase-2 targets: Acc >= 0.78, Macro-F1 >= 0.68, AUC >= 0.84, AD Recall >= 0.82, MCI Recall >= 0.55
- Phase-3 targets: Acc >= 0.80, Macro-F1 >= 0.78, AUC >= 0.86, AD Recall >= 0.90, MCI Recall >= 0.63

### 10.2 Higher publication ambition (SOTA-level direction)
Commonly discussed top-tier direction:
- Accuracy > 0.85
- Macro-F1 > 0.75
- AUC-OVR > 0.90
- AD Recall > 0.65
- MCI Recall > 0.60
(and aspirationally beyond that if methodically justified)

### 10.3 Current known anchors
- Real-only baseline: Macro-F1 0.4704, AUC 0.7253, AD recall 0.0
- Full pipeline had best single-run windows in prior cycles (historically referenced) above baseline but with variance
- Current retained logs emphasize stable references and phase-quality diagnosis

---

## 11. Major Bottlenecks (Small + Major)

### 11.1 Upstream synthetic-quality bottlenecks
- Teacher guidance confidence is weak for AD in strict settings
- Gate strictness can collapse retained synthetic count
- Spectral/topology plausibility can reject too much minority data if untuned

### 11.2 Distribution alignment bottlenecks
- Embedding-space alignment gap remains large (centroid distance high)
- Local neighborhood class purity remains moderate

### 11.3 Pipeline-coupling bottlenecks
- Improvements in one phase can regress another phase
- Hard to optimize all metrics simultaneously (Macro-F1 vs AD recall vs MCI recall vs AUC)

### 11.4 Reproducibility bottlenecks
- Stochastic training variation still impacts final metrics
- Need repeated fixed-config evaluations before claiming stable improvements

### 11.5 Operational bottlenecks
- Git push from cluster currently blocked by auth setup
- Disk pressure mostly from synthetic artifact bins, not logs

---

## 12. Current Cleaned Storage State (Post-Cleanup)

Performed cleanup:
- Removed old temporary run scripts and stale logs
- Pruned old phase1 quality run directories (kept key compare set)
- Pruned old phase2 quality timestamp folders (kept latest)
- Removed duplicate synthetic backups (`*_seed_100.bin`)

Space after cleanup:
- `job_logs`: ~160K
- `results_phase1_quality`: ~872K
- `results_guidance_3class`: ~549M
- `reports`: ~11M

Primary remaining heavy folder is still `results_guidance_3class` due active synthetic bins.

---

## 13. What Is Currently Stable vs Unstable

### 13.1 Stable enough
- Baseline pipeline and baseline run reproducibility reference
- Phase-1 instrumentation and teacher comparison scaffolding
- Phase-2/3 quality-check infrastructure
- Phase-3 embedding QA outputs generation

### 13.2 Not fully stable yet
- End-to-end metric consistency at high target thresholds
- AD/MCI recall balance under strict gate settings
- Real/synthetic embedding alignment gap

---

## 14. Recommended Next Redesign Directions

Priority order:
1. **Phase-1 teacher quality hardening for AD guidance**
   - calibrate teacher confidence distribution (especially AD)
   - keep MLP+dense as practical default unless new evidence emerges

2. **Phase-2 guidance policy redesign**
   - class-specific guidance schedules (AD stronger late-step guidance)
   - enforce min-keep/fallback without over-trusting low-confidence samples
   - targeted regeneration for failed quality bins, not blanket regeneration

3. **Phase-3 alignment-aware filtering**
   - use embedding QA signals as tuning objective, not only pass/fail gates
   - optimize for lower centroid distance while preserving diversity

4. **Phase-5 controlled sweeps under fixed upstream snapshots**
   - avoid moving upstream artifacts while tuning downstream classifier
   - evaluate repeated runs for stability bands, not single-run peaks

---

## 15. Files to Use as Ground-Truth Context for New Agent Handoff

Core context docs:
- `PROJECT_CONTEXT_FOR_NEW_AGENT.md`
- `observation.txt`
- `reports/progress_report_phases1to5.tex`
- `reports/PROJECT_PIPELINE_FULL_CONTEXT.md` (this file)

Core code entrypoints:
- `main_3class.py`
- `src/train_3class.py`
- `src/guided_sampling_3class.py`
- `src/filter_synthetic_3class.py`
- `src/train_contrastive_3class.py`
- `src/finetune_3class.py`
- `src/baseline_3class.py`

Reference logs retained:
- `job_logs/baseline_3class_18960.out`
- `job_logs/frozen_3class_14900.out`
- `job_logs/e2e_p2_base_13959.out`
- `job_logs/e2e_p2_gateA_13960.out`
- `job_logs/p5_stability_v1_19129.out`

---

## 16. Important Git Status Note

Current state known at time of writing:
- local commit exists on `featureBranch` but push from tc2 is blocked due auth mismatch (HTTPS creds / SSH key)
- one local modified file still present in worktree: `test_indices_3class.npy`

This should be decided before final remote sync:
- keep change and commit intentionally, or
- restore it to tracked version.

---

## 17. Executive Summary

The project has evolved from a baseline StandardGCN to a multi-phase synthetic-augmentation pipeline with substantial quality instrumentation. The strongest current technical value is not just metric peaks, but a clear diagnostic framework across phases 1-3. The central unresolved issue remains synthetic-real alignment and AD-focused guidance quality under strict validity constraints. The next redesign should focus on controlled, phase-coupled optimization rather than broad unstructured sweeps.
