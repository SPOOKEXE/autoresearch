# Experiment Plan — Brain-Inspired Learning Mechanisms

**Branch**: `autoresearch/mar18` → continued on `mar18_2`
**Goal**: Reduce `val_bpb` (lower is better) by porting biological learning mechanisms into the GPT training loop.
**Reference**: `Backpropagation Explained.md` (predictive coding, Hebbian learning, metaplasticity, feedback alignment, STDP).

---

## Experiment Queue

Experiments are run sequentially. If an experiment improves val_bpb it becomes the new baseline; otherwise it is reverted.

### [x] Experiment 0 — Baseline
Run `train.py` as committed on `brain-learning` (no changes). Establishes the reference val_bpb.

**Result**: val_bpb = **1.122594** (commit `5e1bef6`, 6.0 GB VRAM)

---

### [x] Experiment 1 — Hebbian Query-Value Gate
**Hypothesis**: The value-embedding gate currently ignores query content. A Hebbian gate that amplifies value residuals when query and value embedding are semantically aligned should improve attention routing.

**Final approach** (after α-sweep): Additive Hebbian potentiation — `v += α * sim * ve_4d` where `sim` is the detached cosine similarity between pre-RoPE query and value embedding. Original gate unchanged (no param count change).

**α sweep results**:
| α | val_bpb | notes |
|---|---------|-------|
| — | 1.122594 | baseline |
| 0.5 | 1.127713 | too strong |
| 0.25 | 1.123053 | approaching baseline |
| **0.1** | **1.121889** | **beats baseline ✓** |
| 0.05 | 1.128514 | too weak |

**Result**: val_bpb = **1.121889** (commit `b762d6b`, −0.0007 vs baseline) — **KEPT**

Earlier rejected variants: augmented gate input (1.125), multiplicative gate modulation (1.127).

---

### [ ] Experiment 2 — Predictive Coding Auxiliary Loss
**Hypothesis**: Early transformer layers receive weak gradient signal due to long backprop paths. A local auxiliary loss (block i predicts block i+1's normalized output) provides direct gradient to each layer.

**Change**: Add 7 `PCPredictor` modules (bottleneck n_embd//4). Each block predicts the next block's RMSNorm output. Target is `.detach()`-ed to prevent representation collapse. Loss: `L_total = L_LM + 0.05 × mean(MSE_i)`. Only active during training (`self.training` guard). Optimizer assertion removed; pc_params added to new AdamW group.

**Risk**: Medium. ~5% compute overhead, ~9% param overhead. Stop-gradient is non-negotiable.

**Expected effect**: Improved early-layer representations, especially for an 8-layer model where gradient attenuation is significant.

---

### [ ] Experiment 3 — Metaplastic Loss Modulation
**Hypothesis**: Surprise-based gradient scaling lets the model learn more from unexpected inputs and less from familiar patterns, improving generalization.

**Change**: Track EMA of training loss. Scale effective loss by `clamp(surprise^1.0, 0.7, 1.4)` where `surprise = loss / EMA`. Activates after 100 warmup steps. Pure training-loop change, no architecture changes. Note: Muon is invariant to gradient magnitude (polar orthogonalization), so this primarily affects AdamW-updated params (embeddings, lm_head, scalars).

**Risk**: Low. Gradient scale bounded to [0.7, 1.4]. No architecture changes.

**Expected effect**: Modest improvement, mainly from better embedding/lm_head training dynamics.

---

### [ ] Experiment 4 — Combined (if 1+2+3 all individually succeed)
Run all three mechanisms together on the same base. Expected synergy: Hebbian gating improves attention, PC loss improves layer representations, metaplasticity modulates the embedding/head optimization.

---

### Fallback Experiments (if above fail)

**4a — Reduce PC_WEIGHT**: If predictive coding hurts, try `PC_WEIGHT = 0.01` (weaker auxiliary signal).

**4b — Hebbian gate only, remove PC**: If PC adds too much overhead or diverges, try Hebbian gate alone at higher MATRIX_LR.

**4c — Feedback alignment**: Replace backprop through attention with fixed random feedback matrices on the projection layer. Simpler gradient path; interesting if standard backprop creates pathologies.

**4d — STDP-inspired temporal credit**: Weight gradient contributions by position distance (recent tokens get higher credit). Modify loss with a position-dependent reweighting.

**4e — Architectural: deeper + narrower**: With PC loss providing local gradients, depth may matter more. Try DEPTH=12 with ASPECT_RATIO=48 (same total param budget).

**4f — Architectural: remove value embeddings, replace with PC**: If PC loss captures the same inductive bias as value embeddings (residual information flow), simplify by removing VE and relying on PC.

---

## Decision Criteria

| Improvement | Action |
|------------|--------|
| val_bpb decreases | Keep commit, proceed to next experiment |
| val_bpb unchanged | Apply simplicity criterion — keep if code is simpler, discard otherwise |
| val_bpb increases | Revert commit (`git reset --hard HEAD~1`) |
| Crash/OOM | Fix and retry or skip; log as crash |

**Simplicity criterion**: A 0.001 improvement that adds 20 lines of complexity is not worth keeping. Deleting code that achieves equal/better results is always a win.

---

## Notes

- `RESUME_FROM_CHECKPOINT = False` is set in Experiments 1–3 (architecture changes).
- All three mechanisms are designed to be graph-traceable under `torch.compile(dynamic=False)`.
- `prepare.py` is never modified. `evaluate_bpb` in `prepare.py` is the ground truth metric.
- The `self.training` guard ensures val_bpb evaluation uses only the LM loss (no PC overhead).
