# Plan: Updating R2-Dreamer to NE-Dreamer

This document describes the concrete steps to extend the existing R2-Dreamer
codebase to implement **NE-Dreamer** ("Next Embedding Prediction Makes World
Models Stronger", Bredis et al., 2026; see `NEDreamer.md` / `NEDreamer.pdf`).

The strategy is to add NE-Dreamer as a new `rep_loss` option alongside
`dreamer`, `r2dreamer`, `infonce`, and `dreamerpro`, so existing experiments
continue to run unchanged. Everything except the world-model representation
objective stays the same (RSSM, reward/continuation heads, actor-critic,
optimizer, training loop).

## 1. What changes vs. R2-Dreamer

R2-Dreamer aligns the *current-step* RSSM feature with the *current-step*
encoder embedding via Barlow Twins on a per-timestep projector.

NE-Dreamer changes only the representation objective (Sec. 3.3 of the paper):

1. A **causal temporal transformer** `T_θ` consumes the history
   `(h_{≤t}, z_{≤t}, a_{≤t})` and predicts the **next-step** encoder embedding
   `ê_{t+1}`.
2. The target is `e*_{t+1} = sg(f_enc(x_{t+1}))` (stop-gradient on the encoder
   output of the next observation).
3. Predicted and target embeddings are aligned with a **Barlow Twins**
   redundancy-reduction loss, computed only over **valid transitions**
   (`c_t = 1`, i.e. non-terminal pairs).
4. Pixel reconstruction is **removed**; the world model loss becomes
   `L_wm = L_rew + L_cont + β_kl L_kl + β_ne L_NE`.

Architecture identity: Eq. (5)-(12) of the paper. Imagination horizon `H=15`,
KL stabilizers, and actor-critic are unchanged from DreamerV3 / R2-Dreamer.

## 2. Code changes

### 2.1 `networks.py` — add the causal temporal predictor

Add a new module `NextEmbeddingPredictor` (name TBD) consisting of:

- A lightweight **input projector** (single Linear+RMSNorm+Act) mapping the
  per-step token `[stoch_t flattened, deter_t, prev_action_t]` into the
  transformer hidden width. This is the "projector" that the ablation
  *w/o projector* removes (Fig. 4 in paper).
- A small **causal transformer** stack:
  - `n_layers` (default 2-4) of pre-norm transformer blocks with multi-head
    self-attention and an MLP block.
  - **Causal mask** so position `t` only attends to `≤ t`.
  - Learned positional embeddings sized to `batch_length` (64).
  - Hidden dim aligned with `model.units` (e.g. 512 for size12M).
- An **output head** (Linear) projecting from transformer hidden dim to
  `embed_size` (= `MultiEncoder.out_dim`).

The forward pass takes `(stoch, deter, action)` of shape `(B, T, ...)` and
returns predicted next-step embeddings of shape `(B, T, E)` where position
`t` is the prediction of `e_{t+1}`.

Notes:
- The input at position `t` should already include `prev_action = a_{t-1}`
  to remain consistent with how RSSM represents the recurrent step. The
  prediction at `t` is for the embedding at `t+1`.
- Use the same RMSNorm / SiLU style already used in `MLP` / `Deter` to keep
  the codebase consistent and `torch.compile`-friendly.

### 2.2 `dreamer.py` — wire NE-Dreamer into `Dreamer`

Add a new branch alongside the existing `rep_loss` branches:

1. **Constructor (`__init__`)**: when `config.rep_loss == "nedreamer"`:
   - Instantiate `self.ne_predictor = NextEmbeddingPredictor(...)`.
   - Store hyperparams: `self.barlow_lambd`, optionally
     `self.use_transformer`, `self.use_shift`, `self.use_projector` for the
     three ablations.
   - Add `ne_predictor` to the `modules` dict that feeds the optimizer.
   - Do **not** create a decoder.
2. **`_cal_grad`**: add a new representation-loss branch:
   - Compute `embed = encoder(data)` (already done above).
   - Run RSSM `observe` to get `post_stoch`, `post_deter` (already done).
   - Build per-step transformer inputs from `(post_stoch, post_deter, action)`
     and run `ne_predictor` to get `ê[:, :T-1]` (predictions for positions
     `1..T-1`).
   - Build `e*[:, 1:T] = embed[:, 1:T].detach()`.
   - Build a **validity mask** from `data["is_terminal"]` (or
     `1 - is_last`): only include `(b, t)` such that the transition `t→t+1`
     is non-terminal. The paper uses `c_t = 1` (Eq. 10). This mask gates the
     Barlow Twins per-dimension normalization and the cross-correlation.
   - Compute Barlow Twins loss exactly as in the existing R2-Dreamer branch
     but on `(predicted, target)` instead of `(projected_feat, embed)`, and
     only over valid `(b, t)`. Reuse the diag/off-diag formulation:
     `loss = Σ_i (1 - C_ii)^2 + λ_BT Σ_{i≠j} C_ij^2`.
   - Store under a new key, e.g. `losses["ne"] = ...`, and add a matching
     `loss_scales.ne` in the config (paper's `β_ne`).
3. **`video_pred`**: keep guarded behind `rep_loss == "dreamer"`. NE-Dreamer
   has no decoder by design, so this method should raise / return early as
   it already does for non-`dreamer` losses.

### 2.3 Ablation switches (Sec. 4.3 of paper)

Expose three flags under `model.nedreamer.*` in the config:

- `use_transformer: True` — when False, replace the causal transformer with
  a shallow MLP applied per timestep (the paper's *w/o transformer*).
- `use_shift: True` — when False, predict the *current-step* embedding
  `e_t` instead of `e_{t+1}` (paper's *w/o shift*); the alignment then
  collapses to per-timestep matching, similar to R2-Dreamer but with the
  transformer in front.
- `use_projector: True` — when False, skip the input projector and feed the
  raw concatenated `(stoch, deter, action)` to the transformer.

These flags should map directly onto branches in
`NextEmbeddingPredictor.forward` and the loss assembly in `_cal_grad` so
the three ablations from Figure 4 are reproducible by config alone.

### 2.4 `configs/model/_base_.yaml`

Add:

```yaml
rep_loss: "nedreamer"  # default switched to nedreamer once validated

loss_scales:
  ...
  ne: 1.0   # β_ne; paper does not specify, start at 1.0 and tune

nedreamer:
  use_transformer: True
  use_shift: True
  use_projector: True
  transformer:
    layers: 4          # tune; small to keep at 12M params
    heads: 4
    hidden: ${model.units}
    dropout: 0.0
    pos_embed: "learned"
    max_len: ${batch_length}
  proj_hidden: ${model.units}
  barlow_lambd: 5e-4   # reuse R2-Dreamer's value as starting point
```

Keep the `r2dreamer.lambd` block; NE-Dreamer can reuse the same Barlow
machinery without sharing the config key.

### 2.5 `configs/configs.yaml`

No structural change, but verify `batch_length: 64` (already true) is
sufficient as the transformer's max sequence length; if larger contexts are
ever needed, `nedreamer.transformer.max_len` is the only knob to update.

### 2.6 Parameter budget

The paper compares everything at **~12M params** (`size12M.yaml`,
"Dreamer-S"). Tune `nedreamer.transformer.layers` and `heads` so that the
total parameter count for the NE-Dreamer variant lands within ±5% of the
R2-Dreamer 12M baseline. Use the per-module parameter print at the end of
`Dreamer.__init__` to verify before launching long runs.

## 3. New environment support: DMLab Rooms

The paper's headline claim (C1) is on four DMLab Rooms tasks:

- `rooms_collect_good_objects_train`
- `rooms_exploit_deferred_effects_train`
- `rooms_select_nonmatching_object`
- `rooms_watermaze`

The current repo has no DMLab support (`envs/` covers DMC, Atari, Crafter,
MetaWorld, MemoryMaze). To reproduce C1 we need:

1. **`envs/dmlab.py`** — a new environment wrapper analogous to
   `envs/atari.py` / `envs/memorymaze.py`. DMLab is built on
   `deepmind_lab`; production-grade wrappers exist in DreamerV3 and IRIS
   that we can adapt. The wrapper should:
   - Expose the standard `(image, action, reward, is_first, is_last,
     is_terminal)` interface used by `buffer.py` and `trainer.py`.
   - Emit 64×64 RGB observations and the discrete action set used by
     DMLab Rooms.
   - Honor `action_repeat` and `eval_episode_num` like other envs.
2. **`configs/env/dmlab.yaml`** — a Hydra env config matching the paper's
   protocol: 50M env steps, 5 seeds, train_ratio matching DreamerV3's DMLab
   recipe (typically 64-128). Include the four task names above as valid
   `env.task` values.
3. **`Dockerfile` / `requirements.txt`** — add the system deps for
   `deepmind_lab` (it requires Bazel + a non-trivial build); document this
   in `docs/dmlab.md` since most users won't have it preinstalled.

If reproducing DMLab is out of scope for the first PR, the algorithmic
changes in §2 can land first and be validated on MemoryMaze (also a
memory/navigation benchmark already supported here), followed by a second
PR adding DMLab. **Recommended:** ship §2 first, run the C3 calibration on
DMC Vision to confirm no regression, then add DMLab in a follow-up.

## 4. Representation diagnostic (paper Fig. 5, optional)

The paper trains a **post-hoc decoder** on frozen NE-Dreamer latents to
visualize what is encoded. This is purely diagnostic and not used during
training. To support it:

- Add a flag `trainer.posthoc_decoder: False`.
- When enabled (typically only at eval / after training), instantiate a
  fresh `MultiDecoder` (the same class already in `networks.py`), train it
  for a short number of steps on the replay buffer with the world model
  frozen, and log reconstructions via the existing `video_pred`-style code
  path. Lives behind a flag so it does not affect the headline runs.

This is nice-to-have, not required to validate the method.

## 5. Validation plan

In order:

1. **Smoke test.** `python3 train.py model.rep_loss=nedreamer
   env=dmc_vision env.task=dmc_walker_walk trainer.steps=2e4` — confirm it
   trains without NaNs, parameter count lands near 12M, and loss curves
   look sane.
2. **C3 (no-regression on DMC).** Run NE-Dreamer on the same DMC Vision
   tasks the existing R2-Dreamer is validated on (1M steps, 5 seeds).
   Expectation per Fig. 6: matches DreamerV3 / R2-Dreamer.
3. **Ablations.** With NE-Dreamer otherwise identical, sweep
   `nedreamer.use_transformer`, `nedreamer.use_shift`,
   `nedreamer.use_projector` — confirm the qualitative ordering from
   Fig. 4 (transformer + shift dominate; projector minor).
4. **C1 (DMLab Rooms).** Once §3 lands, run the four Rooms tasks at 50M
   steps, 5 seeds. Expectation per Fig. 3: substantial gains over
   R2-Dreamer / DreamerV3 / DreamerPro.

## 6. Documentation and cleanup

- Update `README.md`:
  - Title block and abstract paragraph — keep R2-Dreamer billing but add
    "and NE-Dreamer" once §2 lands.
  - Add `nedreamer` to the `model.rep_loss` enumeration.
  - Add NE-Dreamer to the Citation section.
- Add `docs/nedreamer.md` summarizing the new objective (this plan can be
  trimmed into it once implementation is done).
- Keep `docs/tensor_shapes.md` accurate: add the predictor's `(B, T, E)`
  output shape and the per-step token shape.

## 7. Out-of-scope / open questions

- The paper does not specify `β_ne`, transformer depth, or the exact
  Barlow scaling at the new sequence-of-pairs granularity. Treat these as
  hyperparameters to tune in step §5.1.
- DMLab build tooling is heavy. If we want broad reproducibility we may
  want to package a prebuilt `deepmind_lab` wheel or container, separate
  from the main `Dockerfile`.
- Whether to switch the repo's default `rep_loss` from `r2dreamer` to
  `nedreamer` is a release decision to make once §5 is green; keep
  `r2dreamer` as default until then.

## 8. Suggested PR sequencing

1. **PR 1**: §2 (algorithm + config) + §5.1-§5.2 (smoke + DMC). README
   updates: mention NE-Dreamer as an additional option only.
2. **PR 2**: §2.3 ablation flags + reruns of §5.3.
3. **PR 3**: §3 DMLab env + §5.4 Rooms benchmarks.
4. **PR 4** (optional): §4 post-hoc decoder diagnostic.
