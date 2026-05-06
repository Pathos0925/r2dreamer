# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides a PyTorch implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a computationally efficient world model that achieves high performance on continuous control benchmarks. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Installing via a virtual env like uv is recommended.
pip install -r requirements.txt
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro|nedreamer
python3 train.py model.rep_loss=r2dreamer
```

`nedreamer` (NE-Dreamer, Bredis et al., 2026) replaces the pixel decoder
with a causal temporal transformer that predicts the next-step encoder
embedding and aligns it to a stop-gradient target via Barlow Twins. Hyper-
parameters live under `model.nedreamer.*`; the three Sec. 4.3 ablations are
exposed as `model.nedreamer.use_transformer`, `model.nedreamer.use_shift`,
and `model.nedreamer.use_projector`. The implementation plan is in
[`docs/nedreamer_plan.md`](docs/nedreamer_plan.md).

### Curious Replay

[Curious Replay](https://arxiv.org/abs/2306.15934) (Kauvar & Doyle et al.,
ICML 2023) is available as a prioritized sampling option, orthogonal to the
choice of `rep_loss`. Enable with:

```bash
python3 train.py model.rep_loss=nedreamer model.curious_replay.enabled=True env=crafter
```

The buffer's per-transition priority follows Eq. 1 of the paper:

```
p_i = c * beta^v_i + (|L_i| + eps)^alpha
```

where `v_i` is the visit count and `L_i = |dyn + rew + cont|` is the
per-step world-model loss (computed in `dreamer.py:_cal_grad` and threaded
back through `buffer.update_priority`). All five `c, beta, alpha, eps,
p_max` knobs are exposed under `model.curious_replay.*` with the paper's
defaults.

### Validation: Atari-100k (size12M, 5 seeds collapsed to 1, 410k env steps each)

Trained on a single A100 80GB; ~380 env-steps/sec with `model.compile=True`,
~25 minutes wall-clock per game. 3 eval episodes per checkpoint
(`env.eval_episode_num=3`, `trainer.eval_every=2e4`).

| Game     | Init eval | Best eval        | Final eval (400k) | Notes |
|----------|-----------|------------------|-------------------|-------|
| Pong     | -20.7     | **-11.3** @ 400k | **-11.3**         | Monotonic improvement; `loss/ne` 1025 → 40 |
| Breakout |   0.0     | **7.3**  @ 380k  |   3.7             | Eval oscillates between scoring and the no-FIRE time-limit stall (a known Atari-100k Breakout pathology when `autostart: False`) |
| Boxing   | -11.0     | **65.7** @ 380k  | **61.0**          | Strong learning; agent dominates the bot late in training |

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
