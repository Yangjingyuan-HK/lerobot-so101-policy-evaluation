# LeRobot SO-101 Policy Evaluation

> 机械臂算法比较 — Benchmarking state-of-the-art imitation learning & Vision-Language-Action policies on the SO-101 robotic arm for real-world manipulation tasks.

<p align="center">
  <img alt="LeRobot" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

---

## 🤖 Can the Robot Learn to Pick & Place?

**Task:** Grasp the red block and drop it into the box.

The SO-101 follower arm is controlled entirely by a neural policy — no hard-coded motion planning, no traditional inverse kinematics tricks. Everything you see below is pure *learning-from-demonstration* running on real hardware.

Four state-of-the-art policies, one identical dataset, one identical robot.
Which one actually **closes the sim-to-real gap**?

---

## 🎬 Real-World Rollouts (Side-by-Side)

<table>
  <tr>
    <th align="center" width="25%"><b>ACT</b> — Action Chunking Transformer</th>
    <th align="center" width="25%"><b>PI0-Fast</b> — VLA Foundation Model</th>
    <th align="center" width="25%"><b>Diffusion Policy</b> — Denoising Diffusion</th>
    <th align="center" width="25%"><b>Multi-Task DiT</b> — Flow-Matching DiT</th>
  </tr>
  <tr>
    <td align="center"><img src="./media/readme/ACT.gif"       width="100%"></td>
    <td align="center"><img src="./media/readme/PI0.gif"       width="100%"></td>
    <td align="center"><img src="./media/readme/DIFFUSION.gif" width="100%"></td>
    <td align="center"><img src="./media/readme/MULTI-TASK.gif" width="100%"></td>
  </tr>
</table>

---

## 🏆 Current Benchmark Status (2026-08)

> 📍 **The bar:** reach → grasp → lift → transport → release into the box.

| Policy | Paradigm | Reaches Target | Grasps Block | Lifts & Moves | Drops into Box | Success Level |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ACT** | CVAE + Transformer (from scratch) | ✅ | ✅ | ✅ | ❌ | **3 / 5** — partial success (grasps, cannot release) |
| **PI0-Fast** | VLA (fine-tuned last 4 layers) | ⚠️ | ❌ | — | — | **1 / 5** — approaches but misses the grasp |
| **Diffusion Policy** | Denoising Diffusion (from scratch) | ⚠️ | ❌ | — | — | **1 / 5** — approaches but misses the grasp |
| **Multi-Task DiT** | Flow-Matching DiT (from scratch) | ⚠️ | ❌ | — | — | **1 / 5** — approaches but misses the grasp |

**Key takeaway so far:** Only **ACT** reliably closes the gripper around the red block.
All other policies can roughly navigate to the workspace but struggle with the precise millimetre-level alignment needed for a firm grasp — suggesting that *action chunking with temporal consistency* (ACT's signature) is disproportionately valuable for this low-data regime on the SO-101 arm.

> ⚠️ These are **intermediate results**. Experiments continue — release mechanism tuning, longer training runs, and data augmentation are tracked in the [Issues](#) section.

---

## 📚 How Was It Trained? The Dataset

All four policies are trained on **exactly the same demonstrations** so the comparison is fair:

- **Collection method:** Kinesthetic teaching — the SO-101 *leader* arm is manually guided, and the *follower* arm records every joint state, gripper position, and camera frame in sync.
- **Episodes:** 50 full pick-and-place demonstrations (grasp the red block → lift → carry → drop into the target box).
- **Sensors:** 2× RGB cameras (front view + side/wrist view) + 6-DoF joint states + binary gripper state.
- **Format:** [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) — compatible with every policy out of the box.
- **Dataset ID:** `WT/test`

The exact same 50 episodes are fed to every training script in [`examples/tutorial/`](examples/tutorial/). The only variables that change are the **architecture** and the **training recipe** (see below).

---

## 🧠 Four Architectures, Four Philosophies

Each policy represents a different paradigm in robot learning — from classical transformer-based imitation learning to billion-parameter VLA foundation models.

### ① ACT — Action Chunking Transformer

> **"Don't predict one action. Predict a *chunk*."**

ACT replaces the standard single-step regressor with a **Conditional VAE + Transformer** that emits a short *sequence* of future actions (action chunking). This dramatically smooths out jitter and mitigates compounding error — exactly the property that lets it succeed at grasping where the others miss.

- **Training:** Trained **from scratch** on the 50-episode dataset.
- **Recipe:** 100 000 steps, batch size 8, default ACTConfig (CVAE latent + temporal Transformer decoder).
- **Normalization:** MEAN_STD over states & actions inherited from dataset statistics.

<p align="center">
  <img alt="ACT Architecture" src="./media/so101/ACT/ACT.png" width="85%">
</p>

---

### ② PI0-Fast — Vision-Language-Action Foundation Model

> **"Stand on the shoulders of a 3B-parameter giant."**

PI0-Fast is built on **PaliGemma** (≈3 B params, flow-matching action head). Instead of training from scratch, we **freeze the entire vision-language backbone** and only fine-tune the **last 4 transformer layers + lm_head + final norm** — a classic parameter-efficient transfer-learning recipe for VLA models.

- **Pretrained base:** `lerobot/pi0fast-base`
- **Fine-tuning scope:** Last 4 LM layers + lm_head + norm (rest is frozen).
- **Recipe:** 20 000 steps, batch size 1, bfloat16 mixed precision + gradient checkpointing.
- **Language task token:** `"pick up the block"` prepended to every sample.
- **Why it might struggle:** SO-101's kinematics differ substantially from the robots PI0 was pretrained on — even partial fine-tuning may be insufficient to bridge this gap in the low-data regime.

<p align="center">
  <img alt="PI0 Architecture" src="./media/so101/PI0/PI0.png" width="85%">
</p>

---

### ③ Diffusion Policy — Denoising Diffusion for Action Generation

> **"Model the distribution, not the mean — because there's more than one way to grasp a block."**

Diffusion Policy frames action generation as a **denoising diffusion** process over a 64-step action horizon. By gradually removing noise, it naturally captures *multi-modal* action distributions — a property that should help in ambiguous manipulation scenarios. Quantile normalization is used for states/actions to be robust to outlier kinesthetic demonstrations.

- **Training:** Trained **from scratch** on the 50-episode dataset.
- **Recipe:** 100 000 steps, batch size 12, ResNet-18 vision backbone, cosine LR scheduler with warmup.
- **Normalization:** QUANTILES (q01–q99) for state & action, MEAN_STD for pixels.
- **Rollout window:** n_obs_steps=2 → horizon=64 → execute 32 steps / re-plan.

<p align="center">
  <img alt="Diffusion Policy Architecture" src="./media/so101/Diffusion/Diffusion.png" width="85%">
</p>

---

### ④ Multi-Task DiT — Flow-Matching Diffusion Transformer

> **"One DiT, many tasks — conditioned on language."**

Multi-Task DiT is a **flow-matching Diffusion Transformer** that takes both visual tokens (CLIP ViT-B/16) and text tokens (CLIP text encoder) as conditioning. Designed for multi-task transfer across language instructions, it predicts a 32-step action trajectory in a single forward pass of a 6-layer, 512-dim DiT.

- **Training:** Trained **from scratch** on the 50-episode dataset (CLIP encoders are pretrained and lightly fine-tuned with a 0.1× LR multiplier).
- **Recipe:** 20 000 steps, batch size 8, flow-matching objective.
- **Conditioning:** Text task `"pick up the block"` encoded via CLIP text encoder.
- **Rollout window:** n_obs_steps=2 → horizon=32 → execute 24 steps / re-plan.

<!-- ⚠️ Architecture diagram placeholder — to be filled in by the author. -->
<p align="center">
  <img alt="Multi-Task DiT Architecture (coming soon)" src="" width="85%">
  <br>
  <sub><i>🏗 Architecture diagram placeholder — will be updated shortly.</i></sub>
</p>

---

## 🧾 Training Recipe Cheat-Sheet

| Policy | Params approx. | Init | Trainable scope | Steps | Batch | Notes |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| ACT | ~50 M | Scratch | Full model | 100 k | 8 | CVAE + action chunking |
| PI0-Fast | ~3 B (0.4 B trainable) | `lerobot/pi0fast-base` | Last 4 LM layers + head + norm | 20 k | 1 | bf16 + grad ckpt, VLM frozen |
| Diffusion Policy | ~70 M | Scratch | Full model | 100 k | 12 | Quantile norm, cosine schedule |
| Multi-Task DiT | ~230 M | Scratch + CLIP pretrained | Full DiT + 0.1× LR on CLIP | 20 k | 8 | Flow matching, lang-conditioned DiT |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Yangjingyuan-HK/lerobot-so101-policy-evaluation.git
cd lerobot-so101-policy-evaluation
uv sync --locked --extra all
```

### SO-101 Hardware Calibration

Calibrate leader + follower arms following the [SO-101 hardware guide](https://huggingface.co/docs/lerobot/so101):

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3
lerobot-calibrate --robot.type=so101_leader   --robot.port=COM4
```

### Train a Policy (50-episode pick-and-place dataset)

```bash
# ACT (from scratch)
python examples/tutorial/act/act_training_example_so101.py

# PI0-Fast (fine-tune last 4 layers)
python examples/tutorial/pi0_fast/pi0_fast_training_example_so101.py

# Diffusion Policy (from scratch, quantile norm)
python examples/tutorial/diffusion/diffusion_training_example_so101.py

# Multi-Task DiT (from scratch + CLIP)
python examples/tutorial/multi_task_dit/multi_task_dit_training_example_so101.py
```

### Run Inference on the Real Robot

```bash
python examples/tutorial/act/act_using_example_so101.py
# ...and analogously for the other policies.
```

---

## 📁 Project Layout

```
lerobot/
├── examples/tutorial/        # Per-policy train / eval / zero-shot scripts
│   ├── act/                  # ACT — CVAE + Transformer
│   ├── pi0_fast/             # PI0-Fast — VLA fine-tuning (last 4 layers)
│   ├── diffusion/            # Diffusion Policy — denoising diffusion
│   ├── multi_task_dit/       # Multi-Task DiT — flow-matching DiT
│   ├── pi0/  pi05/  smolvla/ # Additional VLA experiments
│   ├── vqbet/                # Legacy VQ-BeT experiments (not in benchmark)
│   ├── async-inf/            # Async inference (policy server / robot client)
│   └── rl/                   # RL experiments (HiLSer-RL, reward classifier)
├── src/lerobot/              # LeRobot library (policies, datasets, robots, …)
├── tests/                    # Test suite
├── docker/                   # Dockerfiles for benchmarks
├── media/
│   ├── readme/               # GIF rollouts & banner images (ACT, PI0, DIFFUSION, MULTI-TASK)
│   └── so101/                # Per-policy architecture diagrams
├── data/                     # Collected datasets (gitignored)
├── outputs/                  # Training checkpoints (gitignored)
└── configs/                  # SO-101 user configs (train / eval / record)
```

---

## 🔮 Next Steps

Work in progress — tracked on this repo:

- [ ] Tune the gripper **release** timing on ACT (it grasps, now it needs to *let go*).
- [ ] Add **data augmentation** (color jitter, random crop) to Diffusion & DiT to improve grasp alignment.
- [ ] Train PI0-Fast beyond 20 k steps & compare to PI0.5 full-expert fine-tune.
- [ ] Quantify success rates with ≥ 20 rollouts per policy (instead of qualitative pass/fail).
- [ ] Upload the full 50-episode dataset & fine-tuned checkpoints to Hugging Face Hub.

---

## 📚 References

- [LeRobot](https://github.com/huggingface/lerobot) — upstream robotics library by Hugging Face
- PI0-Fast & PI0.5 — [PI0/PI0-Fast docs](https://huggingface.co/docs/lerobot/pi0fast)
- ACT — *"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"* ([Chi et al., 2023](https://arxiv.org/abs/2304.13705))
- Diffusion Policy — *"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"* ([Chi et al., 2023](https://arxiv.org/abs/2303.04137))
- [SO-101 hardware guide](https://huggingface.co/docs/lerobot/so101)
- [LeRobotDataset format (v3)](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

---

## 👤 Author

**YANG Jingyuan**

- The Hong Kong Polytechnic University
- GitHub: [@Yangjingyuan-HK](https://github.com/Yangjingyuan-HK)
- Email: `25092109d@connect.polyu.hk`

---

## 🙏 Acknowledgements

Built on top of the excellent [LeRobot](https://github.com/huggingface/lerobot) library from the Hugging Face robotics team. Upstream code and this project are licensed under **Apache 2.0**.
