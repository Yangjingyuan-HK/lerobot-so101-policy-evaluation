# LeRobot SO-101 Policy Evaluation

> A practical evaluation and benchmarking of state-of-the-art imitation learning and Vision-Language-Action (VLA) policies on the SO-101 robotic arm for real-world manipulation tasks.

<p align="center">
  <img alt="LeRobot" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

## Project Overview

This project evaluates and compares multiple robot learning policies on the **SO-101 robotic arm** platform, built on top of the [LeRobot](https://github.com/huggingface/lerobot) library by Hugging Face.

The goal is to empirically compare policies such as **PI0**, **PI0-Fast**, **VQ-BeT**, **ACT**, and **Diffusion Policy** in terms of:

- Zero-shot inference performance (without fine-tuning)
- Fine-tuning convergence speed and final performance
- Robustness to SO-101's specific kinematic configuration

## Hardware Setup

| Component | Specification |
|---|---|
| **Robot arm** | SO-101 (master arm + slave arm, calibrated) |
| **Compute** | NVIDIA GPU with 20 GB VRAM |
| **Cameras** | Multiple OpenCV cameras for visual observation |
| **OS** | Windows 11 + WSL2 |
| **Python** | 3.12+ (managed via Anaconda) |

## Experiments

### Policies Under Evaluation

| Policy | Category | Parameters | Notes |
|---|---|---|---|
| [PI0](./examples/tutorial/pi0/) | VLA | ~3 B | Pretrained foundation model |
| [PI0-Fast](./examples/tutorial/pi0_fast/) | VLA + FAST tokens | ~2.3 B | 5× faster training via DCT+BPE tokenization |
| [VQ-BeT](./src/lerobot/policies/vqbet/) | Vector-Quantized Behavior Transformer | — | Discrete action tokens |
| [ACT](./src/lerobot/policies/act/) | Action Chunking Transformer | — | Chunked action prediction |
| [Diffusion Policy](./src/lerobot/policies/diffusion/) | Diffusion-based | — | Denoising diffusion for actions |
| [SmolVLA](./src/lerobot/policies/smolvla/) | VLA | — | Compact VLA model |

### Experiment Workflow

Each policy is evaluated through a three-stage pipeline:

1. **Zero-shot inference** — Run the pretrained policy directly on SO-101 to measure transferability.
2. **Dataset fine-tuning** — Fine-tune on SO-101 teleoperation demonstrations.
3. **Post-fine-tuning evaluation** — Compare fine-tuned vs. zero-shot performance.

## Project Structure

```
lerobot/
├── examples/
│   └── tutorial/              # Custom experiment scripts (per-policy)
│       ├── pi0/               # PI0 zero-shot / training / inference
│       ├── pi0_fast/          # PI0-Fast zero-shot / training / inference
│       ├── vqbet/             # VQ-BeT experiments
│       ├── act/               # ACT experiments
│       ├── diffusion/         # Diffusion Policy experiments
│       ├── smolvla/           # SmolVLA experiments
│       ├── async-inf/         # Async inference experiments
│       └── rl/                # Reinforcement learning experiments
├── src/lerobot/               # LeRobot library source (policies, datasets, ...)
├── tests/                     # Test suite (artifacts excluded)
├── docs/                      # Documentation source
├── docker/                    # Dockerfiles for benchmarks
├── papers/                    # Reference papers (ACT, Diffusion, VQ-BeT)
├── media/                     # Logos and demo media
├── data/                      # Collected datasets (gitignored)
├── outputs/                   # Training outputs / checkpoints (gitignored)
├── *.yaml                     # Config files for record / eval / train
└── pyproject.toml             # Project metadata and dependencies
```

## Quick Start

### Installation

```bash
git clone https://github.com/Yangjingyuan-HK/lerobot-so101-policy-evaluation.git
cd lerobot-so101-policy-evaluation
uv sync --locked --extra all
```

### SO-101 Hardware Setup

Calibrate the SO-101 arm (master + slave) following the [SO-101 hardware guide](https://huggingface.co/docs/lerobot/so101):

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3
lerobot-calibrate --robot.type=so101_leader  --robot.port=COM4
```

### Training a Policy

Example: fine-tuning PI0-Fast on SO-101 data:

```bash
python examples/tutorial/pi0_fast/pi0_fast_training_example_so101.py
```

### Evaluation

Run inference with a fine-tuned checkpoint:

```bash
python examples/tutorial/pi0_fast/pi0_fast_using_example_so101.py
```

### Zero-shot Inference (no fine-tuning)

```bash
python examples/tutorial/pi0_fast/pi0_fast_zeroshot_so101.py
```

## Key Findings

> Detailed quantitative results will be added as experiments complete.

Expected outcomes (based on LeRobot community reports):

- **Zero-shot**: poor on SO-101 because pretrained policies have not seen this exact arm configuration.
- **Fine-tuned**: significant improvement, especially for PI0 / PI0-Fast which leverage large pretrained foundations.
- **PI0-Fast vs. PI0**: PI0-Fast converges ~5× faster while reaching comparable final performance, thanks to FAST tokenization (DCT + BPE) and KV-cache inference.

## References

- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face robotics library (upstream project)
- [PI0 / PI0-Fast documentation](https://huggingface.co/docs/lerobot/pi0fast)
- [VQ-BeT paper](https://arxiv.org/abs/2403.06009)
- [SO-101 hardware guide](https://huggingface.co/docs/lerobot/so101)
- [LeRobotDataset format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## Author

**YANG Jingyuan**

- The Hong Kong Polytechnic University
- GitHub: [@Yangjingyuan-HK](https://github.com/Yangjingyuan-HK)
- Email: `25092109d@connect.polyu.hk`

## License

Apache License 2.0 — inherited from the upstream [LeRobot](https://github.com/huggingface/lerobot) project. See [LICENSE](./LICENSE) for details.

## 🙏 Acknowledgements

This project builds on the excellent [LeRobot](https://github.com/huggingface/lerobot) library developed by the Hugging Face robotics team. The original LeRobot library is licensed under Apache 2.0.
