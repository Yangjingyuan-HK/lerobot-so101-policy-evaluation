"""VQ-BeT v2 训练脚本 — 最佳性能配置（适配 SO-101 + RTX 5060 Ti）。

相比 v1 的优化：
  1. 训练步数 50k → 100k（20k VQ-VAE + 80k GPT，论文推荐）
  2. VQ-VAE codebook 16 → 32（动作离散化更精细）
  3. GPT 层数 8 → 10（更强的序列建模能力）
  4. spatial_softmax 关键点 32 → 64（提取更多视觉特征）
  5. crop_shape 84x84 → 96x96（模型能看到更多场景）
  6. dropout 0.1 → 0.05（减少正则化，让模型学得更充分）
  7. batch_size 8 → 16（更稳定的梯度估计）

使用方式：
    python examples/tutorial/vqbet/vqbet_training_example_so101_v2.py
"""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.vqbet import VQBeTConfig, VQBeTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将 delta_indices（帧索引列表）转换为时间戳列表（秒）。"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    # === 1. 输出目录 ===
    output_directory = Path("outputs/robot_learning_tutorial/vqbet_so101_v2")
    output_directory.mkdir(parents=True, exist_ok=True)

    # === 2. 设备选择 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，使用 CPU 训练（会非常慢）")

    # === 3. 数据集配置 ===
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"

    print(f"加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)
    print(f"  - 总 episodes: {dataset_metadata.total_episodes}")
    print(f"  - 总 frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # VQ-BeT 只支持单相机，选择 front
    camera_to_remove = "observation.images.side"
    if camera_to_remove in input_features:
        del input_features[camera_to_remove]
        print(f"\n注意: VQ-BeT 只支持单相机，已选择 'observation.images.front'")

    # === 4. 创建 VQ-BeT 策略（v2 优化配置） ===
    cfg = VQBeTConfig(
        input_features=input_features,
        output_features=output_features,
        # ── 观测和动作预测（保持默认，已是最优）──
        n_obs_steps=5,              # 5 步观测历史
        n_action_pred_token=3,      # 预测 3 个 action token
        action_chunk_size=5,        # 每个 token 5 步动作

        # ── 视觉编码器优化 ──
        vision_backbone="resnet18",
        crop_shape=(96, 96),                 # v1: 84x84 → v2: 96x96（看到更多场景）
        crop_is_random=True,                  # 训练时随机裁剪（数据增强）
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_group_norm=False,
        spatial_softmax_num_keypoints=64,     # v1: 32 → v2: 64（更多视觉特征点）

        # ── VQ-VAE 优化 ──
        n_vqvae_training_steps=20000,         # 前 20k 步训练 VQ-VAE
        vqvae_n_embed=32,                     # v1: 16 → v2: 32（更大的 codebook，动作离散化更精细）
        vqvae_embedding_dim=256,
        vqvae_enc_hidden_dim=128,

        # ── GPT 优化 ──
        gpt_block_size=500,
        gpt_input_dim=512,
        gpt_output_dim=512,
        gpt_n_layer=10,                       # v1: 8 → v2: 10（更深的 GPT）
        gpt_n_head=8,
        gpt_hidden_dim=512,
        dropout=0.05,                         # v1: 0.1 → v2: 0.05（减少正则化）

        # ── 损失权重（保持默认，已是最优）──
        offset_loss_weight=10000.0,
        primary_code_loss_weight=5.0,
        secondary_code_loss_weight=0.5,
        bet_softmax_temperature=0.1,
        sequentially_select=False,
    )

    # 打印配置对比
    print(f"\n{'='*60}")
    print(f"VQ-BeT v2 优化配置")
    print(f"{'='*60}")
    print(f"  训练步数: 100000 (v1: 50000)")
    print(f"  batch_size: 16 (v1: 8)")
    print(f"  VQ-VAE codebook: {cfg.vqvae_n_embed} (v1: 16)")
    print(f"  GPT 层数: {cfg.gpt_n_layer} (v1: 8)")
    print(f"  spatial_softmax 关键点: {cfg.spatial_softmax_num_keypoints} (v1: 32)")
    print(f"  crop_shape: {cfg.crop_shape} (v1: (84, 84))")
    print(f"  dropout: {cfg.dropout} (v1: 0.1)")
    print(f"  归一化: STATE/ACTION = MIN_MAX, VISUAL = IDENTITY")
    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")
    print(f"{'='*60}\n")

    policy = VQBeTPolicy(cfg)
    print(f"模型参数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # === 5. 数据集 ===
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    print(f"加载完整数据集: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  - 数据集样本数: {len(dataset)}")

    # === 6. 优化器、调度器、数据加载器 ===
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    batch_size = 16  # v1: 8 → v2: 16
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=0,
    )

    training_steps = 100000  # v1: 50000 → v2: 100000 (20k VQ-VAE + 80k GPT)
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, training_steps)
    grad_clip_norm = 10.0

    log_freq = 100
    save_freq = 10000

    print(f"\n开始训练: {training_steps} 步, batch_size={batch_size}")
    print(f"  - 阶段 1 (step 0~{cfg.n_vqvae_training_steps}): 训练 VQ-VAE")
    print(f"  - 阶段 2 (step {cfg.n_vqvae_training_steps}~{training_steps}): 训练 GPT + 预测头")
    print(f"  - 梯度裁剪: {grad_clip_norm}")
    print(f"  - 日志频率: 每 {log_freq} 步, 保存频率: 每 {save_freq} 步\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)

            loss, info = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                if step < cfg.n_vqvae_training_steps:
                    n_codes = info.get("n_different_codes", "?")
                    n_combos = info.get("n_different_combinations", "?")
                    recon_err = info.get("recon_l1_error", "?")
                    if isinstance(recon_err, float):
                        print(
                            f"step: {step:>6d}  [VQ-VAE] loss: {loss.item():.4f}  "
                            f"codes: {n_codes}/{cfg.vqvae_n_embed * 2}  "
                            f"combos: {n_combos}/{cfg.vqvae_n_embed ** 2}  "
                            f"recon: {recon_err:.4f}  lr: {current_lr:.6f}"
                        )
                    else:
                        print(f"step: {step:>6d}  [VQ-VAE] loss: {loss.item():.4f}  lr: {current_lr:.6f}")
                else:
                    print(f"step: {step:>6d}  [GPT] loss: {loss.item():.4f}  lr: {current_lr:.6f}")

            if step == cfg.n_vqvae_training_steps:
                print(f"\n>>> VQ-VAE 训练完成！切换到 GPT 训练阶段 <<<\n")

            if step > 0 and step % save_freq == 0:
                print(f"  -> 保存 checkpoint (step {step})")
                policy.save_pretrained(output_directory)
                preprocessor.save_pretrained(output_directory)
                postprocessor.save_pretrained(output_directory)

            step += 1
            if step >= training_steps:
                done = True
                break

    print(f"\n训练完成！保存模型到: {output_directory}")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print("\n✓ 完成！用 vqbet_using_example_so101_v2.py 测试模型。")


if __name__ == "__main__":
    main()
