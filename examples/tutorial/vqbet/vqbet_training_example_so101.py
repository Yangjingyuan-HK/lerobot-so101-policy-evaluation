"""在本地数据集上训练 VQ-BeT 策略的示例脚本（适配 SO-101 + Windows + CUDA）。

VQ-BeT (Behavior generation with latent actions) 是基于离散动作 token 的行为克隆：
- 架构：ResNet18 视觉编码器 + VQ-VAE 动作离散化 + miniGPT 序列建模
- 原理：先用 VQ-VAE 将连续动作离散化为 codebook 中的码，再用 GPT 预测这些码
- 优势：对多模态动作分布（同一观测可以有多种合理动作）建模更好
- 两阶段训练：前 20k 步训练 VQ-VAE，之后训练 GPT + 预测头
- 显存需求：~4-6GB（batch_size=8）

注意：VQ-BeT 只支持单相机！本脚本选择 front（正面）相机。

使用方式（在已激活 lerobot 环境的终端中运行）：
    python examples/tutorial/vqbet/vqbet_training_example_so101.py

训练完成后，模型 checkpoint 会保存到 outputs/robot_learning_tutorial/vqbet_so101/ 目录。
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
    output_directory = Path("outputs/robot_learning_tutorial/vqbet_so101")
    output_directory.mkdir(parents=True, exist_ok=True)

    # === 2. 设备选择 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
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

    # 根据数据集特征构建策略的输入/输出特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # === 关键：VQ-BeT 只支持单相机！===
    # 选择 front（正面）相机，移除 side（侧面）相机
    # front 视角通常更适合 pick-place 任务（能看到方块和夹爪）
    camera_to_keep = "observation.images.front"
    camera_to_remove = "observation.images.side"
    if camera_to_remove in input_features:
        del input_features[camera_to_remove]
        print(f"\n注意: VQ-BeT 只支持单相机，已选择 '{camera_to_keep}'，移除 '{camera_to_remove}'")

    # === 4. 创建 VQ-BeT 策略 ===
    # VQ-BeT 默认参数已经针对 PushT 等任务调优，适合大多数场景
    # 关键参数说明：
    #   n_obs_steps=5        : 使用最近 5 步观测（比 ACT/Diffusion 更多）
    #   n_action_pred_token=3: 预测 3 个 action token
    #   action_chunk_size=5  : 每个 token 包含 5 步动作 → 共预测 15 步
    #   n_vqvae_training_steps=20000: 前 20k 步训练 VQ-VAE
    #   crop_shape=(84,84)   : 图像裁剪到 84x84（ResNet18 标准输入）
    #   normalization: MIN_MAX（VQ-BeT 默认，对动作离散化更友好）
    cfg = VQBeTConfig(
        input_features=input_features,
        output_features=output_features,
    )
    print(f"\nVQ-BeT 配置:")
    print(f"  - 观测步数 (n_obs_steps): {cfg.n_obs_steps}")
    print(f"  - 动作预测 token 数: {cfg.n_action_pred_token}")
    print(f"  - 每个 token 的动作步数: {cfg.action_chunk_size}")
    print(f"  - VQ-VAE 训练步数: {cfg.n_vqvae_training_steps}")
    print(f"  - 视觉骨干: {cfg.vision_backbone}")
    print(f"  - 图像裁剪: {cfg.crop_shape}")
    print(f"  - GPT 层数: {cfg.gpt_n_layer}, 隐藏维度: {cfg.gpt_hidden_dim}")
    print(f"  - VQ-VAE codebook 大小: {cfg.vqvae_n_embed}, 维度: {cfg.vqvae_embedding_dim}")
    print(f"  - 归一化方式: STATE/ACTION = MIN_MAX, VISUAL = IDENTITY")
    print(f"  - 输入特征: {list(input_features.keys())}")
    print(f"  - 输出特征: {list(output_features.keys())}\n")

    policy = VQBeTPolicy(cfg)

    # 创建预处理器（归一化等）和后处理器（反归一化等）
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # === 5. 准备数据集（带 delta_timestamps） ===
    # VQ-BeT 需要为 observation.state、action 和图像都提供 delta_timestamps
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 为图像特征也添加 delta_timestamps（只保留 front 相机）
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    print(f"加载完整数据集: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  - 数据集样本数: {len(dataset)}")

    # === 6. 优化器、调度器和数据加载器 ===
    # VQ-BeT 使用多参数组：VQ-VAE 有自己的学习率（1e-3），GPT 用另一个学习率（1e-4）
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    batch_size = 8  # VQ-BeT 的 GPT 较大，batch_size=8 适合 RTX 5060 Ti
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=0,  # Windows 上建议设为 0，避免多进程问题
    )

    # === 7. 学习率调度器 ===
    # VQBeTSchedulerConfig: 前 20k 步保持恒定 LR（VQ-VAE 阶段），之后 cosine 退火 + warmup
    training_steps = 50000  # 20k VQ-VAE + 30k GPT
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, training_steps)
    grad_clip_norm = 10.0  # 梯度裁剪（VQ-BeT 训练稳定性需要）

    log_freq = 100
    save_freq = 10000

    print(f"\n开始训练: {training_steps} 步, batch_size={batch_size}")
    print(f"  - 阶段 1 (step 0~{cfg.n_vqvae_training_steps}): 训练 VQ-VAE（动作离散化）")
    print(f"  - 阶段 2 (step {cfg.n_vqvae_training_steps}~{training_steps}): 训练 GPT + 预测头")
    print(f"  - 梯度裁剪: {grad_clip_norm}")
    print(f"  - 日志频率: 每 {log_freq} 步, 保存频率: 每 {save_freq} 步\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)

            # 前向传播 + 计算 loss
            # 注意：前 20k 步返回 VQ-VAE loss，之后返回 GPT loss
            loss, info = policy.forward(batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪（VQ-BeT 训练稳定性需要）
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

            # 优化器步骤
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 日志
            if step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                # 根据训练阶段显示不同的指标
                if step < cfg.n_vqvae_training_steps:
                    # VQ-VAE 阶段：显示 codebook 使用情况
                    n_codes = info.get("n_different_codes", "?")
                    n_combos = info.get("n_different_combinations", "?")
                    recon_err = info.get("recon_l1_error", "?")
                    print(
                        f"step: {step:>6d}  [VQ-VAE] loss: {loss.item():.4f}  "
                        f"codes: {n_codes}/{cfg.vqvae_n_embed * 2}  "
                        f"combos: {n_combos}/{cfg.vqvae_n_embed ** 2}  "
                        f"recon_err: {recon_err:.4f}  lr: {current_lr:.6f}"
                    )
                else:
                    # GPT 阶段：显示 loss
                    print(f"step: {step:>6d}  [GPT] loss: {loss.item():.4f}  lr: {current_lr:.6f}")

            # 阶段切换提示
            if step == cfg.n_vqvae_training_steps:
                print(f"\n>>> VQ-VAE 训练完成！切换到 GPT 训练阶段 <<<\n")

            # 定期保存 checkpoint
            if step > 0 and step % save_freq == 0:
                print(f"  -> 保存 checkpoint (step {step})")
                policy.save_pretrained(output_directory)
                preprocessor.save_pretrained(output_directory)
                postprocessor.save_pretrained(output_directory)

            step += 1
            if step >= training_steps:
                done = True
                break

    # === 8. 保存最终模型 ===
    print(f"\n训练完成！保存模型到: {output_directory}")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    print("\n✓ 完成！可以用 vqbet_using_example_so101.py 在机械臂上测试模型了。")


if __name__ == "__main__":
    main()
