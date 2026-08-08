"""在本地数据集上训练 ACT 策略的示例脚本（适配 SO-101 + Windows + CUDA）。

本脚本演示如何使用 LeRobot 的 Python API 训练 ACT（Action Chunking with Transformers）策略，
区别于使用 `lerobot-train` 命令行工具。ACT 是 LeRobot 推荐的入门策略：
- 参数量小（~80M），训练快（几小时）
- 显存占用低（~1GB），适合笔记本 GPU
- 数据效率高（50 条演示即可）

使用方式（在已激活 lerobot 环境的终端中运行）：
    python examples/tutorial/act/act_training_example_so101.py

训练完成后，模型 checkpoint 会保存到 outputs/robot_learning_tutorial/act_so101/ 目录。
"""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将 delta_indices（帧索引列表）转换为时间戳列表（秒）。"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    # === 1. 输出目录 ===
    output_directory = Path("outputs/robot_learning_tutorial/act_so101")
    output_directory.mkdir(parents=True, exist_ok=True)

    # === 2. 设备选择 ===
    # Windows + NVIDIA GPU 用 cuda；没有 GPU 用 cpu（会很慢）；Mac 用 mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，使用 CPU 训练（会非常慢）")

    # === 3. 数据集配置 ===
    # 这里使用本地录制的数据集（对应 configs/record_config.yaml 中的配置）
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"

    # 加载数据集元数据（只读取 metadata，不加载全部数据）
    print(f"加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)
    print(f"  - 总 episodes: {dataset_metadata.total_episodes}")
    print(f"  - 总 frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")

    # 根据数据集特征构建策略的输入/输出特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # === 4. 创建 ACT 策略 ===
    # ACTConfig 使用默认超参数即可，适合大多数任务
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)

    # 创建预处理器（归一化等）和后处理器（反归一化等）
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # === 5. 准备数据集（带 delta_timestamps 用于 action chunking） ===
    # ACT 期望一次预测 k 个未来动作（action chunking），需要 delta_timestamps 对齐
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 为图像特征也添加 delta_timestamps
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    print(f"加载完整数据集: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  - 数据集样本数: {len(dataset)}")

    # === 6. 优化器和数据加载器 ===
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 8  # 显存不够时降到 4 或 2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=0,  # Windows 上建议设为 0，避免多进程问题
    )

    # === 7. 训练循环 ===
    # 实际训练建议 100k 步以上，这里设小值用于演示
    training_steps = 100000
    log_freq = 100
    save_freq = 10000

    print(f"\n开始训练: {training_steps} 步, batch_size={batch_size}")
    print(f"日志频率: 每 {log_freq} 步, 保存频率: 每 {save_freq} 步\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)

            # 前向传播 + 计算 loss
            loss, _ = policy.forward(batch)

            # 反向传播 + 优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 日志
            if step % log_freq == 0:
                print(f"step: {step:>6d}  loss: {loss.item():.4f}")

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

    # 如果要上传到 HuggingFace Hub，取消下面的注释（需要先 huggingface-cli login）
    # policy.push_to_hub("你的用户名/act_so101_pickplace")
    # preprocessor.push_to_hub("你的用户名/act_so101_pickplace")
    # postprocessor.push_to_hub("你的用户名/act_so101_pickplace")

    print("\n✓ 完成！可以用 act_using_example_so101.py 在机械臂上测试模型了。")


if __name__ == "__main__":
    main()
