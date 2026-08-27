from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.datasets import DEFAULT_QUANTILES, LeRobotDataset, LeRobotDatasetMetadata, get_feature_stats
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将 delta_indices（帧索引列表）转换为时间戳列表（秒）。"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def compute_quantile_stats_in_memory(
    dataset_id: str,
    dataset_root: str,
    features_to_process: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    在内存中计算数据集的 quantile 统计值（q01, q10, q50, q90, q99）。

    只读取数据，不修改原始数据集文件。返回一个包含 quantile 统计值的字典，
    可以合并到 dataset_metadata.stats 中用于创建处理器。

    Args:
        dataset_id: 数据集 ID（如 "WT/test"）
        dataset_root: 数据集根目录
        features_to_process: 需要计算 quantile 的特征名列表。如果为 None，
            则处理所有非图像、非字符串类型的特征。

    Returns:
        dict，结构为 {feature_name: {q01: ndarray, q10: ndarray, ..., q99: ndarray}}
    """
    print(f"\n=== 在内存中计算 Quantile 统计值（不会修改原始数据集） ===")
    print(f"加载数据集用于计算 quantiles...")
    dataset = LeRobotDataset(dataset_id, root=dataset_root)

    # 确定需要处理的特征
    if features_to_process is None:
        features_to_process = []
        for key, feat_info in dataset.features.items():
            dtype = feat_info.get("dtype", "")
            # 跳过图像、视频、字符串类型
            if dtype in ["image", "video", "string"]:
                continue
            features_to_process.append(key)

    print(f"需要计算 quantile 的特征: {features_to_process}")

    # 收集所有帧的数据
    print(f"收集数据（共 {dataset.meta.total_frames} 帧）...")
    collected: dict[str, list[np.ndarray]] = {k: [] for k in features_to_process}

    for i in tqdm(range(dataset.meta.total_frames), desc="读取帧数据"):
        item = dataset[i]
        for key in features_to_process:
            if key in item:
                val = item[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                collected[key].append(val)

    # 对每个特征计算 quantile 统计值
    quantile_stats: dict[str, dict[str, np.ndarray]] = {}
    quantile_keys = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]

    for key in features_to_process:
        data_list = collected[key]
        if len(data_list) == 0:
            print(f"  跳过 {key}：没有数据")
            continue

        data = np.stack(data_list, axis=0)  # shape: (N, ...)

        # 对于向量特征（如 state, action），axis=0
        feature_shape = dataset.features[key].get("shape", ())
        if len(feature_shape) == 1:
            # 一维向量：shape (N, D)，在 axis=0 上计算统计值
            stats = get_feature_stats(data, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)
        else:
            # 其他情况默认全局计算
            stats = get_feature_stats(data, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)

        quantile_stats[key] = {k: stats[k] for k in quantile_keys if k in stats}

        # 打印对比信息（验证用）
        print(f"  {key}:")
        for qk in quantile_keys:
            if qk in quantile_stats[key]:
                vals = quantile_stats[key][qk]
                print(f"    {qk}: {[round(v, 3) for v in vals]}")

    print("=== Quantile 统计值计算完成（仅在内存中） ===\n")
    return quantile_stats


def merge_quantile_into_stats(
    original_stats: dict, quantile_stats: dict[str, dict[str, np.ndarray]]
) -> dict:
    """
    将内存中计算的 quantile 统计值合并到原始 stats 中。

    返回一个新的字典（深拷贝），不会修改原始 stats。
    """
    merged = deepcopy(original_stats)
    for feature_name, q_stats in quantile_stats.items():
        if feature_name not in merged:
            merged[feature_name] = {}
        for q_key, q_value in q_stats.items():
            merged[feature_name][q_key] = q_value
    return merged


def main():
    # === 1. 输出目录 ===
    output_directory = PROJECT_ROOT / "outputs" / "robot_learning_tutorial" / "diffusion_so101"
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

    # === 3.5 在内存中计算 quantile 统计值 ===
    # 注意：不会修改原始数据集！只在内存中计算，结果合并到 dataset_metadata.stats 的深拷贝中
    # 需要计算 quantile 的特征：observation.state 和 action（这些是 STATE/ACTION 类型）
    quantile_features = ["observation.state", "action"]
    quantile_stats = compute_quantile_stats_in_memory(dataset_id, dataset_root, quantile_features)

    # 合并到 stats（使用深拷贝，不修改原始 dataset_metadata.stats）
    augmented_stats = merge_quantile_into_stats(dataset_metadata.stats, quantile_stats)

    # 打印验证信息
    print("验证合并后的 stats（包含 quantile key）:")
    for feature_name in quantile_features:
        if feature_name in augmented_stats:
            keys = sorted(augmented_stats[feature_name].keys())
            print(f"  {feature_name}: stats keys = {keys}")
            # 检查是否有 q01, q99（QUANTILES 模式需要的 key）
            has_required = "q01" in augmented_stats[feature_name] and "q99" in augmented_stats[feature_name]
            print(f"    QUANTILES 所需 key (q01, q99): {'✓' if has_required else '✗ 缺失!'}")
    print()

    # 根据数据集特征构建策略的输入/输出特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # === 4. 创建 Diffusion 策略 ===
    # 关键参数说明：
    #   n_obs_steps=2   : 使用最近 2 步的观测（当前步 + 前一步）
    #   horizon=64      : 每次预测 64 步的动作范围
    #   n_action_steps=32: 实际执行其中 32 步
    #   vision_backbone  : 默认 resnet18（轻量快速）
    #   normalization    : QUANTILES 对异常值更鲁棒，映射到 [-1, 1]
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    print(f"\nDiffusion 配置:")
    print(f"  - 观测步数 (n_obs_steps): {cfg.n_obs_steps}")
    print(f"  - 动作预测范围 (horizon): {cfg.horizon}")
    print(f"  - 实际执行步数 (n_action_steps): {cfg.n_action_steps}")
    print(f"  - 视觉骨干: {cfg.vision_backbone}")
    print(f"  - 归一化方式: STATE/ACTION = QUANTILES, VISUAL = MEAN_STD")
    print(f"  - 输入特征: {list(input_features.keys())}")
    print(f"  - 输出特征: {list(output_features.keys())}\n")

    policy = DiffusionPolicy(cfg)

    # 创建预处理器（归一化等）和后处理器（反归一化等）
    # 注意：使用包含 quantile 的 augmented_stats（不是原始 dataset_metadata.stats）
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=augmented_stats)

    policy.train()
    policy.to(device)

    # === 5. 准备数据集（带 delta_timestamps） ===
    # Diffusion 需要为 observation.state、action 和所有图像都提供 delta_timestamps
    # observation_delta_indices: [-1, 0]  → 使用前1步 + 当前步观测
    # action_delta_indices:      [-1, 0, 1, 2, ..., 62] → 对应 horizon=64 的动作范围
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 为图像特征也添加观测窗口的 delta_timestamps
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    print(f"加载完整数据集: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  - 数据集样本数: {len(dataset)}")

    # === 6. 优化器、调度器和数据加载器 ===
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 12  # Diffusion 建议 32，更稳定的梯度估计
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=0,  # Windows 上建议设为 0，避免多进程问题
    )

    # === 7. 学习率调度器（Diffusion 关键组件） ===
    # 使用 cosine 退火 + 500 步 warmup，对 Diffusion 收敛至关重要
    training_steps = 100000  # Diffusion 需要较多步数 (80k-150k)
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, training_steps)

    log_freq = 100
    save_freq = 10000

    print(f"\n开始训练: {training_steps} 步, batch_size={batch_size}")
    print(f"学习率调度: cosine 退火 + {cfg.scheduler_warmup_steps} 步 warmup")
    print(f"日志频率: 每 {log_freq} 步, 保存频率: 每 {save_freq} 步\n")
    print("重要提示: 处理器保存时会自动把包含 quantile 的 stats 写入输出目录")
    print("          不会修改原始数据集！\n")

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
            lr_scheduler.step()  # 每步更新学习率
            optimizer.zero_grad()

            # 日志（打印当前学习率便于监控）
            if step % log_freq == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"step: {step:>6d}  loss: {loss.item():.4f}  lr: {current_lr:.6f}")

            # 定期保存 checkpoint
            if step > 0 and step % save_freq == 0:
                print(f"  -> 保存 checkpoint (step {step})")
                policy.save_pretrained(output_directory)
                # preprocessor/postprocessor 会把 augmented_stats（含 quantile）保存到输出目录
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
    # policy.push_to_hub("你的用户名/diffusion_so101_pickplace")
    # preprocessor.push_to_hub("你的用户名/diffusion_so101_pickplace")
    # postprocessor.push_to_hub("你的用户名/diffusion_so101_pickplace")

    print("\n✓ 完成！可以创建 diffusion_using_example_so101.py 在机械臂上测试模型了。")


if __name__ == "__main__":
    main()
