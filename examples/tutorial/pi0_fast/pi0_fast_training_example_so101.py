"""在本地数据集上微调 PI0-FAST 策略（适配 SO-101 + 20GB 显存 GPU）。

PI0-FAST 微调流程：
  1. 从 HuggingFace 加载预训练模型 lerobot/pi0fast-base（~2.3B 参数）
  2. 使用本地数据集的统计值创建处理器
  3. 在本地数据集上微调
  4. 保存微调后的模型

显存优化策略（20GB GPU）：
  - dtype=bfloat16（混合精度训练）
  - gradient_checkpointing=True（用计算换显存）
  - batch_size=2（PI0-FAST 是大模型，batch 不能太大）
  - chunk_size=10（较小的动作跨度，加快收敛）

前置条件：
    1. 安装 pi 依赖：
       pip install transformers scipy
       或：pip install -e ".[pi]"
    2. 首次运行会自动下载预训练模型（约 5-6GB）

使用方式：
    python examples/tutorial/pi0_fast/pi0_fast_training_example_so101.py
"""

import sys
from pathlib import Path

# === 依赖检查 ===
try:
    import transformers  # noqa: F401
    import scipy  # noqa: F401
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请先安装 pi 依赖：")
    print('  pip install transformers scipy')
    print('  或：pip install -e ".[pi]"')
    sys.exit(1)

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0_fast import PI0FastConfig, PI0FastPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

# HuggingFace 上的预训练模型 ID
PRETRAINED_MODEL_ID = "lerobot/pi0fast-base"


def main():
    # === 1. 输出目录 ===
    output_directory = Path("outputs/robot_learning_tutorial/pi0fast_so101")
    output_directory.mkdir(parents=True, exist_ok=True)

    # === 2. 设备选择 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"使用 CUDA 设备: {gpu_name}")
        print(f"  显存总量: {gpu_mem:.1f} GB")
        if gpu_mem < 16:
            print(f"  ⚠ 警告: 显存 < 16GB，可能不够微调 PI0-FAST。")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，无法训练 PI0-FAST")
        return

    # === 3. 数据集配置 ===
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"

    print(f"\n加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)
    print(f"  - 总 episodes: {dataset_metadata.total_episodes}")
    print(f"  - 总 frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # === 4. 创建 PI0-FAST 配置（微调优化）===
    cfg = PI0FastConfig(
        input_features=input_features,
        output_features=output_features,
        # ── 混合精度训练（省显存）──
        dtype="bfloat16",
        # ── 梯度检查点（用计算换显存，大幅减少显存使用）──
        gradient_checkpointing=True,
        # ── 动作跨度（较小的 chunk 加快收敛）──
        chunk_size=10,
        n_action_steps=10,
        # ── KV Cache（加速推理）──
        use_kv_cache=True,
        # ── 学习率（微调用较小的学习率）──
        optimizer_lr=2.5e-5,
        optimizer_weight_decay=0.01,
        optimizer_grad_clip_norm=1.0,
        # ── 调度器 ──
        scheduler_warmup_steps=500,
        scheduler_decay_steps=20000,
        scheduler_decay_lr=2.5e-6,
        # ── 设备 ──
        device=str(device),
    )

    print(f"\n{'='*60}")
    print(f"PI0-FAST 微调配置")
    print(f"{'='*60}")
    print(f"  预训练模型: {PRETRAINED_MODEL_ID}")
    print(f"  dtype: {cfg.dtype}（混合精度）")
    print(f"  gradient_checkpointing: {cfg.gradient_checkpointing}")
    print(f"  chunk_size: {cfg.chunk_size}")
    print(f"  n_action_steps: {cfg.n_action_steps}")
    print(f"  学习率: {cfg.optimizer_lr}")
    print(f"  warmup: {cfg.scheduler_warmup_steps} 步")
    print(f"  decay: {cfg.scheduler_decay_steps} 步")
    print(f"  归一化: STATE/ACTION = MEAN_STD, VISUAL = IDENTITY")
    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")
    print(f"{'='*60}\n")

    # === 5. 加载预训练模型 ===
    print(f"加载预训练模型: {PRETRAINED_MODEL_ID}")
    print("（首次运行会下载模型，约 5-6GB，请耐心等待...）")
    model = PI0FastPolicy.from_pretrained(PRETRAINED_MODEL_ID, config=cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # === 5b. 冻结大部分参数，只训练最后几层 + lm_head ===
    # PI0-FAST 没有 action expert 子模块，整个模型就是 PaliGemma（3.45B）
    # 全参数微调需要 ~45GB 显存，冻结大部分层后只需 ~14GB
    # 策略：冻结视觉编码器 + 前面的语言模型层，只训练最后 4 层 + lm_head + norm
    for param in model.parameters():
        param.requires_grad = False

    # 找到语言模型的最大层数
    layer_nums = set()
    for name, _ in model.named_parameters():
        if "language_model.layers." in name:
            parts = name.split("language_model.layers.")
            if len(parts) > 1:
                layer_nums.add(int(parts[1].split(".")[0]))

    max_layer = max(layer_nums) if layer_nums else 0
    # 只训练最后 4 层
    layers_to_train = set(range(max(0, max_layer - 3), max_layer + 1))

    print(f"\n模型结构: 语言模型共 {max_layer + 1} 层, 只训练最后 {len(layers_to_train)} 层 + lm_head + norm")

    # 解冻：最后几层 + lm_head + final norm
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        should_train = False
        # 解冻 lm_head（动作 token 生成头）
        if "lm_head" in name:
            should_train = True
        # 解冻最后的语言模型层
        elif "language_model.layers." in name:
            parts = name.split("language_model.layers.")
            if len(parts) > 1:
                layer_num = int(parts[1].split(".")[0])
                if layer_num in layers_to_train:
                    should_train = True
        # 解冻 final norm
        elif "model.norm" in name or "language_model.norm" in name:
            should_train = True

        if should_train:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    model.train()
    print(f"\n模型参数量: {total_params / 1e9:.2f}B")
    print(f"  冻结 (VLM backbone): {frozen_count / 1e9:.2f}B")
    print(f"  可训练 (action expert + proj): {trainable_count / 1e9:.2f}B")

    # 清理 CUDA 缓存
    torch.cuda.empty_cache()

    # === 6. 创建处理器（使用本地数据集统计值）===
    print(f"\n创建处理器（使用本地数据集统计值）...")
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # === 7. 数据集 ===
    # PI0-FAST 的 observation_delta_indices 返回 None（不使用观测 delta），
    # 只有 action 使用 delta timestamps（action_delta_indices = range(chunk_size)）
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    # 观测（state 和 images）只用当前帧，不需要 delta timestamps
    delta_timestamps["observation.state"] = [0]
    for k in cfg.image_features:
        delta_timestamps[k] = [0]

    print(f"加载完整数据集: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  - 数据集样本数: {len(dataset)}")

    # === 8. 优化器、调度器、数据加载器 ===
    # 只优化 requires_grad=True 的参数
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = cfg.get_optimizer_preset().build(trainable_params_list)
    batch_size = 1  # 16GB 显存必须用 batch_size=1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=0,
    )

    training_steps = 20000
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, training_steps)

    log_freq = 100
    save_freq = 5000

    default_task = "pick up the block"

    print(f"\n开始微调: {training_steps} 步, batch_size={batch_size}")
    print(f"  - 默认语言指令: '{default_task}'")
    print(f"  - 混合精度: bfloat16 (autocast)")
    print(f"  - 日志频率: 每 {log_freq} 步, 保存频率: 每 {save_freq} 步\n")

    # 混合精度训练的 GradScaler（bfloat16 不需要 scaler，但 autocast 需要）
    use_autocast = (cfg.dtype == "bfloat16")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # 设置默认语言指令
            if "task" in batch:
                batch["task"] = [default_task if not t.strip() else t for t in batch["task"]]

            batch = preprocessor(batch)

            # 前向传播（混合精度）
            if use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, info = model.forward(batch)
            else:
                loss, info = model.forward(batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪（只裁剪可训练参数）
            torch.nn.utils.clip_grad_norm_(trainable_params_list, cfg.optimizer_grad_clip_norm)

            # 优化器步骤
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 日志
            if step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"step: {step:>6d}  loss: {loss.item():.4f}  lr: {current_lr:.6f}  "
                      f"显存: {gpu_mem:.1f}/{gpu_reserved:.1f} GB")

            # 定期保存
            if step > 0 and step % save_freq == 0:
                print(f"  -> 保存 checkpoint (step {step})")
                model.save_pretrained(output_directory)
                preprocessor.save_pretrained(output_directory)
                postprocessor.save_pretrained(output_directory)

            step += 1
            if step >= training_steps:
                done = True
                break

    # === 9. 保存最终模型 ===
    print(f"\n训练完成！保存模型到: {output_directory}")
    model.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print("\n✓ 完成！用 pi0_fast_using_example_so101.py 测试模型。")


if __name__ == "__main__":
    main()
