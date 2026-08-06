"""在本地数据集上训练 SmolVLA 策略的示例脚本（适配 SO-101 + 20GB 显存 GPU）。

SmolVLA 是 HuggingFace 设计的小型 VLA（Vision-Language-Action）模型：
- 架构：SmolVLM2-500M（视觉-语言模型）+ Action Expert（动作专家）
- 原理：用 VLM 理解图像和语言指令，用 Flow Matching 生成连续动作
- 优势：预训练 VLM 提供强大的视觉理解能力，适合复杂任务
- 显存：~15-18GB（batch_size=4），适合 20GB 显存
- 特点：支持多相机 + 语言指令

前置条件：
    1. 安装 smolvla 依赖：
       pip install transformers num2words accelerate
       或：pip install -e ".[smolvla]"
    2. 首次运行会自动下载 SmolVLM2-500M VLM backbone（约 1-2GB）

使用方式：
    python examples/tutorial/smolvla/smolvla_training_example_so101.py
"""

import sys
from pathlib import Path

# === 依赖检查 ===
try:
    import transformers  # noqa: F401
    import num2words  # noqa: F401
    import accelerate  # noqa: F401
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请先安装 smolvla 依赖：")
    print('  pip install transformers num2words accelerate')
    print('  或：pip install -e ".[smolvla]"')
    sys.exit(1)

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def main():
    # === 1. 输出目录 ===
    output_directory = Path("outputs/robot_learning_tutorial/smolvla_so101")
    output_directory.mkdir(parents=True, exist_ok=True)

    # === 2. 设备选择 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"使用 CUDA 设备: {gpu_name}")
        print(f"  显存总量: {gpu_mem:.1f} GB")
        if gpu_mem < 15:
            print(f"  ⚠ 警告: 显存 < 15GB，可能不够训练 SmolVLA。建议降低 batch_size。")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，使用 CPU 训练（会非常慢，不建议）")
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

    # === 4. 创建 SmolVLA 配置 ===
    # SmolVLA 关键参数说明：
    #   chunk_size=50       : 一次预测 50 步动作（action chunking）
    #   n_action_steps=50   : 执行 50 步后重新推理
    #   freeze_vision_encoder=True : 冻结 VLM 视觉编码器（省显存）
    #   train_expert_only=True     : 只训练 action expert（省显存）
    #   load_vlm_weights=False     : VLM 从 HuggingFace 下载预训练权重，expert 从 scratch 训练
    #   resize_imgs_with_padding=(512,512) : 图像 resize 到 512x512
    #   normalization: MEAN_STD（STATE 和 ACTION）
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        # 训练设置
        freeze_vision_encoder=True,    # 冻结视觉编码器（省显存）
        train_expert_only=True,        # 只训练 action expert（省显存）
        train_state_proj=True,         # 训练 state 投影层
        load_vlm_weights=False,        # VLM 用预训练权重，expert 从 scratch
        # 图像处理
        resize_imgs_with_padding=(512, 512),
        # 解码
        num_steps=10,                  # Flow matching 去噪步数
        use_cache=True,                # 使用 KV cache 加速推理
        # 学习率和调度器
        optimizer_lr=1e-4,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=30000,
        scheduler_decay_lr=2.5e-6,
    )

    print(f"\n{'='*60}")
    print(f"SmolVLA 配置")
    print(f"{'='*60}")
    print(f"  VLM backbone: {cfg.vlm_model_name}")
    print(f"  VLM 层数: {cfg.num_vlm_layers}")
    print(f"  Action chunk size: {cfg.chunk_size}")
    print(f"  冻结视觉编码器: {cfg.freeze_vision_encoder}")
    print(f"  只训练 expert: {cfg.train_expert_only}")
    print(f"  图像 resize: {cfg.resize_imgs_with_padding}")
    print(f"  归一化: STATE/ACTION = MEAN_STD, VISUAL = IDENTITY")
    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")
    print(f"{'='*60}\n")

    print("创建 SmolVLA 模型（首次运行会下载 VLM backbone，约 1-2GB）...")
    policy = SmolVLAPolicy(cfg)
    policy.train()
    policy.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.1f}M (可训练: {trainable_params / 1e6:.1f}M)")

    # === 5. 创建预处理器和后处理器 ===
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # === 6. 数据集（带 delta_timestamps） ===
    # SmolVLA 的 delta_indices
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    delta_timestamps |= {
        k: [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
        for k in cfg.image_features
    }

    print(f"\n加载完整数据集: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  - 数据集样本数: {len(dataset)}")

    # === 7. 优化器、调度器、数据加载器 ===
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    batch_size = 4  # SmolVLA 需要较多显存，batch_size=4 适合 20GB
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=0,  # Windows 上建议设为 0
    )

    training_steps = 30000  # SmolVLA 默认 scheduler_decay_steps
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, training_steps)

    log_freq = 100
    save_freq = 5000

    # 默认语言指令（如果数据集 task 为空）
    default_task = "pick up the block and place it"

    print(f"\n开始训练: {training_steps} 步, batch_size={batch_size}")
    print(f"  - 默认语言指令: '{default_task}'")
    print(f"  - 学习率: {cfg.optimizer_lr} (warmup {cfg.scheduler_warmup_steps} 步, decay {cfg.scheduler_decay_steps} 步)")
    print(f"  - 日志频率: 每 {log_freq} 步, 保存频率: 每 {save_freq} 步\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # 设置默认语言指令（如果 task 为空）
            if "task" in batch:
                batch["task"] = [default_task if not t.strip() else t for t in batch["task"]]

            batch = preprocessor(batch)

            # 前向传播
            loss, info = policy.forward(batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer_grad_clip_norm)

            # 优化器步骤
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 日志
            if step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"step: {step:>6d}  loss: {loss.item():.4f}  lr: {current_lr:.6f}")

            # 定期保存
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
    print("\n✓ 完成！用 smolvla_using_example_so101.py 测试模型。")


if __name__ == "__main__":
    main()
