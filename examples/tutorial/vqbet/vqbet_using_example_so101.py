"""在真实 SO-101 机械臂上运行训练好的 VQ-BeT 策略（适配 Windows + CUDA）。

本脚本演示如何加载训练好的 VQ-BeT 模型，在 SO-101 从臂上运行推理。
它会循环执行多个 episode，每个 episode 中：
    1. 从机械臂获取观测（图像 + 关节状态）
    2. 用 VQ-BeT 模型预测动作（基于离散动作 token + GPT）
    3. 将动作发送给机械臂执行

VQ-BeT 与 ACT/Diffusion 的区别：
    - VQ-BeT 只支持单相机（本脚本使用 front 正面相机）
    - VQ-BeT 使用两阶段方法：VQ-VAE 离散化动作 + GPT 预测离散码
    - VQ-BeT 一次预测 action_chunk_size=5 步动作，用队列逐步执行
    - 不需要语言指令（task），是纯视觉-关节策略

使用方式（在已激活 lerobot 环境的终端中运行）：
    python examples/tutorial/vqbet/vqbet_using_example_so101.py

前置条件：
    1. 已用 vqbet_training_example_so101.py 训练好模型
    2. SO-101 从臂已连接（端口 COM9）并校准
    3. 相机已连接（index 0 = front）
"""

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.vqbet import VQBeTPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# === 配置参数 ===
MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 800


def main():
    # === 1. 设备选择 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，使用 CPU 推理（会很慢）")

    # === 2. 加载训练好的模型 ===
    model_path = "outputs/robot_learning_tutorial/vqbet_so101"

    print(f"加载模型: {model_path}")
    model = VQBeTPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 检查 VQ-VAE 是否已经训练好（discretized）
    if not model.vqbet.action_head.vqvae_model.discretized.item():
        print("⚠ 警告: VQ-VAE 未完成训练！模型可能无法正确推理。")
        print("  请确保训练至少达到 n_vqvae_training_steps 步。")
    else:
        print("✓ VQ-VAE 已完成训练（动作离散化就绪）")

    # === 3. 加载数据集元数据（用于构建推理时的数据帧） ===
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    print(f"加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

    # 创建预处理器和后处理器
    # 关键：从保存的模型路径加载，确保归一化参数和训练时完全一致
    print(f"加载预/后处理器: {model_path}")
    preprocess, postprocess = make_pre_post_processors(
        model.config, pretrained_path=model_path
    )

    # 打印数据集中的动作统计信息，用于调试对比
    action_stats = dataset_metadata.stats.get("action", {})
    if action_stats:
        print(f"\n数据集 action 统计:")
        for i, name in enumerate(dataset_metadata.features["action"]["names"]):
            mean_val = action_stats.get("mean", [0] * 6)[i]
            std_val = action_stats.get("std", [1] * 6)[i]
            min_val = action_stats.get("min", [-3.14] * 6)[i]
            max_val = action_stats.get("max", [3.14] * 6)[i]
            print(f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    print()

    # === 4. 配置机械臂和相机 ===
    # 注意：VQ-BeT 只使用 front 相机（训练时配置的）
    # 但机械臂仍可连接两个相机，推理时模型会自动只使用 front
    follower_port = "COM9"
    follower_id = "my_awesome_follower_arm"

    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
    }

    print(f"连接 SO-101 从臂: port={follower_port}, id={follower_id}")
    robot_cfg = SO101FollowerConfig(
        port=follower_port, id=follower_id, cameras=camera_config
    )
    robot = SO101Follower(robot_cfg)

    try:
        robot.connect()
        print("✓ 机械臂已连接")
    except Exception as e:
        print(f"连接失败: {e}")
        return

    # === 5. 推理循环 ===
    print(f"\n开始推理: {MAX_EPISODES} 个 episode, 每个 episode 最多 {MAX_STEPS_PER_EPISODE} 步")
    print("按 Ctrl+C 可随时停止\n")

    # VQ-BeT 需要在每个 episode 开始时重置队列
    model.reset()

    try:
        for episode in range(MAX_EPISODES):
            print(f"--- Episode {episode + 1}/{MAX_EPISODES} ---")

            # 每个 episode 开始时重置模型队列
            model.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                # (a) 获取机械臂观测（图像 + 关节状态）
                obs = robot.get_observation()

                # 打印初始观测的关节状态（前3步调试用）
                if step < 3:
                    raw_state = obs.get("state", None)
                    if raw_state is not None:
                        names = dataset_metadata.features["observation.state"]["names"]
                        print(f"  [Step {step}] 当前关节状态:")
                        for i, name in enumerate(names):
                            print(f"    {name}: {raw_state[i]:.4f}")
                    if step == 0:
                        print(f"  [Step 0] 原始观测 keys: {list(obs.keys())}")

                # (b) 构建推理用的数据帧（匹配数据集特征格式）
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                # 打印构建的帧（前3步调试用）
                if step == 0:
                    print(f"  [Step 0] 构建的观测帧 keys: {list(obs_frame.keys())}")
                    for k, v in obs_frame.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                        else:
                            print(f"    {k}: {v}")

                # (c) 预处理（归一化等）
                obs_processed = preprocess(obs_frame)

                # 打印预处理后的观测（前3步调试用）
                if step == 0:
                    print(f"  [Step 0] 预处理后 keys: {list(obs_processed.keys())}")
                    for k, v in obs_processed.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                        else:
                            print(f"    {k}: {v}")

                # (d) 模型推理：VQ-BeT 用 GPT 预测离散码 + offset
                with torch.no_grad():
                    action_raw = model.select_action(obs_processed)

                # 打印模型原始输出和队列状态（前3步调试用）
                if step < 3:
                    print(f"  [Step {step}] 模型原始输出 (归一化): {action_raw.tolist()}")
                    with torch.no_grad():
                        for k, q in model._queues.items():
                            print(f"    队列 {k}: len={len(q)}, maxlen={q.maxlen}")

                # (e) 后处理（反归一化等）
                action = postprocess(action_raw)

                # 打印反归一化后的动作（前3步调试用）
                if step < 3:
                    print(f"  [Step {step}] 反归一化后 (tensor): {action.tolist()}")

                # (f) 转换为机械臂动作格式并执行
                action_dict = make_robot_action(action, dataset_metadata.features)

                # 打印发送给机械臂的动作（前3步调试用）
                if step < 3:
                    print(f"  [Step {step}] 发送给机械臂:")
                    for name, val in action_dict.items():
                        print(f"    {name}: {val:.4f}")
                    print()

                robot.send_action(action_dict)

                # 打印进度
                if (step + 1) % 10 == 0 or (step + 1) == MAX_STEPS_PER_EPISODE:
                    print(f"  Step: {step + 1}/{MAX_STEPS_PER_EPISODE}", end="\r")

            print(f"\n  Episode {episode + 1} 完成！")

    except KeyboardInterrupt:
        print("\n\n用户中断，停止推理。")
    finally:
        robot.disconnect()
        print("✓ 机械臂已断开")


if __name__ == "__main__":
    main()
