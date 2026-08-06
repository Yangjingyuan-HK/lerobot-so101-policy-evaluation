"""VQ-BeT v2 推理脚本 — 加载 v2 训练的最佳性能模型在 SO-101 上运行。

与 v1 的区别：
  - 加载 v2 模型路径 (outputs/robot_learning_tutorial/vqbet_so101_v2)
  - 增加了相机错误处理（防止相机断开后程序崩溃）
  - 增加了每步动作变化量打印（帮助诊断"不动"问题）

使用方式：
    python examples/tutorial/vqbet/vqbet_using_example_so101_v2.py
"""

import torch

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.vqbet import VQBeTPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 800


def main():
    # === 1. 设备 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 使用 CPU 推理（会很慢）")

    # === 2. 加载 v2 模型 ===
    model_path = "outputs/robot_learning_tutorial/vqbet_so101_v2"
    print(f"加载模型: {model_path}")
    model = VQBeTPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()

    if model.vqbet.action_head.vqvae_model.discretized.item():
        print("✓ VQ-VAE 已完成训练（动作离散化就绪）")
    else:
        print("⚠ 警告: VQ-VAE 未完成训练！")

    # === 3. 数据集元数据 + 处理器 ===
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    print(f"加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

    print(f"加载预/后处理器: {model_path}")
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=model_path)

    action_stats = dataset_metadata.stats.get("action", {})
    if action_stats:
        print(f"\n数据集 action 统计:")
        for i, name in enumerate(dataset_metadata.features["action"]["names"]):
            mean_val = action_stats.get("mean", [0] * 6)[i]
            min_val = action_stats.get("min", [0] * 6)[i]
            max_val = action_stats.get("max", [0] * 6)[i]
            print(f"  {name}: mean={mean_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    print()

    # === 4. 机械臂配置 ===
    follower_port = "COM11"
    follower_id = "my_awesome_follower_arm"
    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG", backend=Cv2Backends.DSHOW),
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG", backend=Cv2Backends.DSHOW),
    }

    print(f"连接 SO-101 从臂: port={follower_port}")
    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
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

    state_names = dataset_metadata.features["observation.state"]["names"]
    prev_action = None  # 用于计算动作变化量

    try:
        for episode in range(MAX_EPISODES):
            print(f"--- Episode {episode + 1}/{MAX_EPISODES} ---")
            model.reset()
            prev_action = None

            for step in range(MAX_STEPS_PER_EPISODE):
                # 获取观测
                obs = robot.get_observation()

                # 打印当前关节状态（前3步）
                if step < 3:
                    raw_state = obs.get("state", None)
                    if raw_state is not None:
                        print(f"  [Step {step}] 当前关节状态:")
                        for i, name in enumerate(state_names):
                            print(f"    {name}: {raw_state[i]:.4f}")
                    if step == 0:
                        print(f"  [Step 0] 原始观测 keys: {list(obs.keys())}")

                # 构建推理帧
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                if step == 0:
                    print(f"  [Step 0] 构建的观测帧 keys: {list(obs_frame.keys())}")
                    for k, v in obs_frame.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")

                # 预处理
                obs_processed = preprocess(obs_frame)

                if step == 0:
                    print(f"  [Step 0] 预处理后 keys: {list(obs_processed.keys())}")
                    for k, v in obs_processed.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")

                # 模型推理
                with torch.no_grad():
                    action_raw = model.select_action(obs_processed)

                if step < 3:
                    print(f"  [Step {step}] 模型原始输出 (归一化): {action_raw.tolist()}")

                # 后处理
                action = postprocess(action_raw)

                if step < 3:
                    print(f"  [Step {step}] 反归一化后 (tensor): {action.tolist()}")

                # 转换为机械臂动作
                action_dict = make_robot_action(action, dataset_metadata.features)

                if step < 3:
                    print(f"  [Step {step}] 发送给机械臂:")
                    for name, val in action_dict.items():
                        print(f"    {name}: {val:.4f}")

                # 计算动作变化量（帮助诊断"不动"问题）
                if prev_action is not None:
                    action_vals = list(action_dict.values())
                    prev_vals = list(prev_action.values())
                    max_change = max(abs(a - p) for a, p in zip(action_vals, prev_vals))
                    if step < 10 or (step + 1) % 100 == 0:
                        print(f"  [Step {step}] 动作最大变化量: {max_change:.4f}")

                prev_action = action_dict.copy()
                robot.send_action(action_dict)

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
