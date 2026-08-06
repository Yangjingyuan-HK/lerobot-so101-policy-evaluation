"""使用微调后的 PI0-FAST 模型在 SO-101 上推理。

与零样本版的区别：
  - 从本地输出目录加载微调后的模型
  - 使用微调时保存的处理器（包含本地数据集的归一化统计值）

使用方式：
    python examples/tutorial/pi0_fast/pi0_fast_using_example_so101.py

前置条件：
    1. 已用 pi0_fast_training_example_so101.py 微调好模型
    2. SO-101 从臂已连接（端口 COM10）并校准
    3. 相机已连接（index 0 = front, index 2 = side）
"""

import sys

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

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0_fast import PI0FastPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 500

# 语言指令（必须和训练时一致！）
TASK_DESCRIPTION = "pick up the block"


def main():
    # === 1. 设备 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 使用 CPU 推理（会很慢）")

    # === 2. 加载微调后的模型 ===
    model_path = "outputs/robot_learning_tutorial/pi0fast_so101"
    print(f"加载微调模型: {model_path}")
    model = PI0FastPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ 模型已加载")

    # === 3. 加载处理器（从微调输出目录，包含本地数据集统计值）===
    print(f"加载处理器: {model_path}")
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=model_path)
    print("✓ 处理器已加载")

    # === 4. 机械臂配置 ===
    follower_port = "COM10"
    follower_id = "my_awesome_follower_arm"
    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
    }

    print(f"\n连接 SO-101 从臂: port={follower_port}")
    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO101Follower(robot_cfg)

    try:
        robot.connect()
        print("✓ 机械臂已连接")
    except Exception as e:
        print(f"连接失败: {e}")
        return

    # === 5. 获取机械臂的特征定义 ===
    robot_type = "so101_follower"
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # === 6. 推理循环 ===
    print(f"\n{'='*60}")
    print(f"PI0-FAST 微调模型推理")
    print(f"{'='*60}")
    print(f"  模型: {model_path}")
    print(f"  语言指令: '{TASK_DESCRIPTION}'")
    print(f"  机器人类型: {robot_type}")
    print(f"  Episodes: {MAX_EPISODES}, 每集最多 {MAX_STEPS_PER_EPISODE} 步")
    print(f"  按 Ctrl+C 可随时停止\n")

    try:
        for episode in range(MAX_EPISODES):
            print(f"--- Episode {episode + 1}/{MAX_EPISODES} ---")
            model.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                # (a) 获取观测
                obs = robot.get_observation()

                if step < 3:
                    print(f"  [Step {step}] 原始观测 keys: {list(obs.keys())}")

                # (b) 构建推理帧
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK_DESCRIPTION,
                    robot_type=robot_type,
                )

                if step == 0:
                    print(f"  [Step 0] 构建的观测帧 keys: {list(obs_frame.keys())}")

                # (c) 预处理
                obs_processed = preprocess(obs_frame)

                if step == 0:
                    print(f"  [Step 0] 预处理后 keys: {list(obs_processed.keys())}")
                    for k, v in obs_processed.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")

                # (d) 模型推理
                with torch.no_grad():
                    action_raw = model.select_action(obs_processed)

                if step < 3:
                    print(f"  [Step {step}] 模型原始输出 (归一化): {action_raw.tolist()}")

                # (e) 后处理
                action = postprocess(action_raw)

                if step < 3:
                    print(f"  [Step {step}] 反归一化后 (tensor): {action.tolist()}")

                # (f) 转换为机械臂动作
                action_dict = make_robot_action(action, dataset_features)

                if step < 3:
                    print(f"  [Step {step}] 发送给机械臂:")
                    for name, val in action_dict.items():
                        print(f"    {name}: {val:.4f}")

                robot.send_action(action_dict)

                if (step + 1) % 20 == 0 or (step + 1) == MAX_STEPS_PER_EPISODE:
                    print(f"  Step: {step + 1}/{MAX_STEPS_PER_EPISODE}")

            print(f"  Episode {episode + 1} 完成！")

    except KeyboardInterrupt:
        print("\n\n用户中断，停止推理。")
    finally:
        robot.disconnect()
        print("✓ 机械臂已断开")


if __name__ == "__main__":
    main()
