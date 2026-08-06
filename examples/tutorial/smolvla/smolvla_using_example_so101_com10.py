"""在真实 SO-101 机械臂上运行训练好的 SmolVLA 策略（COM10 端口版）。

与原版区别：
  - 从臂端口从 COM9 改为 COM10

SmolVLA 推理流程：
    1. 从机械臂获取观测（图像 + 关节状态）
    2. 设置语言指令（如 "pick up the block"）
    3. 用 VLM 编码图像和语言，用 Flow Matching 生成 50 步动作
    4. 从动作队列中逐步执行

使用方式：
    python examples/tutorial/smolvla/smolvla_using_example_so101_com10.py

前置条件：
    1. 已用 smolvla_training_example_so101.py 训练好模型
    2. SO-101 从臂已连接（端口 COM10）并校准
    3. 相机已连接（index 0 = front, index 2 = side）
"""

import sys

# === 依赖检查 ===
try:
    import transformers  # noqa: F401
    import num2words  # noqa: F401
    import accelerate  # noqa: F401
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请先安装 smolvla 依赖：")
    print('  pip install transformers num2words accelerate')
    sys.exit(1)

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 800

# 语言指令（必须和训练时一致！）
TASK_DESCRIPTION = "pick up the block and place it"


def main():
    # === 1. 设备 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 使用 CPU 推理（会很慢）")

    # === 2. 加载模型 ===
    model_path = "outputs/robot_learning_tutorial/smolvla_so101"
    print(f"加载模型: {model_path}")
    model = SmolVLAPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ 模型已加载")

    # === 3. 数据集元数据 + 处理器 ===
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    print(f"加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

    print(f"加载预/后处理器: {model_path}")
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=model_path)

    # 打印 action 统计
    action_stats = dataset_metadata.stats.get("action", {})
    if action_stats:
        print(f"\n数据集 action 统计:")
        for i, name in enumerate(dataset_metadata.features["action"]["names"]):
            mean_val = action_stats.get("mean", [0] * 6)[i]
            std_val = action_stats.get("std", [1] * 6)[i]
            print(f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}")
    print()

    # === 4. 机械臂配置（COM10 端口）===
    follower_port = "COM10"  # 改为 COM10
    follower_id = "my_awesome_follower_arm"
    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
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
    print(f"语言指令: '{TASK_DESCRIPTION}'")
    print("按 Ctrl+C 可随时停止\n")

    state_names = dataset_metadata.features["observation.state"]["names"]
    prev_action = None

    try:
        for episode in range(MAX_EPISODES):
            print(f"--- Episode {episode + 1}/{MAX_EPISODES} ---")
            model.reset()
            prev_action = None

            for step in range(MAX_STEPS_PER_EPISODE):
                # (a) 获取观测
                obs = robot.get_observation()

                if step < 3:
                    raw_state = obs.get("state", None)
                    if raw_state is not None:
                        print(f"  [Step {step}] 当前关节状态:")
                        for i, name in enumerate(state_names):
                            print(f"    {name}: {raw_state[i]:.4f}")
                    if step == 0:
                        print(f"  [Step 0] 原始观测 keys: {list(obs.keys())}")

                # (b) 构建推理帧
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                # 设置语言指令（SmolVLA 需要）
                if "task" in obs_frame:
                    obs_frame["task"] = TASK_DESCRIPTION

                if step == 0:
                    print(f"  [Step 0] 构建的观测帧 keys: {list(obs_frame.keys())}")
                    for k, v in obs_frame.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                        else:
                            print(f"    {k}: {v}")

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
                action_dict = make_robot_action(action, dataset_metadata.features)

                if step < 3:
                    print(f"  [Step {step}] 发送给机械臂:")
                    for name, val in action_dict.items():
                        print(f"    {name}: {val:.4f}")

                # 动作变化量
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
