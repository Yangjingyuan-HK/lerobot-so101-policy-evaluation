"""在真实 SO-101 机械臂上运行训练好的 ACT 策略（适配 Windows + CUDA）。

本脚本演示如何加载训练好的 ACT 模型，在 SO-101 从臂上运行推理。
它会循环执行多个 episode，每个 episode 中：
    1. 从机械臂获取观测（图像 + 关节状态）
    2. 用 ACT 模型预测动作
    3. 将动作发送给机械臂执行

使用方式（在已激活 lerobot 环境的终端中运行）：
    python examples/tutorial/act/act_using_example_so101.py

前置条件：
    1. 已用 act_training_example_so101.py 训练好模型
    2. SO-101 从臂已连接（端口 COM9）并校准
    3. 相机已连接（index 0 = front, index 2 = side）
"""

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# === 配置参数 ===
MAX_EPISODES = 1  # 运行多少个 episode
MAX_STEPS_PER_EPISODE = 800  # 每个 episode 最多执行多少步（每步约 1/fps 秒）


def main():
    # === 1. 设备选择 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA，使用 CPU 推理（会很慢）")

    # === 2. 加载训练好的模型 ===
    # 方式一：从本地路径加载（用 act_training_example_so101.py 训练后的输出）
    model_path = "outputs/robot_learning_tutorial/act_so101"

    # 方式二：从 HuggingFace Hub 加载（如果已上传）
    # model_path = "你的用户名/act_so101_pickplace"

    print(f"加载模型: {model_path}")
    model = ACTPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # === 3. 加载数据集元数据（用于构建推理时的数据帧） ===
    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    print(f"加载数据集元数据: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

    # 创建预处理器和后处理器（必须和训练时一致）
    preprocess, postprocess = make_pre_post_processors(
        model.config, dataset_stats=dataset_metadata.stats
    )

    # === 4. 配置机械臂和相机 ===
    # 注意：相机配置必须和训练数据集时一致（名称和分辨率）！
    # 可以在数据集的 info.json 中查看 camera keys
    follower_port = "COM9"  # 从臂端口（用 lerobot-find-port 查找）
    follower_id = "my_awesome_follower_arm"  # 从臂 ID（用于加载校准文件）

    camera_config = {
        # 相机名称必须和数据集中的 key 一致
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
    }

    print(f"连接 SO-101 从臂: port={follower_port}, id={follower_id}")
    robot_cfg = SO101FollowerConfig(
        port=follower_port, id=follower_id, cameras=camera_config
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    print("✓ 机械臂已连接")

    # === 5. 推理循环 ===
    print(f"\n开始推理: {MAX_EPISODES} 个 episode, 每个 episode 最多 {MAX_STEPS_PER_EPISODE} 步")
    print("按 Ctrl+C 可随时停止\n")

    try:
        for episode in range(MAX_EPISODES):
            print(f"--- Episode {episode + 1}/{MAX_EPISODES} ---")
            reached_max_steps = False
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # (a) 获取机械臂观测（图像 + 关节状态）
                obs = robot.get_observation()

                # (b) 构建推理用的数据帧（匹配数据集特征格式）
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                # (c) 预处理（归一化等）
                obs_processed = preprocess(obs_frame)

                # (d) 模型推理：预测一个动作
                with torch.no_grad():
                    action = model.select_action(obs_processed)

                # (e) 后处理（反归一化等）
                action = postprocess(action)

                # (f) 转换为机械臂动作格式并执行
                action = make_robot_action(action, dataset_metadata.features)
                robot.send_action(action)

                # 打印进度（每10步打印一次，避免刷屏）
                if (step + 1) % 10 == 0 or (step + 1) == MAX_STEPS_PER_EPISODE:
                    print(f"  Step: {step + 1}/{MAX_STEPS_PER_EPISODE}", end="\r")

            reached_max_steps = True
            print(f"\n  Episode {episode + 1} 完成！")
            print(f"  实际运行步数: {MAX_STEPS_PER_EPISODE} 步（已达到最大值）")
            print(f"  ⚠️  如果动作未完成，建议将 MAX_STEPS_PER_EPISODE 从 {MAX_STEPS_PER_EPISODE} 调大")

    except KeyboardInterrupt:
        print("\n\n用户中断，停止推理。")
    finally:
        # 清理：断开机械臂连接
        robot.disconnect()
        print("✓ 机械臂已断开")


if __name__ == "__main__":
    main()
