import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 500


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU 设备")

    model_id = "./outputs/pi0_so101_final/checkpoints/020000/pretrained_model"
    print(f"加载模型: {model_id}")

    model = PI0Policy.from_pretrained(model_id)
    model.to(device)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    follower_port = "COM9"
    follower_id = "my_awesome_follower_arm"

    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
    }

    print(f"连接 SO-101 从臂: port={follower_port}, id={follower_id}")
    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO101Follower(robot_cfg)

    try:
        robot.connect()
        print("机器人连接成功！")
    except Exception as e:
        print(f"连接失败: {e}")
        return

    task = "Grab the red cube"
    robot_type = "so101_follower"

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    try:
        for episode in range(MAX_EPISODES):
            print(f"\n--- Episode {episode + 1}/{MAX_EPISODES} ---")
            for step in range(MAX_STEPS_PER_EPISODE):
                obs = robot.get_observation()

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=task,
                    robot_type=robot_type,
                )

                obs = preprocess(obs_frame)
                action = model.select_action(obs)
                action = postprocess(action)
                action = make_robot_action(action, dataset_features)
                robot.send_action(action)

                if step % 20 == 0:
                    print(f"  Step: {step}/{MAX_STEPS_PER_EPISODE}")
                    if isinstance(action, dict):
                        for k, v in action.items():
                            if hasattr(v, 'shape'):
                                print(f"    {k}: min={v.min():.4f}, max={v.max():.4f}")

            print(f"  Episode {episode + 1} 完成！")

    except KeyboardInterrupt:
        print("\n用户中断，正在关闭连接...")
    finally:
        robot.disconnect()
        print("机器人已断开连接")


if __name__ == "__main__":
    main()
