import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 800


def _reset_usb_cameras() -> None:
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("lerobot-find-cameras") is None:
        return
    tmp = Path(tempfile.mkdtemp(prefix="lerobot_cam_reset_"))
    try:
        subprocess.run(
            ["lerobot-find-cameras", "opencv", "--record-time", "1.5", "--fps", "15"],
            cwd=str(tmp),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("[1/4] model loading...", flush=True)

    model_path = PROJECT_ROOT / "outputs" / "robot_learning_tutorial" / "diffusion_so101"

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            "Please verify:\n"
            "  1. Training has completed successfully\n"
            "  2. Or update model_path to the correct absolute directory\n"
            "  3. For HF Hub loading, use 'username/repo_name' directly"
        )

    model = DiffusionPolicy.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    print("[2/4] model loaded, resetting USB cameras...", flush=True)
    _reset_usb_cameras()

    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

    preprocess, postprocess = make_pre_post_processors(
        model.config, dataset_stats=dataset_metadata.stats
    )

    action_stats = dataset_metadata.stats.get("action", {})
    if action_stats:
        for i, name in enumerate(dataset_metadata.features["action"]["names"]):
            mean_val = action_stats.get("mean", [0] * 6)[i]
            std_val = action_stats.get("std", [1] * 6)[i]
            min_val = action_stats.get("min", [-3.14] * 6)[i]
            max_val = action_stats.get("max", [3.14] * 6)[i]

    follower_port = "COM9"
    follower_id = "my_awesome_follower_arm"

    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=15, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=15, fourcc="MJPG"),
    }

    robot_cfg = SO101FollowerConfig(
        port=follower_port, id=follower_id, cameras=camera_config
    )
    robot = SO101Follower(robot_cfg)

    print("[3/4] connecting SO101 robot...", flush=True)
    try:
        robot.connect(calibrate=False)
    except Exception as e:
        return

    try:
        print("[4/4] running inference...", flush=True)
        for episode in range(MAX_EPISODES):
            for step in range(MAX_STEPS_PER_EPISODE):
                obs = robot.get_observation()

                if step < 3:
                    raw_state = obs.get("state", None)
                    if raw_state is not None:
                        names = dataset_metadata.features["observation.state"]["names"]
                        for i, name in enumerate(names):
                            pass

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                obs_processed = preprocess(obs_frame)

                with torch.no_grad():
                    action_raw = model.select_action(obs_processed)

                action = postprocess(action_raw)

                action_dict = make_robot_action(action, dataset_metadata.features)

                robot.send_action(action_dict)

                if (step + 1) % 10 == 0 or (step + 1) == MAX_STEPS_PER_EPISODE:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
