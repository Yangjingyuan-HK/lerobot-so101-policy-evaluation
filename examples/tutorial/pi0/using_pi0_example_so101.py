import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 500

TASK = "Grab the red cube"
ROBOT_TYPE = "so101_follower"

FOLLOWER_PORT = "COM9"
FOLLOWER_ID = "my_awesome_follower_arm"


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


def _resolve_model_path() -> Path:
    candidates = [
        PROJECT_ROOT / "outputs" / "robot_learning_tutorial" / "pi0_so101",
        PROJECT_ROOT / "outputs" / "pi0_so101_final" / "checkpoints" / "020000" / "pretrained_model",
    ]
    for p in candidates:
        if p.is_dir() and (p / "config.json").exists():
            return p
    raise FileNotFoundError(
        "PI0 model directory not found. Checked:\n"
        + "".join(f"  - {c}\n" for c in candidates)
        + "Please train PI0 first (e.g. outputs/robot_learning_tutorial/pi0_so101/) "
        "or update PROJECT_ROOT/model_path in the script."
    )


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("[1/4] model loading...", flush=True)

    model_path = _resolve_model_path()

    model = PI0Policy.from_pretrained(str(model_path))
    model.to(device)
    print("[2/4] model loaded, resetting USB cameras...", flush=True)
    _reset_usb_cameras()

    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=15, fourcc="MJPG"),
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=15, fourcc="MJPG"),
    }

    robot_cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=camera_config)
    robot = SO101Follower(robot_cfg)

    print("[3/4] connecting SO101 robot...", flush=True)
    try:
        robot.connect(calibrate=False)
    except Exception as e:
        return

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    try:
        print("[4/4] running inference...", flush=True)
        for episode in range(MAX_EPISODES):
            for step in range(MAX_STEPS_PER_EPISODE):
                obs = robot.get_observation()

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )

                obs_processed = preprocess(obs_frame)
                with torch.no_grad():
                    action = model.select_action(obs_processed)
                action = postprocess(action)
                action_dict = make_robot_action(action, dataset_metadata.features)
                robot.send_action(action_dict)

                if (step + 1) % 20 == 0 or (step + 1) == MAX_STEPS_PER_EPISODE:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
