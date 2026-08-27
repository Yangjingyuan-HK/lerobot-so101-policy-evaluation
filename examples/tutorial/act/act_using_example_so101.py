from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Backends
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
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

    model_path = PROJECT_ROOT / "outputs" / "robot_learning_tutorial" / "act_so101"

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            "Please verify training completed successfully or update model_path."
        )

    model = ACTPolicy.from_pretrained(str(model_path))
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

    follower_port = "COM9"
    follower_id = "my_awesome_follower_arm"

    camera_config = {
        "side": OpenCVCameraConfig(
            index_or_path=0, width=640, height=480, fps=15,
            fourcc=None, backend=Cv2Backends.DSHOW, warmup_s=2,
        ),
        "front": OpenCVCameraConfig(
            index_or_path=2, width=640, height=480, fps=15,
            fourcc=None, backend=Cv2Backends.DSHOW, warmup_s=2,
        ),
    }

    robot_cfg = SO101FollowerConfig(
        port=follower_port, id=follower_id, cameras=camera_config
    )
    robot = SO101Follower(robot_cfg)
    print("[3/4] connecting SO101 robot...", flush=True)
    try:
        robot.connect(calibrate=False)
    except Exception:
        return

    try:
        print("[4/4] running inference...", flush=True)
        for episode in range(MAX_EPISODES):
            for step in range(MAX_STEPS_PER_EPISODE):
                if not robot.is_connected:
                    try:
                        robot.connect(calibrate=False)
                    except Exception as reconn_e:
                        raise RuntimeError from reconn_e

                obs = robot.get_observation()

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                obs_processed = preprocess(obs_frame)

                with torch.no_grad():
                    action = model.select_action(obs_processed)

                action = postprocess(action)

                action = make_robot_action(action, dataset_metadata.features)
                robot.send_action(action)

                if (step + 1) % 10 == 0 or (step + 1) == MAX_STEPS_PER_EPISODE:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        if robot.is_connected:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
