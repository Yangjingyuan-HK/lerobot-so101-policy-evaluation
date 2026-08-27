import sys
from pathlib import Path

try:
    import transformers
    import diffusers
    import einops
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install first: pip install transformers diffusers einops")
    sys.exit(1)

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.multi_task_dit import MultiTaskDiTConfig, MultiTaskDiTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    output_directory = PROJECT_ROOT / "outputs" / "robot_learning_tutorial" / "multi_task_dit_so101"
    output_directory.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Error: CUDA required")
        return
    device = torch.device("cuda")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Device: {torch.cuda.get_device_name(0)}, VRAM: {gpu_mem:.1f} GB")

    dataset_id = "WT/test"
    dataset_root = "C:/Users/yangj/Desktop/BaiduNetdiskDownload/lerobot"
    print(f"\nLoading dataset metadata: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)
    print(f"  episodes: {dataset_metadata.total_episodes}, frames: {dataset_metadata.total_frames}, fps: {dataset_metadata.fps}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = MultiTaskDiTConfig(
        input_features=input_features,
        output_features=output_features,
        objective="flow_matching",
        n_obs_steps=2,
        horizon=32,
        n_action_steps=24,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        vision_encoder_name="openai/clip-vit-base-patch16",
        text_encoder_name="openai/clip-vit-base-patch16",
        image_resize_shape=(256, 256),
        image_crop_shape=(224, 224),
        image_crop_is_random=True,
        optimizer_lr=2e-5,
        optimizer_betas=(0.95, 0.999),
        optimizer_weight_decay=0.0,
        vision_encoder_lr_multiplier=0.1,
        scheduler_name="cosine",
        scheduler_warmup_steps=500,
        device=str(device),
    )
    grad_clip_norm = 1.0

    print(f"\nCreating Multi-Task DiT model (first run will download CLIP ~600MB)...")
    model = MultiTaskDiTPolicy(cfg)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params / 1e6:.1f}M, Trainable: {trainable_params / 1e6:.1f}M")

    print("\nCreating processors...")
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    fps = dataset_metadata.fps
    delta_timestamps = {
        "action": [i / fps for i in cfg.action_delta_indices],
        "observation.state": [i / fps for i in cfg.observation_delta_indices],
    }
    for k in cfg.image_features:
        delta_timestamps[k] = [i / fps for i in cfg.observation_delta_indices]

    print(f"Loading full dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  Samples: {len(dataset)}")

    optimizer = cfg.get_optimizer_preset().build(model.get_optim_params())

    batch_size = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
    )

    training_steps = 20000
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, training_steps)

    log_freq = 100
    save_freq = 5000

    default_task = "pick up the block"

    print(f"\n{'='*60}")
    print(f"Multi-Task DiT Training Configuration")
    print(f"{'='*60}")
    print(f"  Objective: flow_matching")
    print(f"  horizon: {cfg.horizon}, n_action_steps: {cfg.n_action_steps}, n_obs_steps: {cfg.n_obs_steps}")
    print(f"  Transformer: {cfg.num_layers} layers, hidden_dim={cfg.hidden_dim}, heads={cfg.num_heads}")
    print(f"  batch_size: {batch_size}, training_steps: {training_steps}")
    print(f"  Learning rate: {cfg.optimizer_lr}, warmup: {cfg.scheduler_warmup_steps} steps")
    print(f"  Language instruction: '{default_task}'")
    print(f"  Log: every {log_freq} steps, Save: every {save_freq} steps")
    print(f"{'='*60}\n")

    step = 0
    done = False

    while not done:
        for batch in dataloader:
            if "task" in batch:
                batch["task"] = [default_task if not t.strip() else t for t in batch["task"]]

            batch = preprocessor(batch)

            loss, info = model.forward(batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"step: {step:>6d}  loss: {loss.item():.4f}  lr: {current_lr:.6f}  "
                      f"VRAM: {gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f} GB")

            if step > 0 and step % save_freq == 0:
                print(f"  -> Saving checkpoint (step {step})")
                model.save_pretrained(output_directory)
                preprocessor.save_pretrained(output_directory)
                postprocessor.save_pretrained(output_directory)

            step += 1
            if step >= training_steps:
                done = True
                break

    print(f"\nTraining complete! Saved to: {output_directory}")
    model.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print("Done! Test the model with multi_task_dit_using_example_so101.py.")


if __name__ == "__main__":
    main()
