"""
对比 lerobot-find-cameras 保存的图片与实时捕获的帧。
使用 PIL 读取（兼容中文路径），使用 cv2 实时捕获。
"""
import time
import os
import platform
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

CAPTURED_DIR = Path("c:/Users/yangj/Desktop/YANG Jingyuan/其他/lerobot/outputs/captured_images")

def analyze_image(img, label):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return {
        "label": label,
        "shape": img.shape,
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": int(img.min()),
        "max": int(img.max()),
    }

def print_stats(stats):
    print(f"  {stats['label']}:")
    print(f"    shape={stats['shape']}, mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
          f"min={stats['min']}, max={stats['max']}")

print("=" * 60)
print("图片对比分析")
print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 1. 用 PIL 读取保存的图片（兼容中文路径）
print("\n--- 保存的图片 (lerobot-find-cameras 输出) ---")
saved_images = {}
for f in sorted(CAPTURED_DIR.glob("*.png")):
    try:
        img_pil = Image.open(f)
        img = np.array(img_pil)
        stats = analyze_image(img, f.name)
        print_stats(stats)
        saved_images[f.stem] = img
    except Exception as e:
        print(f"  {f.name}: 读取失败 - {e}")

# 2. 实时捕获
print("\n--- 实时捕获 ---")
live_images = {}
for idx in [0, 1, 2]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        for _ in range(10):
            cap.read()
            time.sleep(0.02)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stats = analyze_image(frame_rgb, f"live_camera_{idx}")
            print_stats(stats)
            live_images[idx] = frame_rgb
    cap.release()
    time.sleep(0.1)

# 3. 对比
print("\n--- 对比结果 ---")
for idx in [0, 1, 2]:
    saved_key = f"opencv_{idx}"
    if saved_key in saved_images and idx in live_images:
        saved = saved_images[saved_key]
        live = live_images[idx]

        if saved.shape != live.shape:
            print(f"\n  摄像头 #{idx}: 尺寸不同! saved={saved.shape}, live={live.shape}")
            live_resized = cv2.resize(live, (saved.shape[1], saved.shape[0]))
        else:
            live_resized = live

        diff = cv2.absdiff(saved, live_resized)
        max_diff = int(diff.max())
        mean_diff = float(diff.mean())
        diff_pixels = int((diff.sum(axis=2) > 10).sum())
        total_pixels = saved.shape[0] * saved.shape[1]
        diff_percent = diff_pixels / total_pixels * 100

        print(f"\n  摄像头 #{idx}:")
        print(f"    保存图片: mean={saved.mean():.2f}, std={saved.std():.2f}")
        print(f"    实时画面: mean={live_resized.mean():.2f}, std={live_resized.std():.2f}")
        print(f"    最大像素差异: {max_diff}")
        print(f"    平均像素差异: {mean_diff:.4f}")
        print(f"    差异像素占比: {diff_percent:.1f}%")

        if max_diff == 0:
            print(f"    🚨 完全相同！保存的图片 = 实时画面（同一缓存帧）")
        elif diff_percent > 50:
            print(f"    ✅ 差异很大 — 保存的图片和实时画面完全不同（图片是旧的）")
        elif diff_percent > 20:
            print(f"    ℹ️ 有明显差异 — 可能是不同时刻的同一场景")
        else:
            print(f"    ℹ️ 差异较小 — 同一场景不同时刻（正常，如果场景静态）")

print("\n" + "=" * 60)
