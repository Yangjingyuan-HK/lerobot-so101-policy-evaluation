"""
摄像头诊断脚本：检测摄像头是否返回实时画面还是缓存的旧帧。
原理：连续捕获多帧，检查帧之间是否有差异（真实摄像头会有传感器噪声）。
如果所有帧完全相同，说明摄像头返回的是缓存帧（死帧）。
"""
import time
import sys
import numpy as np
import cv2

# Windows MSMF 兼容
import os
import platform
if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def diagnose_camera(index, num_frames=10, interval=0.3):
    """诊断单个摄像头"""
    print(f"\n{'='*60}")
    print(f"诊断摄像头 #{index}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  ❌ 无法打开摄像头 #{index}")
        return False

    backend = cap.getBackendName()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  后端: {backend}")
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {fps}")

    # 先丢弃前几帧（预热）
    print(f"  预热: 丢弃前 5 帧...")
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    frames = []
    timestamps = []
    read_times = []

    print(f"  捕获 {num_frames} 帧 (间隔 {interval}s)...")
    for i in range(num_frames):
        t_start = time.perf_counter()
        ret, frame = cap.read()
        t_end = time.perf_counter()
        read_ms = (t_end - t_start) * 1000

        if not ret:
            print(f"    帧 {i}: ❌ 读取失败 (ret=False)")
            continue

        if frame is None:
            print(f"    帧 {i}: ❌ 返回 None")
            continue

        frames.append(frame)
        timestamps.append(t_end)
        read_times.append(read_ms)
        print(f"    帧 {i}: ✅ shape={frame.shape}, dtype={frame.dtype}, "
              f"mean={frame.mean():.2f}, read_time={read_ms:.1f}ms")

        time.sleep(interval)

    cap.release()

    if len(frames) < 2:
        print(f"  ⚠️ 只捕获到 {len(frames)} 帧，无法比较")
        return False

    # 分析帧差异
    print(f"\n  --- 帧差异分析 ---")
    all_identical = True
    max_diff = 0
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        diff_max = diff.max()
        diff_mean = diff.mean()
        max_diff = max(max_diff, diff_max)
        if diff_max > 0:
            all_identical = False
        print(f"    帧 {i-1} → 帧 {i}: max_diff={diff_max}, mean_diff={diff_mean:.4f}")

    print(f"\n  --- 诊断结论 ---")
    print(f"  最大差异: {max_diff}")
    print(f"  平均读取时间: {np.mean(read_times):.1f}ms")
    print(f"  最大读取时间: {np.max(read_times):.1f}ms")
    print(f"  最小读取时间: {np.min(read_times):.1f}ms")

    if all_identical:
        print(f"  🚨 严重问题: 所有 {len(frames)} 帧完全相同！")
        print(f"     摄像头返回的是缓存/死帧，不是实时画面。")
        print(f"     原因: USB供电不足、驱动崩溃、或摄像头被其他进程占用。")
        return False
    elif max_diff < 5:
        print(f"  ⚠️ 警告: 帧间差异极小 (max_diff={max_diff})")
        print(f"     摄像头可能在返回近似相同的帧（接近死帧）。")
        print(f"     如果镜头前有移动物体但画面不变，确认是死帧。")
        return False
    else:
        print(f"  ✅ 正常: 帧间有差异，摄像头在返回实时画面。")
        return True


def check_camera_processes():
    """检查是否有其他进程占用摄像头"""
    print(f"\n{'='*60}")
    print("检查摄像头占用进程")
    print(f"{'='*60}")
    try:
        import subprocess
        result = subprocess.run(
            ['powershell', '-Command',
             'Get-Process | Where-Object {$_.Modules -ne $null} | '
             'Where-Object {$_.Modules.FileName -match "avicap|mf|mfcapture"} | '
             'Select-Object Id, ProcessName -Unique'],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("  未检测到占用摄像头的进程")
    except Exception as e:
        print(f"  无法检查进程: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("LeRobot 摄像头诊断工具")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 检查 OpenCV 版本
    print(f"OpenCV 版本: {cv2.__version__}")

    # 诊断每个摄像头
    results = {}
    for idx in [0, 1, 2]:
        results[idx] = diagnose_camera(idx, num_frames=8, interval=0.25)

    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    for idx, ok in results.items():
        status = "✅ 正常" if ok else "❌ 异常"
        print(f"  摄像头 #{idx}: {status}")

    if not all(results.values()):
        print(f"\n⚠️ 有摄像头异常！这解释了为什么:")
        print(f"   1. lerobot-find-cameras 保存的图片是旧的（摄像头返回缓存帧）")
        print(f"   2. 推理 1 秒后超时（摄像头停止返回新帧）")
        print(f"\n建议修复步骤:")
        print(f"   1. 拔掉所有 USB 摄像头，等 5 秒，重新插入")
        print(f"   2. 换到主板后置 USB 3.0 接口（不要用前置面板或集线器）")
        print(f"   3. 如果多个摄像头在同一 USB 控制器上，分散到不同控制器")
        print(f"   4. 设备管理器 → 通用串行总线控制器 → USB Root Hub →")
        print(f"      属性 → 电源管理 → 取消'允许计算机关闭此设备以节省电源'")
        print(f"   5. 重启电脑（清除驱动状态）")
