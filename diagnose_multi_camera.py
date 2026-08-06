"""
多摄像头同时运行诊断：模拟推理场景，同时打开多个摄像头。
检测多摄像头同时运行时是否出现超时/帧丢失/死帧。
"""
import time
import os
import platform
import numpy as np
import cv2

if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def test_multi_camera(indices=[0, 1, 2], duration=5.0):
    """同时打开多个摄像头，持续捕获帧，检测超时和死帧"""
    print(f"\n{'='*60}")
    print(f"多摄像头同时测试 (摄像头 {indices}, 持续 {duration}s)")
    print(f"{'='*60}")

    caps = {}
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            caps[idx] = cap
            print(f"  摄像头 #{idx}: 已打开 ({cap.getBackendName()})")
        else:
            print(f"  摄像头 #{idx}: ❌ 无法打开")
            cap.release()

    if not caps:
        print("  ❌ 没有摄像头可用")
        return

    # 预热
    print(f"  预热 1 秒...")
    warmup_end = time.time() + 1.0
    while time.time() < warmup_end:
        for cap in caps.values():
            cap.read()

    # 正式捕获
    print(f"  开始持续捕获 {duration} 秒...\n")
    start = time.perf_counter()
    stats = {idx: {"frames": 0, "timeouts": 0, "errors": 0, "read_times": [],
                    "last_frame": None, "identical_count": 0, "max_diff": 0}
             for idx in caps}

    frame_interval = 1.0 / 30  # 目标 30 FPS
    next_frame_time = time.perf_counter()

    while time.perf_counter() - start < duration:
        current = time.perf_counter()
        elapsed = current - start

        for idx, cap in caps.items():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            read_ms = (t1 - t0) * 1000

            if not ret or frame is None:
                stats[idx]["errors"] += 1
                continue

            stats[idx]["frames"] += 1
            stats[idx]["read_times"].append(read_ms)

            # 检查帧是否变化
            if stats[idx]["last_frame"] is not None:
                diff = cv2.absdiff(frame, stats[idx]["last_frame"])
                max_d = int(diff.max())
                if max_d == 0:
                    stats[idx]["identical_count"] += 1
                stats[idx]["max_diff"] = max(stats[idx]["max_diff"], max_d)
            stats[idx]["last_frame"] = frame.copy()

            if read_ms > 200:
                stats[idx]["timeouts"] += 1

        # 控制帧率
        next_frame_time += frame_interval
        sleep_time = next_frame_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 释放
    for cap in caps.values():
        cap.release()

    # 报告
    print(f"  {'='*60}")
    print(f"  结果统计:")
    print(f"  {'='*60}")
    for idx in sorted(caps.keys()):
        s = stats[idx]
        total = s["frames"] + s["errors"]
        avg_read = np.mean(s["read_times"]) if s["read_times"] else 0
        max_read = max(s["read_times"]) if s["read_times"] else 0
        min_read = min(s["read_times"]) if s["read_times"] else 0
        success_rate = s["frames"] / total * 100 if total > 0 else 0

        print(f"\n  摄像头 #{idx}:")
        print(f"    总尝试: {total}")
        print(f"    成功帧数: {s['frames']}")
        print(f"    读取失败: {s['errors']}")
        print(f"    成功率: {success_rate:.1f}%")
        print(f"    读取时间: avg={avg_read:.1f}ms, min={min_read:.1f}ms, max={max_read:.1f}ms")
        print(f"    超时(>200ms): {s['timeouts']} 次")
        print(f"    相同帧: {s['identical_count']} 次 (max_diff={s['max_diff']})")

        if s["timeouts"] > 0:
            print(f"    🚨 有 {s['timeouts']} 次读取超过 200ms！这就是推理超时的原因。")
        if s["identical_count"] > total * 0.5:
            print(f"    ⚠️ 超过一半的帧是相同的，摄像头可能在返回死帧。")
        if success_rate < 95:
            print(f"    ⚠️ 成功率低于 95%，摄像头不稳定。")


if __name__ == "__main__":
    print("=" * 60)
    print("多摄像头同时运行诊断")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"OpenCV: {cv2.__version__}")

    # 测试 1: 两个摄像头 (模拟推理: front=0, side=2)
    test_multi_camera([0, 2], duration=5.0)

    # 测试 2: 三个摄像头
    test_multi_camera([0, 1, 2], duration=5.0)

    print(f"\n{'='*60}")
    print("诊断完成")
    print(f"{'='*60}")
