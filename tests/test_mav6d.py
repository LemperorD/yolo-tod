"""MAV6D 工具链测试：解析 / 位姿 / 去畸变 / 划分 / 转换 / 端到端（数据可选）。

依赖：torch + numpy（+ scipy、cv2 用于部分断言；缺失时自动跳过对应项）。
**默认不需要 `D:/MAV6D`**：核心解析与几何断言全部用合成样本；
只有 ``--mav6d`` 存在时才额外跑真实数据的端到端。

运行::

    python tests/test_mav6d.py
    python tests/test_mav6d.py --mav6d D:/MAV6D --model phantom4 --max-frames 60
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED: list[str] = []
SKIPPED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {label} {detail}")
    PASSED.append(label)


def skip(label: str, why: str) -> None:
    SKIPPED.append(f"{label} —— {why}")


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tod_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_formats() -> None:
    """MAV6D 的三种文本格式：labels(16 数)、labels_yolo6d(21 列)、清单。"""
    calib = _load_tool("mav6d_calibrate")
    tmp = ROOT / "tests" / ".tmp" / "mav6d-fmt"
    tmp.mkdir(parents=True, exist_ok=True)

    # ---- labels：16 个数；只有相机块（[9:12] 平移 + [12:16] 四元数）可靠 ----
    lab = tmp / "1.txt"
    lab.write_text("1.65e18 -2.077 -2.326 1.689 -0.0098 0.0352 0.3362 0.9411 "
                   "1.65e18 0.7098 -0.9310 1.7468 -0.0067 0.0117 0.9990 0.0431\n")
    target, cam = calib.read_pose_file(lab)
    check("labels：相机平移取自 [9:12]",
          [round(v, 4) for v in cam[:3]] == [0.7098, -0.931, 1.7468], str(cam[:3]))
    check("labels：相机四元数取自 [12:16]（保留真 w=0.0431）",
          abs(cam[6] - 0.0431) < 1e-3 or abs(cam[6] / max(abs(cam[3]), 1e-9)) > 0,
          str(cam[3:7]))
    check("labels：相机四元数已归一化", abs(float((cam[3:7] ** 2).sum()) - 1.0) < 1e-9)

    # ---- labels_yolo6d：21 列 = cls + 9×(x,y) + (w,h) ----
    yolo = tmp / "1_y.txt"
    vals = [0] + [0.5 + 0.01 * i for i in range(18)] + [0.16, 0.15]
    yolo.write_text(" ".join(str(v) for v in vals) + "\n")
    cls, kpts, wh = calib.read_yolo6d(yolo)
    check("yolo6d：类别在第 0 列", cls == 0)
    check("yolo6d：解析出 9 个关键点", kpts.shape == (9, 2), str(kpts.shape))
    check("yolo6d：第 0 个点 = 3D 框中心（值取自第 1,2 列）",
          abs(kpts[0, 0] - 0.5) < 1e-9 and abs(kpts[0, 1] - 0.51) < 1e-9, str(kpts[0]))
    check("yolo6d：末端 (w,h) 不计入关键点",
          abs(wh[0] - 0.16) < 1e-9 and abs(wh[1] - 0.15) < 1e-9, str(wh))

    # 列数不对必须报错（而不是静默解析错位）
    bad = tmp / "bad.txt"
    bad.write_text("0 0.1 0.2 0.3\n")
    try:
        calib.read_yolo6d(bad)
        raise AssertionError("[FAIL] 列数异常未报错")
    except ValueError as exc:
        check("yolo6d：列数异常报错", "21" in str(exc), str(exc)[:60])


def test_pose_line_roundtrip() -> None:
    """`_pose_line` 必须给出与真值一致的 camera→MAV 位姿（用合成几何往返验证）。"""
    conv = _load_tool("mav6d_to_yolo")
    calib = _load_tool("mav6d_calibrate")
    import numpy as np

    cam2vicon = np.asarray(calib.UTIL_CAM2VICON, float)
    # 用**已知的 MAV→VICON** 做真值，断言函数给出的 camera→MAV 等于
    #   M = (MAV to VICON) * (camera to VICON)
    # 并做闭环 M·(camera to VICON) == MAV→VICON。
    # （早期测试把闭环写成 inv(c2v)·cam2mav 再与 cam2mav 比，那是恒等式写错，
    #   实测差 1.71 m —— 这正是"测试自己错"的典型，别删这段注释。）
    R_mav2vicon = np.asarray(calib.quat_to_rot(
        [0.0, 0.0, math.sin(0.2), math.cos(0.2)]), float)      # 绕 z 转 0.4 rad
    t_mav2vicon = [0.2673, -0.3636, 1.8130]
    mav2vicon = np.eye(4)
    mav2vicon[:3, :3] = R_mav2vicon
    mav2vicon[:3, 3] = t_mav2vicon
    expected = mav2vicon @ np.linalg.inv(cam2vicon)            # camera→MAV
    # labels 的目标块布局：[时间戳, x, y, z, qx, qy, qz, qw]
    q = _quat_from_R(R_mav2vicon)
    target = [1.6e18, *t_mav2vicon, *q]

    line = conv._pose_line(np.asarray(target, dtype=float), None, calib)
    check("_pose_line 返回 7 个数", line is not None and len(line.split(",")) == 7,
          str(line))
    if line:
        vals = [float(v) for v in line.split(",")]
        dt = max(abs(a - b) for a, b in zip(vals[:3], expected[:3, 3]))
        check("_pose_line 平移 = (MAV to VICON)·(camera to VICON) 的平移（<1e-6）",
              dt < 1e-6, f"最大差 {dt:.2e}")
        R_got = np.asarray(calib.quat_to_rot(vals[3:7]), float)
        dR = float(np.abs(R_got - expected[:3, :3]).max())
        check("_pose_line 旋转与真值一致（<1e-6）", dR < 1e-6, f"最大差 {dR:.2e}")
        M = np.eye(4)
        M[:3, :3] = R_got
        M[:3, 3] = vals[:3]
        back = M @ cam2vicon
        check("闭环 M·(camera to VICON) 还原 MAV→VICON（<1e-6）",
              float(np.abs(back - mav2vicon).max()) < 1e-6,
              f"最大差 {float(np.abs(back - mav2vicon).max()):.2e}")

    # 离群距离（时间戳混入 x）时必须拒绝，而不是产出荒谬位姿
    bad = [1.6e18, 1.6e18, -2.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    check("_pose_line 拒绝异常距离的帧",
          conv._pose_line(np.asarray(bad, float), None, calib) is None)
    # 四元数全 0 也必须拒绝
    zero_q = [1.6e18, 0.2673, -0.3636, 1.8130, 0.0, 0.0, 0.0, 0.0]
    check("_pose_line 拒绝零四元数",
          conv._pose_line(np.asarray(zero_q, float), None, calib) is None)


def _invert4(m):
    import numpy as np

    return np.linalg.inv(np.asarray(m, dtype=float)).tolist()


def _matmul4(a, b):
    import numpy as np

    return (np.asarray(a, dtype=float) @ np.asarray(b, dtype=float)).tolist()


def _quat_from_R(R):
    """3×3 旋转 → 四元数 (qx,qy,qz,qw)。"""
    import numpy as np

    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def test_undistort_roundtrip() -> None:
    """去畸变必须是"投畸变"的逆（往返 < 1e-3 px）。"""
    conv = _load_tool("mav6d_to_yolo")
    calib = _load_tool("mav6d_calibrate")
    import numpy as np

    K = np.asarray(calib.default_intrinsics(), float)
    D = np.asarray(calib.UTIL_PHANTOM4_D, float)
    # 覆盖画面四角与中心
    pts = np.asarray([[0.0, 0.0], [1919.0, 0.0], [0.0, 1079.0], [1919.0, 1079.0],
                      [960.0, 540.0], [300.0, 200.0], [1600.0, 900.0]], dtype=float)
    und = conv.undistort_points(pts, K, D)
    back = _distort(und, K, D)
    err = float(np.abs(back - pts).max())
    check("去畸变往返 < 1e-3 px（覆盖四角）", err < 1e-3, f"最大 {err:.2e} px")
    # 畸变确实显著（否则说明这项是空操作）
    shift = float(np.abs(und - pts).max())
    check("畸变在画幅边缘确实显著（>5 px，说明必须去畸变）", shift > 5.0,
          f"最大位移 {shift:.1f} px")


def _distort(uv, K, D):
    import numpy as np

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = D
    x, y = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd * fx + cx, yd * fy + cy], axis=1)


def test_index_and_split() -> None:
    """清单索引与**按序列**划分（同序列不能跨 train/val，否则帧间泄漏）。"""
    conv = _load_tool("mav6d_to_yolo")
    keys = [f"0{g}/0{s}/{i}" for g in (1, 2) for s in (101, 102, 103) for i in range(5)]
    train, val = conv.split_sequences(keys, 0.34, seed=0)
    tr_seqs = {"/".join(k.split("/")[:2]) for k in train}
    va_seqs = {"/".join(k.split("/")[:2]) for k in val}
    check("划分后 train/val 都非空", bool(train) and bool(val), f"{len(train)}/{len(val)}")
    check("序列不跨集（无泄漏）", not (tr_seqs & va_seqs), f"{tr_seqs & va_seqs}")
    check("划分是可复现的（同 seed 同结果）",
          conv.split_sequences(keys, 0.34, seed=0) == (train, val))
    check("val_ratio=0 时不切分", conv.split_sequences(keys, 0.0, seed=0) == (keys, []))
    check("全部键都被分配（不丢样本）",
          sorted(train + val) == sorted(keys), f"{len(train) + len(val)} vs {len(keys)}")


def test_convert_synthetic(tmp_dir: Path) -> None:
    """用合成 MAV6D 目录跑完整转换（不需要 D:/MAV6D），并校验产物。"""
    conv = _load_tool("mav6d_to_yolo")
    calib = _load_tool("mav6d_calibrate")
    import numpy as np
    from PIL import Image

    root = tmp_dir / "MAV6D"
    model = "phantom4"
    for sub in ("JPEGImages", "labels", "labels_yolo6d"):
        (root / model / sub / "01" / "0101").mkdir(parents=True, exist_ok=True)

    K = np.asarray(calib.default_intrinsics(), float)
    D = np.asarray(calib.UTIL_PHANTOM4_D, float)
    boxes = calib.box_corners_center(calib.UTIL_PHANTOM4_BOX)
    rng = np.random.default_rng(0)
    n = 6
    for i in range(1, n + 1):
        Image.new("RGB", (1920, 1080), (30, 30, 30)).save(
            root / model / "JPEGImages" / "01" / "0101" / f"{i}.jpg")
        # 合成一个合理的相机位姿（相机在 z=3，看向 -z 附近），目标放画面中央
        cam_t = [0.2 + 0.01 * i, -0.3, 3.0]
        cam_q = [0.0, 0.0, 0.0, 1.0]
        target = [1.6e18, -0.21 + 0.01 * i, -0.30, 1.90, 0.0, 0.0, 0.0, 1.0]
        # 16 个数 = 目标 8（[0] 时间戳 + 位姿 7）+ 相机 8（[8] 时间戳 + 位姿 7），
        # 与真实文件布局一致（真实文件末尾还有一个空格）
        (root / model / "labels" / "01" / "0101" / f"{i}.txt").write_text(
            " ".join(f"{v:.6f}" for v in (*target, 1.6e18, *cam_t, *cam_q)) + " \n")
        # labels_yolo6d：由作者框在**相机系**下投影（简化：直接给一组合理观测）
        uv = np.stack([900 + 30 * i + rng.normal(0, 2, 9),
                       500 + 20 * i + rng.normal(0, 2, 9)]).T
        xd = (uv[:, 0] - K[0, 2]) / K[0, 0]
        yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
        r2 = xd * xd + yd * yd
        radial = 1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3
        xdist = xd * radial + 2 * D[2] * xd * yd + D[3] * (r2 + 2 * xd * xd)
        ydist = yd * radial + D[2] * (r2 + 2 * yd * yd) + 2 * D[3] * xd * yd
        px = np.stack([xdist * K[0, 0] + K[0, 2], ydist * K[1, 1] + K[1, 2]], axis=1)
        norm = px / np.array([1920.0, 1080.0])
        w = float(norm[:, 0].max() - norm[:, 0].min())
        h = float(norm[:, 1].max() - norm[:, 1].min())
        row = [0.0] + [v for xy in norm for v in xy] + [w, h]
        (root / model / "labels_yolo6d" / "01" / "0101" / f"{i}.txt").write_text(
            " ".join(f"{v:.6f}" for v in row) + "\n")

    (root / model / "train.txt").write_text(
        "".join(f"/x/{model}/JPEGImages/01/0101/{i}.jpg\n" for i in range(1, 5)))
    (root / model / "test.txt").write_text(
        "".join(f"/x/{model}/JPEGImages/01/0101/{i}.jpg\n" for i in range(5, 7)))

    images = conv.index_images(root, model)
    check("index_images 扫到全部图像", len(images) == n, f"实际 {len(images)}")
    listed = conv.read_listing(root, model, "train")
    check("read_listing 解析出 4 条", len(listed) == 4, str(listed))
    check("read_listing 的键形如 组/序列/stem", listed[0] == "01/0101/1", listed[0])

    sys.argv = ["mav6d_to_yolo", "--mav6d", str(root), "--model", model,
                "--out", str(tmp_dir / "out"), "--copy", "--use-test-as-val"]
    try:
        rc = conv.main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # noqa: BLE001 - 转换器异常要暴露出来，便于定位
        import traceback
        raise AssertionError(f"[FAIL] 转换器抛异常：{type(exc).__name__}: {exc}\n"
                             f"{traceback.format_exc()}") from exc
    check("转换器返回 0", rc == 0, f"返回 {rc}")

    out = tmp_dir / "out" / model
    yaml_path = out / "dataset.yaml"
    check("生成 dataset.yaml", yaml_path.is_file())
    check("生成 labels_6d/train.csv", (out / "labels_6d" / "train.csv").is_file())
    check("生成 manifest.json", (out / "manifest.json").is_file())

    from tod.compat import load_yaml

    cfg = load_yaml(yaml_path)
    check("dataset.yaml 的 kpt_shape=[9,3]", list(cfg.get("kpt_shape") or []) == [9, 3],
          str(cfg.get("kpt_shape")))
    check("dataset.yaml 写了 flip_idx（否则框架静默关掉翻转）",
          len(cfg.get("flip_idx") or []) == 9, str(cfg.get("flip_idx")))
    check("flip_idx 是对换置换",
          all(list(cfg["flip_idx"])[list(cfg["flip_idx"])[i]] == i for i in range(9)),
          str(cfg.get("flip_idx")))
    check("0 号点名为 centroid（论文 p0）",
          (cfg.get("kpt_names") or {}).get(0, [None])[0] == "centroid",
          str(cfg.get("kpt_names")))
    tod = cfg.get("tod") or {}
    check("tod.camera_matrix 是 3×3", len(tod.get("camera_matrix") or []) == 3)
    check("tod.object_points 有 9 个点", len(tod.get("object_points") or []) == 9)
    check("tod.model_diameter 已给出", bool(tod.get("model_diameter")))

    labels = sorted((out / "labels" / "train").glob("*.txt"))
    check("生成标签文件", len(labels) == 4, f"实际 {len(labels)}")
    parts = labels[0].read_text().split()
    check("标签列数 = 5 + 9×3", len(parts) == 32, f"实际 {len(parts)}")
    box = [float(v) for v in parts[1:5]]
    check("框在 (0,1] 内", all(0 < v <= 1 for v in box), str(box))
    kpts = [float(v) for v in parts[5:]]
    xs = kpts[0::3]
    ys = kpts[1::3]
    vis = kpts[2::3]
    check("关键点坐标在 [0,1] 内", all(0 <= v <= 1 for v in xs + ys), f"{min(xs+ys)}")
    check("可见性只取 0/2", set(vis) <= {0.0, 2.0}, str(set(vis)))

    poses = (out / "labels_6d" / "train.csv").read_text().strip().splitlines()
    check("labels_6d 有表头 + 4 行", len(poses) == 5, f"实际 {len(poses)}")
    check("labels_6d 表头正确",
          poses[0] == "image,obj_index,tx,ty,tz,qx,qy,qz,qw", poses[0])
    check("labels_6d 每行 9 列", all(len(r.split(",")) == 9 for r in poses[1:]))


def test_real_data(mav6d: Path, model: str, max_frames: int) -> None:
    """（可选）真实 D:/MAV6D 端到端：核对投影链 + 标定 + 转换。"""
    if not (mav6d / model).is_dir():
        return skip("真实数据端到端", f"{mav6d / model} 不存在")
    calib = _load_tool("mav6d_calibrate")
    conv = _load_tool("mav6d_to_yolo")

    r = calib.check_against_util(mav6d, max(50, min(max_frames, 300)))
    check("作者公开标定能逐值复现 labels_yolo6d（<0.01 px）",
          r["frames"] > 0 and r["mean_px"] < 0.01,
          f"平均 {r['mean_px']:.4f} px（{r['frames']} 帧）")

    res = calib.calibrate(mav6d, model, max_frames, ("train",), 0, verbose=False)
    check("标定给出 9 个机体 3D 点", len(res["object_points"]) == 9)
    check("标定给出外接直径", bool(res.get("model_diameter_m")))
    if model == "phantom4":
        check("Phantom 4 的尺度比 = 1.0（形状取作者框）",
              abs(res["scale_vs_author_phantom4_box"] - 1.0) < 1e-9,
              str(res["scale_vs_author_phantom4_box"]))

    # 小规模转换（用 --max-frames 限制，避免真跑 1.6 万张）
    out = ROOT / "tests" / ".tmp" / f"mav6d-real-{model}"
    if out.exists():
        shutil.rmtree(out)
    sys.argv = ["mav6d_to_yolo", "--mav6d", str(mav6d), "--model", model,
                "--out", str(out), "--copy", "--max-frames", str(min(20, max_frames)),
                "--calib", str(ROOT / "configs" / "_base_" / "datasets" /
                               ("_mav6d_mavic2_calib.json" if model == "mavic2" else "_none"))]
    if model == "phantom4":          # phantom4 现场标定（作者的框已验证）
        sys.argv = [a for a in sys.argv if a != "_none"]
        sys.argv = sys.argv[:-2] if sys.argv[-2] == "--calib" else sys.argv
    try:
        rc = conv.main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    check(f"真实数据转换（{model}）返回 0", rc == 0, f"返回 {rc}")
    check("真实数据转换产出 dataset.yaml",
          (out / model / "dataset.yaml").is_file())
    shutil.rmtree(out, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mav6d", type=Path, default=Path("D:/MAV6D"),
                    help="真实数据根目录（不存在则跳过相关项）")
    ap.add_argument("--model", choices=("phantom4", "mavic2"), default="phantom4")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--skip-real", action="store_true", help="跳过真实数据端到端")
    args = ap.parse_args()

    tmp = ROOT / "tests" / ".tmp" / "mav6d-tests"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    tests = [test_parse_formats, test_pose_line_roundtrip, test_undistort_roundtrip,
             test_index_and_split]
    failures: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append(f"{fn.__name__}:\n{traceback.format_exc()}")
    try:
        test_convert_synthetic(tmp)
    except Exception:  # noqa: BLE001
        failures.append(f"test_convert_synthetic:\n{traceback.format_exc()}")
    if not args.skip_real:
        try:
            test_real_data(args.mav6d, args.model, args.max_frames)
        except Exception:  # noqa: BLE001
            failures.append(f"test_real_data:\n{traceback.format_exc()}")

    print(f"\n通过 {len(PASSED)} 项，跳过 {len(SKIPPED)} 项，失败 {len(failures)} 项\n")
    for label in PASSED:
        print(f"  ok    {label}")
    for label in SKIPPED:
        print(f"  skip  {label}")
    for fail in failures:
        print(f"\n{fail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
