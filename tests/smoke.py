"""M0 冒烟测试：不需要 torch / GPU，验证 registry 与 compose 的核心逻辑。

运行::

    python tests/smoke.py

只要这一步失败，就不要进入训练阶段 —— 说明架构层已经有问题。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tod.compose import (  # noqa: E402
    Variant,
    apply_type_map,
    inject_p2_head,
    replace_downsample,
)
from tod.registry import (  # noqa: E402
    EXTENSION_POINTS,
    RegistryError,
    catalog,
    get,
    has,
    list_specs,
    register,
)

def has_yaml() -> bool:
    import importlib.util

    return importlib.util.find_spec("yaml") is not None


PASSED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {label} {detail}")
    PASSED.append(label)


# --------------------------------------------------------------- 1. registry


def test_registry() -> None:
    @register(ep="EP3", paper="Dummy Upsample", url="https://example.com",
              year=2024, license="MIT", cost="free", aliases=("DummyUp",))
    class _Dummy:
        pass

    check("注册后可查询", has("_Dummy"))
    check("别名指向同一条目", get("DummyUp").name == "_Dummy")
    check("EP 名称可读", get("_Dummy").ep_name == "上采样")
    check("出现在列表中", any(s.name == "_Dummy" for s in list_specs("EP3")))
    check("EP 过滤生效", all(s.ep == "EP3" for s in list_specs("EP3")))

    # 重复注册必须报错，防止无意覆盖
    try:
        register(ep="EP3")(_Dummy)
        raise AssertionError("[FAIL] 重复注册未报错")
    except RegistryError:
        PASSED.append("重复注册被拒绝")

    # 非法 EP 必须报错，强制归档到扩展点
    try:
        register(ep="EP99")(type("_X", (), {}))
        raise AssertionError("[FAIL] 非法 EP 未报错")
    except RegistryError:
        PASSED.append("非法 EP 被拒绝")

    check("catalog 含表头", "模块总表" in catalog())
    check("扩展点齐全", set(EXTENSION_POINTS) == {f"EP{i}" for i in range(10)})


# ------------------------------------------------------------- 2. compose DSL


def test_compose() -> None:
    v = (Variant("UnitTest_P2_NWD", base="yolov8n", tags=["unit-test"])
         .data("visdrone2019-det", imgsz=1280, slicing={"patch": 1024, "overlap": 0.2})
         .upsample("_Dummy")
         .loss(box="_Dummy")
         .attention("DualAttention", levels=[2, 3, 4, 5])
         .assigner("STAL", small_target_aware=True)
         .strategy(optimizer="MuSGD", teacher="teacher.pt", distill="FeatureAlignKD")
         .train(epochs=10, batch=4, amp=True)
         .model(add_p2=True, nc=10))

    check("EP 归属正确", v.spec()["eps"]["EP3"]["upsample"] == "_Dummy")
    check("EP4 注意力落到 EP4", v.spec()["eps"]["EP4"]["attention"] == "DualAttention")
    check("EP6 分配器落到 EP6", v.spec()["eps"]["EP6"]["assigner"] == "STAL")
    check("EP9 蒸馏落到 EP9", v.spec()["eps"]["EP9"]["distill"] == "FeatureAlignKD")
    check("引用模块被识别（未注册的不会出现）", set(v.used_modules()) == {"_Dummy"})

    depth = Variant("x").upsample("_Dummy")
    depth.without("_Dummy")
    check("without 可剥离模块", depth.used_modules() == [])

    try:
        v.patch("EP99")
        raise AssertionError("[FAIL] 未知 EP 未报错")
    except KeyError:
        PASSED.append("未知 EP 被拒绝")

    # 用工作区内的临时目录（系统 temp 可能被文件沙箱禁止写入）
    tmp = ROOT / "tests" / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if not has_yaml():
        PASSED.append("SKIP 配置落盘 / 卡片（未安装 PyYAML）")
        return
    try:
        yml = v.dump(tmp / "v.yaml")
        check("配置已落盘", yml.is_file())
        card = v.card(tmp / "card.md")
        text = card.read_text(encoding="utf-8")
        check("卡片含来源表", "引用模块来源" in text)
        check("卡片含尺度指标", "AP_small" in text)
        check("卡片含 base", "yolov8n" in text)
    except Exception as exc:  # noqa: BLE001 - 冒烟测试要求给出可读失败原因
        raise AssertionError(f"[FAIL] 配置/卡片落盘失败：{exc!r}") from exc


# --------------------------------------------------------- 3. 模型图改造逻辑

#: 与官方 yolov8.yaml 同构的最小图（含 backbone/head 两段与全局索引）
YOLOV8_LIKE = {
    "nc": 80,
    "scales": {"n": [0.33, 0.25, 1024]},
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],        # 0  P1/2
        [-1, 1, "Conv", [128, 3, 2]],       # 1  P2/4
        [-1, 3, "C2f", [128, True]],        # 2  ← P2 源
        [-1, 1, "Conv", [256, 3, 2]],       # 3  P3/8
        [-1, 6, "C2f", [256, True]],        # 4
        [-1, 1, "Conv", [512, 3, 2]],       # 5  P4/16
        [-1, 6, "C2f", [512, True]],        # 6
        [-1, 1, "Conv", [1024, 3, 2]],      # 7  P5/32
        [-1, 3, "C2f", [1024, True]],       # 8
        [-1, 1, "SPPF", [1024, 5]],         # 9
    ],
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 10
        [[-1, 6], 1, "Concat", [1]],                    # 11
        [-1, 3, "C2f", [512]],                          # 12
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 13
        [[-1, 4], 1, "Concat", [1]],                    # 14
        [-1, 3, "C2f", [256]],                          # 15 P3 → Detect 输入
        [-1, 1, "Conv", [256, 3, 2]],                   # 16
        [[-1, 12], 1, "Concat", [1]],                   # 17
        [-1, 3, "C2f", [512]],                          # 18 P4
        [-1, 1, "Conv", [512, 3, 2]],                   # 19
        [[-1, 9], 1, "Concat", [1]],                    # 20
        [-1, 3, "C2f", [1024]],                         # 21 P5
        [[15, 18, 21], 1, "Detect", [80]],              # 22 Detect(P3,P4,P5)
    ],
}


def test_inject_p2() -> None:
    import copy

    cfg = copy.deepcopy(YOLOV8_LIKE)
    inject_p2_head(cfg, p2_idx=2, p2_channels=64)

    head = cfg["head"]
    check("P2 分支已插入", cfg["_p2_status"] == "injected")
    check("head 长度 +3", len(head) == len(YOLOV8_LIKE["head"]) + 3)

    detect = head[-1]
    check("Detect 仍在末尾", str(detect[2]) == "Detect")
    check("Detect 输入变为 4 路", detect[0] == [24, 15, 18, 21], f"实际 {detect[0]}")

    check("上采样源为 P3 节点", head[12][0] == 15, f"实际 {head[12][0]}")
    check("上采样模块正确", head[12][2] == "nn.Upsample")
    check("Concat 拼接 P2", head[13][0] == [-1, 2] and head[13][2] == "Concat")
    check("融合块通道正确", head[14][2] == "C2f" and head[14][3] == [64])
    check("backbone 未被改动", cfg["backbone"] == YOLOV8_LIKE["backbone"])

    # 幂等：已有 4 路输入时不再重复注入
    inject_p2_head(cfg, p2_idx=2)
    check("重复注入被识别", cfg["_p2_status"] == "already_present")
    check("重复注入不改变长度", len(cfg["head"]) == len(YOLOV8_LIKE["head"]) + 3)

    # 自定义上采样模块（对应 EP3 覆盖）
    cfg2 = copy.deepcopy(YOLOV8_LIKE)
    inject_p2_head(cfg2, p2_idx=2, upsample="DySample", fuse_block="C2f_EMA")
    check("可换成注册的上采样模块", cfg2["head"][12][2] == "DySample")
    check("可换成自定义融合块", cfg2["head"][14][2] == "C2f_EMA")

    # model: 单列表形式也必须支持
    flat = {"nc": 8, "model": YOLOV8_LIKE["backbone"] + YOLOV8_LIKE["head"]}
    inject_p2_head(flat, p2_idx=2)
    check("支持单一 model 列表形式", flat["model"][-1][0] == [24, 15, 18, 21])

    # 回归测试：浅拷贝时节点对象是共享的，改造不得就地污染调用方的原图
    check("浅拷贝输入不污染原图", YOLOV8_LIKE["head"][-1][0] == [15, 18, 21],
          f"实际 {YOLOV8_LIKE['head'][-1][0]}")


def test_type_map() -> None:
    import copy

    cfg = copy.deepcopy(YOLOV8_LIKE)
    apply_type_map(cfg, {"Conv": "SPDConv", "nn.Upsample": "DySample"})
    convs = sum(1 for n in cfg["backbone"] if n[2] == "SPDConv")
    ups = sum(1 for n in cfg["head"] if n[2] == "DySample")
    check("backbone Conv 全部替换", convs == 5, f"实际 {convs}")
    check("上采样全部替换", ups == 2, f"实际 {ups}")
    check("替换统计正确", cfg["_type_map_applied"]["Conv"] == 7)


# ------------------------------------------------- 3b. 主干下采样替换 / P2 预处理


def test_downsample_replacement() -> None:
    import copy

    cfg = copy.deepcopy(YOLOV8_LIKE)
    replace_downsample(cfg, module="ADown", indices=[1, 3, 5, 7])

    check("主干下采样替换 4 处", len(cfg["_downsample_replaced"]) == 4,
          f"实际 {cfg['_downsample_replaced']}")
    check("索引 1（P2/4）已替换", cfg["backbone"][1][2] == "ADown")
    check("节点参数已裁剪到 ADown 的 arity（[128,3,2] -> [128]）",
          cfg["backbone"][1][3] == [128], f"实际 {cfg['backbone'][1][3]}")
    check("索引 3/5/7 已替换",
          [cfg["backbone"][i][2] for i in (3, 5, 7)] == ["ADown"] * 3)
    check("索引 0 的 stem 未被替换（输入通道为奇数）", cfg["backbone"][0][2] == "Conv")
    check("索引 2 的 C2f 未被替换", cfg["backbone"][2][2] == "C2f")
    check("head 段未被误伤", all(n[2] != "ADown" for n in cfg["head"]))

    replace_downsample(cfg, module="ADown", indices=[1, 3, 5, 7])
    check("重复替换幂等", cfg["backbone"][1][2] == "ADown")


def test_p2_pre_node() -> None:
    import copy

    cfg = copy.deepcopy(YOLOV8_LIKE)
    inject_p2_head(cfg, p2_idx=2, p2_channels=128, p2_pre=["Conv", [128, 1, 1]])
    head = cfg["head"]

    check("head 长度 +4（含 P2 预处理）", len(head) == len(YOLOV8_LIKE["head"]) + 4)
    check("P2 侧 1x1 预处理节点已插入", head[12][0] == 2 and head[12][2] == "Conv"
          and head[12][3] == [128, 1, 1], f"实际 {head[12]}")
    check("上采样取自 P3 节点", head[13][0] == 15 and head[13][2] == "nn.Upsample")
    check("Concat 拼接预处理输出（全局索引 22）", head[14][0] == [-1, 22],
          f"实际 {head[14][0]}")
    check("融合块为 Concat 的下一个节点", head[15][2] == "C2f" and head[15][3] == [128])
    check("Detect 输入为 [25, 15, 18, 21]", head[16][0] == [25, 15, 18, 21],
          f"实际 {head[16][0]}")
    check("P2 通道记录正确", cfg["_p2_channels"] == 128)


def test_from_spec_roundtrip() -> None:
    v = (Variant("RoundTrip", base="yolov8n", tags=["rt"], notes="说明")
         .upsample("_Dummy")
         .head("Efficient_UAVDet", per_group=16)
         .loss(box="_Dummy")
         .attention("DualAttention", levels=[2, 3, 4, 5], reduction=16)
         .assigner("STAL", small_target_aware=True)
         .strategy(optimizer="MuSGD", kd_lambda=0.5, temperature=3.0)
         .model(add_p2=True, p2_channels=128)
         .train(epochs=1))
    data = v.spec()
    restored = Variant.from_spec(data)
    check("from_spec 往返一致", restored.spec() == data,
          f"\n{restored.spec()}\n!=\n{data}")


# ------------------------------------------------------------ 4. 可选：真模型


def test_real_model_if_available() -> None:
    from tod import compat

    if not compat.installed():
        PASSED.append("SKIP 真实模型构建（未安装 ultralytics）")
        return
    try:
        cfg = Variant("RealP2", base="yolov8n").model(nc=10, add_p2=True).model_yaml(write=False)
    except compat.CompatError as exc:
        PASSED.append(f"SKIP 真实模型构建（{exc}）")
        return
    check("真实 base 图读取成功", "backbone" in cfg or "model" in cfg)
    detect = (cfg.get("head") or cfg.get("model"))[-1]
    check("真实图 P2 头注入成功", len(detect[0]) == 4)

    # 所选规模必须显式生效：框架的 yaml_model_load 用**文件名**猜规模，猜不到就把
    # d["scale"] 置空，parse_model 于是退化为 next(iter(scales.keys()))。
    # 我们生成的 model.yaml 靠 compose 把所选规模排在首位来保证正确。
    scales = cfg.get("scales")
    if isinstance(scales, dict) and scales:
        check("生成图的 scales 首位 = 所选规模", next(iter(scales)) == "n",
              f"实际 {list(scales)[:3]}")

    # YOLO26 base：原生 DFL-free（reg_max=1）+ NMS-free（end2end）—— SDD-YOLO 依赖这两点
    try:
        compat.base_model_yaml("yolo26")
    except compat.CompatError:
        PASSED.append("SKIP YOLO26 base 检查（该框架版本没有 yolo26.yaml）")
    else:
        cfg26 = (Variant("RealSDD", base="yolo26n")
                 .model(nc=10, add_p2=True, p2_idx=2, p2_channels=128, p2_fuse_block="C3")
                 .model_yaml(write=False))
        check("YOLO26 base：end2end=True（NMS-free 图）", cfg26.get("end2end") is True)
        check("YOLO26 base：reg_max=1（无 DFL 分支）", cfg26.get("reg_max") == 1)
        detect26 = (cfg26.get("head") or cfg26.get("model"))[-1]
        check("YOLO26 图 P2 头注入为 4 路", len(detect26[0]) == 4, f"实际 {detect26[0]}")
        check("P2 融合块按论文 §4.2 用 C3",
              any(str(n[2]) == "C3" for n in cfg26["head"]))

    # 检测头不做 YAML 改名，而是记录为建模后手术
    cfg2 = (Variant("RealHead", base="yolov8n")
            .head("Efficient_UAVDet").model(nc=10, add_p2=True).model_yaml(write=False))
    check("检测头记录为建模后手术", cfg2.get("_head_surgery") == "Efficient_UAVDet")
    check("YAML 中检测头仍是原生 Detect",
          str((cfg2.get("head") or cfg2.get("model"))[-1][2]) == "Detect")


def main() -> int:
    tests = [test_registry, test_compose, test_inject_p2, test_type_map,
             test_downsample_replacement, test_p2_pre_node, test_from_spec_roundtrip,
             test_real_model_if_available]
    for fn in tests:
        fn()
    print(f"\n全部通过：{len(PASSED)} 项检查\n")
    for label in PASSED:
        print(f"  ok  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
