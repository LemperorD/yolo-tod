"""仓库卫生检查：源码与配置目录里不该有文件被 .gitignore 误伤。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import tod

    tod.configure_stdio()
except Exception:
    pass

PASSED: list[str] = []
FAILED: list[str] = []

GUARDED = ("src", "tools", "tests", "configs", "variants", "scripts", "docs")

ALLOWED_PREFIXES = ("tests/.tmp/", ".cache/", "build/", "dist/", "runs/")

SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".onnx", ".engine", ".jpg", ".jpeg",
                 ".png", ".pdf", ".so", ".dll", ".dylib", ".o", ".a", ".exe", ".bin"}


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(label)
        print(f"  ok    {label}")
    else:
        FAILED.append(f"{label} {detail}".strip())
        print(f"  FAIL  {label} {detail}")


def _git(*args: str) -> tuple[int, str]:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return out.returncode, (out.stdout or "") + (out.stderr or "")


def has_git() -> bool:
    code, _ = _git("rev-parse", "--is-inside-work-tree")
    return code == 0


def ignored_with_reason(path: str) -> str:
    """返回命中该路径的忽略规则（形如 ``.gitignore:13:data/``）；未被忽略返回空串。"""
    # --no-index 让已跟踪文件也接受忽略规则检查，避免漏掉目录规则误伤。
    code, out = _git("check-ignore", "-v", "--no-index", "--", path)
    if code != 0 or not out.strip():
        return ""
    return out.strip().split("\t")[0] if "\t" in out else out.strip()


def rule_is_unanchored_dir(rule_line: str) -> bool:
    """``.gitignore:13:data/`` → 规则 ``data/`` 是否「非锚定且目录型」。"""
    parts = rule_line.split(":")
    if len(parts) < 3:
        return False
    rule = ":".join(parts[2:]).strip()
    if rule.startswith("!") or rule.startswith("/"):
        return False
    if "*" in rule or "?" in rule:
        return False
    return rule.endswith("/")


def is_allowed(rel: str) -> bool:
    return any(rel.startswith(p) for p in ALLOWED_PREFIXES) or \
        any(f"/{p}" in rel for p in ("__pycache__/",))


def collect_guarded_files() -> list[str]:
    out: list[str] = []
    for top in GUARDED:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if is_allowed(rel):
                continue
            out.append(rel)
    return out


def test_no_unanchored_ignore_hits() -> None:
    """守卫目录里不得有文件被「非锚定目录规则」命中。"""
    if not has_git():
        print("  skip  没有 git，跳过忽略规则检查")
        return
    files = collect_guarded_files()
    check("守卫目录里扫到源文件/配置", len(files) > 50, f"实际 {len(files)}")

    victims: list[tuple[str, str]] = []
    for rel in files:
        rule = ignored_with_reason(rel)
        if rule and rule_is_unanchored_dir(rule):
            victims.append((rel, rule))

    detail = ""
    if victims:
        shown = "; ".join(f"{p} ← {r}" for p, r in victims[:6])
        detail = (f"{len(victims)} 个文件被非锚定规则误伤（前 6 个：{shown}）。"
                  "请把 .gitignore 里那条规则改成以 / 开头的锚定写法，"
                  "例如 data/ -> /data/")
    check("守卫目录里没有被 .gitignore 误伤的文件", not victims, detail)


def test_key_sources_are_tracked() -> None:
    """几个「曾漏入库」的关键路径必须在 git 索引里。"""
    if not has_git():
        print("  skip  没有 git，跳过索引检查")
        return
    required = [
        "src/tod/data/__init__.py",
        "src/tod/data/pose6d.py",
        "pyproject.toml",
        "requirement.txt",
        "WORKFLOW.md",
    ]
    code, tracked = _git("ls-files", "--", *required)
    tracked_set = {line.strip() for line in tracked.splitlines() if line.strip()}
    check("git 已跟踪 5 项关键文件", code == 0 and tracked_set == set(required),
          f"索引里只有 {sorted(tracked_set)}")

    cfg_dir = ROOT / "configs" / "_base_" / "datasets"
    on_disk = sorted(p.name for p in cfg_dir.glob("*.yaml")) if cfg_dir.is_dir() else []
    code, tracked_cfg = _git("ls-files", "--", "configs/_base_/datasets")
    tracked_names = {Path(line.strip()).name for line in tracked_cfg.splitlines()
                     if line.strip()}
    missing = [n for n in on_disk if n not in tracked_names]
    check("configs/_base_/datasets 下的 yaml 全部已入库", not missing,
          f"漏了 {missing}")


def test_no_tmp_scripts_tracked() -> None:
    """临时探针脚本（tests/.tmp/）不能被提交。"""
    if not has_git():
        print("  skip  没有 git，跳过临时文件检查")
        return
    _, tracked = _git("ls-files", "--", "tests/.tmp")
    names = [line.strip() for line in tracked.splitlines() if line.strip()]
    check("tests/.tmp/ 下没有任何文件被跟踪", not names, f"跟踪了 {names[:5]}")


def main() -> int:
    ap_fast = "--fast" in sys.argv
    print(f"[hygiene] 仓库根：{ROOT}")
    test_no_unanchored_ignore_hits()
    test_key_sources_are_tracked()
    if not ap_fast:
        test_no_tmp_scripts_tracked()

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for f in FAILED:
        print(f"  [FAIL] {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
