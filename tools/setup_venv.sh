#!/usr/bin/env bash
# ===========================================================================
# 为 yolo-tod 建 venv 环境（Linux / macOS 版；Windows 用 tools/setup_venv.ps1）
#
# 步骤：建 .venv -> 引导 pip -> 从 PyTorch 索引装 torch/torchvision
#       -> 装 requirement.txt -> 可编辑安装本仓库 -> 自检
#
# 为什么 torch 单独一步：PyPI 上的 torch/torchvision 是 CPU-only 构建，
# GPU 版只在 https://download.pytorch.org/whl/cuXXX 上。
#
# 用法::
#     bash tools/setup_venv.sh
#     bash tools/setup_venv.sh --torch-index https://download.pytorch.org/whl/cu126
#     bash tools/setup_venv.sh --python python3.12
#     bash tools/setup_venv.sh --skip-checks
# ===========================================================================
set -euo pipefail

PYTHON_BIN="python3"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
SKIP_CHECKS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        --skip-checks) SKIP_CHECKS=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[FAIL] 未知参数：$1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

# 正常 POSIX 布局是 bin/python；但在 Git Bash / Cygwin 下用 Windows 版 python 建 venv
# 会得到 Scripts/python.exe，这里两种都认，脚本就不会在那种环境下白报错。
if [[ -x "$VENV/bin/python" ]]; then
    VENV_PY="$VENV/bin/python"
elif [[ -x "$VENV/Scripts/python.exe" ]]; then
    VENV_PY="$VENV/Scripts/python.exe"
else
    VENV_PY="$VENV/bin/python"
fi
ACTIVATE="source .venv/bin/activate"
[[ "$VENV_PY" == *Scripts* ]] && ACTIVATE=".venv/Scripts/activate"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  [OK] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1" >&2; }
die()  { printf '  [FAIL] %s\n' "$1" >&2; exit 1; }

# pip 的缓存/临时目录固定到工作区内，避免污染 HOME
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
mkdir -p "$PIP_CACHE_DIR"

# ------------------------------------------------------- 1) 选解释器并建 venv
step "1/6 选择解释器并创建 $VENV"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
    die "找不到解释器 '$PYTHON_BIN'。用 --python 指定，例如 --python python3.12"
BASE_VER="$("$PYTHON_BIN" -c 'import sys; print(sys.version)')"
ok "基础解释器：$PYTHON_BIN -> $BASE_VER"

"$PYTHON_BIN" - <<'PY' || die "需要 Python >= 3.10；另外 torchvision 明确排除 3.14.1"
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 必需，当前 {sys.version.split()[0]}")
if sys.version_info[:3] == (3, 14, 1):
    raise SystemExit("torchvision 排除了 Python 3.14.1，请换 3.12/3.13 或 3.14.2+")
PY
ok "版本检查通过"

if [[ -x "$VENV_PY" ]] && "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    ok ".venv 已存在且可用，跳过创建（要重建就先删掉 $VENV）"
else
    [[ -e "$VENV" ]] && { warn ".venv 已存在但不可用，重建"; rm -rf "$VENV"; }
    if ! "$PYTHON_BIN" -m venv "$VENV" 2>/dev/null; then
        warn "python -m venv 失败（ensurepip）；改用 --without-pip 稍后引导 pip"
        "$PYTHON_BIN" -m venv --without-pip "$VENV"
    fi
    [[ -x "$VENV_PY" ]] || die "venv 创建失败：$VENV_PY 不存在"
    ok "已创建 $VENV"
fi

# ------------------------------------------------------- 2) 引导 pip
step "2/6 引导 pip"
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    warn "venv 里没有 pip；用基础解释器装"
    "$PYTHON_BIN" -m pip --python "$VENV_PY" install --upgrade pip \
        || die "pip 引导失败。手工执行： $VENV_PY -m ensurepip --upgrade"
fi
"$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1 || warn "pip 自升级失败（继续）"
ok "$("$VENV_PY" -m pip --version)"

# ------------------------------------------------------- 3) torch 栈
step "3/6 安装 torch 栈（索引 $TORCH_INDEX）"
TORCH_REQ="$ROOT/requirement-torch.txt"
if [[ -f "$TORCH_REQ" ]]; then
    ok "使用 $TORCH_REQ"
    "$VENV_PY" -m pip install -r "$TORCH_REQ" --index-url "$TORCH_INDEX" \
        || die "torch 安装失败。用 --torch-index 换一个 CUDA 版本（cu118/cu126/cu128/cu129/cu130 或 .../whl/cpu）"
else
    ok "无 requirement-torch.txt；直接按索引解析最新轮子"
    "$VENV_PY" -m pip install torch torchvision --index-url "$TORCH_INDEX" \
        || die "torch 安装失败。用 --torch-index 换一个 CUDA 版本"
fi
ok "torch 栈安装完成"

# ------------------------------------------------------- 4) 其余依赖
step "4/6 安装其余依赖（requirement.txt）"
MAIN_REQ="$ROOT/requirement.txt"
[[ -f "$MAIN_REQ" ]] || die "缺少 $MAIN_REQ"
"$VENV_PY" -m pip install -r "$MAIN_REQ" || die "requirement.txt 安装失败"
ok "依赖安装完成"

# ------------------------------------------------------- 5) 本仓库
step "5/6 可编辑安装本仓库（--no-deps）"
( cd "$ROOT" && "$VENV_PY" -m pip install -e . --no-deps ) || die "可编辑安装失败"
ok "tod 已可编辑安装"

# ------------------------------------------------------- 6) 自检
if [[ "$SKIP_CHECKS" -eq 1 ]]; then
    printf '\n已跳过自检。启用环境： source .venv/bin/activate\n'
    exit 0
fi

step "6/6 自检"
IMPORT_OUT="$("$VENV_PY" -c 'import tod, torch, ultralytics; print(torch.__version__, ultralytics.__version__)' 2>&1)" \
    || die "import tod/torch/ultralytics 失败：\n$IMPORT_OUT"
ok "torch/ultralytics：$IMPORT_OUT"

CUDA_OUT="$("$VENV_PY" -c 'import torch; print(int(torch.cuda.is_available())); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")' 2>&1)"
if [[ "$(printf '%s' "$CUDA_OUT" | head -n1)" == "1" ]]; then
    ok "GPU 可用：$(printf '%s' "$CUDA_OUT" | tail -n1)"
else
    warn "torch 看不到 GPU（$CUDA_OUT）。若装成了 CPU 版，重跑："
    warn "  $VENV_PY -m pip install --force-reinstall torch torchvision --index-url $TORCH_INDEX"
fi

( cd "$ROOT" && "$VENV_PY" tests/smoke.py ) || die "tests/smoke.py 未通过"
ok "架构层冒烟测试通过"

printf '\n环境就绪。启用： source .venv/bin/activate\n'
printf '下一步： python3 tests/test_pose.py   或   python3 tools/train_mav6d.py --model phantom4\n'
