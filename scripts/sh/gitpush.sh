#!/usr/bin/env bash
# ============================================================
#  gitpush.sh —— 一键提交并推送（Linux / macOS / Git Bash）
#
#  用法:
#    scripts/sh/gitpush.sh [提交说明]
#    scripts/sh/gitpush.sh --dry-run
#
#  选项:
#    --dry-run         只预览仓库状态，不做任何改动（可安全试跑）
#    --no-push         只提交，不推送
#    --remote <名称>   指定远端，默认 origin
#    -h, --help        显示本帮助
#
#  行为:
#    1) 定位仓库根目录并切过去（脚本放哪都能用）
#    2) git add -A，有改动就提交（没给提交说明则用时间戳兜底）
#    3) 推送到远端；首次推送自动 git push -u 设置上游
#    不做 force push，不做 rebase —— 推送失败会让你自己决定怎么处理。
#
#  首次使用请加可执行位:
#    chmod +x scripts/sh/gitpush.sh
# ============================================================

set -euo pipefail

MSG=""
DRYRUN=0
NOPUSH=0
REMOTE="origin"

# 有 tty 时启用颜色，重定向到文件时自动关闭
if [ -t 1 ]; then
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_DIM=""; C_OFF=""
fi

err()  { printf '%s[x] %s%s\n' "$C_RED" "$1" "$C_OFF" >&2; }
info() { printf '%s%s%s\n' "$C_DIM" "$1" "$C_OFF"; }

usage() {
    cat <<'EOF'

gitpush.sh —— 一键提交并推送

  gitpush.sh [提交说明] [选项]

  选项:
    --dry-run         只预览仓库状态，不做任何改动
    --no-push         只提交，不推送
    --remote <名称>   指定远端，默认 origin
    -h, --help        显示本帮助

  示例:
    scripts/sh/gitpush.sh "feat: 实现 SPAE-YOLOv8n"
    scripts/sh/gitpush.sh --dry-run
    scripts/sh/gitpush.sh "fix: 修 P2 注入" --no-push

EOF
}

# ---------------- 解析参数 ----------------
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --dry-run)  DRYRUN=1; shift ;;
        --no-push)  NOPUSH=1; shift ;;
        --remote)
            if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
                err "--remote 需要一个远端名称，例如 --remote origin"
                exit 2
            fi
            REMOTE="$2"; shift 2 ;;
        --)         shift; MSG="${MSG:+$MSG }$*"; break ;;
        -*)         err "未知选项: $1"; usage; exit 2 ;;
        *)          MSG="${MSG:+$MSG }$1"; shift ;;
    esac
done

# ---------------- 环境检查 ----------------
command -v git >/dev/null 2>&1 || { err "未找到 git"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { err "当前目录不在 Git 仓库中"; exit 1; }

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || { err "无法进入仓库根目录: $ROOT"; exit 1; }

# symbolic-ref 在"还没有任何提交"的空仓库上也能返回分支名
BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo HEAD)"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
    err "远端 \"$REMOTE\" 不存在，可用 --remote 指定"
    exit 1
fi

printf '\n  仓库: %s\n  分支: %s\n  远端: %s  (%s)\n\n' \
    "$ROOT" "$BRANCH" "$REMOTE" "$(git remote get-url "$REMOTE")"

# ---------------- 预览模式（只读） ----------------
if [ "$DRYRUN" -eq 1 ]; then
    if [ -n "$MSG" ]; then
        echo "[预览] 提交说明: $MSG"
    else
        echo "[预览] 提交说明: (未指定，将用时间戳兜底)"
    fi
    echo
    echo "[预览] git status --short:"
    git status --short
    echo
    echo "[预览] 最近 5 次提交:"
    git log --oneline -5 2>/dev/null || true
    echo
    echo "[预览] 未做任何改动。去掉 --dry-run 即真正执行。"
    echo
    echo "[完成]"
    exit 0
fi

# ---------------- 暂存 + 提交 ----------------
git add -A
if ! git diff --cached --quiet; then
    if [ -z "$MSG" ]; then
        MSG="chore: 自动提交 $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo "[1/2] 提交: $MSG"
    git commit -m "$MSG" || { err "提交失败，请检查上面的输出"; exit 1; }
else
    echo "[1/2] 工作区没有需要提交的改动"
fi

# ---------------- 推送 ----------------
if [ "$NOPUSH" -eq 1 ]; then
    echo "[2/2] 已按 --no-push 跳过推送"
    echo
    echo "[完成]"
    git log --oneline -1 2>/dev/null || true
    exit 0
fi

# 空仓库（还没有任何提交）没有可推送的 HEAD
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "[2/2] 仓库还没有任何提交，跳过推送"
    echo
    echo "[完成]"
    exit 0
fi

BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo HEAD)"

if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    echo "[2/2] 推送: $REMOTE"
    git push "$REMOTE"
else
    echo "[2/2] 首次推送，设置上游: $REMOTE/$BRANCH"
    git push -u "$REMOTE" "$BRANCH"
fi

echo
printf '%s[完成]%s\n' "$C_GREEN" "$C_OFF"
git log --oneline -1 2>/dev/null || true
