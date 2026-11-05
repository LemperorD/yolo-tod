#!/usr/bin/env bash
set -euo pipefail


REMOTE="${1:-origin}"


command -v git >/dev/null 2>&1 || {
    echo "[错误] 未找到 Git。" >&2
    exit 1
}


ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "[错误] 当前目录不在 Git 仓库中。" >&2
    exit 1
}

cd "$ROOT"


BRANCH="$(git branch --show-current)"

if [[ -z "$BRANCH" ]]; then
    echo "[错误] 当前处于 detached HEAD 状态，无法自动推送分支。" >&2
    exit 1
fi


git remote get-url "$REMOTE" >/dev/null 2>&1 || {
    echo "[错误] 远端 '$REMOTE' 不存在。" >&2
    exit 1
}


git rev-parse --verify HEAD >/dev/null 2>&1 || {
    echo "[错误] 当前仓库还没有任何提交。" >&2
    exit 1
}

echo
echo "仓库: $ROOT"
echo "分支: $BRANCH"
echo "远端: $REMOTE ($(git remote get-url "$REMOTE"))"
echo


if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git push "$REMOTE" "$BRANCH" || {
        echo "[错误] 推送失败，请检查网络、认证或远端分支状态。" >&2
        exit 1
    }
else
    echo "[首次推送] 设置上游分支"
    git push -u "$REMOTE" "$BRANCH" || {
        echo "[错误] 推送失败，请检查网络、认证或远端分支状态。" >&2
        exit 1
    }
fi

echo
echo "[完成] 推送成功。"
