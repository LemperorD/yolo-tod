@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ============================================================
REM  gitpush.bat —— 一键提交并推送（Windows）
REM
REM  用法:
REM    scripts\bat\gitpush.bat [提交说明]
REM    scripts\bat\gitpush.bat --dry-run
REM
REM  选项:
REM    --dry-run         只预览仓库状态，不做任何改动（可安全试跑）
REM    --no-push         只提交，不推送
REM    --remote <名称>   指定远端，默认 origin
REM    -h, --help        显示本帮助
REM
REM  行为:
REM    1) 定位仓库根目录并切过去（脚本放哪都能用）
REM    2) git add -A，有改动就提交（没给提交说明则用时间戳兜底）
REM    3) 推送到远端；首次推送自动 git push -u 设置上游
REM    不做 force push，不做 rebase —— 推送失败会让你自己决定怎么处理。
REM
REM  提示: 提交说明里含中文时，先在当前窗口执行一次 chcp 65001，
REM        否则 cmd 传参的编码可能和脚本内部不一致。
REM ============================================================

set "MSG="
set "DRYRUN=0"
set "NOPUSH=0"
set "REMOTE=origin"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="-h"        goto usage
if /i "%~1"=="--help"    goto usage
if /i "%~1"=="--dry-run" ( set "DRYRUN=1" & shift & goto parse_args )
if /i "%~1"=="--no-push" ( set "NOPUSH=1" & shift & goto parse_args )
if /i "%~1"=="--remote"  (
    if "%~2"=="" ( echo [x] --remote 需要一个远端名称，例如 --remote origin & exit /b 2 )
    set "REMOTE=%~2" & shift & shift & goto parse_args
)
if defined MSG ( set "MSG=!MSG! %~1" ) else ( set "MSG=%~1" )
shift
goto parse_args

:args_done

REM ---------------- 环境检查 ----------------
where git >nul 2>&1
if errorlevel 1 ( echo [x] 未找到 git，请先安装 Git 并加入 PATH & exit /b 1 )

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 ( echo [x] 当前目录不在 Git 仓库中 & exit /b 1 )

for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do set "ROOT=%%i"
cd /d "!ROOT!"
if errorlevel 1 ( echo [x] 无法进入仓库根目录: !ROOT! & exit /b 1 )

REM symbolic-ref 在"还没有任何提交"的空仓库上也能返回分支名
for /f "delims=" %%i in ('git symbolic-ref --short HEAD 2^>nul') do set "BRANCH=%%i"
if not defined BRANCH set "BRANCH=HEAD"

git remote get-url "!REMOTE!" >nul 2>&1
if errorlevel 1 ( echo [x] 远端 "!REMOTE!" 不存在，可用 --remote 指定 & exit /b 1 )
for /f "delims=" %%i in ('git remote get-url "!REMOTE!" 2^>nul') do set "REMOTE_URL=%%i"

echo.
echo   仓库: !ROOT!
echo   分支: !BRANCH!
echo   远端: !REMOTE!  ^(!REMOTE_URL!^)
echo.

REM ---------------- 预览模式（只读） ----------------
if "!DRYRUN!"=="1" (
    if defined MSG echo [预览] 提交说明: !MSG!
    if not defined MSG echo [预览] 提交说明: ^(未指定，将用时间戳兜底^)
    echo.
    echo [预览] git status --short:
    git status --short
    echo.
    echo [预览] 最近 5 次提交:
    git log --oneline -5 2>nul
    echo.
    echo [预览] 未做任何改动。去掉 --dry-run 即真正执行。
    goto finished
)

REM ---------------- 暂存 + 提交 ----------------
git add -A
git diff --cached --quiet
if errorlevel 1 (
    if not defined MSG set "MSG=chore: 自动提交 %DATE% %TIME%"
    echo [1/2] 提交: !MSG!
    git commit -m "!MSG!"
    if errorlevel 1 ( echo [x] 提交失败，请检查上面的输出 & exit /b 1 )
) else (
    echo [1/2] 工作区没有需要提交的改动
)

REM ---------------- 推送 ----------------
if "!NOPUSH!"=="1" ( echo [2/2] 已按 --no-push 跳过推送 & goto finished )

for /f "delims=" %%i in ('git symbolic-ref --short HEAD 2^>nul') do set "BRANCH=%%i"
if not defined BRANCH set "BRANCH=HEAD"

REM 空仓库（还没有任何提交）没有可推送的 HEAD
git rev-parse --verify HEAD >nul 2>&1
if errorlevel 1 ( echo [2/2] 仓库还没有任何提交，跳过推送 & goto finished )

git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >nul 2>&1
if errorlevel 1 (
    echo [2/2] 首次推送，设置上游: !REMOTE!/!BRANCH!
    git push -u "!REMOTE!" "!BRANCH!"
) else (
    echo [2/2] 推送: !REMOTE!
    git push "!REMOTE!"
)
if errorlevel 1 (
    echo.
    echo [x] 推送失败。常见原因: 远端有新提交、无网络、凭据失效。
    echo     建议先执行 git pull --rebase 后再推送；本脚本不会自动 force push。
    exit /b 1
)

:finished
echo.
echo [完成]
git log --oneline -1 2>nul
exit /b 0

:usage
echo.
echo gitpush.bat —— 一键提交并推送
echo.
echo   gitpush.bat [提交说明] [选项]
echo.
echo   选项:
echo     --dry-run         只预览仓库状态，不做任何改动
echo     --no-push         只提交，不推送
echo     --remote ^<名称^>   指定远端，默认 origin
echo     -h, --help        显示本帮助
echo.
echo   示例:
echo     scripts\bat\gitpush.bat "feat: 实现 SPAE-YOLOv8n"
echo     scripts\bat\gitpush.bat --dry-run
echo     scripts\bat\gitpush.bat "fix: 修 P2 注入" --no-push
echo.
exit /b 0
