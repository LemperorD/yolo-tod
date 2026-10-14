@echo off
chcp 65001 >nul 2>&1
setlocal

REM 一键推送当前分支到远端
REM 用法: scripts\bat\gitpush.bat [remote]
REM 默认远端: origin

set "REMOTE=%~1"
if "%REMOTE%"=="" set "REMOTE=origin"

REM 检查 Git
where git >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Git，请检查 PATH。
    exit /b 1
)

REM 定位仓库根目录
git rev-parse --show-toplevel >nul 2>&1
if errorlevel 1 (
    echo [错误] 当前目录不在 Git 仓库中。
    exit /b 1
)

for /f "delims=" %%i in ('git rev-parse --show-toplevel') do set "ROOT=%%i"
cd /d "%ROOT%"

REM 获取当前分支
for /f "delims=" %%i in ('git branch --show-current') do set "BRANCH=%%i"

if "%BRANCH%"=="" (
    echo [错误] 当前处于 detached HEAD 状态，无法自动推送分支。
    exit /b 1
)

REM 检查远端是否存在
git remote get-url "%REMOTE%" >nul 2>&1
if errorlevel 1 (
    echo [错误] 远端 "%REMOTE%" 不存在。
    exit /b 1
)

for /f "delims=" %%i in ('git remote get-url "%REMOTE%"') do set "REMOTE_URL=%%i"

echo.
echo 仓库: %ROOT%
echo 分支: %BRANCH%
echo 远端: %REMOTE% (%REMOTE_URL%)
echo.

REM 检查是否存在提交
git rev-parse --verify HEAD >nul 2>&1
if errorlevel 1 (
    echo [错误] 当前仓库还没有任何提交。
    exit /b 1
)

REM 有上游则直接推送，没有上游则设置上游
git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >nul 2>&1
if errorlevel 1 (
    git push -u "%REMOTE%" "%BRANCH%"
) else (
    git push "%REMOTE%" "%BRANCH%"
)

if errorlevel 1 (
    echo.
    echo [错误] 推送失败，请检查网络、认证或远端分支状态。
    exit /b 1
)

echo.
echo [完成] 推送成功。
exit /b 0