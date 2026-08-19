# PowerShell build script for RenderSub Standalone Distribution
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Starting RenderSub Windows Standalone Build " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

python build_windows.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build completed successfully!" -ForegroundColor Green
