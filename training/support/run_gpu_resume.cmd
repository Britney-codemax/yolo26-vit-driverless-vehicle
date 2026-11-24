@echo off
setlocal
cd /d "%~dp0..\.." || exit /b 1
if "%~1"=="" (
    echo Usage: run_gpu_resume.cmd "path\to\weights\last.pt" [--check]
    exit /b 2
)
set "YOLO_DISABLE_CUDNN=1"
set "CUDA_LAUNCH_BLOCKING=1"
set "PYTHONNOUSERSITE=1"
echo Resume uses the saved total epochs and a new output folder. CUDA timeout remains unresolved.
".venv-cuda\Scripts\python.exe" train.py --data "training\Driverless Vehicle.yolo26\data.yaml" --resume %*
exit /b %errorlevel%
