@echo off
setlocal
cd /d "%~dp0..\.." || exit /b 1
set "YOLO_DISABLE_CUDNN=1"
set "CUDA_LAUNCH_BLOCKING=1"
set "PYTHONNOUSERSITE=1"
echo GPU compatibility run: cuDNN disabled, CUDA remains enabled. Long-run stability is not yet verified.
".venv-cuda\Scripts\python.exe" training\support\run_experiments.py --data "training\Driverless Vehicle.yolo26\data.yaml" --device 0 --epochs 100 --patience 0 --batch 4 %*
exit /b %errorlevel%
