# 树莓派 5 上的无人车、无人艇图片检测

两个模型来自第一轮第 4、5 组的最佳权重。输入均为单张 640×640、FP32，使用 ONNX opset 17。类别 0 为无人车 UGV，类别 1 为无人艇 USV。

## 在 Windows 重新导出

这台训练电脑已配置单独的导出环境。在 PowerShell 运行以下两行，脚本会读取仓库中对应的两份 best.pt，并在本目录生成同名 ONNX。组合模型在导出期间关闭 PyTorch 注意力快速路径。

```powershell
cd D:\yolo26-vit
& .\.venv-onnx\Scripts\python.exe .\export_pi_onnx.py
```

## 烧卡并连接

1. 确认卡主同意清空 microSD，备份后插入 Windows 读卡器。用 Raspberry Pi Imager 依次选择 Raspberry Pi 5、Raspberry Pi OS Lite (64-bit) 和对应的 64 GB 卡。
2. 将主机名设为 `pi5-yolo`，用户名设为 `vision`，设置密码和 Wi-Fi，打开 SSH 密码登录。等写入、校验都完成再弹出卡。
3. Pi 断电时插卡，接上电源。让电脑和 Pi 连到可以互访的同一网络。在 Windows 输入 `ssh vision@pi5-yolo.local`，登录后输入 `uname -m` 应显示 `aarch64`。主机名无法解析时，可改用路由器显示的 Pi IP。
4. 在另一个 Windows PowerShell 窗口运行以下命令上传整个文件夹。

```powershell
cd D:\yolo26-vit
scp -r .\pi-deploy vision@pi5-yolo.local:~/
```

## 在 Pi 的 SSH 窗口跑检测和测速

```bash
cd ~/pi-deploy
sudo apt update
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install onnxruntime numpy opencv-python-headless
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python infer.py preview
python infer.py bench
```

运行环境应列出 `CPUExecutionProvider`。下次登录后，进入同一目录并执行 `source .venv/bin/activate` 即可。

## 在 Windows 查看结果

在 Windows PowerShell 运行下面的命令，将板端结果复制到本机目录。打开两组带框图片，与部署包中 `windows-reference` 的相同图片对照。每个模型的 `detections.json` 也记录了原图像素坐标、置信度和类别编号，按 x1、y1、x2、y2、score、class 排列。

```powershell
cd D:\yolo26-vit
scp -r vision@pi5-yolo.local:~/pi-deploy/results ./pi5-results
```

测速结果在板上的 `results/benchmark.csv`。两个模型依次使用 4 个 CPU 推理线程，先各预热 10 张，再按相同顺序各处理 93 张验证图。model 为单次 ONNX 推理时间，total 为读图到得到检测框的时间，单位毫秒。P50 为中位数，P95 为第 95 百分位数。images_per_sec 为 93 张除以整批实测秒数。计时区间不含模型加载、预热或输出图片写盘。

表中记录每个模型计时前后的温度和 `vcgencmd get_throttled` 标志。缺失值表示无法采集，`0x0` 表示未检测到该命令报告的异常。历史标志会保留到重启；比较两组时同时看起始值、结束值和散热条件。
