# YOLO26s、视觉 Mamba 与 ViT 的无人车 / 无人艇检测

## 使用步骤

1. **创建模型。** 在仓库根目录打开 [模型融合.ipynb](模型融合.ipynb)，选择项目的 Python 内核，从头运行。五组模型的网络结构写入 [models/fusion.yaml](models/fusion.yaml)。
2. **运行训练。** 在仓库根目录的 PowerShell 中执行下方命令。只训练 P5 Mamba 时，把 `all` 换成 `fit --model p5_mamba`。
3. **查看结果。** 每组正式训练完成后，最佳和末轮权重归入 `models/<模型名>/`，整体及分类 mAP 写入 [training/实验结果汇总.xlsx](training/实验结果汇总.xlsx)。原始训练日志和续训检查点在 `training/artifacts/studies/`。

```powershell
& .\.venv-cuda\Scripts\python.exe train.py all --data training/configs/study-data.yaml
```

若 Excel 被占用，关闭工作簿后用同一解释器运行 `train.py report` 即可重新汇总。恢复中断的训练使用 `train.py resume --data training/configs/study-data.yaml --checkpoint <本次运行的weights/last.pt>`。

在独立的 Python 程序中加载融合权重时，先把仓库的 `training/support` 目录加入 `sys.path`，让 PyTorch 能找到自定义的 `fusion.models` 层。

## 复现与开源

首次安装使用 Python 3.12。安装完成后，在 Jupyter 或 VS Code 选择项目的 `.venv-cuda` 内核。

```powershell
py -3.12 -m venv .venv-cuda
& .\.venv-cuda\Scripts\python.exe -m pip install -r training/configs/requirements-cuda.txt ipykernel==7.4.0
& .\.venv-cuda\Scripts\python.exe train.py --download-weights yolo
```

新版数据集公开链接：

本地 v4 数据位于 `training/dataset/`。保留 93 张验证图、111 个框。从训练集排除 10 张已核实的跨集合副本后，使用 622 张训练图、694 个框。两个集合均无空标签图片，无独立测试集。采集场景和其余近重复的隔离情况未核实。[数据配置](training/configs/study-data.yaml) · [重复图复核清单](training/configs/verified_v4_duplicates.csv)

五组模型为 YOLO26s、P5 ViT、P5 Mamba、P4 Mamba、P4 Mamba＋P5 ViT。统一使用 COCO 初始权重、640 像素、batch 4、AdamW、FP32、100 轮。首轮 seed 0，按验证 mAP50–95 选出 Mamba 候选，与基线、P5 ViT 补跑 seed 1、2；组合胜出时 P4 Mamba 也复验。相差 0.1 个百分点以内优先选参数少、验证前向耗时低的模型。Excel 按同数据、同协议和同种子计算差值。

代码沿用 [AGPL-3.0](LICENSE)。四向选择性扫描参考 [VMamba 论文](https://arxiv.org/abs/2401.10166)与[作者实现](https://github.com/MzeroMiko/VMamba)。初始权重来自 [Ultralytics](https://docs.ultralytics.com/models/yolo26/)。数据与预训练权重分别遵循各自来源许可。[历史 v3 复现说明](training/docs/V3.md) · [历史实验记录](training/docs/EXPERIMENTS.md)
