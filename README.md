# YOLO26–ViT for Driverless Vehicle Detection

基于 YOLO26s 与 Vision Transformer（ViT）P5 门控残差融合的无人车（UGV）与无人艇（USV）目标检测实验项目，提供数据检查、训练、批量实验、评估和结果汇总。

当前使用 **Roboflow Driverless Vehicle 第 3 版**。尚无该版本的完整训练结果，不宣称融合已带来精度提升；旧数据实验不参与当前报告。

## 数据集来源与负样本

- [Roboflow 第 3 版数据来源](https://universe.roboflow.com/yans-workspace-47kt2/driverless-vehicle-iv6qt/dataset/3)
- [Roboflow 项目入口](https://app.roboflow.com/yans-workspace-47kt2/driverless-vehicle-iv6qt/3)
- 工作区：`yans-workspace-47kt2`；项目：`driverless-vehicle-iv6qt`；版本：`3`。
- 本地导出日期：2026-11-18；格式：YOLO26；导出文件标注的数据许可：CC BY 4.0。

| 集合 | 图片数 | 标注框数 | 负样本图片数 |
|---|---:|---:|---:|
| train | 769 | 707 | 137 |
| valid | 168 | 111 | 75 |
| 合计 | 937 | 818 | 212 |

类别编号固定为 `0=UGV`、`1=USV`。

**数据维护者已确认全部 212 张空标签图片都是负样本，其中 119 张为本次新增。** 这些图片中的有人驾驶船、车与目标外观相近，但不属于需要检测的 UGV/USV。保留图片与空 `.txt` 标签，不增加背景类别，不将空标签视为漏标，也不在加载时过滤这些图片。缺失标签文件与空标签不同：前者报错，后者是合法的零目标样本。

本地导出包没有独立测试集；`data.yaml` 中虽列出 test 路径，相应目录并不存在。只有提供真实测试集后才能使用 `--split test`。来源分组和视觉近重复尚未核验；内容哈希检查只检测跨集合完全相同的图片文件。

数据和权重不随代码发布。请从上述来源页面下载并遵循来源许可。代码的 AGPL-3.0 许可不替代数据许可，也不重新授权第三方图片。

## 本地数据结构与接口

将下载包解压到项目根目录：

```text
Driverless Vehicle.yolo26/
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

训练、评估和批量实验共用 `--data`：支持本地 `.yaml/.yml`，也保留检测 NDJSON 兼容。YOLO YAML 的每个集合指定一个图片目录；图片目录段可为 `images` 或 `imgs`，对应的目录段为 `labels`，图片与标签同名。

程序处理本次 Roboflow 导出的 `../train/images` 相对路径，保持原划分与类别顺序，不改写原始图片、标签或 YAML。检查后的数据进入 `artifacts/datasets/` 下的短文件名缓存，负样本生成空标签。首次准备会额外保存一份图片缓存。

YAML 数据指纹包含类别、来源元数据、集合、相对文件名及图片和标签内容。内容变化生成新指纹和缓存，移动整个数据包不改变指纹。图片损坏、缺失标签、非法标注、孤立标签或跨集合相同图片内容会报错。YOLO 输入只读取本地文件，不执行 YAML 中的下载脚本。

## 环境安装

以下命令使用 **Windows PowerShell**，Python 版本为 **3.12**。本机项目目录为 `D:\yolo26-vit`；克隆后的目录名可不同。

```powershell
git clone https://github.com/Britney-codemax/yolo26-vit-driverless-vehicle.git
Set-Location yolo26-vit-driverless-vehicle
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

固定依赖为 PyTorch 2.13.0、Torchvision 0.28.0、Ultralytics 8.4.14、aiohttp 3.14.5。启动时校验前三者版本，不要直接升级。CPU 环境可将 PyTorch 索引改为 `https://download.pytorch.org/whl/cpu`，运行时添加 `--device cpu`。

## 四种模型

| `--model` | P5 新增分支 | 新增部分初始化 |
|---|---|---|
| `baseline` | 无 | 不适用 |
| `p5_vit_small` | 1 层、256 维、4 头 | 随机 |
| `p5_vit_random` | 1 层、768 维、12 头 | 随机 |
| `p5_vit_pretrained` | 与 random 相同 | ViT-B/16 的第一个编码块 |

四种模型均使用 YOLO26s COCO 权重初始化原有检测网络。保留原 C2PSA，在其 P5 输出 `x` 上增加 `x + α × ViT分支(x)`；分支使用输入投影、二维正弦余弦位置编码、单层编码器和输出投影。α 初值为 0，初始输出保持不变；门控离开 0 后新增分支开始获得梯度。

预训练版本只迁移官方 ImageNet ViT-B/16 的第一个 Transformer 编码块，不加载完整 ViT 的 patch embedding、class token、分类头或学习式位置编码。random 与 pretrained 比较编码块初始化；small 与宽分支同时存在容量差异。

## 数据检查与初始权重

在项目根目录运行：

```powershell
$data = ".\Driverless Vehicle.yolo26\data.yaml"
.\.venv\Scripts\python.exe run_experiments.py --data "$data" --check --device cpu
```

`--check` 检查数据和模型结构，不生成图片缓存、不下载、不加载初始权重、不启动训练。检查通过不等于预训练迁移或完整 GPU 训练稳定性已验证。

显式下载官方初始权重：

```powershell
.\.venv\Scripts\python.exe train.py --download-weights all
```

默认保存为 `artifacts/weights/yolo26s.pt` 和 `artifacts/weights/vit_b_16_imagenet1k_v1.pth`。也可使用 `--weights`、`--vit-weights` 指定已有文件。训练发现权重缺失时会报错，不自动下载。

## 训练

以下命令会启动真实训练，由使用者自行执行。

```powershell
$data = ".\Driverless Vehicle.yolo26\data.yaml"
.\.venv\Scripts\python.exe train.py --model baseline --data "$data"
.\.venv\Scripts\python.exe train.py --model p5_vit_random --data "$data"
```

四种模型串行比较：

```powershell
.\.venv\Scripts\python.exe run_experiments.py --data "$data"
```

多种子对照：

```powershell
.\.venv\Scripts\python.exe run_experiments.py --models baseline p5_vit_random --seeds 0 1 2 --data "$data"
```

默认 640、batch=4、最多 100 轮、patience=40、AdamW、lr0=0.001、AMP=false、seed=0。完整设置见 `configs/train.yaml`；支持 `--epochs`、`--batch`、`--imgsz`、`--device` 等覆盖。模型比较须保持数据和训练条件一致；固定 seed 不保证不同 GPU 环境逐位复现。

训练运行默认写入 `artifacts/runs/roboflow_v3/`，批量实验还会生成批次子目录。原始日志、检查点、实际参数及数据指纹保存在各运行目录。`--run-dir` 是显式覆盖；指定到新版目录之外的运行不进入默认汇总。

## 评估与汇总

将示例替换为实际 `best.pt` 路径：

```powershell
.\.venv\Scripts\python.exe evaluate.py --weights ".\artifacts\runs\roboflow_v3\RUN_ID\weights\best.pt" --data "$data" --split val
.\.venv\Scripts\python.exe compare.py
```

独立评估写入 `artifacts/evaluations/roboflow_v3/`。训练后的评估先重载 `best.pt`，显式执行 `eval()`，再调用 `val()` 计算指标。`best.pt` 按 mAP50–95 fitness 选择，结果来自同一检查点，不拼接不同轮次的最高值。P/R 为最大平均 F1 工作点，不是固定 conf=0.25 的值。

`compare.py` 不训练、不推理、不联网，仅汇总新版运行目录，更新 [实验记录](docs/EXPERIMENTS.md) 和 `artifacts/reports/` 下的 JSON/CSV。没有新版记录时显示“暂无新版数据实验结果”。旧目录仍保留在本地，不参与该汇总。

数据指纹、代码、环境或训练参数不同时分组。只有同协议、同模型、不同 seed 才统计均值及样本标准差；同条件重复 seed 全部排除，单 seed 标准差为 N/A。

## 离线验收与限制

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖 YAML/NDJSON、正负样本、模型结构、训练器重建、独立评估、批量参数和新版结果隔离。集成测试禁止外部网络与优化器更新，训练入口在真实训练循环前停止。缺少本地官方权重时，真实权重检查会跳过或使用明确标记的合成检查点。

当前没有新版数据的训练精度结论。此前环境出现过 CUDA 长时间训练异常，数据接口修改不代表该问题已解决；离线合成检查也不能替代完整训练验证。

## 项目结构与许可

`train.py`、`evaluate.py`、`run_experiments.py`、`compare.py` 为四个入口；`fusion/` 放模型与数据实现，`configs/` 放配置，`tests/` 放验收，`docs/` 放当前实验记录。环境、数据、权重及运行产物由 `.gitignore` 排除。

代码采用 **AGPL-3.0**，完整条款见 [LICENSE](LICENSE)。上游主要参考为 Ultralytics 8.4.14；未修改其安装包源文件。

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics 许可说明](https://www.ultralytics.com/license)
- [Torchvision ViT-B/16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)

反馈问题时请提供命令、环境和日志，不要公开邀请链接、访问令牌或带签名的下载地址。
