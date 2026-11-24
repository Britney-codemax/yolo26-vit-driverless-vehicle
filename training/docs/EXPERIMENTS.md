# YOLO26–ViT：Roboflow 第 3 版实验记录

## 当前数据与协议

数据来源：[Driverless Vehicle v3](https://universe.roboflow.com/yans-workspace-47kt2/driverless-vehicle-iv6qt/dataset/3)。工作区 `yans-workspace-47kt2`，项目 `driverless-vehicle-iv6qt`，本地导出日期 2026-11-18；导出元数据标注 CC BY 4.0。

| 集合 | 图片 | 标注框 | 已确认负样本 |
|---|---:|---:|---:|
| train | 769 | 707 | 137 |
| valid | 168 | 111 | 75 |
| 合计 | 937 | 818 | 212 |

类别 `0=UGV`、`1=USV`。维护者确认全部 212 张空标签图片为有人驾驶或外观相近载具的负样本，其中 119 张为本次新增。保留空标签，不新增类别。无独立 test 集；来源分组和视觉近重复未核验。

本次 baseline 协议：YOLO26s COCO 初始化，640、batch=4、总计 100 轮、patience=0、AdamW、lr0=0.001、AMP=false。第 53 轮后中断续训，第 78 轮为最佳权重；实际配置、数据指纹与环境以运行记录为准。后续四模型实验统一使用项目内 `Driverless Vehicle.yolo26/data.yaml`。

## 记录范围

当前汇总只读取 `artifacts/runs/roboflow_v3/` 与 `artifacts/evaluations/roboflow_v3/`。2026-11-19 按要求清理其他实验、诊断和测试产物，以及旧数据集缓存；现只保留 `baseline_seed0_20261119_141852_696039` 的完整结果。

本次目录保留模型权重、100 轮 results.csv、最终 metrics.json、运行配置和日志；run.json 中的续训来源 ID 作为本次溯源信息保留，对应历史目录已移出项目。

下方区域由 `python compare.py` 自动更新。

<!-- AUTO_RESULTS_START -->

## 自动实验结果

best.pt 按 mAP50–95 fitness 选择；P/R 为最大平均 F1 工作点。指标缺失不补 0。
数据、代码、实际配置或环境不同时分组；独立评估不参与训练种子统计。

| 实验 | 类型 | 模型 | seed | 数据哈希 | 协议 | 最佳轮 | 已跑轮 | P | R | mAP50 | mAP50–95 | 参数量 | 状态 | 本地相对路径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/roboflow_v3/baseline_seed0_20261119_141852_696039 | train | baseline | 0 | b36ff7170795 | 2827de5bb1fd | 78 | 100 | 0.659550 | 0.753489 | 0.768932 | 0.600439 | 9949412 | 完成 | artifacts/runs/roboflow_v3/baseline_seed0_20261119_141852_696039 |
| runs/roboflow_v3/batch_20261119_174931_006326/p5_vit_small_seed0 | train | p5_vit_small | 0 | b36ff7170795 | 1428334a56ab | 34 | 34 | N/A | N/A | N/A | N/A | 11002085 | 失败 | artifacts/runs/roboflow_v3/batch_20261119_174931_006326/p5_vit_small_seed0 |
| runs/roboflow_v3/batch_20261119_205212_523242/p5_vit_random_seed0 | train | p5_vit_random | 0 | b36ff7170795 | 1428334a56ab | 6 | 8 | N/A | N/A | N/A | N/A | 17824997 | 失败 | artifacts/runs/roboflow_v3/batch_20261119_205212_523242/p5_vit_random_seed0 |
| runs/roboflow_v3/p5_vit_random_seed0_20261119_210412_300188 | train | p5_vit_random | 0 | b36ff7170795 | 6b8516378fef | 97 | 100 | 0.752532 | 0.663059 | 0.705751 | 0.554502 | 17824997 | 完成 | artifacts/runs/roboflow_v3/p5_vit_random_seed0_20261119_210412_300188 |
| runs/roboflow_v3/p5_vit_small_seed0_20261119_183315_459278 | train | p5_vit_small | 0 | b36ff7170795 | bd6a22ac1950 | 98 | 100 | 0.688842 | 0.730019 | 0.767816 | 0.625271 | 11002085 | 完成 | artifacts/runs/roboflow_v3/p5_vit_small_seed0_20261119_183315_459278 |
| runs/roboflow_v3/batch_20261119_174931_006326/p5_vit_random_seed0 | train | p5_vit_random | 0 |  |  | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未运行 | artifacts/runs/roboflow_v3/batch_20261119_174931_006326/p5_vit_random_seed0 |
| runs/roboflow_v3/batch_20261119_174931_006326/p5_vit_pretrained_seed0 | train | p5_vit_pretrained | 0 |  |  | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未运行 | artifacts/runs/roboflow_v3/batch_20261119_174931_006326/p5_vit_pretrained_seed0 |
| runs/roboflow_v3/batch_20261120_102741_937061/p5_vit_pretrained_seed0 | train | p5_vit_pretrained | 0 |  |  | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未运行 | artifacts/runs/roboflow_v3/batch_20261120_102741_937061/p5_vit_pretrained_seed0 |

## 每类指标

| 实验 | 类别 | 实例数 | P | R | AP50 | AP50–95 |
| --- | --- | --- | --- | --- | --- | --- |
| runs/roboflow_v3/baseline_seed0_20261119_141852_696039 | UGV | 57 | 0.813681 | 0.766237 | 0.834322 | 0.661678 |
| runs/roboflow_v3/baseline_seed0_20261119_141852_696039 | USV | 54 | 0.505418 | 0.740741 | 0.703542 | 0.539199 |
| runs/roboflow_v3/p5_vit_random_seed0_20261119_210412_300188 | UGV | 57 | 0.924102 | 0.640933 | 0.782893 | 0.605365 |
| runs/roboflow_v3/p5_vit_random_seed0_20261119_210412_300188 | USV | 54 | 0.580962 | 0.685185 | 0.628608 | 0.503640 |
| runs/roboflow_v3/p5_vit_small_seed0_20261119_183315_459278 | UGV | 57 | 0.798156 | 0.719298 | 0.799053 | 0.651227 |
| runs/roboflow_v3/p5_vit_small_seed0_20261119_183315_459278 | USV | 54 | 0.579529 | 0.740741 | 0.736578 | 0.599315 |

## 同条件复验汇总

标准差使用 ddof=1；单种子标准差为 N/A。同模型同条件重复 seed 全部排除，不挑最高值。
ΔmAP50 仅在 baseline 与候选具有相同有效 seed 集合时给出。

| 协议 | 模型 | seeds | n | mAP50 均值 | 标准差 | mAP50–95 均值 | 标准差 | ΔmAP50 | 排除的重复 seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2827de5bb1fd | baseline | [0] | 1 | 0.768932 | N/A | 0.600439 | N/A | 0.000000 | [] |
| 6b8516378fef | p5_vit_random | [0] | 1 | 0.705751 | N/A | 0.554502 | N/A | N/A | [] |
| bd6a22ac1950 | p5_vit_small | [0] | 1 | 0.767816 | N/A | 0.625271 | N/A | N/A | [] |

<!-- AUTO_RESULTS_END -->
