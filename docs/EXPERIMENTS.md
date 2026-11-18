# YOLO26–ViT：Roboflow 第 3 版实验记录

## 当前数据与协议

数据来源：[Driverless Vehicle v3](https://universe.roboflow.com/yans-workspace-47kt2/driverless-vehicle-iv6qt/dataset/3)。工作区 `yans-workspace-47kt2`，项目 `driverless-vehicle-iv6qt`，本地导出日期 2026-11-18；导出元数据标注 CC BY 4.0。

| 集合 | 图片 | 标注框 | 已确认负样本 |
|---|---:|---:|---:|
| train | 769 | 707 | 137 |
| valid | 168 | 111 | 75 |
| 合计 | 937 | 818 | 212 |

类别 `0=UGV`、`1=USV`。维护者确认全部 212 张空标签图片为有人驾驶或外观相近载具的负样本，其中 119 张为本次新增。保留空标签，不新增类别。无独立 test 集；来源分组和视觉近重复未核验。

默认协议：YOLO26s COCO 初始化，640、batch=4、最多 100 轮、patience=40、AdamW、lr0=0.001、AMP=false。四模型为 baseline、p5_vit_small、p5_vit_random、p5_vit_pretrained；实际配置、数据指纹与环境以各运行记录为准。

## 记录范围

当前汇总只读取 `artifacts/runs/roboflow_v3/` 与 `artifacts/evaluations/roboflow_v3/`。旧实验原始文件保留在本地，不参与当前报告，也不作为新版结果引用。

2026-11-18 接口验收：18 项测试全部通过，四种模型的实际数据 `--check --device cpu` 均通过。Ultralytics 数据加载器确认 937 张图片、818 个框和 212 张负样本全部保留；检查前后 1,877 个源文件内容一致。

日志见 `artifacts/checks/yolo_interface_tests.log`、`yolo_v3_cli_check.log` 和 `yolo_v3_data_acceptance.json`（后两项也在 `artifacts/checks/` 下）。本次未启动真实训练或优化器更新；上述验收不代表训练精度或长期 CUDA 稳定性已验证。

下方区域由 `python compare.py` 自动更新。

<!-- AUTO_RESULTS_START -->

## 自动实验结果

暂无新版数据实验结果。旧运行目录不参与当前汇总。

<!-- AUTO_RESULTS_END -->
