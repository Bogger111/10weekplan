# 10 周科研代码工作区

研究方向：AI / Computer Vision / 3D–4D Human / Motion Modeling。

本仓库保存现有代码整理快照，以及后续计划、配置和科研日志。首次提交仅记录整理时的文件状态，不认定历史代码作者，不表示 Week1 作业已完成。

## 目录用途

```text
/home/bogger/10weekplan/
├── README.md                 目录说明与状态
├── FILE_INVENTORY.md         现有文件及产物清单
├── .gitignore                版本管理边界
├── .venv/                    根目录旧环境（不纳入 Git）
└── week1/
    ├── *.py                  独立练习与参考实验脚本
    ├── computation_graph.svg 已有计算图产物
    ├── *.pt                  已有权重和 checkpoint（不纳入 Git）
    ├── .venv/                Week1 环境（不纳入 Git）
    └── week1_work/           综合手写作业，尚未完成
        └── checkpoints/     当前为空，运行产物目录（不纳入 Git）
```

保留现有源码、环境和产物路径。独立练习与综合作业分开记录。后续计划、配置和科研日志可以纳入 Git；目前尚无独立计划文件或持久化科研日志。CSV、图片和日志不按扩展名笼统排除。

## 环境与入口

Week1 解释器：`/home/bogger/10weekplan/week1/.venv/bin/python`。

此前只读环境检查：Python 3.12.3，PyTorch `2.14.0+cu130`，CUDA 13.0，CUDA available=True，GPU 为 RTX 5070 Laptop GPU。根目录 `.venv` 未安装 PyTorch。此记录不是依赖锁定或训练成功证明。

独立练习入口包括 `week1/sgd_one_step.py`、`week1/train_toy.py`、`week1/debug.py` 和 `week1/resume_check.py`。使用上述解释器，并以 `week1/` 为工作目录；部分脚本使用相对路径保存或读取 checkpoint。多个脚本会在导入或直接执行时训练，请先阅读源码。

综合作业 `week1/week1_work/main.py` 为空，当前没有有效运行入口。`data.py`、`model.py`、`trainer.py` 有部分实现；训练器存在未定义变量、调用接口不一致、梯度裁剪参数缺失和评估调用错误。`data.py` 尚未使用传入的 seed。`checkpoints.py`、`debug.py`、`test.py` 为空。修复和补全这些内容属于后续本人实现任务，本轮未修改训练代码。

## Mac 与 WSL 对应关系

Mac 资料根目录为 `/Users/mac/顶会刊收割机/`，其中 `01_10week_plan/` 暂与本工作区对应，具体作业要求尚待核对；`02_papers/` 尚无已确认的 WSL 对应目录。

Mac 路径只是资料位置说明，当前 WSL 无法直接访问。尚未配置远程仓库或跨设备同步。

## 现有运行产物

权重和 checkpoint 保留原位，详见 `FILE_INVENTORY.md`。现有恢复实验权重比较支持连续训练与完整恢复权重一致，但不能替代源码复现、训练日志或综合作业验收。
