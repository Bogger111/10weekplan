# 现有文件清单

整理日期：2026-10-11。所有下列相对路径均基于 `/home/bogger/10weekplan/`。本清单不认定代码作者，不把预期输出或模板当作已完成实验。

## 纳入 Git

| 路径 | 用途 / 状态 |
|---|---|
| `.gitignore`、`README.md`、`FILE_INVENTORY.md` | 本轮新增的版本管理边界、目录说明与清单 |
| `week1/micrograd.py` | 手写标量自动微分和 MLP；包含顶层训练循环 |
| `week1/sgd_one_step.py` | 标量 SGD 单步练习 |
| `week1/autograd_gate.py` | PyTorch 自动微分训练练习 |
| `week1/train_toy.py` | 合成回归数据、模型、训练和评估 |
| `week1/inspector.py` | 梯度统计辅助函数 |
| `week1/optimizer_norm.py` | 优化器与归一化练习 |
| `week1/debug.py` | detach、非有限值和异常检测练习 |
| `week1/amp.py`、`week1/amp_debug.py`、`week1/amp_compare.py` | AMP、环境检查和 FP32 对比 |
| `week1/dataloader_bench.py` | DataLoader 性能对比 |
| `week1/profile_train.py`、`week1/profile_dataloader.py` | 训练与数据加载 profiler 脚本 |
| `week1/resume_check.py` | 连续训练、恢复训练和保存权重比较 |
| `week1/gate.py` | 集成实验脚本；当前评估变量作用域及 checkpoint 接口存在问题 |
| `week1/computation_graph.svg` | 已有计算图产物；来源和生成过程未确认 |
| `week1/week1_work/data.py` | 综合作业部分实现；seed 未使用 |
| `week1/week1_work/model.py` | 综合作业 MLP，支持 Identity / BN / LN |
| `week1/week1_work/trainer.py` | 综合作业训练器草稿，尚不能正常运行 |
| `week1/week1_work/main.py`、`week1/week1_work/checkpoints.py`、`week1/week1_work/debug.py`、`week1/week1_work/test.py` | 空文件，尚待本人实现 |

## 保留原位，排除出 Git

| 路径 | 已知用途 / 证据 |
|---|---|
| `.venv/` | 根目录 Python 3.12.3 虚拟环境，未安装 PyTorch |
| `week1/.venv/` | Week1 专用 Python / PyTorch 环境 |
| `week1/__pycache__/` | Python 字节码缓存 |
| `week1/week1_work/checkpoints/` | 空运行产物目录；Git 不追踪空目录 |
| `week1/step50.pt` | 含模型、Adam 状态、global_step=50、配置和 CPU RNG |
| `week1/continuous_final.pt` | 连续训练保存权重 |
| `week1/resumed_final.pt` | 完整恢复保存权重；与 continuous 的最大参数绝对差为 0.0 |
| `week1/model_only_final.pt` | 仅恢复模型的保存权重；与 continuous 最大参数绝对差为 0.18361039459705353 |
| `week1/checkpoint.pt` | 含模型、optimizer、rng、step=49；缺少当前 gate.py 要求的 global_step 和 scaler |

以上权重关系来自此前只读加载和比较，未重新训练。没有发现综合作业运行结果或 Week1 持久化训练日志。现有产物不保证当前源码可以复现。

`.gitignore` 同时排除后续本地数据集、模型权重、大型输出目录、profiler trace 和 TensorBoard 事件；科研日志、计划、配置、CSV 和图片默认保留版本管理资格。大型新文件仍需提交前检查。
