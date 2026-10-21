# Week 1 — Training & Debugging Fundamentals

> 3D/4D Human 10 周科研计划 v2.2 · Week 1 执行细化 · 创建于 2026-09-28  
> 总投入：5 天 × 约 3.5–4h，合计 **19h 专注学习**；休息另计。  
> 当前状态：任务已布置，学习与实验结果待本人完成、记录和验收。  
> 结构：Day 1–Day 5 为五个并列章节，使用完全相同的栏目顺序。

## Week Goal

本周结束时，你应能独立解释并调试一个小型 PyTorch 训练循环：知道梯度从哪里来、为什么累积、参数如何更新，能恢复训练、定位断梯度与非有限值，并用测量结果判断性能瓶颈。

形成四项能力：

- **Understand / Trace**：从 loss 追踪到参数，解释 reverse-mode AD、VJP 与梯度累积。
- **Run / Manipulate**：实现标量 autograd；在统一小模型上切换 optimizer、normalization、训练与评估模式。
- **Experiment / Analyze**：用控制变量实验检查 checkpoint / resume、AMP、gradient debugging、Profiler / DataLoader。
- **Research Log**：留下问题、假设、代码、命令、结果、结论和限制，供后续人体重建项目复用。

本文件只细化 Week 1，**不修改 10 周总路线**。继续保留 v2.2 的 HMR2.0 / 4DHumans → Multi-HMR → Human3R 三个核心 Demo 优先级；本周以小型合成数据训练实验打基础，不新增大型 repo 复现或数据集下载要求。当前依据为已提供的 v2.2 对话内容与本次任务要求；原始 Excel 未作为本地附件提供，未做逐单元格核对。

## Prerequisites

- 数学：一元求导、偏导、链式法则、矩阵乘法与维度；卡住时只补当前例子需要的知识。
- Python：类、闭包、递归、集合；会运行 `.py` 文件。
- 环境：一个能运行 PyTorch 的 Python 环境；基础实验可用 CPU，dtype 与加速设备要求见各日安排。
- 数据：Day 1 只有几个标量；Day 2–5 使用固定随机种子的合成回归数据，无外部数据权限依赖。
- 记录：填写 [environment.md](environment.md)；记录解释器路径、Python / PyTorch 版本、设备和随机种子。

先用你实际学习用的解释器检查：

```bash
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

若没有可用环境，按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/)选择与系统及 Python 兼容的安装方案。环境排查先限时 15 分钟，超时先做手算与纯 Python 实现；依赖 PyTorch 的实验记为待补，未实测的验收项不标记通过。

## 统一科研资料结构

当前目录即资料根目录。以下内容已创建；代码中的 TODO 是你的实现任务，空日志不是已完成的实验。

```text
10weekplan/
├── README.md              # 使用入口与仓库衔接
├── week1.md               # Day 1–Day 5 并列执行计划
├── environment.md         # 环境与复现记录
├── .gitignore
├── references/            # 原计划及参考资料索引
└── week01/
    ├── src/               # 各日实现
    ├── experiments/       # 各日实验入口
    ├── notes/             # 推导与概念笔记
    ├── results/           # 各日输出与报告
    ├── research_log.md    # 每日研究日志
    └── fail_log.md        # 每日失败与修复记录
```

| 日程 | 代码与资料准备状态 | 执行状态 |
| --- | --- | --- |
| Day 1 | 已提供 autograd 练习骨架、验收与对照脚本、笔记模板 | 本人实现、实验与验收待完成 |
| Day 2 | 训练器及 optimizer / normalization 实验按当日任务新增 | 实验与验收待完成 |
| Day 3 | 保存/恢复功能及 resume 实验按当日任务新增 | 实验与验收待完成 |
| Day 4 | AMP / gradient debugging 实验按当日任务新增 | 实验与验收待完成 |
| Day 5 | Profiler / DataLoader 实验与周总结按当日任务新增 | 实验与验收待完成 |


命令均从 `10weekplan/` 目录运行。准备状态与学习完成状态分开记录，计划、模板和空目录不计为实验成果。

## 五天安排总览

| 日程 | 专注时间 | 模块与时间分配 | 当日交付与验收证据 |
| --- | --- | --- | --- |
| Day 1 | 3h45m | 概念与 trace 50m；mini autograd 70m；PyTorch 对照与 VJP 70m；分析与日志 35m | 完成 `src/mini_autograd.py`；两份实验输出；`notes/autograd.md` 中计算图及 Jacobian；Research / Fail Log |
| Day 2 | 3h45m | optimizer 与训练循环 45m；搭建基线 70m；optimizer / normalization 对照 75m；分析日志 35m | 新增 `src/train_toy.py`；`experiments/optimizer_norm.py`；`results/d2_comparison.csv`；说明梯度与参数更新的区别、BN/LN 的归一化轴及 train/eval 行为 |
| Day 3 | 3h45m | checkpoint 状态 trace 40m；保存/恢复实现 75m；连续训练 vs 恢复实验 75m；日志 35m | 在训练器增加 resume；新增 `experiments/resume_check.py`、`results/d3_resume.md`；记录恢复后的 loss、学习率、参数差异 |
| Day 4 | 4h | AMP 原理与代码 trace 45m；FP32/AMP 实验 65m；梯度故障定位 90m；分析日志 40m | 新增 `experiments/amp_debug.py`、`results/d4_debug.md`；至少两类故障的定位与修复证据；AMP 实测或明确的设备限制与待补项 |
| Day 5 | 3h45m | Profiler / DataLoader 阅读与 trace 40m；测量 70m；单变量优化 45m；整体验收与周总结 70m | 新增 `experiments/profile_loader.py`、`results/d5_profile.md`、`week01/weekly_review.md`；瓶颈证据、前后测量和最终 Gate |
| **合计** | **19h** | 每天沿用 Understand → Trace → Run → Manipulate → Experiment → Analyze → Research Log | 不以读完资源数量计算进度 |

## 共同执行约定

以下每天都按 **Understand → Trace → Run → Manipulate → Experiment → Analyze → Research Log** 推进。时间表使用“累计专注时间”：暂停计时休息，休息不计入 19h。建议每 60–90 分钟休息 10–15 分钟。

五天均使用相同的九个栏目：当日目标、前置条件、时间安排、核心资源、执行步骤、交付物、验收标准、受阻处理、下一步。执行步骤统一为四个阶段，内容与时间按当天主题分配。

每天完成后，把实际运行命令和输出路径写入 Research Log。核心资源阅读计入当日预算；官方入口已于 2026-09-28 核对，阅读时选择与本机安装版本对应的文档。

**统一基线与控制变量（Day 2–5 共用）**

Day 2 创建一个 CPU 小模型，Day 3–5 复用：`Linear(8,32) → ReLU → Linear(32,1)`；默认 float32、seed=42，256 个训练样本、64 个验证样本；合成目标 `y=Xw+0.1ε`，训练和验证共用同一个真实 `w`；MSE loss、batch size=32，先运行 100 个更新步骤。

约定 `X.shape=(N,8)`、`y.shape=(N,1)`，先检查预测与标签形状完全相同，避免广播造成假训练。生成数据后固定保存于内存，重复使用；初始模型权重用独立副本保存，不能把后续会变化的状态引用当作初始快照。

每个对照从相同初始线性层权重、数据、样本顺序开始，一次只改一个因素。记录配置、初始/最终 loss、梯度范数、参数变化及异常。此处是训练机制练习，单个种子的玩具实验不能支持“某 optimizer 普遍更好”的结论。

## Day 1：Autograd & Reverse-Mode AD｜3h45m

### 1. 当日目标

解释 backward 在做什么，并亲手实现、验证一个最小标量 autograd。

### 2. 前置条件

具备 Python 类、闭包和链式法则基础；先检查学习环境。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:50 | 50m | Understand → Trace | 环境检查、概念、手算与代码 trace | 计算图、局部导数和概念解释 |
| 00:50–02:00 | 70m | Run | 实现 mini autograd 并初步验证 | Value、拓扑序、局部反传 |
| 02:00–03:10 | 70m | Manipulate → Experiment | PyTorch 基础对照 45m；VJP 验证 25m | 预测/实测/解释表、Jacobian 与 VJP |
| 03:10–03:45 | 35m | Analyze → Research Log | 结果分析、日志与口述验收 | 实验结论、失败记录、Gate 状态 |

### 4. 核心资源

- [micrograd](https://github.com/karpathy/micrograd)：只看 Value 与 backward，理解后关掉参考自行实现。
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：计算图、梯度记录与 grad modes。
- [torch.autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)：VJP、grad_outputs 与返回梯度。

阅读包含在任务时间内；按需查阅，不要求安装 micrograd 或跑完整网络 demo。

### 5. 执行步骤

#### A. Understand → Trace

1. 用不超过 10m 检查环境，并填写 `environment.md`。
2. 用约 20m 阅读下方 Day 1 核心资源，只聚焦 computational graph、局部导数和 reverse-mode。
3. 用约 20m 手算并追踪一个标量计算图。在 `notes/autograd.md` 画出节点和依赖箭头，标出 forward 值、局部导数、反传顺序及每条边的梯度贡献。

先解释五个概念：computational graph、chain rule、reverse-mode AD、VJP、gradient accumulation。Reverse-mode 从输出端的种子梯度向输入传递；forward-mode 从输入方向传播导数。不要把反向传播等同于 optimizer 更新参数。

主例：

```text
a = 2, b = -3, c = 10
u = a * b
d = u + c
q = ReLU(d)
L = q * q + a
```

先手算再核对：`d=4, q=4, L=18`；`∂L/∂a=-23, ∂L/∂b=16, ∂L/∂c=8`。

这里既有 `q*q` 的同一操作数重复使用，也有 `a` 通往 `L` 的两条路径。它们能暴露“把梯度覆盖掉”的错误。补画 `a*a+a`，取 `a=2`，梯度应为 `5`。

#### B. Run

打开 `week01/src/mini_autograd.py`，自行填写 TODO。必须支持：

- `Value` 保存 `data`、`grad`、`_prev`、`_backward`。
- 加法、乘法、常数正整数幂、ReLU；支持与 Python 数字相加/相乘。
- forward 建立依赖；DFS 得到拓扑序；从输出向输入逆序调用局部反传。
- 根节点种子梯度为 `1.0`；局部贡献用 `+=` 累加。
- 本练习约定：每张新建图调用一次 `backward()`；重复实验创建新对象。不要把练习引擎的重复反传行为等同于 PyTorch。

建议分配：类与加法 15m → 乘法/幂/ReLU 20m → 拓扑排序与 backward 20m → 验收及修改 15m。

实现后运行：

```bash
python week01/experiments/check_mini_autograd.py
python week01/experiments/check_mini_autograd.py --torch
```

验收覆盖四类情况：主例、共享变量、幂、ReLU 正/负区间；还用中心差分检查主例梯度。解析值使用 `1e-9` 绝对误差，中心差分使用 `1e-5` 绝对误差。不在 ReLU 的零点做数值梯度判定。

必须能指出：为什么反向拓扑序能让节点收齐下游贡献后再向上传播；为什么即便 `_prev` 去重了，`a*a` 的两个局部导数贡献仍然都要计算。

**小型故障实验**：完成正确版本后，临时把一个乘法反传中的 `+=` 改成 `=`，重跑共享变量例子，记录差异，再恢复正确实现。先记录你实际看到的现象，不把预期 bug 当作已发生的实验结果。

#### C. Manipulate → Experiment

**PyTorch 基础对照（45m）**

先在日志预测各组结果，再运行提供的实验脚本：

```bash
python week01/experiments/pytorch_autograd_basics.py --section basics > week01/results/d1_basics.txt
```

| 实验 | 操作 | 预期 / 要解释的现象 |
| --- | --- | --- |
| 基本 backward | 对主例反传 | 与手算及 mini autograd 对齐 |
| 跨 backward 累积 | 同一个叶子 `x=2`，每次重新计算 `x*x`，反传两次 | `.grad` 为 `4 → 8`；清零后重新前向与反传得到 `4` |
| 图的生命周期 | 同一 `loss=x*x` 默认连续反传两次 | 第二次预期报错，因为所需中间量已释放；新建图再试首次 `retain_graph=True`，可以反传两次 |
| leaf / non-leaf | `x=2, y=3*x, loss=y*y` | `x.grad=36`；默认不保留 `y.grad`；在反传前 `y.retain_grad()` 后 `y.grad=12` |
| detach / no_grad | 比较 `(3*x).detach()+x` 与在 `no_grad` 内算出 `3*x` 后再加 `x` | 两者该例 `x.grad=1`；解释张量脱离已有图和上下文内不记录运算的区别 |

必须分清：**图内多路径贡献相加、跨 backward 的 `.grad` 缓冲区累积、为有效大 batch 主动做 microbatch accumulation** 是三个不同层次。本日完成前两种，Day 2 再讨论第三种。

`retain_graph` 管理反传所需图信息的保留，`zero_grad` 管理梯度缓冲区，两者互不替代。默认给需要梯度的叶子保留 `.grad`，不代表中间张量没有导数。细节参见 [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

**VJP 对照（25m）**

定义 `f: R² → R²`：

```text
y₁ = x₁² + x₂
y₂ = x₁ x₂
J = [[2x₁, 1],
     [ x₂, x₁]]
```

固定 `x=(2,3)`、`v=(1,2)`：`y=(7,6)`，`J=[[4,1],[3,2]]`，`Jᵀv=(10,5)`。

一般情况下 `J ∈ R^(m×n)`，输出端种子 `v ∈ R^m`；以列向量记法，返回输入端 `Jᵀv ∈ R^n`；行向量记法为 `vᵀJ`。这不是直接返回完整 Jacobian。接口依据：[torch.autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)。

```bash
python week01/experiments/pytorch_autograd_basics.py --section vjp > week01/results/d1_vjp.txt
```

完成三个判断：

1. 解释 `grad_outputs=v` 的角色，以及为什么非标量输出需要指定种子。
2. 分别改 `v=(1,0)`、`v=(0,1)`，每次重新 forward，验证返回 Jacobian 对应行的转置。
3. 检查 `torch.autograd.grad` 返回的张量和 `x.grad`；默认情况下该调用返回梯度，不把结果累积进 `x.grad`。

#### D. Analyze → Research Log

- 15m：整理预测与实测，包括一条解释多路径累积的证据、一条图生命周期的证据。
- 10m：填写 `research_log.md` 和 `fail_log.md`。日志注明哪些代码独立写出、哪些参考了 micrograd 或助手提供的实验。
- 10m：不看资料口述下面的 Day 1 Gate；不能回答的项目保持未勾选。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week01/src/mini_autograd.py` | 本人完成 Value 与 backward |
| 实验 | `week01/experiments/check_mini_autograd.py`、`pytorch_autograd_basics.py` | 使用已提供入口，记录实际运行命令 |
| 结果 | `week01/results/d1_mini.txt`、`d1_basics.txt`、`d1_vjp.txt` | 解析梯度、差分与 PyTorch 对照输出 |
| 推导 | `week01/notes/autograd.md` | 计算图、Jacobian、VJP 与机制解释 |
| 日志 | `week01/research_log.md`、`week01/fail_log.md` | Day 1 的预测、实测、解释、失败或困惑及下一步 |

### 7. 验收标准（Gate）

- [ ] `mini_autograd.py` 是本人完成的实现；纯 Python 验收通过。
- [ ] mini autograd、手算、PyTorch 在主例及共享变量例子上对齐。
- [ ] 留下 PyTorch 基础实验、VJP 输出与预测/实测/解释记录。
- [ ] 能解释反向拓扑序、图内 `+=`、跨 backward 清零三者的作用。
- [ ] 能解释 VJP 的维度、`grad_outputs`、leaf/non-leaf、`detach` 与 `no_grad`。
- [ ] 能说明重复 backward 的图释放错误与梯度累积的区别。
- [ ] 完成一条真实的失败/困惑记录、修复或解释，以及下一步。

### 8. 受阻处理

工程证据和口述解释都通过才记 `PASS`。超时优先删去额外笔记美化和扩展阅读；达到当日专注时间预算后记录待补。环境排查先限时 15m；没有可用 PyTorch 时先做手算与纯 Python 实现，对照实验记待补。

### 9. 下一步

基于梯度累积与清零机制，手算并验证一次 SGD 参数更新。

## Day 2：Optimizer + Normalization｜3h45m

### 1. 当日目标

自己写出可解释的训练循环，分清梯度计算、参数更新和归一化各自的作用。

### 2. 前置条件

Day 1 的梯度累积与清零机制能讲清，当前环境能运行 PyTorch。打开 `week01/src/`，本日新增 `train_toy.py`；把实验入口放在 `experiments/optimizer_norm.py`。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | optimizer 阅读、一步 SGD 手算与 trace | 参数更新对照与循环注释 |
| 00:45–01:55 | 70m | Run | 建立合成数据与训练基线 | 训练器和基线 loss |
| 01:55–03:10 | 75m | Manipulate → Experiment | optimizer / normalization 对照 | 实验指标表和模式差异 |
| 03:10–03:45 | 35m | Analyze → Research Log | 结果分析、日志与口述验收 | 实验结论、失败记录、Gate 状态 |

### 4. 核心资源

- [torch.optim](https://docs.pytorch.org/docs/stable/optim.html)：zero_grad、step、SGD/Adam 状态。
- [BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) / [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)：统计轴与 train/eval。

阅读包含在任务时间内，分别安排 15m optimizer 与 15m normalization。

### 5. 执行步骤

#### A. Understand → Trace

1. **15m 阅读**：[torch.optim](https://docs.pytorch.org/docs/stable/optim.html)，只看 `zero_grad`、`step`、SGD/Adam 的参数与状态。写下 `backward()` 和 `step()` 分别修改什么。
2. **15m 手算并验证**：设参数 `θ=2`、`L=(θ-5)²`、lr=0.1，SGD 无 momentum、无 weight decay。先推导梯度 `-6`、更新后 `θ=2.6`；用一个 PyTorch 标量参数验证，建议用 float64、绝对误差 `1e-9`。
3. **15m Trace**：打印 `backward` 前后及 `step` 后的参数和梯度。说明为什么 `step` 本身不等于清空梯度，以及为什么下一轮要显式清零。确认 `model.eval()` 与关闭梯度记录是不同操作。

#### B. Run

1. **15m 数据与形状**：实现 `make_data(seed)`，打印训练/验证张量形状；二者使用同一真实 `w`，分别采样输入与噪声。
2. **20m 模型与循环**：实现 `build_model()`、`train_step()`、`evaluate()`。每步顺序为清零 → forward → loss → backward → 记录梯度 → step。验证时切换 eval 模式并关闭梯度记录，结束后恢复 train 模式。
3. **20m 小 batch 调试**：固定同一个 32 样本 batch，SGD lr=0.01 训练 100 步，保存 loss。以最终 loss 低于初始的 20% 为首个调试目标；这是排查目标，不是理论保证。若未达到，先看形状、梯度和参数是否变化，再单独调整学习率或步数并记录。
4. **15m 正式基线**：使用全部 256 个样本，固定 batch 顺序循环至 100 步。每 10 步记录训练 loss；最后记录整个训练集与验证集的 MSE。后续几天沿用这套数据和循环。

在 `train_toy.py` 保留可复用函数，把直接运行的入口放入 `if __name__ == '__main__':`，避免其他实验导入时意外开始训练。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 固定项与改变量 | 要记录 / 要回答 |
| --- | --- | --- | --- |
| SGD vs Adam | 20m | 同初始权重、数据顺序、100 步、lr=0.01，仅换 optimizer；SGD 无 momentum，两者 weight decay=0 | 初末 loss、第一步参数变化范数、optimizer state 的键；相同 lr 不表示公平调优，也不预设赢家 |
| BN/LN 阅读与轴 | 15m | 输入隐藏特征 `(N,32)`，只看官方定义 | BN1d 每个特征在 batch 轴统计；LN(32) 每个样本在最后 32 维统计；手写两种均值张量的形状 |
| 三种 normalization | 30m | 固定 SGD 与 batch=32；在第一层 Linear 后、ReLU 前放 Identity / BN1d(32) / LN(32)，各跑 100 步 | 同一固定输入的 train/eval 输出差异；BN running mean/variance 的前后值；各组 loss |
| Microbatch 机制复述 | 10m | 无 BN、无 dropout，mean loss，4 个等大小 microbatch | 解释为什么每份 loss 除以 4、只在开头清零且最后 step 一次；代码对照留作有余力时的扩展 |

归一化实验再做一个短 probe：保持样本 A 不变，分别与 3 个、31 个其他样本组成 batch，比较 A 的输出。BN 的训练模式可能随同 batch 样本而改变；LN 在固定参数下不依赖其他样本。使用独立副本做 train/eval probe，避免探测过程修改 BN 统计量后污染另一组比较。依据：[BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)、[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。

#### D. Analyze → Research Log

用 15m 整理表格，10m 写日志，10m 口述检查。`d2_comparison.csv` 至少记录 `run_id, optimizer, lr, norm, batch_size, steps, initial_loss, final_train_mse, val_mse, grad_norm, notes`。不要用不同阶段的 loss 数值直接冒充同一口径的对照。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week01/src/train_toy.py` | 可复用的数据、模型、训练与验证函数 |
| 实验 | `week01/experiments/optimizer_norm.py` | 一步 SGD、optimizer 与 normalization 对照 |
| 结果 | `week01/results/d2_comparison.csv` | 配置、loss、梯度与参数更新指标 |
| 日志 | `week01/research_log.md`、`week01/fail_log.md` | Day 2 的预测、实测、解释、失败或困惑及下一步 |

### 7. 验收标准（Gate）

- [ ] 一步 SGD 手算与 PyTorch 对齐，能区分梯度和参数更新。
- [ ] `train_toy.py` 能运行，固定小 batch 的 loss 明显下降；异常有排查证据。
- [ ] 完成 SGD/Adam 和 Identity/BN/LN 对照，保存配置与结果。
- [ ] 能解释 BN/LN 的统计轴、train/eval 差异，以及为什么 `eval()` 不等于 `no_grad()`。
- [ ] 日志回答：“loss 不降时，先检查哪三件事？”并关联一条实际困惑或故障记录。

### 8. 受阻处理

先保证无 normalization 的基线正确，再做 BN/LN。超时删除 microbatch 扩展代码和曲线美化；未完成必做项如实记待补。

### 9. 下一步

在已验证的训练器里列出恢复训练需要的状态。

## Day 3：Checkpoint / Resume｜3h45m

### 1. 当日目标

训练中断后在新进程接着训练，并用数值对照证明恢复的位置和状态正确。

### 2. 前置条件

Day 2 的无 normalization 基线能跑通。本日固定 CPU、float32、Adam lr=0.01、无 dropout；预先固定 100 步 batch 序列。这里的 checkpoint 指训练状态保存，区别于 activation checkpointing。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:40 | 40m | Understand → Trace | 状态清单、保存时机与恢复 trace | checkpoint 字段表 |
| 00:40–01:55 | 75m | Run | 实现保存与恢复，新进程短程检查 | 恢复函数与 10+1 步记录 |
| 01:55–03:10 | 75m | Manipulate → Experiment | 连续训练、完整恢复、只恢复权重对照 | loss 接续与参数差异 |
| 03:10–03:45 | 35m | Analyze → Research Log | 结果分析、日志与口述验收 | 实验结论、失败记录、Gate 状态 |

### 4. 核心资源

- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：general checkpoint、optimizer state 与恢复流程。

阅读包含在任务时间内，限时约 20m；其余时间用于实现和新进程对照。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 阅读**：[Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) 的 general checkpoint 部分。
2. **10m 列清单**：逐项写出“为什么需要”，使用下面的字段表。
3. **10m 定义边界**：`global_step` 表示已经完成的参数更新次数；在 `optimizer.step()` 后保存。保存到 50 时，下次应执行第 51 次更新，不能重复第 50 个 batch。

| 状态 | 本日要求 | 遗漏可能导致什么 |
| --- | --- | --- |
| model state | 必须；包括模型参数和 buffer | 起点不同 |
| optimizer state | 必须；Adam 的动量估计与步数 | 更新轨迹不同 |
| global_step / batch 位置 | 必须；记录下一个要取的 batch | 重复或漏掉数据 |
| 配置与初始数据生成信息 | 必须；模型形状、lr、seed、dtype | 无法重建相同实验 |
| RNG 状态 | 保存实际使用的随机源；本日至少 PyTorch CPU | 后续随机操作不同；若用 Python/NumPy/CUDA 也记录对应状态 |
| scheduler / scaler | 启用才保存；本日可先不启用 | 后续学习率或缩放状态错误 |

主实验先采用固定 batch 序列，避免同时研究复杂 sampler。不要声称仅保存 seed 就能恢复任意时刻的随机状态。

#### B. Run

1. **25m 保存**：在 `train_toy.py` 增加 `save_checkpoint(...)`，保存上表状态；大文件放 `checkpoints/`，报告只记录路径与配置。
2. **25m 恢复**：增加 `load_checkpoint(...)`。先按配置创建模型及绑定这些参数的 optimizer，再加载状态；恢复步数和数据位置。实际使用的随机状态在重建对象之后、继续训练之前恢复，避免初始化再次消耗随机数。
3. **15m 短程试跑**：训练 10 步保存并退出，新进程读取，先确认模型参数与保存瞬间一致，再运行第 11 步。打印恢复的 step、lr、optimizer state 是否非空。
4. **10m 整理实验入口**：新增 `experiments/resume_check.py`，能分别执行 continuous、save-first-half、resume、model-only、compare。命令行具体写法由你实现后写入报告；训练器保持共享，不复制四套训练逻辑。

#### C. Manipulate → Experiment

| 实验 | 时间 | 执行方法 | 比较项目 |
| --- | --- | --- | --- |
| A 连续基线 | 15m | 相同初始状态完整跑 100 步 | 保存第 51 步更新前 loss、最后参数、逐步 loss |
| B 完整恢复 | 25m | 相同初始状态跑 50 步，退出；新进程恢复再跑 50 步 | 第 51 步 loss/lr/batch 位置；最终参数最大绝对差 |
| C 只载入权重 | 20m | 从同一个第 50 步 checkpoint 恢复模型与数据位置，但新建 Adam，不加载其状态 | 与 B 比较接下来的参数更新和 loss 轨迹 |
| 定位与复测 | 15m | 补回 C 缺失状态；若 B 已不一致，按下方顺序排查 | 修复后的同条件结果 |

注意：C 与 B 的**第 51 步更新前 loss 可能相同**，因为此时参数和输入相同；差异主要从该步参数更新后显现。不要用首次 forward 一致就判断恢复成功。

计算所有对应参数的 `max_abs_diff = max(abs(param_A - param_B))`，本日目标 `≤1e-6`。若不满足，依次比对：初始状态 → 第 51 个 batch → global step → optimizer state → lr → 随机状态 → dtype/设备。先找第一次分歧的位置，不只盯最终 loss。

#### D. Analyze → Research Log

用 15m 完成 `results/d3_resume.md`，10m 写 Research / Fail Log，10m 口述验收。报告包括三组的实际命令、checkpoint 字段、损失接续表、参数差值和结论。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week01/src/train_toy.py` | 新增保存与恢复函数 |
| 实验 | `week01/experiments/resume_check.py` | 连续、完整恢复、只恢复权重三组入口 |
| 结果 | `week01/results/d3_resume.md` | 命令、状态字段、损失接续和参数差异；记录 checkpoint 路径 |
| 日志 | `week01/research_log.md`、`week01/fail_log.md` | Day 3 的预测、实测、解释、失败或困惑及下一步 |

### 7. 验收标准（Gate）

- [ ] 保存/恢复可在不同进程执行；第 51 步使用正确数据。
- [ ] 连续与完整恢复最终参数差 `≤1e-6`；有数值输出作证据。
- [ ] 已实际观察只恢复模型的结果，能解释 Adam 状态为何影响下一次更新。
- [ ] 能列出至少四类必须恢复的状态，并解释保存时机与 step 定义。
- [ ] 日志包含一次恢复实验的失败或困惑，以及排除原因的证据。

### 8. 受阻处理

先把实验缩为 10+10 步找分歧，再恢复 50+50 验收；先不开 scheduler、不启用随机 shuffle。

### 9. 下一步

冻结这套正确 FP32 基线，加入 AMP 与梯度诊断。

## Day 4：AMP + Gradient Debugging｜4h

### 1. 当日目标

解释混合精度训练的执行顺序，并能定位“没有梯度”和“梯度不正常”的原因。

### 2. 前置条件

无 normalization 的 FP32 基线正常，保存一份初始权重。新建 `experiments/amp_debug.py`，提供 clean、detach、nonfinite 三种独立实验路径，每组都重新初始化。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | AMP 阅读、执行顺序与设备检查 | 流程图和设备分支 |
| 00:45–01:50 | 65m | Run | 运行 FP32 / AMP 对照 | 数值、耗时、显存或待补项 |
| 01:50–03:20 | 90m | Manipulate → Experiment | detach 与非有限值故障、修复与复测 | 两条完整故障证据 |
| 03:20–04:00 | 40m | Analyze → Research Log | 结果分析、日志与口述验收 | 诊断流程、失败记录、Gate 状态 |

### 4. 核心资源

- [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)：autocast、GradScaler、unscale 与裁剪顺序。
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：回查梯度记录与断图行为。

阅读包含在任务时间内；AMP 集中阅读约 20m，梯度问题按需回查。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 阅读**：[AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)，关注 autocast 与 GradScaler 各自解决什么问题，使用与本机版本匹配的 API。
2. **15m Trace**：写出 zero_grad → autocast 内 forward/loss → scaled backward → unscale → 检查/必要时裁剪 → scaler step → scaler update。标明 backward 在 autocast 上下文外执行；裁剪与真实梯度范数检查应在 unscale 之后。
3. **10m 环境决策**：确认是否有兼容 CUDA；没有时走下面 CPU 分支。记录设备、dtype、是否使用 scaler，不默认 Apple GPU 与 CUDA 用同一套实验配置。

#### B. Run

1. **15m FP32 控制组**：相同数据和初始权重跑 100 步，确认 loss 与参数有限，保存初末 loss 和梯度范数。
2. **25m 实验组**：有兼容 CUDA 时启用 autocast 与 scaler，重置初始权重和数据顺序，再跑相同步数。混合精度与 FP32 不要求逐位一致，先检查能否稳定训练以及差异量级。
3. **15m 测量**：各配置独立热身 10 步，再测 50 步，重复 3 次；设备异步执行时在计时区间边界同步。记录 step time 中位数、峰值分配显存和测量范围；显存统计在热身后重置。小模型可能没有速度收益。
4. **10m 整理**：将 `device, dtype, scaler, initial_loss, final_loss, grad_norm, nonfinite_count, step_ms, peak_memory` 写入报告，说明性能比较只适用于本次设备与模型。

**CPU 分支**：复用同样时间完成 FP32 基线；CPU autocast bfloat16 若可用，做数值对照，并明确它与 CUDA FP16/scaler 实验的区别。若不支持，完成 AMP 流程 trace、保存将来需测试的配置；余下时间用于下面的故障复测，不耗费整天调设备。CUDA AMP 实测保持待补，显存等不适用字段填 NA。

#### C. Manipulate → Experiment

先做 clean 控制组，使用 `zero_grad(set_to_none=True)`；每次记录参数名、`grad is None`、非有限元素数和有限梯度的范数。检查发生在 backward 之后、step 之前。故障定位先用 FP32，避免精度与代码错误混在一起。

| 故障 | 操作与预期 | 定位过程 | 修复 / 复测 |
| --- | --- | --- | --- |
| 意外 detach（40m） | 在两层 Linear 之间把隐藏特征 detach；前层梯度通常变为 None，后层仍有梯度，loss 仍可能下降 | 先看第一层参数是否变化，再查隐藏特征的 `requires_grad/grad_fn`，追到 detach 位置 | 移除 detach，重置同初始状态及输入；第一层恢复梯度和更新，并与 clean 对照 |
| 非法数值（40m） | 加一个受控错误项，如对一个取值为 -1 且 requires_grad 的标量求 sqrt 并加入 loss；预期出现 NaN | 先检查 forward loss 是否有限，再检查各层梯度；必要时把 forward 和 backward 都放入 anomaly detection 上下文定位算子 | 移除错误项，从干净状态重跑；要求 loss/梯度/更新后参数均有限；不对 NaN 盲目裁剪 |
| 回归复测（10m） | 恢复 clean 模式，连续执行 10 步 | 确认修复没有引入新问题 | 保存成功输出，并关联两条 Fail Log |

非有限值实验只在新建的临时模型/参数上运行，检测到故障就停止 step；这样可以保留正常基线供对照。预期触发的报错也记录为“受控故障实验”。如果实际现象不同，记录并分析，不照抄预期。

#### D. Analyze → Research Log

用 15m 整理报告，15m 完成两条 Fail Log，10m 口述检查。报告末尾写一个可复用的排查顺序：输入/标签 → forward/loss → 图是否连通 → backward 梯度 → step 参数变化。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week01/src/train_toy.py` | 保留正确 FP32 基线及可复用训练步骤 |
| 实验 | `week01/experiments/amp_debug.py` | 精度对照与 clean/detach/nonfinite 路径 |
| 结果 | `week01/results/d4_debug.md` | AMP 实测或待补项；两类故障修复前后的证据 |
| 日志 | `week01/research_log.md`、`week01/fail_log.md` | Day 4 的预测、实测、解释、失败或困惑及下一步 |

### 7. 验收标准（Gate）

- [ ] 能区分 autocast、loss scaling、gradient clipping 的作用。
- [ ] 能说明为什么检查/裁剪前要 unscale，为什么 AMP 不保证加速。
- [ ] 完成 detach 与非有限值两类故障的症状、定位、修复、同条件复测。
- [ ] 能区分 `.grad=None`、零梯度、小梯度、Inf/NaN，并说明各自至少一个可能原因。
- [ ] `amp_debug.py`、`d4_debug.md`、两条 Fail Log 齐全；AMP 实测或设备待补状态明确。

### 8. 受阻处理

先缩成一个 batch、两层模型定位错误。无法使用 AMP 的机器仍完成 FP32 故障实验；待补项需写清未来设备与验收指标。

### 9. 下一步

对正确的 clean 训练循环做性能测量。

## Day 5：Profiler / DataLoader + Final Gate｜3h45m

### 1. 当日目标

用测量找出时间花在哪里，完成一次有证据的性能对照，并整理本周可复现成果。

### 2. 前置条件

使用 Day 4 修复后的 clean FP32 训练循环，保持模型、batch size、数据和 seed 固定。本日新增 `experiments/profile_loader.py`；先在 CPU 上完成，CUDA 项按设备能力补充。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:40 | 40m | Understand → Trace | profiler / loader 阅读与测量设计 | 计时边界和瓶颈假设 |
| 00:40–01:50 | 70m | Run | 基线采集、workers 对照与重复测量 | 耗时表、原始测量 |
| 01:50–02:35 | 45m | Manipulate → Experiment | 单变量改动与复测 | 改前/改后指标和结论 |
| 02:35–03:45 | 70m | Analyze → Research Log | 整体验收 45m；日志与周总结 25m | 周验收状态与周总结 |

### 4. 核心资源

- [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：采集、标记与解释算子表。
- [DataLoader](https://docs.pytorch.org/docs/stable/data.html)：workers、数据加载与传输参数。

阅读包含在任务时间内，两份资料合计约 25m。

### 5. 执行步骤

#### A. Understand → Trace

1. **25m 阅读**：[Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[DataLoader](https://docs.pytorch.org/docs/stable/data.html)，只看采集、算子表、workers 与数据传输相关参数。
2. **15m Trace**：在脚本中用 `record_function` 标记 `load_batch / transfer / forward / backward / optimizer_step`。`load_batch` 要包住实际 `next(iterator)`；只标记已拿到 batch 后的循环体，会漏掉取数据的等待。
3. 运行前写一个可证伪的假设，例如：“数据已经在内存中，2 个 workers 可能因为进程开销比 0 个更慢。”定义 step time 包含哪些阶段。

#### B. Run

1. **20m profiler 基线**：先用 `num_workers=0`，热身后采集约 10–20 步，记录主要 CPU 算子耗时。CUDA 可用时同时采集设备活动。区分总时间与 self time，避免把嵌套事件重复相加。
2. **20m 无 profiler 测量**：关闭 profiler，先热身 10 步，再计时 50 步；重复 3 次，记录每次 step_ms 与 samples/s，并报告中位数、最小值和最大值。明确 iterator 重建是否在计时内，所有组保持一致。
3. **20m workers 对照**：只把 `num_workers` 从 0 改为 2，其余不变，按相同方法测量。macOS 的运行入口放在 `if __name__ == '__main__':` 下，Dataset 类放模块顶层。记录启动开销，避免把一次启动与长期吞吐混为一谈。
4. **10m 整理证据**：保存前 5 个主要耗时项、两种 workers 的测量表；判断瓶颈来自取数据、计算还是启动/调度开销。CPU 时间表不能单独当作 CUDA kernel 耗时结论。

若小模型太快、计时噪声大，可统一把所有配置的测量窗口增至 500 步，再重做同口径比较；不要只给其中一组增加工作量。

#### C. Manipulate → Experiment

1. **10m 选择一个动作**：依据上一段证据，只选一项，例如减少 workers；或在 CUDA 数据传输场景单独测试 pin_memory。其余可选因素本日不同时改变。
2. **20m 复测**：使用相同输入、步数、计时边界和 3 次重复，记录改前/改后指标。性能测量关闭 profiler；需要解释原因时另开一次短 profiler 采集。
3. **15m 分析**：写明收益是否超过重复测量波动；无收益或变慢同样是有效结论。不要把增加 workers、启用 pin_memory 和扩大 batch 一次性打包后归因给其中一个。

CPU 场景把 CUDA 传输选项标为不适用。若现有基线没有加载瓶颈，先保留此结论；只有时间允许时，额外做含固定 CPU 预处理的数据集实验，并明确它是另一个工作负载，不替换原始结果。

`results/d5_profile.md` 至少包含：设备与配置、假设、计时边界、主要耗时项、每次原始测量、汇总表、单变量变化、结果解释和限制。

#### D. Analyze → Research Log

**整体验收（45m）**

1. **15m 检查材料**：代码能定位、实际命令有记录、配置/环境齐全、结果路径存在；清除“预期值被误填为实测”的记录。
2. **20m 复现关键证据**：重跑 Day 1 autograd/VJP、Day 3 resume 对照和 Day 4 clean/修复后的最小例子。某项失败就标明失败位置，不覆盖旧结果制造通过状态。
3. **10m 口述**：从一个 loss 开始讲到参数更新；解释完整恢复为何需要 optimizer；给出梯度异常排查顺序；指出 profiler 中支撑性能结论的证据。

**日志与周总结（25m）**

新增 `week01/weekly_review.md`，用 15m 回答下列问题，再用 10m 更新 Day 5 Research Log 和本周 Gate：

- 本周能独立完成的三件事是什么？分别链接哪份代码或结果？
- 哪两个失败案例最值得复用？根因和检查方法是什么？
- 一个最初的假设如何被实验支持或推翻？
- 哪些尚未验证？是理解、代码、时间还是设备原因？下一步动作是什么？
- 后续阅读人体重建训练代码时，如何检查 loss 路径、梯度、optimizer、模式切换、resume、数值和数据加载？

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week01/src/train_toy.py` | 复用 clean 训练循环 |
| 实验 | `week01/experiments/profile_loader.py` | 阶段标记、workers 对照与计时 |
| 结果 | `week01/results/d5_profile.md` | 原始测量、汇总表、单变量结论与限制 |
| 总结 | `week01/weekly_review.md` | 本周机制、失败案例、复用检查表与待补事项 |
| 日志 | `week01/research_log.md`、`week01/fail_log.md` | Day 5 的预测、实测、解释、失败或困惑及下一步 |

### 7. 验收标准（Gate）

- [ ] 能正确采集与解释主要耗时项，未混淆嵌套时间或 profiler 开销。
- [ ] workers=0/2 对照及一次单变量改动均有同口径测量；无收益也如实报告。
- [ ] `profile_loader.py`、`d5_profile.md`、`weekly_review.md` 和每日日志可定位。
- [ ] 已更新“Week 1 最终 Gate”中的所有项目的实际状态，待补项都有具体行动与通过标准。
- [ ] 能根据保存的命令重现关键实验，而不仅是展示截图。

### 8. 受阻处理

先做 CPU profiler 与 workers=0 基线；多进程错误作为 Fail Log 记录，修复后再比较。优先交付可信测量和周总结，不追加大型性能优化。

### 9. 下一步

完成本周待补项并整理仓库提交；下一周内容依 v2.2 原路线展开。

## Fail Log / Research Log 模板

可直接填写 [Research Log](week01/research_log.md) 和 [Fail Log](week01/fail_log.md)。每天沿用以下结构。

### Research Log

```markdown
## YYYY-MM-DD / Week 01 Day N / 主题
状态：未开始 / 进行中 / PASS / 待补
计划时间 / 实际时间：
环境、seed、代码版本或文件快照：
Question：本日要回答的具体问题？
Hypothesis：改变什么，预期出现什么，为什么？
Understand / Trace：概念、关键代码位置、数据/梯度路径。
Run：复现命令、配置、输入、结果文件路径。
Manipulate / Experiment：控制组、实验组、唯一改变的变量。
Prediction / Observation：预测与实测数值，附证据。
Analyze：结论、机制解释、替代解释、适用条件与限制。
Implementation provenance：哪些自己写，哪些参考了资料或助手？
One thing I can now explain：不看资料能讲清楚的一件事。
Bugs / confusion：关联 Fail Log ID。
Gate：逐项状态，未通过项的补做动作。
Tomorrow first step：一个可直接执行的动作。
```

### Fail Log

```markdown
## W01-DN-F01 / 简短标题
时间 / 状态：Open / Explained / Fixed / Retested
Context：环境、代码版本、输入与复现命令。
Expected：预期值或行为。
Observed：真实输出、报错、错误梯度或现象；附文件路径。
Minimal reproduction：最小复现步骤。
Hypotheses：候选原因与排除证据。
Root cause：有证据支持的原因，未确认则写待验证。
Fix / explanation：改动或对机制的解释。
Retest：同输入复测的命令、结果与判据。
Lesson：下次在哪里加什么检查。
```

## Week 1 最终 Gate

| 项目 | 通过证据 | 当前状态 |
| --- | --- | --- |
| Autograd | 分支与共享变量梯度正确；能解释反向拓扑序；手算、数值差分、PyTorch 对齐 | 待完成 |
| VJP / 累积 | Jacobian 维度正确；VJP 数值正确；解释图内与跨 backward 累积 | 待完成 |
| Optimizer / Normalization | 可复现训练基线；SGD 一步手算对齐；BN/LN 对照及适用条件 | 待完成 |
| Checkpoint / Resume | 新进程恢复；固定实验的参数差 ≤1e-6；保存配置和状态说明 | 待完成 |
| Gradient debugging | 至少两类故障有最小复现、原因、修复与复测证据 | 待完成 |
| AMP | FP32 对照与流程解释；支持设备上的数值实验，或明确的待补项 | 待完成 |
| Profiler / DataLoader | 一份瓶颈报告；单变量改动前后测量，说明无收益也成立 | 待完成 |
| Research habit | 五日日志、结果路径、失败证据、环境记录、周总结可供他人复现 | 待完成 |

所有必需证据齐全才记完整 `PASS`。仅受设备限制的 AMP 项可记“核心通过，AMP 硬件实测待补”，列出未来设备、命令与验收指标；这不等于全部 Gate 通过。未通过概念/代码项优先在本周预算内修补，超出预算如实登记，不擅自重排 Week 2–10。

## 与 research-journey 衔接

建议后续把这个目录整体放入 `research-journey/10weekplan/`，根 README 只保留导航链接；本目录作为计划、代码和日志的唯一维护位置，避免再复制一份并行更新。具体迁移约定见 [README](README.md)。

每天完成对应代码、实验和日志后，再整理提交。原始 v2.2 总计划仍是后续各周路线依据；新建 Week 2–10 文件时引用原计划，不在这里另拟路线。
