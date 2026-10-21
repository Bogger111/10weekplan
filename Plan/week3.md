# Week 3 — 3D Human Representation｜Skeleton → Rotation → FK → SMPL

> 3D/4D Human 10 周科研计划 · Week 3 执行细化 · 创建于 2026-10-13  
> 总投入：5 天 × 约 3.5–4h，合计 **19h 专注学习**；休息另计。  
> 当前状态：延续会话中 Week 2 已由本人确认 PASS；本周任务已布置，实现、实验与验收待本人完成。  
> 结构：Day 1–Day 5 为五个并列章节，沿用 `week2.md` 相同的九个栏目、四个执行阶段与证据要求。

## Week Goal

本周结束时，你应能把“关节坐标”与“生成关节 / 表面的参数”分开：给定局部旋转、骨架父子关系与根位姿，自己计算全局关节位置；给定 SMPL 参数，能追踪它们如何经过 shape / pose blend shapes、FK 与 LBS 产生 joints 和 mesh，并用一个动作片段检查表示、坐标系与动画是否一致。

形成四项能力：

- **Understand / Trace**：解释 coordinate frame、SO(3)、四种旋转表示、SE(3)、kinematic tree、FK、LBS，以及 SMPL / SMPL-X / AMASS 的关系；追踪参数到 joints / vertices 的主要 shape。
- **Run / Manipulate**：实现 rotation playground、手写 6D→R 并对照 PyTorch3D；自写 `forward_kinematics()` 和最小 LBS；用 `smplx` 分别改变人体模型的四组参数。
- **Experiment / Analyze**：完成旋转边界检查、局部 / 全局旋转干预、rest-pose skinning 检查及人体参数单变量实验；生成 root trajectory、3D skeleton 与 mesh animation。
- **Research Log**：保留坐标约定、公式、预测、实测、源码位置、失败与复测，为 Week 4 的 SMPL → Camera → HMR 提供可以直接复用的表示接口。

**总计划依据（原计划内容）**

本次已只读读取本地 [总计划表](00_plan/10周研究计划_曾奕博.xlsx)，以工作表 `10周计划!B12:D12` 为本周事实依据。正式要求如下，保留原表的研究主线与交付边界：

| 字段 | 原计划内容 |
| --- | --- |
| Focus（B12） | 3D Human Representation｜Skeleton → Rotation → FK → SMPL。重点：coordinate frame、SO(3)、axis-angle / quaternion / rotation matrix / 6D、SE(3)、kinematic tree、FK、LBS、SMPL / SMPL-X、AMASS。 |
| Core Resources（C12） | Modern Robotics；PyTorch3D Transforms；SMPL / SMPL-X；AMASS。 |
| Deliverable 1（D12） | 实现 rotation playground：多表示互转；自己实现 6D→R 并用 PyTorch3D 校验。 |
| Deliverable 2（D12） | 自定义最小 kinematic tree，完成 forward_kinematics() 与 skeleton animation。 |
| Deliverable 3（D12） | 用 smplx 控制 betas / body_pose / global_orient / transl，观察参数效应。 |
| Deliverable 4（D12） | 若 AMASS 已可用，则读取一个小子集输出 root trajectory、3D skeleton 与 SMPL mesh animation；若暂时拿不到数据，则用自造 SMPL motion sequence 完成同等表示与可视化练习。 |
| Gate（D12） | 能区分 joints / rotations / SMPL parameters / mesh，并解释 local/global rotation、FK 与 LBS。 |

补充依据：`资源清单!B20:G21` 将 Modern Robotics 与 **6D Rotation Representation** 分别列为 Week 3 核心资源与精读资源；`资源清单!B3:G5、B8:G9` 给出模型 / 数据入口和工具优先级。AMASS 按需获取，不要求下载完整数据集。`10周计划!B4:B7` 规定能力驱动、≈20h / 周、至少 50% 时间用于代码 / 实验 / 调试；前 9 周必跑 Demo 仍是 HMR 2.0 / 4DHumans、Multi-HMR、Human3R。

**执行建议与版本边界**

下面的 19h 拆分、toy 数据、实现接口、容差、文件名和逐日 Gate 均为**执行建议**，用于落实原表四项交付，不是表格提供的实验事实。本文件依据当前 `.xlsx`；未核对同目录 `.numbers` 与它是否逐单元格一致，不额外指定总计划版本号。

Week 2 的 PASS 是延续会话中本人明确确认的学习状态；现有 [week2.md](week2.md) 中的初始待完成表不自动反映后来的学习进度，本次不修改或重新验收它。Week 3 不要求重做 Transformer，也不提前引入 Week 4 的相机投影、MPJPE / PA-MPJPE 或 HMR 网络复现。

## Prerequisites

**执行建议**：确认直接依赖的能力和资源；模型 / 数据访问与学习任务分开记录。

- 数学：矩阵乘法、转置、点积、叉积、向量范数；能说明正交矩阵保持长度、行列式为 -1 的反射不是合法 SO(3) 旋转。
- Python / PyTorch：会写 batched tensor 运算、`reshape / transpose`、finite 检查与基本 autograd；沿用本人 Week 1–2 的可用环境。
- Week 2 衔接：知道 `[B,T,J,3]` 是关节坐标、`[B,T,J,C]` 可以是特征；特征宽度 C 不等于旋转自由度。已有 toy 的任意单位 / joint 顺序保留为背景，不假装它采用 SMPL 骨架。
- 环境：CPU 足够完成旋转、FK 与短 SMPL 序列；优先沿用 PyTorch、matplotlib，`smplx` 用于模型调用。PyTorch3D 本周先用 transforms，不把完整 GPU renderer / CUDA 扩展作为轻量任务前置条件。
- 模型：`smplx` 软件包与 SMPL / SMPL-H / SMPL-X 参数文件是不同资源。实际使用前检查已有授权文件、模型类型、gender、扩展名、shape 系数数量与路径；包安装成功不等于模型已可用。
- 数据：Day 1–3 用自造点、旋转和骨架；Day 4 需要可用的人体模型文件；Day 5 优先只取已有 AMASS 的一个动作，缺少数据时按原表生成短 SMPL motion sequence。
- 记录：执行时新建 `week03/environment.md`，记录解释器、Python / PyTorch / PyTorch3D / smplx 版本、device、dtype、seed、源码访问日期 / commit、模型文件来源及本地路径。

先在本人学习用的解释器运行基础检查：

```bash
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Day 2 / Day 4 到达对应依赖时再分别检查 `pytorch3d.transforms` 与 `smplx` 的 import 和实际调用。环境排查每项先限时 15m；需要安装 / 编译时交由环境专项处理，不为旧教程整套环境无限耗时。没有 PyTorch3D 可以继续自写与手算，但“PyTorch3D 校验”仍待补；没有 AMASS 可以走自造序列；没有人体模型文件则 SMPL 实跑 / mesh Gate 仍待补，不能用 toy 骨架代替。

## 统一科研资料结构

**执行建议**：当前资料根目录是 `/Users/mac/顶会刊收割机/01_10week_plan/`。本次只生成 `week3.md`；下方 `week03/` 为本人执行时按需新建的结构，尚未提供练习脚本、模型文件或实验结果。

```text
01_10week_plan/
├── 00_plan/                       # 原计划表，只读使用
├── week1.md
├── week2.md                       # 已有计划，不改动
├── week3.md                       # Day 1–Day 5 并列执行计划
└── week03/                        # 执行时按需创建
    ├── environment.md
    ├── src/                       # rotations.py / kinematics.py / skinning.py
    │                              # body_model.py / motion_sequence.py
    ├── experiments/               # 各日入口，本人实现
    ├── notes/                     # frames / rotations / FK / SMPL / AMASS
    ├── results/                   # 原始参数、CSV、图、动画与实验报告
    ├── research_log.md
    ├── fail_log.md
    └── weekly_review.md
```

| 日程 | 代码与资料准备状态 | 执行状态 |
| --- | --- | --- |
| Day 1 | 坐标约定、SO(3) / Rodrigues 与点旋转检查按当日任务新增 | 本人实现、实验与验收待完成 |
| Day 2 | 多表示互转、手写 6D→R 与 PyTorch3D 对照按当日任务新增 | 实验与验收待完成 |
| Day 3 | SE(3)、最小 tree、FK 与 skeleton 动画按当日任务新增 | 实验与验收待完成 |
| Day 4 | 两骨骼 LBS、smplx trace 与四组参数实验按当日任务新增 | 模型可用性、实验与验收待完成 |
| Day 5 | AMASS / 自造动作、三类可视化与周总结按当日任务新增 | 实验与验收待完成 |

所有建议入口从上述资料根目录运行；入口实现后把实际命令和参数写进日志。实验导出、临时文件和代码留在 `week03/` 内。已有模型 / 数据可从原位置只读加载，日志记录路径，不移动、覆盖或把授权模型复制进同步项目 `sources/`。空目录、计划与示例结果表均不计作学习成果。

## 五天安排总览

**执行建议**：先确定坐标语义和旋转合法性，再组装骨架与表面；阅读、排错和复测全部包含在预算内。

| 日程 | 专注时间 | 模块与时间分配 | 当日交付与验收证据 |
| --- | --- | --- | --- |
| Day 1 | 3h45m | 坐标系 / SO(3) 50m；旋转实现 70m；点变换 / 顺序 / 小角度实验 70m；分析日志 35m | `rotations.py`；坐标约定；手算与矩阵合法性；Research / Fail Log |
| Day 2 | 3h45m | 多表示 / 6D 精读 45m；互转与自写 6D 75m；参考对照 / 边界实验 70m；分析日志 35m | rotation playground；PyTorch3D 对照；符号 / π 边界 / 退化输入证据 |
| Day 3 | 3h45m | SE(3) / tree trace 45m；FK 与动画 75m；父子 / 全局干预 70m；分析日志 35m | `forward_kinematics()`；手算 / 骨长；skeleton animation；局部 / 全局实验 |
| Day 4 | 4h | LBS / SMPL trace 45m；toy LBS 与人体调用 75m；四参数控制 80m；分析日志 40m | rest-pose / 单骨骼 LBS；SMPL shape trace；betas / pose / orient / transl 对照 |
| Day 5 | 3h45m | AMASS / SMPL-X 与片段定义 40m；序列和三类图 75m；root / body 干预复测 45m；最终验收 / 周总结 65m | root trajectory、3D skeleton、mesh animation；最小实验报告；最终 Gate 与 Week 4 交接 |
| **合计** | **19h** | 每天沿用 Understand → Trace → Run → Manipulate → Experiment → Analyze → Research Log | Run / Experiment 阶段共 **11h45m**；以实现、数值和可解释产物计算进度 |

## 共同执行约定

**执行建议**：五天均使用相同九个栏目和四个执行阶段。时间表为累计专注时间；每 60–90m 可休息 10–15m，休息不计入 19h。先写 Prediction 再填 Observation；每个 Gate 同时检查本人解释和实际产物。

**独立实现与参考边界**

1. 本人手写 Rodrigues 的主要逻辑、6D→R、SE(3) inverse / composition、tree FK 与两骨骼 toy LBS；可用 tensor 运算和 autograd，不用封装 FK / skinning 函数代替独立主路径。
2. 多表示互转可以使用 PyTorch3D 的 axis-angle / quaternion / matrix 转换作为已记录的工具路径；必须说明函数输入、输出和约定。6D→R 主路径自己写，库函数仅作独立对照。不强加手写所有逆转换的任务。
3. 完整 SMPL 通过 `smplx` 调用，不要求重写其全部实现或训练人体模型。Trace 只覆盖 `forward → lbs → blend_shapes / vertices2joints / batch_rigid_transform → joints / vertices` 的关键路径。
4. 正确性检查先 CPU、float64、seed=42；人体模型按实际支持的 dtype（通常 float32）运行。所有输入 / 输出应有限；不把绘图用的 `detach()` 放回需要反传的主路径。
5. 执行建议容差：float64 一般对照 `atol=1e-6、rtol=1e-5`，float32 / SMPL 对照 `atol=1e-5、rtol=1e-4`。报告最大绝对差与实际 dtype；近 π / 近退化例单独解释，不能靠放宽全部阈值掩盖错误。

**坐标、旋转与单位约定**

- toy 使用右手系，列向量公式 `p_A=R_AB p_B+t_AB`；`T_AB` 表示把 B 坐标映射到 A，组合为 `T_AC=T_AB T_BC`。批量点按行存成 `[...,N,3]` 时，同一公式实现为 `P_A=P_B @ R_AB.transpose(-1,-2)+t_AB`，广播轴单独确认。
- 区分主动旋转与坐标变换：主动旋转改变同一 frame 中的点；被动变换改变同一点的坐标表示。toy 的主动 +90° z 旋转应把 `(1,0,0)` 送到 `(0,1,0)`；换 frame 时必须给出源 / 目标，不能只说“乘 R”。
- 角度全部用 radians；axis-angle 向量 `ω=θu`，不是三项 Euler angles。四元数统一 `(w,x,y,z)`、单位范数；`q` 与 `-q` 表示同一个旋转。
- `SO(3)={R∈R^(3×3): RᵀR=I, det(R)=1}`；`SE(3)` 的 4×4 矩阵最后一行为 `[0,0,0,1]`。6D rotation 是 SO(3) 的冗余表示，不是 SE(3) 或六个旋转自由度。
- **6D 统一用 PyTorch3D 的行约定**：`matrix_to_rotation_6d` 取前两行，Gram–Schmidt 后沿 `dim=-2` 堆叠为三行。采用列约定的资料须先显式转换，不能直接和行约定参考比较。依据已核对的 [PyTorch3D conversion 源码](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html)。
- toy 点 / 骨长使用任意单位，图中明示。人体序列按模型 / 数据说明核对米与坐标轴，记录确认依据；为了显示而改轴时对 joints、mesh、trajectory 一起变换，原始参数保留。
- 本周的“global”是相对当前声明的外部 frame；它可能是 toy frame、模型 frame 或数据 frame。没有 camera pose / scene 定义，不能仅凭 `global_orient` 或 AMASS `trans` 声称得到后续 world-space reconstruction。

**统一 shape 与人体模型边界**

| 对象 | 本周接口 / 典型 shape | 必须讲清楚 |
| --- | --- | --- |
| Joint positions | `[B,T,J,3]` | 位置、frame、单位、joint convention；不包含每个局部 frame 的完整朝向 |
| Rotations | axis-angle `[...,3]`；quaternion `[...,4]`；matrix `[...,3,3]`；6D `[...,6]` | 表示同一个 SO(3) 元素；先确定 local / global，不能用维度推断语义 |
| Toy tree | `parents[J]`、`offsets[J,3]`、`R_local[B,T,J,3,3]` | parents 是拓扑关系，offset 是父 frame 中的静态骨长向量 |
| Toy FK 输出 | `R_global[B,T,J,3,3]`、`P_global[B,T,J,3]` | 相对声明的 toy frame；root translation 与 local rotation 分开 |
| SMPL 参数 | `betas[B,Kβ]`、`body_pose[B,69]`、`global_orient[B,3]`、`transl[B,3]` | 这里是基础 SMPL 的 axis-angle 接口，69=23×3；Kβ 由已加载文件与配置确定 |
| SMPL 输出 | `vertices[B,V,3]`；`joints[B,J_out,3]`；`faces[F,3]` | 标准 SMPL 常见 V=6890；J_out 可能含额外 keypoints，未必等于 24 个运动学 joints |
| SMPL-H / SMPL-X | body pose 常为 `[B,63]`，另有 hands；SMPL-X 还有 face / expression | 同一个库支持不同模型，不等于参数直接互换；手部 PCA / full axis-angle 需单独核对 |

人体调用先用 `pose2rot=True` 的 axis-angle 路径、`joint_mapper=None`，并显式传四组参数，避免模型内部默认参数污染对照。SMPL-X 以源码 / 接口比较为必做背景；有可用模型时再作轻量扩展调用，不额外要求下载第二套模型。实际输出数量按本次打印和 [smplx 官方代码](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py) 确认。

## Day 1：Coordinate Frame + SO(3) + Rotation Matrix｜3h45m

### 1. 当日目标

建立坐标约定；解释合法旋转矩阵、主动 / 被动变换与旋转顺序；自写 axis-angle→R 的 Rodrigues 主逻辑，用手算与数值确认点真的转向预期方向。

### 2. 前置条件

会点积、叉积和矩阵乘法；使用本人环境完成基础检查。**执行建议**：新增 `week03/src/rotations.py`、`week03/experiments/frame_rotation_check.py`；本日先不加入 FK 或人体模型。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:50 | 50m | Understand → Trace | frame、SO(3)、Rodrigues 手算与广播 | 坐标声明、公式、手算表 |
| 00:50–02:00 | 70m | Run | 自写旋转、点变换和矩阵合法性检查 | 可批量运行的主路径 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 方向 / 顺序 / 长度 / 小角度对照 | 预测、实测、误差与修复 |
| 03:10–03:45 | 35m | Analyze → Research Log | 整理报告、日志与闭卷解释 | Day 1 证据与 Gate 状态 |

### 4. 核心资源

- [Modern Robotics · Book Resources](https://modernrobotics.northwestern.edu/nu-gm-book-resource/)：定位 3.2.1 Rotation Matrices 与 3.2.3 Exponential Coordinates；只看 orientation / frame change / point rotation 与指数坐标。
- [PyTorch3D Transforms](https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html)：核对 `axis_angle_to_matrix` 的输入语义，参考调用放到 Day 2。

阅读合计约 25m；不展开 twist、机器人控制或完整 Lie theory，先把具体点和坐标算对。

### 5. 执行步骤

#### A. Understand → Trace

1. **25m 阅读与画图**：画 frame A / B 的三根轴和同一个点；分别写主动 `p'=Rp` 与被动 `p_A=R_AB p_B`。说明 `R_BA=R_ABᵀ`，逆变换不是把角度符号和 frame 名随意替换。
2. **15m 手算**：写出 `Rz(π/2)`，计算 `e_x→e_y`、`e_y→-e_x`；再算 `Rx(π/2)Rz(π/2)e_x` 与逆顺序。先填预测，运行后对照。
3. **10m Trace**：令 `Ω=[ω]×`、`θ=||ω||`，写出 `R=I+(sinθ/θ)Ω+((1-cosθ)/θ²)Ω²`；确认 `ω:[...,3] → Ω / R:[...,3,3]`。推导 θ→0 时两个系数分别趋于 1、1/2。

这是 Rodrigues 的小角度稳定形式；零旋转不应产生 NaN，也不应该返回全零矩阵。用 normalized axis 写法时同样要处理 θ=0。

#### B. Run

1. **20m 构建基础函数**：本人写 `skew(omega)`、`axis_angle_to_matrix(omega)`，支持任意 batch 前缀；identity 从输入的 device / dtype 创建。
2. **20m 处理小角度**：用稳定 sinc 或 Taylor 系数处理零点。若用 `where`，让未选中分支也避免 0/0，以免反传出现非有限值；记录具体方案。
3. **15m 点变换**：实现列公式对应的行存储点变换；先用单点，再用 `[2,4,3]` 点与 `[2,3,3]` 旋转。显式插入点轴，确认不同 batch 没被混乘。
4. **15m Run**：identity、绕 x/y/z 的 π/2，以及固定三项非零的 axis-angle `[0.3,-0.2,0.4]`。打印 R、变换点、`max|RᵀR-I|`、`|det(R)-1|` 与 finite 状态。

建议接口只规定语义，不提供已完成实现：

```python
axis_angle_to_matrix(omega)  # [..., 3] -> [..., 3, 3]
apply_rotation(R, points)   # [B, 3, 3], [B, N, 3] -> [B, N, 3]
check_so3(R)                # finite / orthogonality / determinant
```

#### C. Manipulate → Experiment

| 子任务 | 时间 | 固定项与改变量 | 要记录 / 要回答 |
| --- | --- | --- | --- |
| 方向与逆变换 | 20m | 相同点；主动 Rz(π/2)；再用 Rᵀ 恢复 | 与手算误差、round-trip；写清两次操作的 frame |
| 非交换组合 | 15m | 同一个 e_x 与两次 90°；只交换 Rx / Rz 顺序 | 两个终点和矩阵差；说明右侧矩阵先作用 |
| 长度 / 反射 | 15m | 固定 8 个点；合法 R 对比 diag(-1,1,1) | 长度与点距；反射虽正交却 det=-1，必须拒绝 SO(3) 标签 |
| 零点 / 小角度 | 20m | ω=0、`[1e-8,0,0]`、一般旋转；相同 dtype | 输出误差与有限梯度；故意去掉零点保护，记录并恢复 |

梯度检查用加权矩阵元素的标量，例如 `sum(R * fixed_weights)`，不用 `||R||²` 这类合法旋转上恒定的量来要求非零梯度。检查正常 forward / backward 有限；记录梯度值，不规定每个元素都必须非零。

#### D. Analyze → Research Log

15m 整理 `results/d1_frames.md` 与 `notes/frames.md`；10m 写 Research / Fail Log；10m 闭卷解释“这个 R 把哪个 frame 变到哪个 frame”，并用手算说明为什么旋转顺序不能交换。把反射和小角度故障的复测证据留下。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week03/src/rotations.py` | 自写 skew / Rodrigues / 点旋转 / SO(3) 检查 |
| 实验 | `week03/experiments/frame_rotation_check.py` | 主例、batch、顺序、反射、小角度与 backward |
| 推导 | `week03/notes/frames.md` | frame 图、主动 / 被动、行 / 列、角度 / 单位 |
| 结果 | `week03/results/d1_frames.md` | 手算与实测、合法性、误差与受控故障复测 |
| 日志 | `week03/research_log.md`、`week03/fail_log.md` | Day 1 的问题、预测、观察、解释与下一步 |

### 7. 验收标准（Gate）

- [ ] 能闭卷写出 SO(3) 条件并排除正交反射。
- [ ] Rz(π/2) 点方向、逆变换与 Rx / Rz 两种顺序有手算及数值证据。
- [ ] batch 点变换符合共同约定，没有把点轴 / batch 轴混用。
- [ ] 自写 Rodrigues 在零、小角度、一般角度下输出和正常梯度有限。
- [ ] 有明确 frame 声明和一次小角度 / 方向错误的定位、修复、复测。

### 8. 受阻处理

先退回单点与显式 Rz(π/2)，按矩阵每一行手算；再恢复 Rodrigues 和 batch。反传出 NaN 时逐个检查系数和未选中分支。超时删去额外 frame 美化和理论扩展；本人环境不可用时先留手算，实跑 Gate 待补。

### 9. 下一步

复用已检查的 R 作为四种表示互转的统一比较对象；先核对 quaternion 顺序与 6D 行 / 列约定。

## Day 2：Axis-Angle / Quaternion / Matrix / 6D Playground｜3h45m

### 1. 当日目标

建立多表示互转 playground；自己实现 6D→R 并与 PyTorch3D 校验；通过符号、π 边界与退化输入实验理解“参数不同”不等于“旋转不同”。

### 2. 前置条件

Day 1 手算、Rodrigues 与 frame 约定可用。**执行建议**：在同一 `rotations.py` 增加 6D 主路径与工具互转接口；新增 `week03/experiments/rotation_playground.py`，先不训练旋转回归网络。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 多表示、6D 精读、库约定与数据流 | 表示表、论文判断、shape |
| 00:45–02:00 | 75m | Run | 互转入口、手写 6D、PyTorch3D 对照 | 可运行 playground 与对照输出 |
| 02:00–03:10 | 70m | Manipulate → Experiment | round-trip、q 符号、π 边界与退化输入 | 表示误差 / 旋转误差 / 故障记录 |
| 03:10–03:45 | 35m | Analyze → Research Log | 整理结果、日志与闭卷解释 | Day 2 证据与 Gate 状态 |

### 4. 核心资源

- [6D Rotation 原论文入口（原表资源）](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)：定位 continuity 定义、5D / 6D 构造与 rotation regression 实验；入口受阻时读 [作者论文 arXiv](https://arxiv.org/abs/1812.07035) 或 [项目页](https://zhouyisjtu.github.io/project_rotation/rotation.html)。
- [PyTorch3D Transforms](https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html) 与 [conversion 源码](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html)：只查本日六个转换接口与 quaternion / 6D 的约定。

论文精读约 20m、源码 trace 约 10m；不要求从头读完证明或安装作者训练仓库。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 精读**：在 `notes/rotation_representations.md` 写出 problem、旧表示的连续性限制、6D 的构造、作者支持结论的实验设置及适用边界。区分作者的拓扑 / 连续性贡献与本周使用它作为可解码网络输出的工程意义；不声称有六个自由度或任意任务一定更准。
2. **15m 表示推导**：axis-angle `ω=θu`；单位 quaternion `q=(cos(θ/2),u sin(θ/2))`；R 有 9 个存储元素但 3 个旋转自由度；6D 是两向量构造。说明 q 的符号等价、π 附近 axis-angle 的分支与参数距离问题。
3. **10m Trace**：检查库的 `axis_angle_to_matrix / matrix_to_axis_angle`、`quaternion_to_matrix / matrix_to_quaternion`、`matrix_to_rotation_6d / rotation_6d_to_matrix`。写清每一步末尾 shape；只在有效 R 上调用逆转换。

本周自写 6D 主路径固定：`a1=d6[..., :3]、a2=d6[..., 3:]`；归一化 a1 得 b1；去掉 a2 在 b1 上的投影再归一化得 b2；`b3=b1×b2`；把 b1/b2/b3 堆成**行**。记录零向量和近共线时投影范数可能过小，连续性构造不等于所有 R⁶ 输入都能稳定解码。

#### B. Run

1. **20m 互转入口**：用 R 作中间表示，接通 AA↔R、q↔R、6D↔R；工具路径注明库版本。先比 Day 1 自写 AA→R 与库输出，不复制库函数作为“自写”。
2. **25m 独立 6D**：关掉参考后实现 Gram–Schmidt；返回 `[...,3,3]`。检查 `||a1||` 与投影后 `||a2_perp||`，float64 的 toy 默认退化阈值 1e-8；退化输入明确报错或标 invalid，不能默默宣称是 SO(3)。
3. **20m 参考校验**：identity / 90° / 一般角度、64 个固定 seed 的非退化随机 6D，以及 `[2,4,5,6]` 的 batch 输入，分别对照 PyTorch3D 的 6D 解码；记录最大差、RᵀR、det、finite。合法输入才要求对照一致。
4. **10m Run / backward**：对一个远离退化的随机 6D 输入用固定加权标量反传；保存输入与梯度。另用 R→6D→R 回环，不要求任意原始 d6 经回环仍逐元素相等。

输出结果表建议字段为 `case_id, source_repr, target_repr, shape, max_R_error, orth_error, det_error, finite, valid`。有意退化的输入单列 valid=False；不能混进合法样本误差均值后把它隐藏。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 唯一改变量 / 对照 | 要记录 / 要回答 |
| --- | --- | --- | --- |
| 多表示回环 | 20m | 相同有效 R，经 AA / q / 6D 再解码 | 比较解码 R 与旋转后的同一点；参数可不唯一，R 应对齐 |
| 四元数符号 | 10m | 单位 q 与 -q，其他不变 | 参数 L2 可非零；解码 R / 点应一致；零 quaternion 明确拒绝 |
| π 边界扫角 | 20m | z 轴 θ 取 170°、175°、179°、181°、185°、190°，输入换算 radians | 逆 AA / 标准化 q 可能跳变，R 和前两行 6D 随此路径连续；相邻参数差与真实旋转差分开 |
| 退化 / 约定故障 | 20m | a1=0、a2 与 a1 共线、近共线；另把三行误堆成列 | 退化策略与条件；错误转置仍可能满足 SO(3)，参考 / 方向检查才能抓住 |

旋转距离可记录 `δ=acos(clamp((trace(RaᵀRb)-1)/2,-1,1))`（radians）；接近 0 时同时看矩阵最大差，避免只靠 acos 的数值精度。此处仅做评估，不用 acos 端点梯度证明训练稳定。近共线例记录扰动和误差敏感性；禁止把数值不稳解释为“6D 不成立”。

#### D. Analyze → Research Log

15m 整理 `results/d2_rotations.md / csv`；10m 日志；10m 闭卷写 6D→R 与四种 shape。说明 `6D→R→6D` 为什么未必还原原始两向量，以及本周扫角只是机制检查，不是 6D 相对 quaternion 的完整网络性能消融。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week03/src/rotations.py` | 自写 6D；多表示工具路径；退化输入处理 |
| 实验 | `week03/experiments/rotation_playground.py` | 互转 / 回环、参考校验、符号 / 边界 / 退化检查 |
| 推导 / 论文 | `week03/notes/rotation_representations.md` | 四种表示、6D 构造、源码约定、作者贡献与边界 |
| 结果 | `week03/results/d2_rotations.csv`、`d2_rotations.md` | 逐例误差、π 扫角数据、梯度与修复复测 |
| 日志 | `week03/research_log.md`、`week03/fail_log.md` | Day 2 的预测、实测、参考程度与下一步 |

### 7. 验收标准（Gate）

- [ ] AA / q / R / 6D 的互转和有效旋转回环实际运行，比较口径为旋转而非原参数唯一性。
- [ ] 6D→R 核心由本人实现；非退化 batch 对照 PyTorch3D 有数值证据。
- [ ] 能说明 quaternion `(w,x,y,z)`、q/-q 与 6D 前两行约定。
- [ ] π 边界与零 / 共线输入有实际记录；不把非法解码宣称为合法旋转。
- [ ] 能解释论文连续性问题、冗余维度与退化边界，不从 toy 推断普遍性能优势。

### 8. 受阻处理

先在 identity 的 `d6=(1,0,0,0,1,0)` 上检查，再用一个非对称 R 抓行 / 列错误。PyTorch3D 不可用时继续手算、合法性与独立点旋转检查，正式库对照仍标待补；可参考官方源码解释约定，但“照抄公式跑了”不代替原表的 PyTorch3D 校验。超时不加训练和更多旋转表示。

### 9. 下一步

把合法 R 用作每个关节的 local rotation；引入 parent 索引和骨长后再谈 global joint positions。

## Day 3：SE(3) + Kinematic Tree + Forward Kinematics｜3h45m

### 1. 当日目标

自己实现 SE(3) inverse / composition 与 batched tree FK；用手算、骨长和局部干预说明父旋转如何传播；生成可定位参数的 skeleton animation。

### 2. 前置条件

Day 1–2 的旋转与坐标约定可用；Day 2 的库校验如仍待补要单独登记，不让它阻止已具备的 FK 手算。**执行建议**：新增 `week03/src/kinematics.py`、`week03/experiments/fk_playground.py`。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | SE(3)、最小 tree、local / global 手算 | tree 表、递推与 frame 声明 |
| 00:45–02:00 | 75m | Run | SE(3)、FK、batched 输入与动画 | 可运行骨架路径与短动画 |
| 02:00–03:10 | 70m | Manipulate → Experiment | rest / 单关节 / root 干预与参考对照 | 位置、骨长、传播范围与修复 |
| 03:10–03:45 | 35m | Analyze → Research Log | 整理证据、日志与闭卷验收 | Day 3 结果与 Gate 状态 |

### 4. 核心资源

- [Modern Robotics · Book Resources](https://modernrobotics.northwestern.edu/nu-gm-book-resource/)：3.3.1 Homogeneous Transformation Matrices；Chapter 4 的 FK 只用于理解多次刚体变换组合，不要求照搬机器人 POE 接口。
- [smplx · batch_rigid_transform](https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py)：独立 FK 完成后再对照 parent-chain 递推；注意人体源码后面另做 rest-pose correction，不能和 joint 位置提取混为一件事。

阅读合计约 20m；先读自己画的 tree 和 Day 1 frame 声明，再用源码排错。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 阅读与推导**：`T=[R,t;0,1]`、`T⁻¹=[Rᵀ,-Rᵀt;0,1]`。手算“先旋转后平移”与“先平移后旋转”，说明直接把 t 取负不一定是正确 inverse。
2. **15m 定义 tree**：使用下表的最小双分支骨架；root index=0、parent=-1，输入按父在子前的拓扑顺序排列。offset 在父 frame 中表达，静态不随帧变化。
3. **10m Trace**：root `P0=t_root、Rg0=Rlocal0`；非 root `Pj=Pp+Rgp offsetj、Rgj=Rgp Rlocalj`。画出“自己的 rotation 改变子节点位置，但不改变自身 joint center”的路径。

| index | name | parent | offset（父 frame、任意单位） |
| --- | --- | --- | --- |
| 0 | root | -1 | `(0,0,0)` |
| 1 | left_shoulder | 0 | `(-0.5,0,0)` |
| 2 | left_elbow | 1 | `(-0.7,0,0)` |
| 3 | left_wrist | 2 | `(-0.6,0,0)` |
| 4 | right_shoulder | 0 | `(0.5,0,0)` |
| 5 | right_elbow | 4 | `(0.7,0,0)` |
| 6 | right_wrist | 5 | `(0.6,0,0)` |

这是一棵自定义示意骨架，不采用 SMPL / Human3.6M 的 joint 编号。共同接口为 `R_local[B,T,7,3,3]、root_trans[B,T,3] → R_global / P_global`；不从 joint 坐标直接推回唯一局部朝向。

#### B. Run

1. **15m SE(3)**：写构造、inverse、点变换与 composition；用一般旋转 + 非零 t 检查 T 与 T⁻¹ round-trip，不只检查纯旋转或零平移。
2. **25m FK**：自写 `forward_kinematics(R_local, offsets, parents, root_trans)`。先 B=T=1，再支持 `[2,4,7,3,3]`；检查一个 root、parent 范围、拓扑顺序、输入合法旋转和无 cycle。非法 tree 明确报错，不用 root=-1 索引误取最后一个关节。
3. **15m 参考路径**：独立构建 `T_parent,j=[Rlocalj,offsetj;0,1]`，root 为 `[Rlocal0,t_root;0,1]`，沿 parent 连乘；取每个 T 的平移和旋转，与递推 FK 对齐。参考不用同一个 FK helper，避免同错同对。
4. **20m 动画**：自造 T=60、fps=30 的序列，`s_k=k/59`；仅 left_elbow z 角为 `(π/4)sin(2πs_k)`，其他旋转 identity、root_trans=0。保存角度 / R / P 和带 frame、joint 名、坐标轴 / 单位的 GIF，所有帧使用固定等比例范围。

建议导出 `results/d3_fk_sequence.npz` 与 `skeleton_fk.gif`。实际入口 / 命令由本人写入报告；动画必须来自保存的同一组参数，不能用另一个随手画的骨架代替。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 组别 / 唯一变化 | 数值判据与解释 |
| --- | --- | --- | --- |
| Rest 与手算 | 15m | 全部 identity、root=0；再只给 left_elbow Rz(π/2) | rest wrist=(-1.8,0,0)；旋转后 elbow=(-1.2,0,0)、wrist=(-1.2,-0.6,0)；right branch 不变 |
| Root 刚体干预 | 15m | 固定所有 local body rotations；root 旋转 Q 或平移 Δ | `P'_j=Q(P_j-t_root)+t_root+Δ`，同帧骨长保持；只平移时所有差相同 |
| Local / global 故障 | 20m | 先给 left_shoulder Rx(π/2)；再给 left_elbow Rz(π/2)，错误路径把 local 直接当 global | 正确 wrist=(-1.2,0,-0.6)；错误传播方向不同；两个旋转不交换，不能只用共轴测试 |
| Batch / 骨长 / 复测 | 20m | 相同参数；递推对照矩阵连乘；两个 batch 用不同根平移 | 位置 / R 最大差、每条边的骨长等于 offset 的范数；恢复后重跑动画参数 |

rest tree 的 left_elbow 自转不改变 elbow 位置，改变 wrist；left_shoulder 自转影响其后代，不影响 right branch。root rotation 则影响两条分支。上述传播范围和骨长是必要检查，但不能单靠它们排除所有朝向错误，必须结合非共轴手算与独立参考。

#### D. Analyze → Research Log

15m 整理 `results/d3_fk.md` 与 `notes/fk.md`；10m 日志；10m 闭卷画 parent 链，计算一个没背过的 root 旋转 + elbow 旋转例子。解释 forward kinematics 由参数生成位置，inverse kinematics 则是另一类问题，单靠 positions 不足以恢复所有 bone twist。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week03/src/kinematics.py` | SE(3) inverse / composition、独立 batched FK 与非法 tree 检查 |
| 实验 | `week03/experiments/fk_playground.py` | 手算、独立连乘、骨长、local / root 干预与 batch |
| 推导 | `week03/notes/fk.md` | tree 表、offset / frame、递推公式、局部 / 全局区别 |
| 结果 / 参数 | `week03/results/d3_fk.md`、`d3_fk_sequence.npz` | 位置、误差、骨长和实际动画参数 |
| 动画 / 日志 | `week03/results/skeleton_fk.gif`；Research / Fail Log | 固定尺度、轴 / 帧标注；Day 3 预测、复测与下一步 |

### 7. 验收标准（Gate）

- [ ] SE(3) inverse / composition 对一般 R、非零 t 有 round-trip 证据。
- [ ] 自写 FK 在 B/T batch 下与独立齐次连乘参考对齐。
- [ ] rest、elbow 单旋转、root 干预和非共轴传播有手算 / 数值证据；骨长保持。
- [ ] skeleton animation 可打开，参数文件、joint 顺序、fps 与坐标标注一致。
- [ ] 能闭卷解释 local / global、父子传播、位置与朝向区别；错误 tree 或 local-as-global 故障已定位并复测。

### 8. 受阻处理

先缩成 root→elbow→wrist 三节点，用 identity、单个 Rz(π/2) 手算；再恢复双分支和非共轴旋转。动画受阻先保存 P 与三个关键帧，动画 Gate 继续待补。正确性未过时优先修 FK，删去绘图美化；不改用库 FK 来替代独立交付。

### 9. 下一步

保留 tree / FK 输出，下一日增加 mesh vertices 与 skinning weights；先解释为什么对每个顶点直接乘一个 joint transform 不足以描述关节附近的表面。

## Day 4：LBS + SMPL / SMPL-X Trace + Parameter Control｜4h

### 1. 当日目标

用两骨骼 toy 实现并解释 LBS 的 rest-pose correction；追踪 SMPL 的 shape / pose / FK / skinning；用 `smplx` 单独控制 betas、body_pose、global_orient、transl，留下数值与可视化证据。

### 2. 前置条件

Day 3 FK 与坐标定义可用；检查已有授权人体模型能否加载，记录类型 / gender / Kβ。**执行建议**：新增 `week03/src/skinning.py`、`body_model.py` 与 `week03/experiments/smpl_parameter_probe.py`；不重写完整 SMPL。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | LBS、SMPL 核心数据流与模型访问检查 | 公式、源码 shape、资源状态 |
| 00:45–02:00 | 75m | Run | 两骨骼 LBS、人体模型零参数调用 | toy rest / one-hot；人体 joints / mesh |
| 02:00–03:20 | 80m | Manipulate → Experiment | 四组参数的单变量对照与机制检查 | 预测 / 实测表、参数与图 |
| 03:20–04:00 | 40m | Analyze → Research Log | 整理数值、日志与口述验收 | Day 4 证据、限制与 Gate |

### 4. 核心资源

- [SMPL 官方项目](https://smpl.is.tue.mpg.de/) 与 [SMPL-X 官方项目](https://smpl-x.is.tue.mpg.de/)：定位 template、shape / pose 控制与 SMPL-X 的身体 / 手 / 面部范围；不把两个模型视为同一种参数接口。
- [smplx · body_models.py](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)：选 `SMPL.forward`，核对 full_pose、LBS、extra joints、transl；SMPL-X 的接口比较放到 Day 5。
- [smplx · lbs.py](https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py)：只追 `blend_shapes / vertices2joints / lbs / batch_rigid_transform`。

本日阅读 / trace 限时约 30m；按参数流定位，不从文件第一行读到最后一行。官方入口与关键函数已于 2026-10-13 只读核对，执行时保存实际版本。

### 5. 执行步骤

#### A. Understand → Trace

1. **15m LBS 推导**：对 rest vertices `v_i` 与非负归一化 weights `w_ij`，写 `v'_i=Σ_j w_ij A_j [v_i;1]`、`A_j=G_j(θ)(G_j^rest)⁻¹`。解释先撤销 rest joint placement，再施加 posed placement；LBS 加权后的矩阵一般不是刚体变换。
2. **15m SMPL Trace**：记录 `v_shaped=template+B_S(β)`，`J(β)=J_regressor·v_shaped`，`v_posed=v_shaped+B_P(θ)`，FK / rest correction、weights、vertices / posed joints、最后加 transl。pose blend 的 feature 使用非 root 旋转相对 identity 的偏差；不把它等同于 FK 或 root translation。
3. **15m 模型 / shape**：检查实际 `model_type、gender、num_betas、parents、faces、weights、joint_mapper`；调用默认与显式参数的区别写进 `environment.md`。用下表对齐基础 SMPL 路径，按实载模型修正数量。

| 环节 | 基础 SMPL 典型 shape | 要核对的语义 |
| --- | --- | --- |
| betas → shape blend | `[B,Kβ] → [B,V,3]` | shape 改变 rest 表面，也改变回归 rest joints |
| global + body pose | `[B,3]+[B,69] → [B,72]` | root + 23 个 local body rotations，不是位置向量 |
| rotations / pose feature | `[B,24,3,3]`；`[B,207]` | 207=23×9；root 不进入基础 SMPL pose corrective feature |
| FK / corrected transforms | `[B,24,3]`；`[B,24,4,4]` | posed joints 与顶点用 corrected transforms 是不同产物 |
| weights / skinning | `[V,24] → [B,V,4,4] → [B,V,3]` | weights 是模板 skinning weights，不是 Week 2 attention |
| output / transl | `[B,V,3]` 与 `[B,J_out,3]` | 顶点与 joints 都加 translation；extra keypoints 来源单独记录 |

SMPL 的 blend shapes / regressor / weights 是模型文件中的已学习参数；本周仅做推理与机制理解。自写两骨骼 LBS 是解释性 toy，没有人体 shape / pose correctives，不能叫完整 SMPL 复现。

#### B. Run

1. **20m Toy LBS**：两 joints 的 rest centers 为 `(0,0,0)`、`(1,0,0)`，parent=[-1,0]；三 vertices 为 `(0.5,0.2,0)`、`(1,0.2,0)`、`(1.5,0.2,0)`；weights 分别 `[1,0]、[0.5,0.5]、[0,1]`。本人写 corrected transforms、加权和与 homogeneous point；检查 weights≥0、每行和=1。
2. **15m Toy 校验**：θ=0 时输出必须等于 rest vertices；将第二 joint 绕 z 旋转 π/2，验证 one-hot vertex 等于对应单骨骼刚体结果，混合 vertex 等于两结果的加权和。临时去掉 rest inverse 看零姿态是否也发生错误位移，恢复后复测。
3. **20m 人体调用**：在已可用 SMPL 文件上创建 B=1、Kβ 按加载配置的模型，显式 β=0、body_pose=0、global_orient=0、transl=0，`return_verts=True`；保存输出、faces、运动学 joint 索引与实际 shape。只跑 forward，不下载网络 checkpoint。
4. **20m 渲染 / trace**：画 zero pose 的 3D skeleton 和三角 mesh；可用 matplotlib 的三角面绘图和 GIF，不引入完整 GPU renderer。导出绘图对应参数 / vertices，并把函数与 shape 写入 `notes/smpl_trace.md`。

检查 joints 时使用与 parents 对齐的运动学子集。`output.joints` 可能经过 vertex selector / mapper；数量大于 24 时先追源码和索引，不能给所有输出 keypoints 随意套 24-joint parent 表。

#### C. Manipulate → Experiment

**最小实验：四组人体参数的单变量控制（60m）**

同一 model、gender、dtype、joint convention；每组从独立零参数副本开始，禁止累积上组改动。先写预测，再用相同视角与固定轴范围画图。

| 组别 | 唯一改变 | 数值检查 / 输出 | 需要解释 |
| --- | --- | --- | --- |
| A reference | 四组参数均为 0 | 保存 V0、J0、faces、参数 | 比较基线是实际模型输出，不把 canonical root 强制设为原点 |
| B transl | `(0.2,-0.1,0.3)`，单位按模型核实 | `(VB−V0)−Δ` 与 `(JB−J0)−Δ` 最大残差；root-centered joints 差 | 整体平移；root trajectory 与 transl 数值可能差一个 rest pelvis offset |
| C global_orient | z 轴 π/4，其他为 0 | 距离 / 骨长；围绕 baseline pelvis c 的 `c+R(V0−c)` 残差 | 身体整体朝向；旋转中心需按模型实现确认，不默认 world origin |
| D body_pose | 核对基础 SMPL 的 left_elbow joint index 后，局部 z 角 π/4 | 该 joint、后代、另一分支位移；受影响表面的图与数值 | pose 经 FK 与 pose corrective / skinning 影响关节和表面；不能要求只有后代 vertices 改变 |
| E betas | β0 分别取 -1、+1，其余 shape 为 0 | mean / max vertex displacement、rest 骨长和 bounding box；两组 root-centered joints | shape 改变体型与 rest skeleton；不预设 β0 恰好等于“身高” |

基础 SMPL 该 elbow 的 `body_pose` slice 为 `(joint_index−1)*3 : joint_index*3`，因为 root 在 `global_orient`；以模型规范核对名字 / index，再写 slice，不套 toy index。C 组的 c 从模型的 root joint 定义取得；如果公式不对齐先查 pivot / 输出 mapper，不改图来遮掩。

**正确性、受控故障与复测（20m）**

- 10m 用相同数据复测 toy 的 rest / one-hot 与 B 组 translation；查 weights 行和、rest correction、transl 是否被重复添加。对 float32 记录实际残差，阈值用共同口径。
- 10m 复查 D / E 的参数副本和 slice，保存至少一条真实或受控的“pose index / 默认参数 / 双平移”故障及修复结果。区分 kinematic joint 的传播与 mesh correctives 的影响，不凭视觉猜根因。

建议 `results/d4_params.csv` 记录 `condition, parameter_name, parameter_value, model_type, joint_subset, vertex_displacement, root_centered_joint_difference, rigid_residual`；只在有意义的组计算 rigid residual，其他填不适用而非伪造 0。

#### D. Analyze → Research Log

20m 整理 `results/d4_smpl.md`、原始参数 / 输出与 `smpl_parameter_grid.png`；10m 日志；10m 闭卷讲“β / θ → rest skeleton / pose corrective → FK → LBS → mesh”。解释 SMPL 相对纯 skeleton 保留了哪些表面信息，LBS 的关节附近形变限制与 pose correctives 的作用。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week03/src/skinning.py`、`body_model.py` | 自写 toy LBS；显式人体模型调用与输出索引 |
| 实验 | `week03/experiments/smpl_parameter_probe.py` | rest / one-hot 与四组参数独立对照、复测 |
| Trace | `week03/notes/smpl_trace.md` | 函数位置、主要 shape、模型参数与 learned components |
| 结果 / 参数 | `week03/results/d4_params.csv`、`d4_smpl.md`、`d4_body_outputs.npz` | 原始四组输入、joints / vertices、数值与限制 |
| 图 / 日志 | `week03/results/smpl_parameter_grid.png`；Research / Fail Log | 同视角 / 尺度、参数标签；Day 4 预测、观察与待补 |

### 7. 验收标准（Gate）

- [ ] toy LBS 由本人实现；rest identity、one-hot、混合权重与去掉 rest correction 故障有数值证据。
- [ ] 实际 `smplx` forward 产生可定位的 joints / vertices / faces；模型文件、类型、shape 数量有记录。
- [ ] betas / body_pose / global_orient / transl 四组控制均实际运行，参数和结果分别保存。
- [ ] translation 残差、root pivot、运动学 joint 子集与 pose slice 已核对；结论不只靠图片。
- [ ] 能解释 shape、pose、FK、LBS 与额外 keypoints 的关系，不把 toy LBS 当完整人体模型。

### 8. 受阻处理

模型文件缺失时先完成 toy LBS、源码 trace、参数 / shape 预测，并登记访问状态及下一步；原表交付 3 和人体 mesh 部分保持待补。AMASS 的 synthetic fallback 同样依赖人体模型，不能解决模型缺失。若只有 SMPL-X 已可用，可用其明确接口完成四参数控制，记录 body_pose=63、hands / face 默认设置；不要伪装成 SMPL=69 的结果。渲染受阻优先检查原始 vertices / faces，再缩减画面；图的 Gate 待补。

### 9. 下一步

复用相同人体模型与渲染接口，准备一个短动作片段；先判断 AMASS 的模型族、pose 分段与 frame / fps，再决定真实数据或自造序列分支。

## Day 5：AMASS / Synthetic SMPL Motion + Minimal Experiment + Final Gate｜3h45m

### 1. 当日目标

把动作参数接回人体模型，输出 root trajectory、3D skeleton 与 mesh animation；用 root / body 控制实验检查它们的语义；比较 SMPL / SMPL-X 接口并完成全周验收和 Week 4 交接。

### 2. 前置条件

Day 4 有可用人体模型与保存输出的接口；Day 3 的动画方法可复用。**执行建议**：新增 `week03/src/motion_sequence.py`、`week03/experiments/motion_visualize.py`；本日不引入 motion generation 模型或完整 AMASS 预处理 pipeline。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:40 | 40m | Understand → Trace | SMPL-X 对比、AMASS schema 或自造片段、实验预测 | 参数映射、时间轴、frame、假设 |
| 00:40–01:55 | 75m | Run | 短序列解码与 root / skeleton / mesh 可视化 | 参数、输出、轨迹 CSV 与两个动画 |
| 01:55–02:40 | 45m | Manipulate → Experiment | 同序列 root / body 干预和复测 | 数值残差、变化范围与结论限制 |
| 02:40–03:45 | 65m | Analyze → Research Log | 最终验收 45m；日志 / 周总结 20m | 最终 Gate、周总结、Week 4 接口 |

### 4. 核心资源

- [AMASS 官方项目](https://amass.is.tue.mpg.de/) 与 [官方仓库](https://github.com/nghorbani/amass)：按需获取一个动作；本周不复制完整训练数据制备流程。
- [AMASS 官方可视化 notebook](https://github.com/nghorbani/amass/blob/master/notebooks/01-AMASS_Visualization.ipynb)：核对 `poses / trans / betas / gender / mocap_framerate` 与 body / hand 分段；读数据流，不要求运行其旧整套 renderer 环境。
- [SMPL-X 官方项目](https://smpl-x.is.tue.mpg.de/) 与 [smplx 模型源码](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)：查 SMPL / SMPLH / SMPLX 的接口与 output joints 区别。

阅读 / trace 限时约 25m；优先使用本周已写的 FK、模型调用与图形入口。

### 5. 执行步骤

#### A. Understand → Trace

1. **10m 模型比较**：在 `notes/amass_and_models.md` 写 SMPL 主要建模 body，SMPL-H 扩展 hands，SMPL-X 扩展 body / hands / face；核对 body_pose 长度、hands PCA 开关、expression、jaw / eye 参数。记录 verts / joints 数量来自实际模型或官方定义，不用同一个 pose 向量直接试塞。
2. **15m 片段 / 映射**：按下述 A / B 选一条可执行分支，记录模型、时间轴、单位、frame、joint subset 与序列来源。真实数据不兼容时先记具体阻塞，不无限尝试完整资产下载。
3. **15m 假设 / 冻结变量**：写“去掉 transl，是否只移除整体平移？”以及“冻结 body_pose，是否保留 root 路径与朝向变化？”；冻结 β、model、原始 root pose、fps、frame indices 与渲染尺度，定义待测指标。

**A：AMASS 已可用且模型兼容**

- 只取一个动作的至多 60 帧，保存原始 frame indices。先列 keys、shape、gender 与 fps；按 `time=original_frame_index / source_fps` 保留时间轴。若降采样，显式记录 stride / 有效 fps，不能取隔帧后仍按原 fps 播放。
- 官方示例的 `poses[:, :3]` 是 root，`poses[:, 3:66]` 是 21 个 body joints，余下是 hands；常见 SMPL-H 轴角布局为 156=3+63+90。这是已核对的官方示例布局，**执行时仍以具体文件 / 模型 schema 为准**，不是所有未来下载格式的保证。
- 使用匹配的模型族 / gender；若 smplx 的 SMPL-H 使用 full hand axis-angle，确认 `use_pca=False`，并核对 pose mean 设置。betas 的数量与模型支持范围一致；必要截断 / 固定 mean / 忽略 DMPLs 必须记录，输出称“该参数子集的重建”，不称完整官方动态表面复现。
- 156 维 SMPL-H pose 不直接等于 72 维 SMPL，也不等于 SMPL-X full pose。只有 SMPL 文件时，不把截取前 72 维当作正确映射；需要明确 joint 对应与缺失值，否则改用 B 分支。真实 AMASS 数据与模型文件的可用性分别登记。

**B：AMASS 暂不可用，使用原表许可的自造 SMPL motion sequence**

- 使用 Day 4 已可用的同一模型，T=60、fps=30、`s_k=k/59`；β 在全序列固定为 0，其余参数从零开始，保存模型族对应的完整 pose shape。
- `transl_k=(0.3*s_k,0,0)`；`global_orient_k=(0,0,(π/6)*sin(2πs_k))`；按 Day 4 已核对的 left_elbow slice 设置局部 z 角 `(π/4)*sin(2πs_k)`，其他局部 body pose 为 0。角度 radians，translation 单位按模型核实。
- 所有参数显式生成、同一时间轴；标注“自造动作 / 参数控制，无 mocap ground truth”。它满足本周表示与可视化练习，不支持 AMASS 分布、真实运动质量或泛化结论。

如果 A 的数据或对应模型暂时不可用，走 B 并记录原因；这不修改原表。B 仍需可用人体模型。SMPL-X only 的情形按其 body / hands / face 接口明确生成，不把输出标成基础 SMPL。

#### B. Run

1. **20m 解码序列**：加载固定参数，逐帧或小 batch 调用同一模型；保存 `params、joints、vertices、faces、frame_indices、time、fps、model_type、joint_names / parents`。检查有限值、T 与轴对应；不要把 frame 轴误当 joint。
2. **15m Root trajectory**：同时保存原始 `transl[t]` 与经模型输出的 `pelvis_position[t]`，root 索引来自已核对定义；另存 `P_root[t]−P_root[0]`。绘制带时间 / 单位的轨迹，避免把 pelvis offset 或首帧对齐隐藏在图里。
3. **20m Skeleton animation**：沿运动学 parents 连线，显示当前 frame 与时间；坐标范围由整段固定得到。可另存 root-centered 图作比较，但主动画保留序列外部 frame，不能每帧居中后声称展示了整体 root motion。
4. **20m Mesh animation**：用同一 vertices / faces、同一时间轴与变换绘制短 GIF，保存 `motion_mesh.gif`；抽查首 / 中 / 尾三个帧的 skeleton 与 mesh 对齐。CPU 慢时减少显示帧 / 绘图面，数值与原参数保留完整，记录可视化抽样方式。

`results/root_trajectory.csv` 至少含 `source, frame_index, time_s, transl_x/y/z, pelvis_x/y/z, relative_root_x/y/z`。`motion_outputs.npz` 与图使用相同模型 / 参数快照；gif 可打开、帧数与播放时长符合记录，不只检查文件名存在。

#### C. Manipulate → Experiment

**最小实验：同一动作的 root / articulation 分解（25m）**

用 A 或 B 的同一片段，以实际实现的模型重跑以下组；所有组不重新拟合 β、不修改 pose convention。

| 组别 | 唯一变化 | 固定项 | 指标与要回答的问题 |
| --- | --- | --- | --- |
| A original | 原 motion 参数 | 同一 model、β、fps / frame indices | 作为 joints / vertices / trajectory 基线 |
| B no translation | 所有 transl=0 | global_orient、body / hands / face pose 均不变 | `VA−VB≈transl`、`JA−JB≈transl`；root-centered joints 应对齐 |
| C fixed articulation | body_pose 全帧固定为首帧；其他 articulation 若有也固定首帧 | root orientation、transl、β、model 保留 | root trajectory 保留；去掉 root rigid motion 后 local body motion 应消失；不要只看整体动画 |

去 root rigid motion 的检查可用 `Q_t[j]=R_root(t)ᵀ(P_t[j]−P_root(t))`。C 组在固定 shape / articulation 下，Q_t 应在容差内不随时间变化；A 组是否变化由实际动作决定，站立片段可能变化很小。B / C 的结果验证参数与模型路径，不支持人体运动建模方法的性能或 novelty。

**复测与分析（20m）**

1. **10m 新进程复测**：从已保存参数 / 配置重新加载，重算 A / B，比较同帧 joints / vertices、translation 残差与 root-centered 输出；不换模型、dtype 或 joint mapper。
2. **10m 故障定位**：若 root 轨迹不合理，先查重复加 transl、axis swap、每帧居中、fps / stride 与 pose split；按数值定位一个真实或受控错误后恢复。若 C 组 root 变化，检查 root index 与 body articulation slice 是否被混改。

保存 `results/d5_motion_experiment.md`：假设、模型 / 数据、三组唯一变化、实测误差、解释与最大不确定性。真实 AMASS 分支明确忽略项；synthetic 分支明确不存在 ground-truth 动作评测，不能编造 benchmark 数字。

#### D. Analyze → Research Log

**整体验收（45m）**

1. **15m 材料检查**：原表四项交付逐一映射到代码 / 数值 / 图 / 日志；检查动画确实可打开，模型 / frame / fps / 单位一致，预期与实测分栏。
2. **20m 重跑关键证据**：Rodrigues 零点与手算、手写 6D 的库对照、FK 独立连乘与骨长、LBS rest / one-hot、SMPL transl、motion A/B 复测。复用已有入口，不另写一套“只供验收”的实现。
3. **10m 闭卷验收**：给 `[B,T,J]=[2,3,7]`，写四种 rotation shape、FK 输入 / 输出与 parent 递推；讲清 β / body_pose / global_orient / transl 到 joints / mesh。再解释有 joints 为什么未必有唯一 rotations。

**日志与周总结（20m）**

15m 写 `week03/weekly_review.md`，再用 5m 更新日志与最终 Gate：

- 本周哪三条路径能不看参考独立写出？对应哪份代码和数值证据？
- 最容易混淆的 frame / quaternion / 6D / pose index 是什么？哪个实验抓住了错误？
- 四组人体参数分别改变什么？什么现象由实测支持，什么仍只是预测？
- AMASS 或 synthetic 分支完成了哪些表示产物？数据 / 模型不兼容或未建模 DMPLs 留下什么限制？
- 最大不确定性是什么？下一次最小验证需改变哪个变量、观察哪个指标？
- 哪项 Gate 待补，缺环境、模型、实现还是解释？下一步和通过证据是什么？
- Week 4 要接的 joints / vertices 位于哪个 frame？为什么还需要 camera projection，而 SMPL 输出本身不等于从 RGB 估计人体？

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 / 实验 | `week03/src/motion_sequence.py`、`week03/experiments/motion_visualize.py` | AMASS 或自造序列、解码 / 导出、A/B/C 与复测 |
| 数据定义 | `week03/notes/amass_and_models.md` | 模型差异、pose 分段、β / hand 设置、fps / frame / 来源 |
| 原始参数 / 输出 | `week03/results/motion_params.npz`、`motion_outputs.npz` | 可复现参数、joints / vertices / faces、时间轴与模型信息 |
| 轨迹 | `week03/results/root_trajectory.csv`、`root_trajectory.png` | transl、pelvis、首帧相对路径分别保存，轴与单位明确 |
| 动画 | `week03/results/motion_skeleton.gif`、`motion_mesh.gif` | 同一片段 / 模型、fixed axes、时间 / 帧、skeleton / mesh 对齐 |
| 报告 / 日志 | `week03/results/d5_motion_experiment.md`；`weekly_review.md`、Research / Fail Log | 真实数值、边界、最终 Gate、待补与 Week 4 第一步 |

### 7. 验收标准（Gate）

- [ ] AMASS 兼容小片段或自造人体序列实际解码；来源 / 模型 / pose split / 时间轴记录完整。
- [ ] root trajectory、3D skeleton animation、mesh animation 三项可打开并对应同一输入。
- [ ] A/B/C 参数控制有同口径实测与复测；transl / pelvis / root-centered 定义分开。
- [ ] 能区分 SMPL / SMPL-H / SMPL-X 与 AMASS，说明 body / hands / face 参数与 pose 不可直接混用。
- [ ] 能闭卷区分 joints / rotations / parameters / mesh，解释 local / global、FK、LBS。
- [ ] 已更新最终 Gate；没有 AMASS 不自动判失败，没有模型 / mesh 不自动判完整通过。

### 8. 受阻处理

真实数据缺失、格式不兼容或资产获取超时，走明确标注的 synthetic 分支。若人体模型仍不可用，保留 Day 3 toy animation 与 Day 4 trace，但人体参数 / mesh / 完整最终 Gate 待补。绘图慢先用 12–20 帧预览，再在预算内完成约定短动画；保存参数和实际帧数，不把三个静态帧称作动画。删去第二模型实跑、美化与更大数据，保留三项可视化和数值证据。

### 9. 下一步

根据最终 Gate 优先补数值 / 模型 / frame 问题。进入 Week 4 时复用已经确认的模型参数与 joints / vertices，把它们接入最小 camera projection / reprojection toy；论文与 Metrics Protocol I 按总计划展开，不因本周访问问题重排后续主线。

## Fail Log / Research Log 模板

**执行建议**：执行时自行新建 `week03/research_log.md` 与 `week03/fail_log.md`。沿用 Week 2 的字段，新增源 / 目标 frame、旋转 / 单位约定、模型族与参数 split；所有示例字段待本人填实测。

### Research Log

```markdown
## YYYY-MM-DD / Week 03 Day N / 主题
状态：未开始 / 进行中 / PASS / 待补
计划时间 / 实际时间：
环境、seed、代码版本或文件快照：
Question：本日要回答的具体问题？
Hypothesis：改变什么，预期出现什么，为什么？
Understand / Trace：概念、函数位置、关键 shape 与参数路径。
Conventions：源/目标 frame、主动/被动、行/列、radians、单位、quaternion/6D。
Model / Data：model_type、gender、Kβ、pose split、joint subset、fps/frame indices。
Run：实际命令、配置、输入、模型与结果路径。
Manipulate / Experiment：控制组、实验组、唯一改变的变量。
Prediction / Observation：预测与实测分开；误差/位置/shape/动画证据。
Analyze：结论、机制、替代解释、适用条件与限制。
Implementation provenance：自己写/工具调用/参考源码的边界。
One thing I can now explain：不看资料能讲清楚的一件事。
Bugs / confusion：关联 Fail Log ID。
Gate：逐项状态、未通过项的补做动作与证据。
Tomorrow first step：一个可直接执行的动作。
```

### Fail Log

```markdown
## W03-DN-F01 / 简短标题
时间 / 状态：Open / Explained / Fixed / Retested
Context：环境、版本、输入、frame、shape、单位、模型与复现命令。
Expected：预期点位置、合法性、传播范围或输出行为。
Observed：真实输出、报错、误差或梯度；附文件路径。
Minimal reproduction：缩到哪几个点/帧/joints/vertices与哪个旋转？
Hypotheses：候选原因与排除证据。
Root cause：有证据支持的原因，未确认则写待验证。
Fix / explanation：改动或机制解释。
Retest：同输入/参数/模型复测的命令、结果与判据。
Lesson：下次在哪一环加什么方向/frame/index/finite检查。
```

**常见失败点 / Debug 提示（执行建议）**

| 症状 | 优先假设 | 最小验证 / 修复方向 |
| --- | --- | --- |
| 旋转方向反了但 R 合法 | 主动 / 被动混用，行存储忘 transpose | 用 e_x 和 Rz(π/2) 手算，写出源 / 目标 frame |
| 零旋转 NaN 或 backward 非有限 | θ=0 除法、未选中分支仍有 0/0 | 单个 ω=0，逐查系数；稳定 sinc / Taylor 与 finite 复测 |
| quaternion 回环参数不同 | q/-q 等价、标准化符号、顺序混用 | 比较 R 与点；核对 `(w,x,y,z)`，零 q 拒绝 |
| 6D 得到正交 R 却参考不对齐 | 行 / 列 stack 错，输入取法不一致 | 非对称 R 的前两行编码，逐行对照 PyTorch3D |
| 6D 近共线输出敏感 | a1 或投影后 a2 的范数小 | 记录两个范数、输入扰动和 valid；不要只调容差 |
| FK 骨长正常但腕方向错 | local 当 global、矩阵顺序错 | Rx parent + Rz child 的非共轴手算与齐次连乘 |
| root / branch 位置错 | parent=-1 被索引为最后 joint、offset frame 不一致 | 三节点 / 双分支 rest 检查；root 单独处理 |
| LBS 零姿态也偏移 | 缺 rest inverse、weights 不归一 | identity / one-hot 顶点检查，核对 corrected transforms |
| SMPL pose 修改了错误关节 | body_pose 不含 root、模型族维度不同 | 核对 joint index、slice 与 SMPL=69 / SMPLH/X=63 |
| 人体整体平移两次 | wrapper / renderer 重复加 transl | 同输入 transl=0 / Δ，检查逐点差与 Δ |
| root-centering 后 betas “没用” | shape 作用不只在 root，观察口径太粗 | rest 骨长、root-centered joints 与 surface displacement |
| output.joints 无法接 parents | extra keypoints / mapper 不同 | 追 joint selector，明确运动学 subset / names |
| AMASS 播放过快 / pose 不对 | fps / stride 错、SMPL-H 参数错塞 SMPL/X | 原始 frame/time 表、模型族与 pose split 检查 |
| animation 看不出 root motion | 每帧居中、自动缩放、坐标改轴不一致 | 固定范围，保存 raw transl / pelvis；对三类输出统一显示变换 |

## Week 3 最终 Gate

下面以原表四项交付为主，将细化证据映射回 `10周计划!D12`。**本周状态初始均为待完成**；计划文件通过结构检查不代表学习 / 实验通过。

| 项目 | 对应原计划 | 通过证据 | 当前状态 |
| --- | --- | --- | --- |
| Rotation playground | 交付 1 | AA / q / R / 6D 互转、有效旋转回环、边界与输入约定可解释 | 待完成 |
| 自写 6D→R + PyTorch3D | 交付 1 | 本人实现；非退化 batched 数值对照；合法性、方向与退化策略 | 待完成 |
| Minimal tree / FK | 交付 2 | 本人 `forward_kinematics()`；手算、独立参考、骨长、父子 / root 干预 | 待完成 |
| Skeleton animation | 交付 2 | 自定义 tree 动画可打开；参数 / frame / fps / joint 顺序一致 | 待完成 |
| smplx 参数控制 | 交付 3 | 四组参数实际运行；模型与 raw joints / vertices、数值 / 图可定位 | 待完成 |
| Motion 与三类可视化 | 交付 4 | 兼容 AMASS 或原表许可的自造人体序列；root trajectory、skeleton 与 mesh animation 齐全 | 待完成 |
| Local / global、FK / LBS | 原 Gate / Focus | 闭卷解释；SE(3)、rest correction、one-hot 与非共轴传播证据 | 待完成 |
| 表示与模型族区别 | 原 Gate / Focus | joints / rotations / parameters / mesh，SMPL / SMPL-X / AMASS 及 frame 定义可解释 | 待完成 |
| Minimal experiment / Research habit | 执行建议 | 四参数 / root-articulation 对照、五日日志、真实失败 / 复测、周总结 | 待完成 |

本人独立解释与工程证据都通过才记完整 `PASS`。AMASS 暂不可用但按原表完成自造人体 motion 及三项可视化，可以通过交付 4；只有 toy skeleton、源码阅读、模型包 import，不能替代实际人体参数控制或 mesh。PyTorch3D 校验缺失时保留其待补，不凭自写合法性检查勾选原表交付 1 完整通过。

参数控制结果不符合直觉时先定位 frame、pose split、模型默认值与输出定义；不要求 β0 必须使人体变高，也不强求真实 motion 的某项变化幅度。所有未过项写清缺口、最小补做动作与证据。超出 19h 的必做待补先登记，优先补原表四项，不擅自重排 Week 4–10。

## 与 research-journey 衔接

沿用 Week 1–2 的“计划、代码、实验与日志只有一份维护位置”约定：当前实际目录为 `/Users/mac/顶会刊收割机/01_10week_plan/`，本周产物放入 `week03/`。不因旧文件中的历史路径再次迁移目录，不改 `week1.md / week2.md`、原计划表或同步参考文件。

**Week 4 交接（依据 `10周计划!B13:D13`，具体接口为执行建议）**

- 保留已核对的模型族 / gender / β 宽度、pose split、joint subset、faces、frame / 单位、旋转接口、原始参数与 joints / vertices；先能回答“这个点现在在哪个坐标系”。
- 复用 Day 4 reference / 参数控制和 Day 5 序列，进入 camera projection / reprojection，检查 global_orient / transl / scale 的作用；本周表示输出不等于已完成相机估计或图像人体重建。
- Week 4 按原表读 HMR 2018 / HMR 2.0 的方法图，串联 image → feature → SMPL parameters → joints / vertices → projection；PoseFormer / MotionBERT 仍只占 2–4h 背景回顾预算。
- MPJPE / PA-MPJPE、root / Procrustes alignment 放在 Week 4 的 Metrics Protocol I；本周的矩阵误差、骨长与参数残差是实现检查，不冒充人体 benchmark。
- 需要补验的 PyTorch3D / 模型 / 动画项单独列行动与证据；保留已完成模块，不重新开一套目录或用更大 Demo 掩盖基础表示问题。

**尚未确认的源文件 / 执行事项**

- 当前同步项目 `sources/` 没有可用参考文件；已优先读取实际科研工作区的 `.xlsx`、`week1.md` 与 `week2.md`，未引入历史版本替换它们。
- 本次核对了官方资源与关键转换 / 模型 / AMASS 代码路径；访问成功不证明本人学习环境、授权模型或 AMASS 已就绪，执行时据实检查。
- 本文件交付的是完整执行计划；代码、参数、结果和动画均由本人按日实现与记录，所有实验结论待实际运行，不预填数值或 PASS。
