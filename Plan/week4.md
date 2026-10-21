# Week 4 — Human Reconstruction Bridge｜Skeleton / Motion → SMPL → Camera → HMR

> 3D/4D Human 10 周科研计划 · Week 4 执行细化 · 2026-10-19  
> 总投入：5 天，**19h 专注学习**；Day 1/2/3/5 各 3h45m，Day 4 为 4h；休息另计。  
> 当前状态：旋转、SE(3)、FK、LBS 已学习；SMPL、HMR/HMR2、Humans in 4D 已有阅读基础。本周新增实验与验收均待本人执行。  
> 沿用 [Week3](week3.md) 的五天并列结构：每天九个栏目、四个执行阶段，先预测，再动手、测量、解释与复测。

## Week Goal

把已学的人体表示接到图像观测和评测上。给定一组 SMPL 参数，能说清 joints / vertices 所在的坐标系，将其投到图像上，并用受控实验解释体型、姿态、整体朝向、平移、焦距和深度分别改变什么。自己实现 MPJPE / PA-MPJPE，知道指标消除了哪些误差、保留了哪些误差。最后能根据 HMR2 源码列出 Week5 单图 Demo 的输入、输出、资产与检查项。

本周最重要的可执行闭环是：

```text
已知 SMPL 参数 → joints / vertices → camera transform → 2D projection
                                  ↓                       ↓
                           3D 指标检查             合成 2D 观测与重投影
                                  └──────── 控制实验 ────────┘

HMR / HMR2：RGB → image feature → 人体参数与相机预测 → 复用上面的生成路径
```

**总表原定路线**

依据当前本地 [10周总计划 Excel](00_plan/10周研究计划_曾奕博.xlsx) 的 `10周计划!B13:D13`，并核对同目录 [Numbers 原表](00_plan/10周研究计划_曾奕博.numbers)。原表要求与本周落实位置如下：

| 原表要求 | 本周落实 | 验收证据 |
| --- | --- | --- |
| Human Reconstruction Bridge：Skeleton / Motion → SMPL → Camera → HMR | 五天围绕表示、投影、重投影、指标、官方接口形成闭环 | 一张统一数据流图、一套可复跑的小实验 |
| 1）2–4h 快速回顾 PoseFormer / MotionBERT；不跑完整 Repo | Day1 **120m**，包括阅读、shape trace 与轻量张量实验 | 两类时空表示对照、关键 shape、输入输出边界 |
| 2）读 HMR 2018 + HMR2 方法图，画统一数据流 | 已读内容改为 Day2/3 的定向回查、Day5 的源码核对 | 方法图到函数/张量的映射，训练与推理区别 |
| 3）基于 Week3 smplx 实现最小 camera projection / reprojection toy，检查 coordinate、scale、global_orient / transl | Day1 检查人体接口；Day2 投影；Day3 人体参数与重投影实验 | joints / vertices、相机配置、2D 叠图、残差与失败例 |
| 4）Metrics Protocol I：自己实现/校验 MPJPE、PA-MPJPE，理解 root / Procrustes；知道 PVE / Accel Error 用途 | Day4 实现与独立参考对照；Day5 综合判读 | 单元案例、指标协议、对照误差与解释 |
| 5）为 Week5 HMR2 Demo 预热，明确输入输出、checkpoint、主要模块与准备检查项 | Day5 只读 trace、轻量接口校验与 readiness 清单 | 官方路径、配置/资产状态、第一条运行命令 |
| 过关：串联 skeleton / SMPL / camera / HMR，读懂 HMR2 输入输出与基础评测 | 最终 Gate 逐项回扣五项交付 | 实际结果与本人闭卷解释共同通过 |

另外，`10周计划!B4:B7` 要求能力驱动、约 20h/周、至少一半时间用于代码/实验/调试；`B14:D14` 将正式 HMR2/4DHumans Demo、输出解析与 failure analysis 放在 Week5。`资源清单!B22:G31、B56:G59` 分别限定背景回顾、HMR 阅读和指标资源。**本文件中的天数拆分、toy 数据、接口、容差和实验阈值是执行建议，不是总表中的原始数据或已测结果。**

版本核对：完整比较Numbers的三张业务表与Excel对应工作表，`10周计划`和`开发环境`内容一致；`资源清单`仅E15/F15的Week1 Karpathy链接/显示值有差异。Week4路线与相关资源没有差异；本次未修改任何源文件。

本周不增加新的必读论文，不训练 HMR，不将 PoseFormer/MotionBERT 变成复现项目，也不提前展开 Week7 的 SfM、三角化与 COLMAP。HMR2 与 Humans in 4D 属于同一篇论文中的重建网络与完整系统，不按两篇重新精读。

## Prerequisites

**已有进度与证据边界**

| 模块 | 目前依据 | Week4 如何处理 |
| --- | --- | --- |
| rotation 6D / AA / quaternion、SE(3)、FK | 本人已学习；历史聊天记录包含手写与数值实验 | 直接复用接口，不重新安排整天推导 |
| LBS | 本人确认已学习 | 只在 SMPL 输出异常时回查 rest correction；不要求重写一套 LBS |
| SMPL / HMR / HMR2 / Humans in 4D | 2026-10-19 学习总结与本人说明支持已有理论和架构基础 | 阅读改成带问题查公式、方法图和源码 |
| smplx 完整前向、人体参数控制、motion 可视化 | 最新学习总结写明 SMPL 完整前向实践尚未全部完成 | Day1 检查可复用证据；缺少人体 forward 时登记具体阻塞 |
| HMR2 / 4DHumans 官方 Demo | 最新学习总结明确尚未跑通 | Week4 准备，Week5 正式运行；不提前记 PASS |
| PyTorch3D 6D 对照、FK 非平凡 batch / 非法 tree | 较早聊天曾列待补，本次没有新的完整复测证据 | 只有仍缺证据才登记 Week3 backlog，不机械重做已完成内容 |

这里的“已学习”以本人陈述与现有记录为准；本次制作计划没有重跑这些实验。旧计划中的“待完成”不是最新学习状态，聊天中的解释也不自动等于代码通过。已有代码可能位于学习机，不能因本机没有实验目录就判断从未完成。

**开工依赖与推进顺序**

- 继续使用本人已有 PyTorch 环境；CPU 足够完成本周 tensor、相机、SVD 与短人体 forward。不要为了 Week4 安装完整 PyTorch3D renderer、Detectron2 或视频跟踪环境。
- 复用 Week3 的 rotation、人体模型调用和已保存的参数/输出。需要明确 model_type、gender、betas 数量、pose 接口、joint names/indices、faces、单位与 frame。
- Day1 给最小 SMPL forward / 导出留 35m；这足够检查已有路径，**不等于把所有 Week3 剩余实验压进 35m**。模型或代码缺失时，Day2/4 的点投影与合成指标可先做，Day3 的人体交付仍待补。
- 单项环境排查限时 15m；需要安装、CUDA、服务器或授权处理，整理错误、版本与最小复现，交给“🌐 环境网络大师”。本计划仍保留相应科研 Gate。
- 不要求 Human3.6M、3DPW 或完整 AMASS。用合成已知 GT 验证实现；没有真实 GT 的图片不报告真实人体 MPJPE。
- 不把 Week3 所有缺口作为整周的启动阻塞。优先级：**能读懂人体输出 → 投影正确 → 指标正确 → 人体重投影闭环 → Week5 就绪**。

本人执行时先检查基础环境：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

能否 import `smplx`、能否读取授权模型、能否 forward、能否生成可解释结果分成四项记录，不用一个“环境好了”覆盖它们。

## 统一科研资料结构

维护位置沿用 `/Users/mac/顶会刊收割机/01_10week_plan/`。本次交付 `week4.md`；下面是本人执行时按需建立的结构，**不是已提供的代码或结果**。若实验在 WSL/Linux，保持相同相对结构并在日志写实际根目录。

```text
01_10week_plan/
├── 00_plan/                      # 总表，保持原样
├── week3.md
├── week4.md
└── week04/                       # 本人执行时创建
    ├── environment.md
    ├── src/                      # camera.py / metrics.py / body_adapter.py
    ├── experiments/              # d1_bridge.py / d2_projection.py
    │                             # d3_reprojection.py / d4_metrics.py / d5_integration.py
    ├── notes/                    # representation_bridge.md / camera_protocol.md
    │                             # hmr_bridge.md / metrics_protocol.md / hmr2_trace.md
    ├── results/                  # 原始参数、NPZ、CSV、图、实测报告
    ├── research_log.md
    ├── fail_log.md
    ├── week5_readiness.md
    └── weekly_review.md
```

模型和数据从原有位置只读加载，不复制到同步 `sources/`。正式论文笔记继续留在长期 `02_papers/`，这里放本周问题和实验引用，避免复制整套阅读笔记。下文代码文件名是建议接口；本人已有同等实现时直接复用，并记录真实路径与命令。

## 五天安排总览

| 日程 | 专注时间 | 模块与四阶段分配 | 当日检查点 |
| --- | --- | --- | --- |
| Day 1 | 3h45m | 表示与进度 45m；shape / SMPL 接口 75m；对照 70m；日志 35m | 2h 背景回顾封顶；人体输出能说明 frame、单位、joint convention |
| Day 2 | 3h45m | 相机阅读与推导 45m；投影实现 75m；相机干预 70m；日志 35m | 手算、独立参考、batch 与无效深度均有证据 |
| Day 3 | 3h45m | 重投影与监督 45m；人体闭环 75m；参数/歧义实验 70m；日志 35m | 人体投影与 camera-only fitting；能展示低 2D 误差的局限 |
| Day 4 | 4h | 协议与 SVD 45m；指标实现 75m；合成/参考检查 80m；日志 40m | raw / root / PA 三种口径分开，反射与退化处理明确 |
| Day 5 | 3h45m | 官方数据流 40m；集成与就绪检查 75m；失效分析 45m；验收总结 65m | 原表五项交付可定位；Week5 第一动作明确 |
| **合计** | **19h** | Run + Manipulate → Experiment 共 **11h50m**，约 62% | 阅读、调试和复测包含在预算内 |

每天每 60–90m 可休息 10–15m，休息另计。Day1 的 120m 背景预算为 35m 定向阅读 + 40m shape 实验 + 45m 背景对照；不是再额外增加两小时。

## 共同执行约定

**证据与实现**

1. 本人手写投影、mask/reduction、root alignment、similarity Procrustes 与 MPJPE 主路径；允许使用矩阵运算、autograd、`torch.linalg.svd`，库函数放到独立参考路径。
2. 完整 SMPL 继续调用已有 `smplx`。本周的 camera fitting 只优化少数相机变量，不训练图像网络、不宣称复现 HMR。
3. 正确性优先 CPU / float64 / seed=42；SMPL 按可用模型使用 float32。toy 数值默认 `atol=1e-6, rtol=1e-5`；float32 人体/官方接口默认 `atol=1e-5, rtol=1e-4`。报告单位、最大差、均值与有效样本数，不能只写 allclose。
4. 所有结果先留空，预测单独写。实际运行后才填数值或勾 Gate。合成数据用于实现和机制检查；不支持真实性能、泛化或 novelty 结论。
5. 每天保留一条真实或明确标注“人为注入”的故障，写症状 → 假设 → 最小验证 → 根因 → 修复 → 同条件复测。不要为凑日志制造大量错误。

**坐标与相机口径**

- 基础投影 toy 使用右手相机坐标：x 向右、y 向下、z 向前；图像 u 向右、v 向下，单位 pixel。人体模型坐标通过显式 `R_cm,t_cm` 转到 camera，禁止靠翻图修坐标。
- 列向量公式为 `Xc=R_cm Xm+t_cm`；点按 `[B,N,3]` 行存储时为 `Xc=Xm @ R_cmᵀ+t_cm[:,None,:]`。记录方向；平移向量不自动等于相机中心。
- 自定义 toy 中，SMPL `transl` 若已经加入 `Xm`，wrapper 不再加一次。外部 `t_cm` 与模型平移在变换链中的位置必须写清。
- 像素透视：`u=fx Xc/Zc+cx`，`v=fy Yc/Zc+cy`。toy 的 weak perspective 明确定义为 `u=fx Xc/Zbar+cx`，`v=fy Yc/Zbar+cy`，其中 Zbar 是声明的参考深度。
- `Zc<=z_min` 或非有限输入标 invalid；toy 用 `z_min=1e-4 m`。不要用 `abs(Z)` 或 clamp 后把相机背后的点当成有效投影。几何有效不等于在图像内，也不等于没有遮挡，三种 mask 分开。
- PyTorch3D 的 view/NDC 轴约定与这里不同；查 [官方 camera 文档](https://pytorch3d.org/docs/cameras) 后显式适配。本周不要求安装 renderer；官方 HMR2 的纯 tensor 投影函数可作为轻量参考。
- HMR2018 的弱透视参数与 HMR2 源码中的 `pred_cam / pred_cam_t` 不能按同名直接交换。具体归一化、crop、焦距与分辨率放在 Day5 核对。

**评测协议先于数字**

每次报告至少记录 `source, sample/frame_ids, joints或vertices定义, frame, unit, alignment, mask, reduction, GT来源`。`Metrics Protocol I` 是总表的第一个指标学习模块，不自动等于某数据集的 Protocol #1。

本周保留三个独立结果名：`MPJPE_raw`、`MPJPE_root`、`PA-MPJPE`。PA 拟合逐样本的尺度、合法旋转与平移；root alignment 只分别减去同名根关节。没有标注时不随意把某个 root-centered 指标称为所有论文通用的“MPJPE”。

## Day 1：Representation Bridge + SMPL 接口检查｜3h45m

### 1. 当日目标

用两小时完成原表规定的 PoseFormer/MotionBERT 背景回顾，连接 Week2 的时空特征与 Week3 的人体表示。保留已学旋转/FK/LBS，检查一个可用于相机实验的真实 SMPL 输出。

### 2. 前置条件

准备旧的 shape 笔记、Week2 tensor 操作和 Week3 模型调用。新增 `d1_bridge.py`，不下载两个背景方法的权重，不配置其完整环境。若只有模型文件没有调用代码，先用现有 smplx 最小 forward，不重写人体层。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 进度/依赖 10m；两篇背景定向阅读 35m | 已完成/待补表、表示差异 |
| 00:45–02:00 | 75m | Run | shape 实验 40m；SMPL 接口检查 35m | shape trace、人体参数/输出快照 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 背景对照 45m；人体接口干预 25m | 维度与语义检查、平移残差 |
| 03:10–03:45 | 35m | Analyze → Research Log | 汇总 15m；日志 10m；闭卷 10m | Day1 报告与 Gate |

### 4. 核心资源

- [PoseFormer 官方代码](https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py)：只定位 `Spatial_forward_features / forward_features / forward`，对照方法图。
- [MotionBERT 原论文](https://arxiv.org/abs/2210.06551) 与 [DSTformer 官方代码](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py)：看 motion pretraining、双分支的 spatial/temporal 顺序与输入输出。
- [smplx 模型接口](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)：仅在已有 wrapper 无法说明参数/输出时查 `SMPL.forward`。

35m 阅读分为 PoseFormer 15m、MotionBERT 20m。已经能闭卷画出的部分，用源码确认一个关键 shape 后立即进入动手。

### 5. 执行步骤

#### A. Understand → Trace

1. **10m 进度检查**：列“能复用的文件/函数/结果、尚缺证据、是否阻塞本周”。明确 SMPL forward 的状态；不根据旧计划的空勾选推断未学习。
2. **15m PoseFormer**：读图与三处函数，写输入 2D joints、中心帧 3D pose 输出。原版将一帧各 joint 的特征拼成 temporal token，不把所有时空 Transformer 都画成 `[BJ,T,C]`。
3. **20m MotionBERT**：回答带噪/部分 2D 观测如何用于 3D motion representation 学习；motion feature、3D joints 和 SMPL rotations 是不同对象。双分支的 ST/TS 顺序与融合只画到理解所需层级。

**本日 shape 表**

| 路径 | 教学输入/中间 shape | 需要验证 |
| --- | --- | --- |
| 通用 spatial attention | `[B,T,J,C] → [BT,J,C]` | attention 比较同帧 joints |
| 逐 joint temporal attention | `[B,T,J,C] → [BJ,T,C]` | attention 比较同一 joint 的帧序列 |
| 原版 PoseFormer | `[B,T,J,2] → [BT,J,Cs] → [B,T,J*Cs] → [B,1,J,3]` | temporal token 是整帧表示，输出中心帧 |
| MotionBERT/DSTformer | 输入通道与数据约定核对；内部保留 joint/time 结构 | 2D confidence 不能当 z；输出/representation 由调用接口决定 |
| 基础 SMPL | β、root/body rotations → `J[B,Jout,3]、V[B,6890,3]` | Jout 及其顺序由模型/mapper 决定，不用维度猜语义 |

#### B. Run

1. **40m shape playground**：取 B=2、T=9、J=17、C=32 的 toy features。用现有 block 或只做 `permute/reshape` 实现 spatial、逐关节 temporal、整帧 temporal 三条路径，再恢复到可比较的轴顺序。给每个元素编码 batch/time/joint 身份，确认回环完全一致。这里 J=17 是教学设置，不表示 SMPL 采用同一关节表。
2. **35m SMPL 输出检查**：复用已验证模型，B=1，显式记录 β、body_pose、global_orient、transl；保存 reference 的 joints、vertices、faces、模型配置和 joint subset。已有结果先复跑一次；缺少完整 wrapper 时只完成最小调用与导出。

建议输出 `d1_shapes.json`、`body_reference.npz` 和 `body_reference_meta.json`。meta 要说明是否包含 model transl、是否 root-centered、真实单位、关节索引来源。避免后面“有一个数组，却不知道点在哪里”。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 改变量/对照 | 检查内容 |
| --- | --- | --- | --- |
| Token 轴检查 | 20m | 保持相同 features，换三种组织方式 | attention 末尾为 J×J 或 T×T；列出 batch/head 轴；确认逆排列 |
| 缺失观测语义 | 15m | 仅隐藏指定 frame/joint 的 2D 观测，另存 mask | 坐标为 0 不必然表示缺失；真实 3D/SMPL 参数不被误改 |
| 背景小结 | 10m | 并排列 PoseFormer / MotionBERT / HMR 输入、目标、表示 | 说明 pose lifting 不能凭输出 joints 唯一恢复 SMPL mesh |
| 人体平移接口 | 15m | β/pose/orient 固定，只把 transl 加 Δ=(0.1,0,0)m | vertices/joints 差应为 Δ；若单位未核实先不使用 m 标签 |
| Joint 与 frame 核对 | 10m | 运动学 joints 对比全部输出 keypoints | parents 只接匹配子集；记录 model frame，不擅称世界坐标 |

若没有模型，最后 25m 用于检查授权/文件/接口与保存精确阻塞，或复用已导出的可信人体数据做只读分析；不得用 toy 点冒充本日实际 SMPL forward。背景实验不训练网络，不从随机特征响应推出方法优劣。

#### D. Analyze → Research Log

15m 写 `representation_bridge.md` 与 Day1 结果；10m 日志；10m 闭卷解释位置、旋转、人体参数、mesh、motion feature 的差别。画出统一 HMR 数据流的初稿，把尚未接通的 camera 箭头标出来。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验 | `week04/experiments/d1_bridge.py` | 三条 shape 路径与回环，复用人体接口 |
| 背景笔记 | `week04/notes/representation_bridge.md` | 输入输出、表示、实际 shape；120m 预算记录 |
| 人体快照 | `week04/results/body_reference.npz`、`body_reference_meta.json` | 参数、joints/vertices/faces、frame/单位/joint subset |
| 数值报告 | `week04/results/d1_bridge.md`、`d1_shapes.json` | 回环与平移残差，缺口单列 |
| 日志 | Research / Fail Log | Day1 的预测、观测、证据、下一步 |

### 7. 验收标准（Gate）

- [ ] 背景回顾含原版 PoseFormer 与 MotionBERT，累计 120m，未配置完整 Repo。
- [ ] 三条 shape 路径有实际输入与回环；解释整帧 temporal 和逐 joint temporal 的区别。
- [ ] 能区分 2D confidence、3D z、feature channel、rotation 参数。
- [ ] 实际人体 forward 与输出元数据可定位；平移只加一次。若缺模型，此项待补。
- [ ] 已标出 Week3 真正影响本周的依赖，不重做已完成整套基础实验。

### 8. 受阻处理

旧 Week2 block 不在当前机器时，仅用张量身份编码与 reshape 完成 shape 检查；这属于数据流验证，不称网络复现。SMPL 缺失不阻止 Day2/4，Day3 人体部分保持待补。超时删除 attention 可视化美化，保留原表的两小时背景和人体接口检查。

### 9. 下一步

带着 `body_reference` 的 frame/单位进入相机。第一步先用五个手算点检查投影，再把人体放进去。

## Day 2：Camera Projection / Weak Perspective / Crop｜3h45m

### 1. 当日目标

自己实现带 valid mask 的透视与弱透视投影，解释焦距、深度和坐标变换。区分像素、crop 与归一化坐标，给人体重投影建立可靠的前向算子。

### 2. 前置条件

复用 SE(3) 变换；先用自造点，不依赖模型文件。新增 `camera.py` 和 `d2_projection.py`。只研究单视图投影，不展开标定、极线和多视图重建。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 定向阅读 25m；公式/手算 20m | frame 图、相机公式与手算表 |
| 00:45–02:00 | 75m | Run | 透视 30m；弱透视 15m；参考与 batch 30m | 可运行投影与对照 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 焦距/深度 20m；弱透视 20m；crop 20m；非法输入 10m | 数值表、故障与复测 |
| 03:10–03:45 | 35m | Analyze → Research Log | 报告 15m；日志 10m；闭卷 10m | camera protocol 与 Gate |

### 4. 核心资源

- [CS231A Camera Models](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)：15m，只读 pinhole、intrinsics/extrinsics 与 weak perspective。
- [HMR 2018](https://arxiv.org/abs/1712.06584)：10m，回查相机与二维投影定义；本地可用 [HMR.pdf](../02_papers/HMR.pdf)，不重读整篇。
- [HMR2 geometry.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/utils/geometry.py) 的 `perspective_projection`：独立实现完成后再用作参考。PyTorch3D 只查共同约定中链接的 camera 文档。

### 5. 执行步骤

#### A. Understand → Trace

1. **25m 阅读**：写 pinhole 中除以 Z 的位置、K 的单位，以及 weak perspective 用单一 Zbar 替代逐点深度的条件。HMR 弱透视符号要抄到自己的协议中，不和 toy 像素公式混用。
2. **10m frame 图**：`model → camera → pixel → crop` 每条箭头标参数、输入/输出单位。外部 camera center 在 model frame 为 `C_m=-R_cmᵀ t_cm`，与 t_cm 分开。
3. **10m 手算**：固定 W=640、H=480、fx=fy=800px、cx=320、cy=240、R=I、t=0；用下表检查。

| camera point（m） | 理论像素/状态，非实测 |
| --- | --- |
| (0,0,2) | (320,240) |
| (0.1,0,2) | (360,240) |
| (0,0.1,2) | (320,280) |
| (0.1,0,4) | (340,240) |
| (0,0,0) / (0,0,-1) | invalid；不进入误差均值 |

#### B. Run

1. **30m 透视**：写 `to_camera(points,R,t)` 与 `project_perspective(points_c,K)`。返回 `uv,valid`；支持 `[B,N,3]`、批量 K；除法前安全处理非法点，输出约定与日志一致。
2. **15m 弱透视**：写 `project_weak(points_c,K,z_ref)`。z_ref 显式传入，先不要在函数内部悄悄换成每组点的新均值；需要估计 Zbar 时记录估计规则。
3. **30m 独立对照**：先手算五点，再与官方 HMR2 纯 tensor 投影函数比较，输入全部 Z>0、相同 R/t/f/c。记录参考 URL、commit/访问日期。再用 B=2、N=8、两组不同 R/t/K，比较 batch 与逐样本循环。

合法点要求 forward 有限；选一般旋转和非零平移，不能全用 identity 掩盖广播问题。只提取官方纯函数时保留来源与许可证，不加载整个 HMR2 模型。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 唯一变化/对照 | 指标和解释 |
| --- | --- | --- | --- |
| 焦距 / 深度 | 20m | 固定点，f=400/800/1200；另固定 f，只增加整体 z 平移 | 记录距主点的像素距离；解释放大与远近，不能把 f 和物体尺度混成一个参数 |
| 弱透视适用范围 | 20m | 固定 X/Y、Zbar=3m，令深度偏差幅值 δ=0/0.1/0.5/1m | mean/max pixel difference；δ=0 对齐；用实际误差说明条件，不承诺所有构型单调 |
| Crop / resize | 20m | 固定透视 uv；只用已知 crop affine A，再改尺寸 | 手算 `u'=ax(u-x0), v'=ay(v-y0)`；K'=AK；两条路径应对齐 |
| 深度与有限性 | 10m | 单点 Z=0、负值、极近值、NaN；其余不变 | valid count、故障提示、零有效点状态；不能报“平均误差=0” |

crop 采用连续坐标的教学 affine，记录 x0/y0、缩放与像素中心约定。真实图像库的半像素规则、padding 和矩形 crop 在 Day5 另行核对；不能拿教学 A 当所有预处理的精确实现。

#### D. Analyze → Research Log

15m 整理 `camera_protocol.md` 与 `d2_projection.csv`；10m 记录一次“R/t 方向、y 轴、pixel/normalized 或 crop 主点”的错误；10m 闭卷算一个新点，并解释只知道 2D 像素为什么不能唯一恢复深度。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week04/src/camera.py` | 变换、两类投影、valid mask、crop affine |
| 实验 | `week04/experiments/d2_projection.py` | 手算、官方参考、batch 与单变量干预 |
| 协议 | `week04/notes/camera_protocol.md` | 源/目标 frame、轴、单位、K、Zbar、crop 定义 |
| 结果 | `week04/results/d2_projection.csv`、`d2_projection.md`、`projection_grid.png` | 逐例误差、有效点数、固定坐标叠图 |
| 日志 | Research / Fail Log | 错误的最小复现、修复与同输入复测 |

### 7. 验收标准（Gate）

- [ ] 手算点正确，非法深度明确排除。
- [ ] 自写透视与独立参考对齐，B=2 非平凡 R/t/K 对照通过。
- [ ] 能解释逐点 Z 与 Zbar 的区别，保存 approximation error。
- [ ] crop 后 uv 与更新 K 的两条路径一致。
- [ ] frame/单位明确；pixel、normalized、NDC 没有混报。

### 8. 受阻处理

先 R=I、t=0、单点排查，再恢复 batch。官方函数导入牵涉重依赖时只隔离该纯函数；无法完成则保留手算与另一条独立矩阵实现，官方对照记待补。缺 SMPL 不阻塞本日。渲染不是本日依赖，散点和连线足够。

### 9. 下一步

把 Day1 已知人体输出接入同一相机，生成合成二维观测。先冻结人体，仅拟合相机少数变量，再讨论重投影不能约束的东西。

## Day 3：SMPL → Reprojection + 可辨识性实验｜3h45m

### 1. 当日目标

完成原表的人体投影/重投影 toy，验证 global_orient、transl 与 scale。用一次小型 camera fitting 和一个尺度—深度反例，解释“2D 对齐”为什么不足以证明三维恢复正确。

### 2. 前置条件

Day2 投影正确；Day1 有真实 SMPL 参数与 forward。新增 `d3_reprojection.py`，复用 `body_adapter.py` 与 `camera.py`。无模型时可先做几何反例，但人体交付不得勾选通过。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | HMR 监督回查 20m；实验协议与预测 25m | 约束分析、冻结变量表 |
| 00:45–02:00 | 75m | Run | 合成观测 25m；camera fitting 35m；保存复跑 15m | 参数、loss 曲线与重投影叠图 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 人体参数 30m；歧义 25m；mask/梯度 15m | 单变量表、反例与复测 |
| 03:10–03:45 | 35m | Analyze → Research Log | 分析 15m；日志 10m；闭卷 10m | Day3 报告与 Gate |

### 4. 核心资源

- [HMR 2018](https://arxiv.org/abs/1712.06584)：10m，回查 reprojection、人体先验与有/无配对 3D 监督的区别。
- [Humans in 4D](https://arxiv.org/abs/2305.20091)：10m，只回查 HMR2 方法图/训练约束；本地 [HumanIn4D.pdf](../02_papers/HumanIn4D.pdf) 可直接定位。不要重读 ViT 和 cross-attention 基础。
- [smplx](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)：按需核对模型平移、global_orient 与 pose2rot 接口。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 论文回查**：写三行“图像约束了什么、人体先验约束了什么、仍然无法确定什么”。区分作者提出的学习框架与本日只有几何参数的最小优化。
2. **15m 实验协议**：固定 model、β、body_pose、global_orient、R_cm、f、主点和 joint subset。检查点经变换后 Z>0。保存一组真实人体 joints/vertices，不改成每帧 root-center 后又宣称保留整体平移。
3. **10m 预测**：预测增加 tx、改变 global_orient、同时缩放整个相机坐标点云各会怎样影响 2D。先写预期，再跑。

本日的最小优化只回答“已知 3D 与相机其余参数时，能否恢复 x/y 平移”。未知人体形状、全姿态、焦距和深度同时优化会引入额外歧义，不作为入门成功条件。

#### B. Run

1. **25m 合成观测**：选真实人体的匹配 joint subset。令模型 transl=0、R_cm 为已声明方向，t_cm=(0.10,-0.05,3.0)m，K 沿用 Day2；若有 Z 无效则先修 frame/位置并记录实际配置。把投影保存为合成 GT `uv_gt`，同时保存生成它的参数与全部 3D。
2. **35m camera-only fitting**：固定 Z 平移、R、K、β/pose，只优化 tx、ty，初值设为(0,0)m。采用有效 joint 上的归一化平方重投影损失，建议从Adam、lr=0.03、最多200步开始；记录实际optimizer/LR/seed。比较优化前后mean pixel error，并用线性最小二乘解或手推梯度作独立校验。此处LR仅是小型平移toy的起点，不是网络训练推荐值。
3. **15m 保存/复跑**：保存每步 loss、tx/ty、最终 uv、K、mask 与配置。新进程加载相同参数，确认重投影一致；画 GT/initial/fitted 三组相同轴范围的叠图。

损失建议定义为 `mean_j ||(uv_pred-uv_gt)/[W,H]||²`，mask 只计有效配对点；报告另算 mean Euclidean pixel error，不把训练 loss 直接叫 px。无噪声、固定正确 3D/Z/K 的建议 Gate 为最终 mean pixel error <0.1px 且明显低于初始值；未达标先查收敛/归一化/梯度，而不是更改 GT。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 改变量/固定项 | 检查与结论边界 |
| --- | --- | --- | --- |
| 模型 transl | 10m | 只加 Δ=(0.1,0,0)m，camera 固定 | 验证 camera 点差为 R_cmΔ；2D 位移由逐点 Z 决定，不要求恒定像素偏移 |
| global_orient | 10m | 只改模型 root 旋转，记录其模型 pivot；β/body_pose 固定 | 3D/2D 图、关节间距离；不是随便绕世界原点转整张图 |
| body_pose 对照 | 10m | 只改已知 elbow 局部旋转，其他不变 | 3D joint 传播与2D变化；局部姿态不等价于相机平移 |
| 尺度—深度歧义 | 15m | 对已在 camera frame 的全部点取 λ=0.7/1/1.3，K 固定 | `Π(λXc)=Π(Xc)`，2D 残差≈0，但3D尺度/深度不同 |
| 沿观测射线改深度 | 10m | 单个有效点改为 αXc，其他点不动 | 该点2D不变、3D结构改变；这是几何反例，未必属于合法 SMPL 姿态 |
| mask / 梯度故障 | 15m | 屏蔽一个合成离群2D点；或人为 detach 优化变量一次后恢复 | 有效点数、loss/梯度；不把低loss归因于“人体更真实” |

全点缩放是**几何尺度干预**，不是给 SMPL 的某个 beta 设置 λ。β 改体型通常不构成精确的均匀缩放。也不能只缩放人体 model coordinates，却忘记 camera translation，然后期待透视完全不变。

#### D. Analyze → Research Log

15m 写 `d3_reprojection.md`，用实测分别回答“实现对了吗”和“解唯一吗”；10m 日志；10m 闭卷解释一个真实人体先验合理但与图像不符的可能案例。补全统一 HMR 图中的 camera、reprojection loss、训练监督位置；推理图上不放 GT 或判别器必经箭头。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验 | `week04/experiments/d3_reprojection.py` | 人体生成、合成观测、camera-only fitting 与干预 |
| 参数 | `week04/results/d3_reprojection_inputs.npz`、`d3_config.json` | 真值/初值、模型、K/R/t、mask与joint定义 |
| 结果 | `week04/results/d3_fit.csv`、`d3_controls.csv`、`reprojection_overlay.png` | loss/px分开；参数效果与反例实测 |
| 分析 | `week04/results/d3_reprojection.md`、`notes/hmr_bridge.md` | 可辨识性、先验作用、统一图与适用边界 |
| 日志 | Research / Fail Log | 至少一次梯度/坐标/mask故障复测 |

### 7. 验收标准（Gate）

- [ ] 真实 SMPL 输出经自写相机生成2D，参数与模型可复用。
- [ ] tx/ty 最小优化有初始/最终误差、曲线与独立校验；合成GT标注明确。
- [ ] global_orient / transl / body_pose 三组独立干预有预测和实测。
- [ ] 全camera点云缩放与单射线反例实际验证，解释适用条件。
- [ ] 能解释2D误差低不证明深度、尺度、表面或真实人体姿态正确。

### 8. 受阻处理

模型缺失时继续五点/非共面点的 fitting 与歧义实验，报告称“几何子任务”，人体 Gate 待补。优化不收敛先固定所有变量仅解一个 tx，检查梯度与单位；用解析解定位问题，不能用它替换后仍声称 autograd 已验证。不要增加人体全参数联合拟合。

### 9. 下一步

保留“2D几乎相同但3D不同”的两组数据。明天用 raw、root 与 PA 指标说明不同评测口径会掩盖或暴露什么。

## Day 4：Metrics Protocol I｜MPJPE / PA-MPJPE｜4h

### 1. 当日目标

本人实现并校验 MPJPE 与 similarity Procrustes；用已知误差案例理解 root alignment 和 PA。建立可复用的协议记录，知道 PVE 与 Accel Error 的用途，为 Week5 留出完整实现与真实数据评测。

### 2. 前置条件

会 L2 范数、均值、矩阵乘法；可调用 SVD。新增 `metrics.py` 与 `d4_metrics.py`。第一轮用非共面合成点，全可见、同一单位；不下载 benchmark，不在 T-pose 的共面骨架上验证反射检测。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 指标协议 20m；Procrustes推导 25m | 公式、对齐定义、SVD方向 |
| 00:45–02:00 | 75m | Run | MPJPE/root 20m；PA 35m；参考 20m | 主实现与独立参考输出 |
| 02:00–03:20 | 80m | Manipulate → Experiment | 已知变换 30m；反射/局部错误 20m；mask/退化/单位 30m | 指标案例表与故障复测 |
| 03:20–04:00 | 40m | Analyze → Research Log | 结果15m；PVE/Accel10m；日志/闭卷15m | metrics cheat sheet、Day4 Gate |

### 4. 核心资源

- [MMPose keypoint_mpjpe](https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/functional/keypoint_eval.html#keypoint_mpjpe)：读 alignment、mask、reduction；函数自身不自动替你选择 pelvis。
- [MMPose similarity transform](https://github.com/open-mmlab/mmpose/blob/main/mmpose/evaluation/functional/mesh_eval.py)：独立实现后对照 SVD、det correction 和 scale。
- [VideoPose3D 指标代码](https://github.com/facebookresearch/VideoPose3D/blob/main/common/loss.py)：第二来源，用于核对 MPJPE / P-MPJPE 定义。
- [VIBE evaluation](https://github.com/mkocabas/VIBE/blob/master/lib/utils/eval_utils.py)：只查 PVE 与 acceleration error 定义，不展开新论文精读。

总表给出的 MMPose `models/losses/regression_loss.py` 主要是训练 loss 入口；本周校验使用更直接的 evaluation functional 路径，并保留这一资源定位调整。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 评测协议**：令预测 P、GT Q 为 `[B,J,3]`，mask 为 `[B,J]`。逐点误差 `e_bj=||P_bj-Q_bj||₂`。默认按所有有效 joint 等权求均值，同时保存每样本均值/有效数。零有效样本标 invalid，不填零。
2. **10m root alignment**：分别计算 `P-P_root`、`Q-Q_root` 后取同一评测子集。root 可以是一个模型定义的 pelvis 或指定 hips 均值，必须两侧同定义；若 root 无效，则该样本 root 指标无效。本周默认评测子集保留root，比较其他协议时明确是否排除root，防止分母悄悄变化。
3. **15m PA 推导**：每个样本在同一有效 joint 集上求 `min ||s P R+t-Q||_F²`，s>0、det(R)=+1。这里 P 为行存储，所以公式右乘 R；逐步确定方向，禁止从列公式直接复制矩阵顺序。

行存储的推导检查表（学习提示，不是已经完成的实现）：

- 用有效点计算质心 μP/μQ，中心化为 Pc/Qc；H=PcᵀQc。
- H=U diag(σ) Vᵀ；D=diag(1,1,det(UVᵀ))；R=UDVᵀ。
- `s=sum(σ*diag(D))/sum(Pc²)`，`t=μQ-s μP R`；对齐预测 `P_aligned=s P R+t`。
- 最后报告 aligned prediction 与 GT 的平均欧氏距离。SVD 拟合最小化平方误差；不因此断言每个样本的 PA 平均欧氏误差必然小于 raw 平均欧氏误差。
- 对零方差、少于3个有效点、共线点集或非有限值明确标 invalid。本周对中心化点集 rank<2 拒绝稳定PA；平面点可算，但反射测试必须用满3D点集。

#### B. Run

1. **20m MPJPE/root**：本人实现 `mpjpe_raw(pred,gt,mask)`、root 对齐与协议记录。先返回逐样本/逐点值，再汇总。masked NaN 必须在运算前被安全排除，不能靠 `NaN*0`。
2. **35m PA**：先写 B=1 的 SVD 主路径，再支持 batch。返回 aligned points、s/R/t、valid status、PA误差。检查 R正交、det=+1、s正值；PA用于评价，不要求对退化SVD反传。
3. **20m 独立参考**：用固定 seed 的 B=4、J=8、满秩点集，至少两组不同旋转/尺度/平移，与 MMPose或VideoPose3D官方实现对照。全可见先过，再讨论mask差异。保留来源和实际版本，不将同一helper包装两次称“独立实现”。

注意官方 MMPose 参考路径在 similarity fitting 时先用传入的全部点，mask 用于最终误差选择。**有缺失点时不能直接把 masked垃圾值塞进去比较。**本周masked PA按有效子集拟合，对照时先取同样子集再调用参考；不同定义要说明。

#### C. Manipulate → Experiment

固定一个非对称、非共面、至少8点的 Q，单位 m、seed=42；保证后续保持同一joint对应。下表是**理论预期**，实测另填CSV，不预填PASS。

| 案例 | 时间分组 | 预测构造 | 必查现象 |
| --- | --- | --- | --- |
| identity | 前30m | P=Q | raw/root/PA≈0 |
| 纯平移 | 前30m | P=Q+(0.1,-0.2,0.3)m | raw=√0.14m≈374.166mm；root/PA≈0 |
| 旋转+缩放+平移 | 前30m | P=1.3 Q R0+t0，R0合法 | raw/root一般非零，PA≈0；每个样本单独拟合 |
| 局部错误 | 中20m | 只偏移一个非root joint 0.1m | 全可见raw=0.1/J m；PA一般仍非零；不能要求PA等于root |
| 反射 | 中20m | P=Q diag(-1,1,1) | 满3D、固定对应下合法旋转不能消除；det correction抓住错误 |
| mask | 后30m | 一个无效joint填大值/NaN；另全mask=False | 有效子集结果正确；全无效明确invalid，不返回0 |
| 单位 / 退化 | 后30m | 两侧同时×1000；另一侧单独×1000；全点重合/共线 | 两侧单位转换后mm结果一致；单侧单位错可被PA尺度掩盖，必须由协议检查发现；退化拒绝 |

三个时间块合计 80m：30m 已知变换、20m 局部/反射、30m mask/单位/退化。对齐前后的点都保存。使用两种不同刚体变换的 batch，专门排查“把整个batch一起SVD”的错误。

CSV建议字段：`case_id,B,J,valid_count,unit,root_definition,alignment,raw_m,root_m,pa_m,raw_mm,root_mm,pa_mm,det_R,scale,reference_error,status`。invalid值填缺失并给原因，不与真实0混用。

#### D. Analyze → Research Log

15m 解释案例结果；10m 补充以下两个后续指标；15m 更新日志并闭卷回答“PA很低，是否证明尺度和全局方向正确”。

| 指标 | 本周要知道 | 不能忽略的协议 |
| --- | --- | --- |
| PVE | 同拓扑对应vertices的平均3D距离，衡量表面恢复 | vertex对应、单位、对齐；不能直接比较SMPL与SMPL-X不同拓扑 |
| Accel Error | 预测与GT的二阶时间差分之差，评价运动误差 | 时间间隔、frame mask、坐标系、是否除以Δt²；没有GT的acceleration magnitude不是Accel Error |

写出 `a_t=(X[t+1]-2X[t]+X[t-1])/Δt²`。部分公开实现使用未除Δt²的帧差分，因此不同fps/单位不能直接比较。平滑本身不保证准确；恒定错误姿态可以很平滑。完整PVE/Accel实现与实际模型评测仍按原表放Week5。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week04/src/metrics.py` | raw/root/PA分离，mask/reduction/退化处理明确 |
| 实验 | `week04/experiments/d4_metrics.py` | 合成7类案例、独立官方参考与batch检查 |
| 原始数据 | `week04/results/d4_metric_cases.npz` | P/Q/mask、生成参数、对齐参数与输出 |
| 结果 | `week04/results/d4_metrics.csv`、`d4_metrics.md` | 预测/实测分开，单位与有效数完整 |
| 协议/日志 | `week04/notes/metrics_protocol.md`；Research / Fail Log | 四指标cheat sheet；反射/单位/mask故障与复测 |

### 7. 验收标准（Gate）

- [ ] raw/root/PA为本人主实现；知道各自消除的自由度。
- [ ] identity、平移、相似变换与局部错误符合理论预期，单位换算正确。
- [ ] 反射不被当作合法旋转；不同batch独立拟合。
- [ ] 与独立参考同口径对齐；缺失点拟合子集与误差子集明确。
- [ ] 零有效点、退化点集、非有限值有状态；没有用0伪装成功。
- [ ] 能说明PVE/Accel所需GT、对应关系、时间单位与Week5边界。

### 8. 受阻处理

SVD错时先固定s=1、t=0的已知旋转例，逐查行/列约定与源→目标方向，再恢复scale/translation。缺少完整MMPose环境可隔离NumPy参考函数，无须安装整个框架；官方参考未执行则该Gate待补。不要用root-only结果冒充PA。

### 9. 下一步

用同一个指标实现解释Day3的2D/3D歧义，再核对HMR2返回值如何接入现有模型、相机和评测协议。

## Day 5：HMR2 Code Trace + 集成验收 + Week5 准备｜3h45m

### 1. 当日目标

把方法图、真实源码接口与本人toy连起来。修正容易混用的rotation/camera/crop约定，完成全周Gate和Week5就绪清单。保留正式单图/视频Demo、真实样例failure analysis在Week5。

### 2. 前置条件

Day2相机与Day4指标可运行，Day3有保存的人体实验；缺口据实登记。新增`d5_integration.py`，只读官方源码，轻量测试使用已写函数或隔离的纯函数，不加载大型checkpoint作为本日必做项。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:40 | 40m | Understand → Trace | 官方输入/主干/人体/相机路径 | 方法图到源码的映射 |
| 00:40–01:55 | 75m | Run | 6D与camera接口30m；集成20m；就绪检查25m | 约定对照、集成输出、资源状态 |
| 01:55–02:40 | 45m | Manipulate → Experiment | 2D/3D失效案例30m；论文判断15m | 指标解释、待验证问题 |
| 02:40–03:45 | 65m | Analyze → Research Log | 原表逐项验收45m；总结与交接20m | 最终Gate、周总结、Week5第一步 |

### 4. 核心资源

- [4DHumans 官方仓库](https://github.com/shubham-goel/4D-Humans)：README中的image demo、model资产、tracking入口，按实际访问版本记录。
- [hmr2.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/hmr2.py)、[smpl_head.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/heads/smpl_head.py)、[geometry.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/utils/geometry.py)：只追一条forward与两个纯函数。
- 已读HMR/Humans in 4D方法图和本周笔记。出现“论文图”和“实际配置”不一致时，把二者分别标注，不靠印象填网络尺寸。

### 5. 执行步骤

#### A. Understand → Trace

1. **10m 输入**：定位image crop/normalization的调用与配置。记RGB顺序、resize/crop尺寸、bbox、原图大小和crop元数据；图像预处理后的batch tensor不直接等于最终ViT输入。
2. **15m 图像到参数**：定位backbone→SMPL head→pose/shape/camera输出；标记features转tokens、query来源与cross-attention。只用配置或代码证据填C/H/W，不把教学用的192 tokens当所有版本常量。
3. **15m 人体到投影**：定位rotation matrix→SMPL、3D输出、camera translation换算和2D projection。区分人体局部/模型坐标结果与加相机平移后的camera点。

本次核对官方主分支后，应重点核验以下事实；执行时仍记录自己的版本与配置：

| 环节 | 本周要定位的接口证据 | 要避免的误读 |
| --- | --- | --- |
| image backbone | `forward_step`在backbone前还有宽度切片 | 不能仅看batch输入shape就断言patch数 |
| SMPL head | `pred_smpl_params`含root/body矩阵与betas；另有`pred_cam` | head内部6D与最终matrix输出分开 |
| decoder / 参数更新 | 配置有`IEF_ITERS`，读出值加到当前估计；query来源也由配置决定 | 不能绝对声称“HMR2没有迭代更新”或“所有配置都是可学习query” |
| body layer | 调用SMPL时`pose2rot=False` | 不把matrix再当axis-angle转换一次 |
| camera | `pred_cam`与派生的`pred_cam_t`不同；投影还用归一化焦距 | 不能把前者直接当3D位置或world trajectory |
| video | tracking是另一入口，依赖PHALP等组件 | 单图mesh不等于视频tracking已通过 |

#### B. Run

1. **30m 约定适配**：用同一个非对称合法R，分别编码为Week3/PyTorch3D的“前两行”与当前HMR2的“前两列向量”6D，按各自解码器验证回到同一个R。错误地交换6D输入做一次故障并恢复。再用一组声明的正s、f、image size，手算HMR2的camera深度换算，与隔离的源码运算比较；不从该toy推断真实人体尺度。
2. **20m 集成**：从Day3保存的参数重跑 `SMPL → camera → uv`，用Day4指标比较原始/扰动的匹配joints。每个输出都带frame/单位/alignment标签；记录真实入口命令。SMPL缺失时只能报告已验证的geometry/metrics子链。
3. **25m readiness检查**：按下表填写现状、证据、阻塞和下一步，区分文件存在、可读取和已实跑。只做检查；资源需要安装或下载时记录后交对应环境任务，不把准备工作写成Demo成功。

HMR2的6D实现和PyTorch3D行约定不能直接互换，即使输出都满足SO(3)。比较最终R和对同一点的作用才有意义。当前代码的camera深度换算含 `tz=2f/(image_size*s+ε)`；需同时核对crop、normalization以及tx/ty定义，不能只记这一个式子就转换任意模型相机。

| Week5准备项 | 本日记录内容 | 就绪证据 |
| --- | --- | --- |
| Repo/config | 官方URL、commit或访问日期、准备使用的配置 | 能定位`demo.py`和核心forward路径 |
| 输入 | 先选一张全身、无遮挡的本地图片，记录来源/尺寸 | 能读取；不必先组织stress-test数据集 |
| Checkpoint | 按官方README/loader记录预期权重、缓存位置和状态 | 已有就核对路径；未有写待获取，不猜文件名 |
| SMPL资产 | 官方要求的neutral模型文件与放置位置；模型授权状态 | 能定位匹配资产；smplx包存在不算模型存在 |
| 运行环境 | 执行主机、解释器、PyTorch/CUDA/driver、GPU/显存 | 本人环境实查；总表RTX5070 Laptop 8GB只是计划记录 |
| 输出 | 预期render/mesh；参数数组若需额外导出，定位代码 | 不假设所有内部返回值都会自动写成文件 |
| fallback | 单图、batch_size=1；显存/编译失败的远程Linux路径 | 已知联系人/主机或明确待确认项，不承诺8GB必跑 |
| tracking | 单独记录PHALP/视频入口状态 | 留Week5；未装不阻塞Week4源码预热 |

**Week5首个运行动作**

以下是**资产和环境就绪后**的官方单图入口计划命令，尚未执行。先从仓库根目录运行，使用本人选定的输入目录；选项执行时再对照实际版本：

```bash
python demo.py --img_folder example_data/images --out_folder demo_out --batch_size=1 --save_mesh --full_frame
```

先保存单图输入、配置、渲染和mesh，再加forward hooks解析真实shape。视频tracking放在单图输出解释清楚之后。

#### C. Manipulate → Experiment

**30m 三组失效对照**：复用Day3相同GT和Day4指标，不写第二套评测函数。

| 组别 | 干预 | 同时观察 | 应形成的判断 |
| --- | --- | --- | --- |
| A reference | 相同数据与参数 | reprojection、raw/root/PA | 端到端一致性基线 |
| B 全局尺度歧义 | 全camera点云整体乘λ>0 | 2D残差、raw/root、PA | 投影与PA可能仍近零，不能证明绝对尺度正确 |
| C 局部深度错误 | 一个joint沿原射线改变深度 | 2D残差、raw/root/PA与3D图 | 单视图观测可隐藏结构误差；反例不必是合法SMPL输出 |

B/C中的GT是人工构造的已知3D，不能把它们报告成HMR2预测结果。若PA未反映C的预期变化，先检查点数/退化/对应关系和错误幅度，再分析，而不是强写结论。

**15m 论文判断**：在一页对照中各写HMR和HMR2的Problem、Previous limitation、Core insight、Method、Training/Inference、Metrics/Results、Ablation、Limitations、Connection。每项1–2句即可，引用自己已读的页/图/表；明确“作者声称的贡献”和“本周实验让我理解的机制”，不补造指标数字。已有这些字段就直接引用旧笔记，只新增相机/指标纠正。

若保留“24个joint queries”想法，写入parking lot：这是结构假设，novelty目前无法判断；未来先查已有joint-query方法，再固定backbone/数据/训练预算与1-query baseline做最小实验。它不占本周主线，不把模块拼接当成果。

#### D. Analyze → Research Log

**45m 整体验收**：15m按原表五项检查产物；20m从已保存配置重跑投影手算、SMPL平移/重投影、指标相似变换/反射/缺失案例、6D约定对照；10m闭卷讲统一数据流，回答“这个tensor是什么、在哪个frame、用什么指标能比较”。

**20m 总结与交接**：15m写`weekly_review.md`，5m更新最终Gate与`week5_readiness.md`。记录：当前结论、支持证据、最大不确定性、最小验证实验、下一步一个可执行动作。缺模型、未完成参考校验或未解决指标错误，逐项写补做条件，不用“总体完成”掩盖。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 源码映射 | `week04/notes/hmr2_trace.md` | 输入预处理、backbone、head、SMPL、camera和返回值；版本证据 |
| 集成入口 | `week04/experiments/d5_integration.py` | 复用已有相机/指标，独立约定适配与同输入复跑 |
| 结果 | `week04/results/d5_integration.csv`、`d5_failure_analysis.md` | A/B/C的2D/3D指标、坐标/单位与结论边界 |
| 准备清单 | `week04/week5_readiness.md` | checkpoint/模型/环境/输入/输出的真实状态与第一条命令 |
| 总结/日志 | `week04/weekly_review.md`；Research / Fail Log | 五项交付映射、最终Gate、待补与下一步 |

### 7. 验收标准（Gate）

- [ ] 统一图已关联官方函数和真实接口，训练监督与推理输出分开。
- [ ] 6D行/列约定有数值对照；camera参数/模型transl/camera point不混用。
- [ ] 集成实验从保存的输入重跑，2D与raw/root/PA指标同口径可解释。
- [ ] Week5准备清单全部有状态与下一步，未把预热写成Demo完成。
- [ ] HMR/HMR2对照能给出机制、证据和限制；当前想法未被提前宣称novel。
- [ ] 最终Gate与周总结完整，未通过项保留且可行动。

### 8. 受阻处理

官方主分支与当前配置不同，以实际版本为准，记录差异。checkpoint/大型依赖受阻时继续纯函数、源码和清单，不消耗本周整日安装。若必须压缩，先删图美化与额外查询方案，保留投影正确性、指标正确性和原表五项证据；未完成的必做内容标待补。

### 9. 下一步

Week5先完成单图HMR2 Demo，保存输入/输出/配置，随后追真实tensor与解析SMPL参数，再进入短视频tracking、PVE/Accel和复杂样例。若Week4人体forward或指标仍有阻塞，优先修复能直接影响输出解释的那一项。

## Fail Log / Research Log 模板

下面字段执行时填写，保持Prediction与Observation分离。所有结果路径指实际文件，不能只写“见代码”。

### Research Log

```markdown
## YYYY-MM-DD / Week 04 Day N / 主题
状态：未开始 / 进行中 / PASS / 待补
计划时间 / 实际时间：
Question：本日具体要判断什么？
Hypothesis：唯一改变量、预期与理由。
Understand / Trace：论文页/图/表，源码路径/版本，主要shape。
Conventions：源/目标frame、行/列、单位、crop/normalization、joint定义。
Model / Data：模型族、参数、GT来源、样本/帧、mask。
Run：实际命令、环境、seed、配置和输入输出路径。
Manipulate / Experiment：控制组、干预组、固定项。
Prediction：运行前预期。
Observation：真实数值、valid_count、图与误差单位。
Metrics protocol：raw/root/PA、root定义、fit/eval子集、reduction。
Analyze：结论、替代解释、适用范围、最大不确定性。
Implementation provenance：本人实现 / 库调用 / 隔离参考函数。
Gate：逐项证据；待补条件和最小动作。
Tomorrow first step：一个可直接执行的动作。
```

### Fail Log

```markdown
## W04-DN-F01 / 简短标题
状态：Open / Explained / Fixed / Retested
来源：真实遇到 / 人为注入
Context：环境、输入、frame、unit、shape、mask、复现命令。
Expected：手算或理论预期，及适用条件。
Observed：实测/报错，不能填推测结果。
Hypotheses：候选原因与排除方法。
Minimal reproduction：最少点/joints/样本、最小参数差异。
Root cause：证据支持的根因；未确认则写待验证。
Fix：改动及原因。
Retest：同输入、同协议的数值与结果路径。
Lesson：下次在哪个边界加检查。
```

**高优先级Debug提示**

| 症状 | 首先怀疑 | 最小验证 |
| --- | --- | --- |
| 人体上下颠倒/在相机后 | model→camera轴、R方向、z平移 | 三个轴与一个已知点，不靠翻图修正 |
| 人体投影整体偏移 | transl重复加入、主点/crop未更新 | 只加一次已知Δ，比较camera点差与pixel差 |
| 焦距改了，尺度结论相反 | pixel/normalized/crop混用 | 同一3D点、固定Z与分辨率手算 |
| 优化loss下降却3D错误 | 单视图歧义、变量过多、GT/预测混用 | 冻结3D只拟合tx/ty，检查独立解析解 |
| raw高、root近零 | 可能只有平移差 | 检查逐joint差是否为相同向量 |
| PA低、尺度却错 | PA去掉了scale；可能单位不一致 | 同时报raw/root、拟合s及两侧单位 |
| 反射也PA≈0 | det correction缺失或点集共面 | 固定对应的非共面非对称点集 |
| mask后仍NaN或异常 | NaN乘0、拟合时未排除无效点 | 显式取有效子集；零有效点返回状态 |
| 两个6D函数都合法却方向不同 | 行/列编码和stack不同 | 同一个非对称R分别编码、解码、旋转同一点 |
| SMPL接HMR2输出形状错误 | matrix被当AA、root/body分段或joint mapper错 | 打印pose2rot、root/body shapes与实际joint names |

## Week 4 最终 Gate

全部初始状态均为“待执行/待验证”。文档结构检查通过，只说明计划已准备好；不表示本人实验已通过。

| 项目 | 对应原表 | 通过证据 | 当前状态 |
| --- | --- | --- | --- |
| PoseFormer / MotionBERT背景 | D13交付1 | 120m阅读/shape实验；任务与表示区别；没有完整Repo复现负担 | 待执行 |
| HMR / HMR2统一数据流 | D13交付2 | 已读笔记引用、方法图、函数/shape/训练推理映射 | 待验证 |
| 人体camera projection | D13交付3 | 真实SMPL输出、明确frame/单位、手算/参考/batch对照 | 待执行 |
| Reprojection与参数效应 | D13交付3 | orient/transl/scale实验、相机小优化、歧义反例 | 待执行 |
| MPJPE / PA-MPJPE | D13交付4 | 本人实现、官方参考、root/PA、反射/mask/单位/退化检查 | 待执行 |
| PVE / Accel概念 | D13交付4 | GT/对应关系/时间单位/用途可解释，完整实践留Week5 | 待验证 |
| Week5预热 | D13交付5 | 输入输出/主要模块、checkpoint/模型/环境状态、第一命令 | 待执行 |
| 四层表示贯通 | 原Gate | 闭卷从image到SMPL/camera/projection/metrics，指出坐标与不可辨识性 | 待验证 |
| 研究习惯与可复现性 | 执行建议 | 五日日志、预测与实测分栏、原始参数、失败复测、周总结 | 待执行 |

**通过规则**：本人能独立解释，且对应运行证据齐全，才勾选PASS。没有SMPL模型时允许完成相机与指标子任务，但整周人体重投影交付仍待补；官方Demo没跑不影响Week4原定预热Gate，却不能把Week5记完成。需要额外时间时，单列待补工时与原因，不把它隐含在19h里。

## 与 Week3 / Week5 及长期工作区衔接

- Week3保留原文件与已完成能力，不重新学rotation/FK/LBS全套。只补会影响人体输出解释的关键接口/证据，其他旧缺口单列。
- 当前周学习重点由“更多论文”转为“已学方法的相机、指标和代码证据”。HMR/HMR2阅读以旧笔记回查和实验解释为主。
- Week5保留总表的Core Demo①，继续输出解析、真实forward trace、PVE/Accel和failure analysis；Week4不会用合成结果冒充这个Demo。
- 历史`research-journey`名称不再触发目录迁移。总表、Week3与同步`sources/`保持原样；本周实验在`week04/`维护。

**本次依据与完成边界**

完整阅读本地总表的10周路线、开发环境与资源清单，以及750行`week3.md`；结合本Project“修正旋转表示理解”聊天中2026-10-19学习总结更新进度。原“制作Week4 md”聊天只包含制作请求，未提供另一套Week4路线。资源链接与关键官方相机/指标/HMR2接口于2026-10-19核对；执行时固定实际版本。本次交付执行计划，未替本人运行科研实验，也未改变旧计划或学习完成记录。
