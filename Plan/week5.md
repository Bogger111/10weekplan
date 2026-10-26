# Week 5 — HMR 2.0 / Humans in 4D｜Core Demo ①：RGB → SMPL Mesh → Video Tracking

> 3D/4D Human 10 周科研计划 · Week 5 执行细化 · 2026-10-22  
> 总投入：5 天，**19h 专注学习**；Day 1/2/3/5 各 3h45m，Day 4 为 4h；休息另计。  
> 沿用 [Week4](/Users/mac/workspace/10weekplan/Plan/week4.md) 的五天并列结构：每天九个栏目、四个执行阶段，先预测，再运行、测量、解释与复测。  
> 本文是执行计划与代码附录。正式 Demo、模型回放、真实图像 failure analysis、视频 tracking 均须在执行主机留下证据后再验收；旧计划的待办不代表当前实际进度。

## Week Goal

完成总表规定的第一个核心 Demo：给定一张 RGB 图像，跑通官方 HMR2，保存输入、配置、人体渲染与 mesh；能够把实际输出重新送入相同人体层，解释 pose、betas、global_orient、camera、joints 和 vertices。沿一条实际 forward 追到主要 shape，完成 PVE / Accel Error 的实现检查，并用受控复杂输入区分检测、crop、人体恢复和跟踪的失效。

本周闭环：**真实 RGB → 检测与 crop → backbone → SMPL head → SMPL 参数 → mesh/joints → 相机与渲染 → 输出回放 → failure analysis**。视频分支再接 tracker、逐帧 ID 与时序结果。单图的 detection index 与跨帧 tracking ID 分别记录。

**总表原定路线**

完整读取本地总表 `/Users/mac/顶会刊收割机/01_10week_plan/00_plan/10周研究计划_曾奕博.xlsx` 的 `10周计划`、`开发环境`、`资源清单`，以下以 `10周计划!B14:D14` 为 Week5 依据。未修改总表；本次读取 Excel 导出表，未重新解析 Numbers 原件。

| 原表要求 | 本周落实 | 验收证据 |
| --- | --- | --- |
| 1）必跑官方 Demo，图像或短视频；保存输入、渲染、运行配置 | Day1 单图必做；Day5 在 PHALP 就绪时追加短视频 | 原图、渲染、OBJ、日志、commit/config/checkpoint 与实际命令 |
| 2）解析 pose / betas / global_orient / camera / joints / vertices / tracking ID，以实际输出为准 | Day1 导出真实 forward；Day2 相同人体层回放；Day5 核对视频 ID | NPZ、字段字典、参数/输出 shape、回放残差；未出现字段写不适用 |
| 3）追 preprocessing → backbone → HMR head / SMPL → renderer / tracker | Day3 核心追踪，Day1 同步记录 forward hooks | 方法图到函数映射、实际 shape、crop 与 full-image camera 区别 |
| 4）Metrics Protocol II：理解/校验 PVE 与 Accel Error，与 MPJPE/PA-MPJPE 同表 | Day4 新实现 PVE/Accel，复用 Week4 的 joint 指标 | 已知 GT 的数值案例、时间/mask/对应关系协议、独立参考记录 |
| 5）遮挡、侧身、截断、异常姿态、视频运动等自选样例，至少一份图文分析 | Day5 baseline + 至少两种自选干预；推荐五类各留一例 | 样例清单、同条件输出、图文报告、错误来源与验证动作 |
| 原 Gate：能运行、解释输出来源，并指出容易失败的输入条件 | 五天最终验收 | 可复跑证据与本人闭卷讲解共同通过 |

总表 `B4:B7` 规定能力驱动、约 20h/周、至少一半时间用于代码/实验/调试；本计划安排 **19h，其中 Run + Manipulate → Experiment 为 11h50m，约 62%**。`资源清单!B28:G31、B57:G59` 是本周主要资源。总表下一周为 Multi-HMR；本周不展开 Multi-HMR、SLAM、世界轨迹或大规模 benchmark。

五天拆分、文件名、最少三个图像案例、时间盒和数值容差是执行建议，不是总表原文，也不是已经测得的结果。HMR2 是 Humans in 4D 系统中的人体重建网络；论文理解以已读笔记和针对性回查为主，不重做整周精读。

## Prerequisites

| 依赖 | 开工时核实 | 缺失时如何推进 |
| --- | --- | --- |
| Week4 投影、frame、MPJPE/root/PA | 查看本人已有代码和实测报告，不按旧勾选推断 | 优先补影响输出解释的接口；不重做全部旋转/FK/LBS |
| 执行主机 | 实查解释器、PyTorch、CUDA/driver/GPU；8GB 是总表设备记录 | 正式模型优先独立 Linux/WSL 环境；Mac 保留资料与轻量分析 |
| 官方仓库与 checkpoint | 固定 commit、记录工作区差异；权重及相邻 model_config.yaml 可读取 | 使用既有就绪路径；新环境问题限时排查后单列阻塞 |
| SMPL 与资源 | 授权 neutral 模型、HMR2 配置所需 regressor/mean params、检测器权重 | Python 包可 import 不等于资产完整；分别记录 |
| 图像与视频 | 一张单人全身 baseline，来源/尺寸明确；复杂样例逐步增加 | 不要求下载 Human3.6M、3DPW、完整 AMASS |
| 代码附录 | NumPy；绘图另需 matplotlib/Pillow；模型脚本需要官方依赖 | 不把官方旧依赖整套装入当前主环境；不自动迁移 Mac/WSL 文件 |

总表的通用 Python 3.12.x 记录不能直接当作旧 Repo 的兼容保证；官方 README 提供独立环境示例，执行时以实际仓库与主机测试为准。单项安装/编译排查先限时 15m；严重 CUDA、显存或渲染问题汇总最小报错、版本和 fallback 主机，另行解决。保留未完成 Gate，不把 19h 学习预算改成无限安装时间。

开工只记录已有基础，不在本文预填“Week4 已全部完成”或“官方模型已跑通”。即使本机找不到旧实验，也先查学习机实际路径。

## 统一科研资料结构

本文保存到 `/Users/mac/workspace/10weekplan/Plan/week5.md`，与现有 week4.md 同目录。实验执行时在实际代码工作区建立 `week5/`；长期论文与总表保留原位置。以下是运行时目录约定，尚未替本人创建这些科研结果。

```text
10weekplan/
├── Plan/week4.md
├── Plan/week5.md
└── week5/
    ├── exp/
    │   ├── d1_demo_check.py
    │   ├── d2_output_check.py
    │   ├── d3_trace_check.py
    │   ├── d4_metrics_check.py
    │   ├── d5_failure_check.py
    │   └── visualization.py
    ├── inputs/                   # 本人选定的 baseline/stress 图片、cases.json
    ├── results/                  # 每次使用新目录，不覆盖旧结果
    ├── notes/                    # output_schema / code_trace / metrics_cheat_sheet
    ├── environment.md
    ├── research_log.md
    ├── fail_log.md
    ├── failure_analysis.md
    └── weekly_review.md
```

末尾提供 **5 个独立的每日 Python 代码块 + 最后一个 visualization.py 代码块**。按标题保存到同一 `week5/exp/` 后使用。代码包含主要实验检查；论文解释、字段语义、样例人工观察和日志仍由本人完成，脚本退出 0 不能自动勾选整日所有 Gate。

## 五天安排总览

| 日程 | 专注时间 | 四阶段分配 | 当日检查点 |
| --- | --- | --- | --- |
| Day 1 | 3h45m | 45m / 75m / 70m / 35m | 官方单图、输入/配置、渲染/mesh、真实输出与运行状态 |
| Day 2 | 3h45m | 45m / 75m / 70m / 35m | 同人体层回放、betas 单变量干预、字段与关节约定 |
| Day 3 | 3h45m | 45m / 75m / 70m / 35m | 实际 shape、归一化投影、crop/full-image camera 与失效定位 |
| Day 4 | 4h | 45m / 75m / 80m / 40m | PVE/Accel 的已知 GT、mask、帧间隔、单位与参考对照 |
| Day 5 | 3h45m | 40m / 75m / 45m / 65m | 多条件失败分析、视频分支状态、图文结果与最终 Gate |
| **合计** | **19h** | **1140m；代码/实验阶段 710m** | 休息另计；超时任务另列补做工时 |

每天每 60–90m 可休息 10–15m。已有合格结果先做最小复跑并引用证据，省下的时间用于复杂样例或解释，避免重复安装与重复阅读。

## 共同执行约定

1. **版本与可复现性**：记录实际主机、解释器、commit/dirty 状态、checkpoint hash、配置、原始图片 hash、检测器、实际 batch 和命令。固定 seed=42；相同 seed 不保证所有 GPU 算子逐位一致，容差按实测环境登记。
2. **源码核对**：2026-10-22 核对官方主分支。当前 demo 虽解析 `--batch_size`，实际 DataLoader 使用固定 8；不要声称传 1 就保证 batch=1。先选单人图，检查导出的真实 batch。附录不改官方仓库；Day1 包装 loader 注册观察 hooks 后调用原 `demo.main()`。
3. **输出与保存**：原 demo 主要写渲染和可选 OBJ，参数 NPZ 是附录观察脚本新增的导出；`demo_ok` 要求实际 forward、渲染和 OBJ 均存在，`no_detection` 与程序失败单列。官方启动流程可能下载缺少的公开权重/缓存；运行前先按本人执行环境准备资源。
4. **表示**：最终 `global_orient/body_pose` 是旋转矩阵时使用 `pose2rot=False`；head 内部的 6D 表示单独追踪。HMR2 的 SMPL wrapper 有 joint mapping/额外回归，不按输出数量猜 joint 语义；回放使用完全相同 wrapper。
5. **相机**：`pred_cam`、crop 的 `pred_cam_t`、转换后的 `cam_t_full` 分开保存；它们不直接给出 world trajectory。`pred_keypoints_2d` 是当前网络归一化 crop 投影；画原图叠加必须使用 full-image camera 或明确的预处理逆变换。
6. **几何容差**：归一化投影建议最大绝对差 <1e-5；旋转正交/det 残差 <1e-4；相机换算 <1e-4；CPU/GPU 同模型回放建议最大坐标差 <5e-4 model unit。均为执行起点，失败需定位误差，不能先放宽阈值。人体单位核实为 m 后才换算 mm。
7. **准确率边界**：参数回放证明接口一致；同一预测的重投影只检查实现。没有配对 GT 的真实图片不报告 MPJPE/PVE/Accel Error。合成指标结果与真实模型结果分别标注；图像干预后的输出变化不是精度下降的数值证据。
8. **实验习惯**：每次只改一个变量，先写 Prediction，再写 Observation；每天保留一个真实故障或明确标注的人为注入故障，并做同条件复测。未通过项写最小补做动作。

## Day 1：官方单图 Demo + 真实输出留存｜3h45m

### 1. 当日目标

完成 Core Demo① 的最小必跑路径：一张单人全身图产生实际渲染和 OBJ，保存能支持后续解释的参数与运行配置。将环境可用、检测到人、模型 forward 成功和渲染成功分成不同状态。

### 2. 前置条件

确认独立官方环境、授权模型和 checkpoint/config；查看 Week4 readiness 的真实状态。先用官方自带/本人已有合适图片，不从复杂视频开始。将附录全部脚本保存到同一 exp 目录，准备 `d1_demo_check.py`。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 依赖/版本 15m；入口/配置 15m；输入与预测 15m | 就绪状态、入口图、baseline 预测 |
| 00:45–02:00 | 75m | Run | 官方单图与日志 45m；参数/渲染核对 30m | 原图、OBJ、渲染、NPZ、manifest |
| 02:00–03:10 | 70m | Manipulate → Experiment | 同图复跑 25m；单因素输入对照 30m；故障定位 15m | 两次证据、输入/输出对应、根因候选 |
| 03:10–03:45 | 35m | Analyze → Research Log | 结果 15m；日志 10m；闭卷 10m | Day1 报告与 Gate |

### 4. 核心资源

- [官方 README](https://github.com/shubham-goel/4D-Humans)：安装资产、单图入口与保存项。
- [demo.py](https://github.com/shubham-goel/4D-Humans/blob/main/demo.py)：检测、人框、模型调用与 renderer；只读本日路径。
- [模型 loader](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/__init__.py)：checkpoint/config 与 SMPL 资产查找。

### 5. 执行步骤

#### A. Understand → Trace

15m 核实环境与文件，不将安装成功当推理成功；15m 从 `demo.main` 标出检测器、数据集、模型、renderer 和写文件位置；15m 记录图像尺寸/来源，预测遮挡与 crop 可能影响哪里。官方图像 Demo 中 `personid` 只是当前图的检测索引。

#### B. Run

45m 用附录 Day1 入口运行一张图片，记录实际主机与命令；该脚本调用官方 demo 并额外保存参数。30m 检查 `manifest.json` 状态、run.log、model_config.yaml、capture NPZ、renders 下 PNG/OBJ。看原图与叠图是否对应同一人，不能只看到程序退出就判成功。

`demo_ok` 只是本日运行证据；资产导入失败、未检出人、网络失败、渲染失败分别记录。官方内部转换/导出的 OBJ 可能含相机平移及 renderer 轴变换，不直接拿 OBJ 顶点和未加平移的网络 vertices 比较。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 唯一变化 | 需要记录 |
| --- | --- | --- | --- |
| 同图复跑 | 25m | 输入/配置不变，使用新结果目录 | 参数最大差、检测人数、设备与容差；不要求 PNG/OBJ 文件字节相同 |
| 输入对照 | 30m | 同一图仅添加一块遮挡；保留原图 | 检测人数、bbox/crop、姿态/表面变化；不报告无 GT 的 3D accuracy |
| 故障定位 | 15m | 从真实故障选择一处；没有则人为传错输入路径后恢复 | 失败发生在输入、模型还是渲染；恢复后同命令复跑 |

首次下载/编译超过预算时停止扩展样例，保存阻塞；模型 Gate 仍待补。已有资产就绪时优先把同图复跑做完整。

#### D. Analyze → Research Log

15m 写运行状态与文件索引；10m 日志；10m 闭卷从原图讲到渲染，指出检测器与人体网络的不同职责，并说明单图输出没有跨帧 ID。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验入口 | `exp/d1_demo_check.py` | 单图官方 demo + 观察 hooks；不改官方源码 |
| 运行证据 | `results/d1_baseline/`、`results/d1_repeat/` | 输入、manifest、日志、配置、captures、PNG、OBJ |
| 输出字典初稿 | `notes/output_schema.md` | key、shape、dtype、语义、单位/坐标待核实项 |
| 日志 | Research / Fail Log | 预测、实际状态、失败位置与复测路径 |

### 7. 验收标准（Gate）

- [ ] 官方单图实际 forward、渲染与 OBJ 都可定位，未把 import 当成功。
- [ ] 输入/config/checkpoint/commit/主机/实际 batch 有证据；复跑保留新目录。
- [ ] NPZ 为真实 forward 导出，和原 demo 的原生文件分清来源。
- [ ] 对照保留唯一改变量，能区分未检出人和人体恢复失败。
- [ ] 能解释当前图 detection index 与 tracking ID 的区别。

### 8. 受阻处理

显存不足先用单人图核实实际 batch；考虑较轻 detector 时必须单列配置变化，不能与主实验混报。模型 forward 成功但 OpenGL/render 失败，保存参数证据，Demo 渲染项待补。不得用随机 SMPL 或预制图替代官方运行。

### 9. 下一步

带真实参数与输出进入 Day2。先列清字段，再使用相同模型配置和 wrapper 回放，不重新做一次从 RGB 到参数的预测来冒充输出回放。

## Day 2：Output Parsing + SMPL 回放与参数干预｜3h45m

### 1. 当日目标

解释真实输出各字段是什么、来自哪一层、处于什么坐标约定；用相同人体层重建保存的 vertices/joints，并只改 beta0 观察体型效果。把模型回放一致性和人体恢复准确率分开。

### 2. 前置条件

Day1 有 `demo_ok`、真实 capture 与相同 checkpoint/config/SMPL 资产。新增 `d2_output_check.py`；继续复用 HMR2 自带的 SMPL wrapper，不另换 gender、joint regressor 或拓扑。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 字段 25m；人体接口/单位 20m | 完整 schema、参数到人体映射 |
| 00:45–02:00 | 75m | Run | 同层回放 45m；shape/数值/图核验 30m | replay JSON/NPZ、最大差与平均距离 |
| 02:00–03:10 | 70m | Manipulate → Experiment | beta0 干预 30m；矩阵/AA 故障 20m；joint/OBJ 边界 20m | 单变量结果、故障恢复、坐标说明 |
| 03:10–03:45 | 35m | Analyze → Research Log | 字典与结论 15m；日志 10m；闭卷 10m | Day2 报告与 Gate |

### 4. 核心资源

- [hmr2.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/hmr2.py)：`pred_smpl_params` 与 body forward。
- [smpl_wrapper.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/smpl_wrapper.py)：joint mapping、额外回归和基础人体层。
- [smplx body_models.py](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)：按需核对 rotation matrix / axis-angle 与 transl。

### 5. 执行步骤

#### A. Understand → Trace

25m 把参数、几何和相机分列；20m 查 wrapper/config，填写 model family/gender/betas 数量、joint mapper、是否额外 joints、坐标/单位、是否已经加 translation。不把某张图的 joint 总数写成所有配置的常量。

**最少字段字典**：`smpl_global_orient`、`smpl_body_pose`、`smpl_betas`、`pred_cam`、`pred_cam_t`、`pred_keypoints_3d`、`pred_vertices`、`pred_keypoints_2d`、`cam_t_full`、box 元数据、当前图 `personid`。pose 若没有独立同名 key，就说明其由 root/body 参数组成。

#### B. Run

45m 用 Day2 脚本加载相同 checkpoint，直接把已保存参数送入相同 `model.smpl(..., pose2rot=False)`；30m 比较 replay 与原 forward 的 shape、最大坐标差、平均 L2 距离，保存数值。先确认单位，再换算 mm。GPU 推理到 CPU 回放可能有浮点差，阈值超限要查配置/精度/资产差异。

不把回放平均顶点距离命名为真实图像 PVE；其参照是同一模型之前的输出，不是真实 GT。无 GT 时只报告 replay residual。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 唯一变化 | 预期与观察 |
| --- | --- | --- | --- |
| beta0 | 30m | 仅 beta0 +1，root/body rotations 与 camera 固定 | 保存位移与并列人体图；beta 并非统一 scale 开关，不保证某身体部位单调变大 |
| rotation 接口故障 | 20m | 在独立副本误用 matrix/AA 后恢复 | 比较报错/shape 与正确结果；不修改原参数，恢复后复测 |
| joint 与 OBJ | 20m | 对照同一输出的 joints/vertices/已渲染 OBJ | 核对 joint map；写清 OBJ 额外变换，不用数组大小猜坐标 |

#### D. Analyze → Research Log

15m 完成 `output_schema.md` 与回放结论；10m 日志；10m 闭卷回答“改变 global_orient 与改变 camera translation 有何不同”“为何两个 3D 数组不能只按 shape 比较”。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验 | `exp/d2_output_check.py` | 相同 wrapper 回放，beta0 单变量干预 |
| 结果 | `results/d2_replay/replay.json`、回放 NPZ | vertices/joints 残差、shape、阈值、参数干预位移 |
| 字段说明 | `notes/output_schema.md` | 每字段来源、表示、frame/单位、joint/拓扑协议 |
| 日志/图 | Research / Fail Log；参数对照图 | 只改 beta0；矩阵接口故障与同条件恢复 |

### 7. 验收标准（Gate）

- [ ] root/body rotation matrix 与 betas 的实际 shape 有记录，`pose2rot` 使用正确。
- [ ] vertices 与相同 mapper 的 joints 均完成回放对照，并解释残差单位。
- [ ] betas 单变量干预有保存参数、图和数值，未同时改变相机。
- [ ] 说明输出 joints、运动学 joints、mesh 与 OBJ 变换的区别。
- [ ] 回放一致性没有被表述为真实人体重建准确率。

### 8. 受阻处理

vertices 一致但 joints 不一致，优先检查 mapper、regressor 与 wrapper；两者都不同，查 checkpoint/config、SMPL gender/资产与 pose 表示。不要先拟合对齐来消除本应相同的回放误差。CPU 内存不足时只加载已有独立人体层，但需证明配置完全相同；未证明则待补。

### 9. 下一步

使用这组已解释的几何检查真实网络数据流与相机。先验证 crop 归一化投影，再验证 full-image camera，不直接把网络 2D 数组画在原图上。

## Day 3：真实 Forward Trace + Camera / Renderer 契约｜3h45m

### 1. 当日目标

把论文方法图接到实际函数和 tensor shape，完成 backbone/head/SMPL 的运行追踪。通过相机公式、有效深度和原图叠点定位投影错误，解释 renderer 和 tracker 在主路径中的位置。

### 2. 前置条件

Day1 的 hooks trace 和 capture 可读取，Day2 已解释参数/关节语义；准备 `d3_trace_check.py`。不需要 GT，不额外训练 backbone/head。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 方法图 15m；preprocessing/head 30m | 函数与张量映射、query/QKV 说明 |
| 00:45–02:00 | 75m | Run | trace/旋转 25m；两类 camera 30m；叠点 20m | trace-camera JSON、原图点投影 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 归一化错用 25m；crop/full 相机对照 25m；renderer/tracker 20m | 错误最小例、恢复后数值、分支图 |
| 03:10–03:45 | 35m | Analyze → Research Log | 路径笔记 15m；日志 10m；闭卷 10m | Day3 报告与 Gate |

### 4. 核心资源

- [ViTDetDataset](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/datasets/vitdet_dataset.py)：bbox、RGB、crop/normalization 与元数据。
- [SMPL head](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/heads/smpl_head.py)：decoder、query、迭代参数更新；按实际配置解释。
- [geometry.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/utils/geometry.py)、[renderer.py](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/utils/renderer.py)：透视与 crop-to-full。

### 5. 执行步骤

#### A. Understand → Trace

15m 用已读方法图写 backbone feature、decoder/query、SMPL regression 的职责；30m 对照实际代码，标出检测框→crop→标准化→backbone前宽度切片→head→SMPL→projection→renderer。hook 只覆盖模块边界，crop/renderer 要补源码位置和输入对象，不能声称已自动追完整仓库。

写实际 backbone 输入/输出、head 输入/输出与 SMPL 参数 shape，不用教学维度代填。继续定位 decoder 中 Q/K/V 的来源、`IEF_ITERS` 与 readout 更新；hooks 不自动解释这些语义。训练监督和推理主路分开画。

#### B. Run

25m 校验真实 trace 存在、旋转正交且 det≈+1；30m 用脚本独立重算 `pred_cam_t`、`cam_t_full` 和 crop normalized projection，记录有效/无效点与最大差；20m 用 visualization 的 overlay 模式把 full-image 投影画到同一原图，和官方 mesh 叠图并看。

脚本额外记录“crop normalized 点经 bbox affine”与 full-camera 投影的像素差。**这个诊断量不要求为零**：当前 crop-to-full 相机换算与透视深度、焦距设定存在近似关系，需结合公式说明，不能把差异直接归为代码 bug 或把 weak-perspective 换算当精确透视等价。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 干预/对照 | 需要解释 |
| --- | --- | --- | --- |
| normalized 当 pixel | 25m | 单独错用网络 normalized 2D 后恢复 full-image 投影 | 散点位置、轴范围、单位；正确图保留同一原图尺寸 |
| crop/full camera | 25m | 只替换相机平移，其余固定；另核对主点/焦距 | 为什么相同点不能直接交换两套 translation；避免双加平移 |
| renderer/tracker 分支 | 20m | 只读各入口与对象 | renderer 轴转换不回写网络参数；视频 ID 来自 tracker，不由单图 personid 保证 |

#### D. Analyze → Research Log

15m 写 `code_trace.md`，每个箭头标函数、实际 shape、frame 与用途；10m 日志；10m 闭卷说明“图像好看但 camera 协议错”如何被独立投影发现。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验 | `exp/d3_trace_check.py` | 检查真实 trace、旋转、两套相机与归一化投影 |
| 数值/图 | `results/d3_trace/`、`results/d3_overlay.png` | 残差/valid_count、原图投影、近似差异说明 |
| 路径笔记 | `notes/code_trace.md` | preprocessing 到 renderer/tracker；decoder QKV/更新说明 |
| 日志 | Research / Fail Log | 一个坐标/归一化故障与正确恢复 |

### 7. 验收标准（Gate）

- [ ] 主要 shape 来自实际运行，backbone 前切片与 head 更新有源码证据。
- [ ] root/body rotations 合法；crop 投影与网络返回值同口径一致。
- [ ] 两套 camera 使用正确，非法深度有数量；原图叠点有单位和轴说明。
- [ ] crop-affine 与 full-camera 的差异据公式分析，未预设必须为零。
- [ ] 能说明 renderer 与 tracker 的输入/输出，不把 camera-space 当 world-space。

### 8. 受阻处理

trace key 不匹配时先检查实际 commit/config；不凭旧接口硬适配。叠图异常先用一个点手算，再查 box_size、图像 W/H、主点和 translation。只缺渲染依赖时保留几何检查；缺真实 capture 时不得用 toy 标成本日网络 trace。

### 9. 下一步

相机协议固定后进入指标。真实图像先保留可视化观察；PVE/Accel 的实现正确性使用已知 GT 单独验证，不凭当前 demo 图生成虚假的 accuracy。

## Day 4：Metrics Protocol II｜PVE / Accel Error｜4h

### 1. 当日目标

实现并校验 PVE 与 Accel Error，与 Week4 的 MPJPE/raw-root-PA 放进同一张 cheat sheet。通过已知答案解释拓扑对应、mask、坐标、帧间隔和单位，以及“平滑但错误”的反例。

### 2. 前置条件

NumPy 即可执行附录 `d4_metrics_check.py`，不依赖 HMR2 或真实数据集。Week4 的 MPJPE/PA 实现与参考对照若还缺证据，登记待补；不因本日新代码通过就替旧项勾 PASS。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 四指标协议 20m；Accel 单位/窗口 25m | cheat sheet、公式与预测 |
| 00:45–02:00 | 75m | Run | PVE 25m；Accel 30m；参考 20m | 已知 GT 数值与独立对照 |
| 02:00–03:20 | 80m | Manipulate → Experiment | 刚体/单位 25m；时序/平滑 30m；缺失/退化 25m | 案例表、mask/gap 与故障恢复 |
| 03:20–04:00 | 40m | Analyze → Research Log | cheat sheet 20m；日志 10m；闭卷 10m | Day4 报告与 Gate |

### 4. 核心资源

- [VIBE eval_utils.py](https://github.com/mkocabas/VIBE/blob/master/lib/utils/eval_utils.py)：PVE 与三帧 acceleration-error 参考；不运行 VIBE 模型。
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)：mesh/keypoint convention 参考。
- Week4 的 MPJPE/PA 笔记与实现；HMR2 论文指标表按实际协议阅读，不移植数字到本人样例。

### 5. 执行步骤

#### A. Understand → Trace

20m 写四指标协议；25m 推三帧公式。`PVE=mean_v ||Vpred−Vgt||`，要求同拓扑同顶点顺序。`a_t=(X[t+1]−2X[t]+X[t−1])/Δt²`；Accel Error 对比预测和 GT 的加速度向量。缺 GT 时只能报告预测 acceleration magnitude，名称与 error 分开。

| 指标 | 比较对象 | 必须声明 | 不能据此证明 |
| --- | --- | --- | --- |
| MPJPE_raw / MPJPE_root | 对应 joints | root、joint subset、frame、mask、单位、reduction | 单个 joint 指标不能证明 mesh 表面正确 |
| PA-MPJPE | similarity 对齐后的对应 joints | 每样本 s/R/t、fit/eval 子集、反射/退化 | 绝对尺度、全局方向或全局位置正确 |
| PVE | 同拓扑对应 vertices | SMPL family/gender/顶点顺序、frame、alignment | SMPL 与 SMPL-X 不能仅按索引直接比较 |
| Accel Error | 匹配时刻/关节的二阶差分与 GT 之差 | fps/Δt、时间连续性、mask、frame、root 对齐、单位 | 平滑不等于姿态准确，也不等于 world 运动准确 |

附录按所有有效 vertex 或 joint-window 等权求均值，分别报告计数；不自动 root-center、PA 或改单位。参考 VIBE 使用每帧二阶差分，附录乘 fps² 后为物理时间单位：坐标 m 时得到 m/s²；换 mm 再乘 1000。对比前必须统一 fps/坐标/对齐和汇总口径。

#### B. Run

25m 检查 identity、已知平移和两侧同步单位转换的 PVE；30m 生成匀速 GT，加 `A*t²` 的已知误差轨迹，理论 Accel Error 为 `2||A||`，保存曲线与原始序列；20m 对照官方参考文件的两个函数，记录来源与 hash。

Day4 脚本默认执行解析答案检查；加 `--vibe-ref` 才执行官方参考对照。不传参考时报告 `official_reference=not_run`，这部分 Gate 仍待补。该选项从本人已保存的官方 eval_utils.py 提取指定函数定义，保留原参考文件与其许可证；不安装/执行 VIBE 模型。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 唯一变化/案例 | 必查现象 |
| --- | --- | --- | --- |
| 平移与单位 | 25m | Vpred=Vgt+(0.1,-0.2,0.3)m；两侧同步×1000 | 理论 PVE=√0.14m；单位数值转换正确，alignment 不隐式变化 |
| 时序与平滑 | 30m | 二次误差、恒定错误姿态、只改变 fps 解释口径 | 二次误差有解析答案；恒定错误可有 Accel Error≈0；说明 smoothness 不足以表示 accuracy |
| 缺失/帧断点 | 25m | 单帧 mask=False/NaN；frame_ids 有跳号；全无效 | 三帧窗口全部检查，禁止跨缺帧拼接；全无效报错，不能用 0 伪装 |

先排除无效值再作差，不能用 NaN×0。不同 topology 即使顶点数相同也可能错序；代码只能检查 shape/数值，语义对应需由元数据证明。若手动试平滑序列，保留原 GT/输入并同时看位置误差，不能只凭 acceleration 下降宣称改善。

#### D. Analyze → Research Log

20m 完成 metrics cheat sheet 与 `metrics.json` 解释；10m 日志；10m 闭卷回答“无 GT 视频能否报 Accel Error”“为什么 30fps 的 frame-difference 数字不能直接与 60fps 比较”。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验 | `exp/d4_metrics_check.py` | PVE、Accel、有效 mask/连续窗口、解析答案/可选参考 |
| 原始数据/结果 | `results/d4_metrics/` | synthetic 标注、GT/pred、fps、窗口数、数值与参考状态 |
| 指标表/图 | `notes/metrics_cheat_sheet.md`、`results/d4_accel.png` | 四指标协议与物理时间单位、真实/合成边界 |
| 日志 | Research / Fail Log | missing frame/fps/unit 故障与同条件恢复 |

### 7. 验收标准（Gate）

- [ ] PVE 的 identity/平移/单位案例符合已知答案，拓扑/顶点对应有说明。
- [ ] Accel 二次轨迹解析答案通过，frame² 与 s² 单位转换可解释。
- [ ] 缺失帧影响三个窗口、跳帧不拼接、全无效不返回 0。
- [ ] 与独立官方参考同协议对照已执行；缺文件时此项待补。
- [ ] 四指标 cheat sheet 完整，未把 synthetic 或 replay 当真实 demo accuracy。

### 8. 受阻处理

无人体模型仍可完成本日 synthetic 实现检查；不代表 Day1 已完成。参考文件无法取得时保留解析结果并标记独立参考待补。真实视频没有 GT 时不计算“Accel Error”；只做预测平滑性诊断，并声明坐标系/相机运动影响。

### 9. 下一步

携带可靠指标协议分析复杂输入。图像案例以实际检测/渲染和人工描述为主；只有匹配 GT 与时序协议齐全时才加入真实 accuracy。

## Day 5：Failure Analysis + Video Tracking 分支 + 全周验收｜3h45m

### 1. 当日目标

形成至少一份可追溯的图文失败分析，解释问题出在输入/检测、crop、人体恢复、相机/渲染还是 tracking。复核单图核心 Demo 的五项交付，并记录短视频分支的真实状态。

### 2. 前置条件

Day1–3 有合格 baseline 与协议；准备 baseline 加至少两种自选复杂输入，推荐遮挡、截断、侧身、异常姿态各取一例。短视频只有在 PHALP/视频依赖与主机就绪时运行；单图路线本身满足总表“图像或短视频”的 Demo 入口要求。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:40 | 40m | Understand → Trace | 样例/假设 20m；tracker/ID 20m | cases 清单、视频分支协议 |
| 00:40–01:55 | 75m | Run | 图像批量 45m；短视频或现有结果解析 30m | failure JSON/渲染；视频实际状态 |
| 01:55–02:40 | 45m | Manipulate → Experiment | 三类问题对照 30m；复测 15m | 图文对照、分层根因、最小验证 |
| 02:40–03:45 | 65m | Analyze → Research Log | 图文报告 30m；全周 Gate 20m；总结/闭卷 15m | failure analysis、weekly review、Week6 衔接 |

### 4. 核心资源

- [官方 tracking 说明](https://github.com/shubham-goel/4D-Humans#run-tracking-demo-on-videos)、[track.py](https://github.com/shubham-goel/4D-Humans/blob/main/track.py)：PHALP 集成、视频入口与实际保存格式。
- [PHALP](https://github.com/brjathu/PHALP)：仅查询当前版本的 tracklet 输出与 ID 约定，不把依赖安装扩成第二个长期复现。
- [Humans in 4D 论文/项目](https://shubham-goel.github.io/4dhumans/)：结合实际失败解释，区分作者结论和本周观察。

### 5. 执行步骤

#### A. Understand → Trace

20m 制作 cases.json：每例写 id、图片路径、condition、唯一 change、来源以及是否由 baseline 人工生成；20m 定位 track 入口，解释 single-frame person index、track ID、帧号/时间戳和 lost/reappeared 的关系。不同独立图片的检测索引不可跨图当作同一个人。

#### B. Run

45m 用 Day5 脚本保持 checkpoint/detector/config 不变运行 cases；记录未检出、运行失败、相机协议失败与成功。脚本不自动裁决“重建好/坏”，人工查看原图与官方渲染并填观察。

30m 若视频环境已就绪，选择 3–5s 单人短视频先跑官方 `python track.py video.source=/实际/视频路径`，记录 commit、实际配置、原始 fps/帧数、输出视频与 tracklet 文件；具体路径/键以实际版本为准。只读取本人产生/可信来源的 pickle，不为了查看未知文件执行反序列化。

若 PHALP 尚未就绪，这 30m 用于定位入口/预期字段并核对已有样例输出，状态写“视频未运行”，不能写 tracking PASS。若运行成功，抽检开头/中间/结尾及遮挡前后帧：同一人 ID 是否连续，是否丢检/跳变，pose/betas/camera 的实际对象 shape 是什么。ID 需由 tracklet 输出证明。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 对照 | 分析证据 |
| --- | --- | --- | --- |
| 三类分层问题 | 30m | baseline/遮挡/截断，或本人选定的等价复杂条件 | 分别看检测与 bbox、crop、姿态/mesh、相机叠图；写实际变化及替代解释 |
| 最小复测 | 15m | 恢复一个输入因素或复查一个 frame | 同模型同配置；不能同时换检测器和 crop 后归因单因素 |

侧身/异常姿态若来自不同照片，作为自然案例分析，不能声称只改变了视角/姿态。若干预后检出多人，脚本保留全部 personid；先人工建立对应关系，再谈人体输出变化。没有对应关系时不直接按数组位置比较。

#### D. Analyze → Research Log

30m 用 visualization 生成 input/render 图板，填写 `failure_analysis.md`：现象→候选层→证据→替代解释→最小下一实验；20m 按总表五项逐一核验路径；15m 周总结和闭卷解释。没有找到显著失效时如实写“在这些样例上未观察到”，保留覆盖局限，不人为编造失败。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验/样例 | `exp/d5_failure_check.py`、`inputs/cases.json` | baseline + 至少两类条件；来源/干预明确 |
| 图文结果 | `results/d5_cases/`、`results/d5_board.png`、`failure_analysis.md` | 全部真实运行状态、原图/渲染、分层解释与下一动作 |
| 视频记录 | `notes/video_tracking.md` | 就绪/已跑/未跑；实际 ID/shape/帧率/输出证据 |
| 周总结 | `weekly_review.md`；Research / Fail Log | 原表映射、Gate、缺口/工时、Week6 首动作 |

### 7. 验收标准（Gate）

- [ ] 至少一份实际图文分析覆盖 baseline 与自选复杂输入，未混报不同变量。
- [ ] 区分检测、crop、人体、相机/渲染和 tracking 问题；根因未证实则保留候选。
- [ ] 视频状态明确；只有实际 tracklet/视频证据齐全才写 tracking 通过。
- [ ] 无 GT 图片未报告真实 PVE/MPJPE；无 GT 视频未报告 Accel Error。
- [ ] 原表五项均有证据或具体待补动作，周总结与闭卷解释完成。

### 8. 受阻处理

Day5 脚本退出 0 仅说明 baseline 成功且批量流程完成，复杂样例失败仍需报告，整周 Gate 不会自动通过。若视频安装受阻，保留单图路线与视频待补项。若 baseline 自身仍失败，先修 baseline；复杂样例即使有渲染也不能替代必跑交付。

### 9. 下一步

Week6 按总表进入 Multi-HMR：重点从逐人检测/crop 回归转向 single-shot multi-person、SMPL-X 与 camera-space layout。携带本周输出字段、joint/拓扑约定、failure taxonomy；本周未完成的官方 Demo/人体接口优先补齐，不把单图 personid 扩写成跨帧或世界坐标能力。

## Fail Log / Research Log 模板

### Research Log

```markdown
## YYYY-MM-DD / Week 05 Day N / 主题
状态：未开始 / 进行中 / 已验证 / 待补
Question / Hypothesis：唯一改变量、固定项、预期理由。
Run：主机、解释器、commit/dirty、checkpoint/hash、config、实际命令。
Input：来源/hash/尺寸、sample/frame IDs、fps、bbox/person对应。
Trace：函数路径、实际 shape、训练/推理区别。
Conventions：model/crop/camera/world、单位、joint/vertex定义、pose表示。
Prediction：运行前写。
Observation：真实数字/报错、有效数、原始文件与图。
Metric protocol：GT来源、mask、alignment、时间连续性、reduction。
Analyze：证据支持的结论、替代解释、最大不确定性。
Gate：逐项路径；未完成原因、最小动作与预计补做工时。
Tomorrow first step：一个可直接执行的动作。
```

### Fail Log

```markdown
## W05-DN-F01 / 简短标题
来源：真实遇到 / 人为注入；状态：Open / Explained / Fixed / Retested
Context：输入、配置、frame/unit/shape、复现命令与实际失败阶段。
Expected / Observed：理论预期与真实观测分开。
Hypotheses：输入/检测/crop/人体/相机/渲染/tracker，如何逐层排除。
Minimal reproduction：最少图片/点/帧，固定项和唯一变化。
Root cause：有证据才确认；否则写尚待验证。
Fix / Retest：改动、同输入同协议结果、原始证据路径。
Lesson：下次在哪个接口加检查。
```

## Week 5 最终 Gate

| 项目 | 对应总表 | 通过证据 | 初始状态 |
| --- | --- | --- | --- |
| 官方 Core Demo① | D14-1 | 原图、实际 forward、渲染、mesh、环境/config/命令 | 待执行 |
| 输出解析与回放 | D14-2 | 实际字段、同人体层残差、参数干预、frame/joint/单位解释 | 待执行 |
| 核心代码追踪 | D14-3 | preprocessing→backbone→head→SMPL→renderer/tracker，实际 shape | 待执行 |
| Metrics Protocol II | D14-4 | PVE/Accel 解析答案、mask/fps/单位、独立参考、四指标表 | 待执行 |
| Failure analysis | D14-5 | 实际输入/渲染图板、分层分析、复测、结论边界 | 待执行 |
| 视频 tracking 分支 | 原表图像或视频入口；本计划追加项 | 真实视频/tracklet/ID/帧率/字段证据；未运行则明确说明 | 待核实环境 |
| 解释能力与研究记录 | 原 Gate | 闭卷讲来源/限制，五日日志与周总结可追溯 | 待验证 |

**通过规则**：总表五项必做交付与解释能力有实际证据后才记整周通过。单图路线可满足 Demo 入口；视频未运行时必须保留状态，不宣称具备已验证的 tracking 能力。代码语法/合成检查通过只证明相应检查脚本，不能替代真实模型 Demo。

## 本次依据与完成边界

依据实际 Excel 总表与完整 week4.md 编写；保留原计划文件与总表，不更改学习完成记录。2026-10-22 对照官方 README、demo、HMR2 forward、dataset、SMPL wrapper、renderer、track 和 VIBE 指标代码确认接口；执行时仍固定实际 commit/config。文末代码是本次新增的参考实验实现，真实模型与视频部分尚需在执行环境运行验证。

交付前已检查五天栏目/四阶段一致、时间连续且合计 19h、六个 Python 代码块语法和最终代码块位置；已运行 PVE/Accel 合成解析答案、VIBE 官方函数同口径对照、双相机合成投影与注入错误检查，以及 visualization 三种模式的合成样例绘图。以上均通过。没有运行官方模型推理、SMPL 参数回放、真实图像 stress test 或视频 tracking；这些 Gate 仍保持待执行。

## 实验代码使用说明

将下面六段分别保存到 `week5/exp/`，不要把多段拼成一个 Python 文件。假设从实际 `10weekplan` 根目录运行，把 `<REPO>`、`<CHECKPOINT>` 换成已准备好的绝对路径：

```bash
python week5/exp/d1_demo_check.py --repo "<REPO>" --checkpoint "<CHECKPOINT>" --image week5/inputs/baseline.jpg --out week5/results/d1_baseline
python week5/exp/d2_output_check.py --run week5/results/d1_baseline --out week5/results/d2_replay
python week5/exp/d3_trace_check.py --run week5/results/d1_baseline --out week5/results/d3_trace
python week5/exp/d4_metrics_check.py --out week5/results/d4_metrics
python week5/exp/d5_failure_check.py --repo "<REPO>" --checkpoint "<CHECKPOINT>" --cases week5/inputs/cases.json --out week5/results/d5_cases
python week5/exp/visualization.py overlay --run week5/results/d1_baseline --projection week5/results/d3_trace --out week5/results/d3_overlay.png
python week5/exp/visualization.py metrics --metrics week5/results/d4_metrics --out week5/results/d4_accel.png
python week5/exp/visualization.py failures --summary week5/results/d5_cases/failure_summary.json --out week5/results/d5_board.png
```

每次重跑换新的结果目录/图片文件名；附录主动拒绝覆盖旧结果。Day1 同图复跑或遮挡对照分别重复 Day1 命令并换 `--image/--out`。Day4 对照参考时在第一次运行命令增加 `--vibe-ref /实际/已保存的/eval_utils.py`；不要往已有结果目录补跑覆盖。

Day2 的 beta0 对照数据在其输出 NPZ 的 `vertices/variant_vertices` 中；可用现有 mesh 工具按相同相机/比例并列展示。最后 visualization.py 提供原图 joint overlay、合成 Accel 曲线和真实 input/render 图板三种模式，人体 mesh 渲染继续使用官方 demo/本人现有工具。

`cases.json` 示例（路径相对 cases.json 所在目录；图像需由本人准备；condition 建议使用英文以避免绘图库缺中文字体）：

```json
[
  {"id":"baseline", "image":"baseline.jpg", "condition":"clear full body", "change":"none"},
  {"id":"occlusion", "image":"occlusion.jpg", "condition":"occlusion", "change":"baseline with one declared occluder"},
  {"id":"truncation", "image":"truncation.jpg", "condition":"truncation", "change":"baseline with declared lower-body crop"}
]
```

Day5 会从各图片独立重跑检测；personid 不保证跨案例对应。最后在 `failure_analysis.md` 人工写实际观察，脚本生成的 `human_observation` 只是待填字段。没有视频 GT 时无需构造虚假的 Accel Error；视频额外实验依官方 track 入口另留日志。

## 每日 exp 完整代码

### d1_demo_check.py

```python
"""Run the official single-image demo and capture real tensors without editing it."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shape_tree(x):
    if hasattr(x, "shape"):
        return list(x.shape)
    if isinstance(x, dict):
        return {k: shape_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [shape_tree(v) for v in x]
    if hasattr(x, "vertices"):
        return {"vertices": shape_tree(x.vertices), "joints": shape_tree(x.joints)}
    return type(x).__name__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--detector", choices=["vitdet", "regnety"], default="vitdet")
    a = p.parse_args()
    repo, image, ckpt, out = [x.resolve() for x in
                              (a.repo, a.image, a.checkpoint, a.out)]
    if not (repo / "demo.py").is_file() or not image.is_file() or not ckpt.is_file():
        p.error("Missing demo.py, input image, or checkpoint; prepare assets first")
    if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        p.error("Use a jpg/jpeg/png image")
    out.mkdir(parents=True, exist_ok=False)  # Every run has a new evidence directory.
    inp = out / "input"
    inp.mkdir()
    copied = inp / ("source" + (".jpg" if image.suffix.lower() == ".jpeg"
                               else image.suffix.lower()))
    shutil.copy2(image, copied)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True)
    manifest = {"status": "started", "repo": str(repo),
                "commit": git.stdout.strip(), "git_status": dirty.stdout,
                "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
                "image": str(copied), "source_image": str(image),
                "input_sha256": sha256(copied), "python": sys.executable,
                "platform": platform.platform(), "detector": a.detector,
                "seed": 42, "captures": [], "trace": []}
    old_cwd, old_argv = Path.cwd(), sys.argv[:]
    handles = []
    code = 1
    try:
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        with open(out / "run.log", "w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import numpy as np
                import torch
                import demo
                torch.manual_seed(42)
                np.random.seed(42)
                manifest.update(torch=torch.__version__, cuda=torch.version.cuda,
                                device="cuda" if torch.cuda.is_available() else "cpu")
                original_loader = demo.load_hmr2

                def observed_loader(*args, **kwargs):
                    model, cfg = original_loader(*args, **kwargs)
                    (out / "model_config.yaml").write_text(cfg.dump())

                    def trace(name):
                        def hook(module, args, result):
                            manifest["trace"].append({"module": name,
                                "input": shape_tree(args), "output": shape_tree(result)})
                        return hook

                    for name in ("backbone", "smpl_head", "smpl"):
                        handles.append(getattr(model, name).register_forward_hook(trace(name)))

                    def capture(module, args, result):
                        batch = args[0]
                        data = {k: v.detach().cpu().numpy() for k, v in result.items()
                                if torch.is_tensor(v)}
                        for k, v in result["pred_smpl_params"].items():
                            data["smpl_" + k] = v.detach().cpu().numpy()
                        for k in ("box_center", "box_size", "img_size", "personid"):
                            data[k] = batch[k].detach().cpu().numpy()
                        focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * batch["img_size"].max()
                        full_t = demo.cam_crop_to_full(result["pred_cam"],
                            batch["box_center"].float(), batch["box_size"].float(),
                            batch["img_size"].float(), focal)
                        data.update(cam_t_full=full_t.detach().cpu().numpy(),
                                    full_focal=np.asarray(float(focal)),
                                    image_size=np.asarray(cfg.MODEL.IMAGE_SIZE),
                                    faces=np.asarray(module.smpl.faces))
                        fn = f"capture_{len(manifest['captures']):03d}.npz"
                        np.savez_compressed(out / fn, **data)
                        manifest["captures"].append(fn)
                        manifest["trace"].append({"module": "hmr2",
                            "input": shape_tree(batch), "output": shape_tree(result)})
                    handles.append(model.register_forward_hook(capture))
                    return model, cfg

                demo.load_hmr2 = observed_loader
                sys.argv = ["demo.py", "--checkpoint", str(ckpt), "--img_folder", str(inp),
                            "--out_folder", str(out / "renders"), "--save_mesh",
                            "--full_frame", "--detector", a.detector]
                manifest["argv"] = sys.argv[:]
                demo.main()
                pngs = list((out / "renders").glob("*.png"))
                objs = list((out / "renders").glob("*.obj"))
                if not manifest["captures"]:
                    manifest["status"], code = "no_detection", 2
                elif not pngs or not objs:
                    raise RuntimeError("Forward ran, but render/mesh evidence is missing")
                else:
                    manifest["status"], code = "demo_ok", 0
                manifest.update(render_count=len(pngs), mesh_count=len(objs))
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
    finally:
        for h in handles:
            h.remove()
        os.chdir(old_cwd)
        sys.argv = old_argv
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out / "manifest.json", manifest["status"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
```

### d2_output_check.py

```python
"""Replay the exact body layer; replay residual is not reconstruction accuracy."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--atol", type=float, default=5e-4)
    a = p.parse_args()
    run = a.run.resolve()
    m = json.loads((run / "manifest.json").read_text())
    if m["status"] != "demo_ok":
        p.error("Day 1 must contain real successful inference evidence")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    os.chdir(m["repo"])
    sys.path.insert(0, m["repo"])
    from hmr2.models import load_hmr2
    model, cfg = load_hmr2(m["checkpoint"])
    model = model.cpu().eval()
    reports = []
    with torch.no_grad():
        for name in m["captures"]:
            with np.load(run / name, allow_pickle=False) as z:
                params = {k: torch.from_numpy(z["smpl_" + k]).float()
                          for k in ("global_orient", "body_pose", "betas")}
                body = model.smpl(**params, pose2rot=False)
                row = {"capture": name, "params": {k: list(v.shape) for k, v in params.items()}}
                for key, value in (("vertices", body.vertices), ("joints", body.joints)):
                    ref = z["pred_vertices" if key == "vertices" else "pred_keypoints_3d"]
                    pred = value.cpu().numpy()
                    if pred.shape != ref.shape or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid {key} shape/values")
                    row[key + "_max_abs_model_unit"] = float(np.max(np.abs(pred - ref)))
                    row[key + "_mean_l2_model_unit"] = float(np.linalg.norm(pred - ref, axis=-1).mean())
                changed = {k: v.clone() for k, v in params.items()}
                changed["betas"][:, 0] += 1.0
                variant = model.smpl(**changed, pose2rot=False)
                row["beta0_plus1_mean_vertex_displacement_model_unit"] = float(
                    torch.linalg.vector_norm(variant.vertices - body.vertices, dim=-1).mean())
                row["replay_ok"] = all(row[k + "_max_abs_model_unit"] <= a.atol
                                       for k in ("vertices", "joints"))
                row["atol_model_unit"] = a.atol
                np.savez_compressed(out / name, vertices=body.vertices.numpy(),
                    joints=body.joints.numpy(), variant_vertices=variant.vertices.numpy(),
                    faces=z["faces"], personid=z["personid"])
                reports.append(row)
    result = {"scope": "same-output replay, not GT accuracy", "device": "cpu",
              "joint_mapping": "same HMR2 SMPL wrapper; inspect cfg and smpl_wrapper.py",
              "runs": reports}
    (out / "replay.json").write_text(json.dumps(result, indent=2))
    print(out / "replay.json")
    return 0 if reports and all(r["replay_ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### d3_trace_check.py

```python
"""Validate actual trace, rotation matrices and the two camera contracts."""
import argparse
import json
from pathlib import Path
import numpy as np


def project(points, translation, focal, center):
    x = points + translation[:, None, :]
    valid = np.isfinite(x).all(-1) & (x[..., 2] > 1e-6)
    uv = np.full(x.shape[:-1] + (2,), np.nan)
    b, j = np.nonzero(valid)
    uv[b, j] = x[b, j, :2] / x[b, j, 2, None] * focal[b] + center[b]
    return uv, valid


def analyze(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    m = json.loads((run / "manifest.json").read_text())
    if m["status"] != "demo_ok":
        raise ValueError("Need successful Day 1 capture")
    out.mkdir(parents=True, exist_ok=False)
    modules = {r["module"] for r in m["trace"]}
    if not {"backbone", "smpl_head", "smpl", "hmr2"}.issubset(modules):
        raise ValueError("Missing real forward-hook trace")
    rows = []
    for name in m["captures"]:
        with np.load(run / name, allow_pickle=False) as z:
            cam = z["pred_cam"].astype(float)
            pts = z["pred_keypoints_3d"].astype(float)
            b = len(cam)
            size = float(z["image_size"])
            if not np.isfinite(cam).all() or np.any(cam[:, 0] <= 0):
                raise ValueError("Invalid camera scale")
            f = z["focal_length"].astype(float)
            expected_crop = np.column_stack((cam[:, 1], cam[:, 2],
                2 * f[:, 0] / (size * cam[:, 0] + 1e-9)))
            bs = z["box_size"].reshape(-1) * cam[:, 0] + 1e-9
            full_f = float(z["full_focal"])
            expected_full = np.column_stack((
                2 * (z["box_center"][:, 0] - z["img_size"][:, 0] / 2) / bs + cam[:, 1],
                2 * (z["box_center"][:, 1] - z["img_size"][:, 1] / 2) / bs + cam[:, 2],
                2 * full_f / bs))
            uv_norm, valid = project(pts, z["pred_cam_t"], f / size, np.zeros((b, 2)))
            if not valid.any():
                raise ValueError("No positive-depth finite joints")
            residual = float(np.max(np.abs(uv_norm[valid] - z["pred_keypoints_2d"][valid])))
            uv_full, full_valid = project(pts, z["cam_t_full"],
                np.full((b, 2), full_f), z["img_size"] / 2)
            rotations = np.concatenate((z["smpl_global_orient"].reshape(-1, 3, 3),
                                        z["smpl_body_pose"].reshape(-1, 3, 3)))
            orth = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
            det = float(np.max(np.abs(np.linalg.det(rotations) - 1)))
            crop_err = float(np.max(np.abs(expected_crop - z["pred_cam_t"])))
            full_err = float(np.max(np.abs(expected_full - z["cam_t_full"])))
            row = {"capture": name, "batch_size": b, "valid_crop_joints": int(valid.sum()),
                "invalid_crop_joints": int((~valid).sum()), "rotation_orth_max": orth,
                "rotation_det_max_error": det, "crop_camera_max_abs": crop_err,
                "full_camera_max_abs": full_err, "projection_max_abs_normalized": residual,
                "ok": max(orth, det) < 1e-4 and max(crop_err, full_err) < 1e-4
                      and residual < 1e-5}
            # This affine comparison is diagnostic: camera conversion need not make it zero.
            affine_uv = uv_norm * z["box_size"].reshape(b, 1, 1) + z["box_center"][:, None, :]
            both = valid & full_valid
            row["crop_affine_vs_full_mean_px"] = (float(np.linalg.norm(
                affine_uv[both] - uv_full[both], axis=-1).mean()) if both.any() else None)
            np.savez_compressed(out / name, uv=uv_full, valid=full_valid,
                                joints_camera=pts + z["cam_t_full"][:, None, :],
                                personid=z["personid"])
            rows.append(row)
    (out / "trace_camera.json").write_text(json.dumps({"trace": m["trace"], "checks": rows}, indent=2))
    return bool(rows) and all(r["ok"] for r in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    ok = analyze(a.run, a.out)
    print(a.out / "trace_camera.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### d4_metrics_check.py

```python
"""PVE and acceleration-error checks using synthetic, known ground truth."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np


def pve(pred, gt, mask=None):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if pred.shape != gt.shape or pred.ndim < 2 or pred.shape[-1] != 3:
        raise ValueError("Need matching [...,V,3] arrays and matching vertex ordering")
    valid = np.ones(pred.shape[:-1], bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != pred.shape[:-1] or not valid.any():
        raise ValueError("Invalid/empty vertex mask")
    pp, qq = pred[valid], gt[valid]
    if not (np.isfinite(pp).all() and np.isfinite(qq).all()):
        raise ValueError("Nonfinite unmasked vertices")
    return float(np.linalg.norm(pp - qq, axis=-1).mean())


def acceleration_error(pred, gt, fps, valid=None, frame_ids=None):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3 or len(pred) < 3:
        raise ValueError("Need matching [T,J,3], T>=3")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    mask = np.ones(pred.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    if mask.shape != pred.shape[:2]:
        raise ValueError("Need [T,J] visibility mask")
    ids = np.arange(len(pred)) if frame_ids is None else np.asarray(frame_ids)
    if ids.shape != (len(pred),) or np.any(np.diff(ids) <= 0):
        raise ValueError("Need strictly increasing frame IDs")
    triple = mask[:-2] & mask[1:-1] & mask[2:]
    consecutive = (np.diff(ids)[:-1] == 1) & (np.diff(ids)[1:] == 1)
    triple &= consecutive[:, None]
    t, j = np.nonzero(triple)
    if not len(t):
        raise ValueError("No valid consecutive three-frame windows")
    pp = np.stack([pred[t + k, j] for k in range(3)])
    qq = np.stack([gt[t + k, j] for k in range(3)])
    if not (np.isfinite(pp).all() and np.isfinite(qq).all()):
        raise ValueError("Nonfinite unmasked trajectory")
    delta = (pp[2] - 2 * pp[1] + pp[0]) - (qq[2] - 2 * qq[1] + qq[0])
    errors = np.linalg.norm(delta, axis=-1) * fps ** 2
    per_joint = np.full((len(pred) - 2, pred.shape[1]), np.nan)
    per_joint[t, j] = errors
    return float(errors.mean()), len(errors), per_joint


def reference_functions(path):
    # Extract only these official pure-function definitions, avoiding VIBE's torch imports.
    source = Path(path).read_text()
    wanted = {"compute_error_accel", "compute_error_verts"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError("Reference functions not found; check source version")
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns, hashlib.sha256(source.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--vibe-ref", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    gt = rng.normal(size=(2, 32, 3))
    translated = gt + np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(pve(gt, gt), 0, atol=1e-12)
    np.testing.assert_allclose(pve(translated, gt), np.sqrt(0.14), atol=1e-12)
    np.testing.assert_allclose(pve(translated * 1000, gt * 1000),
                               1000 * pve(translated, gt), atol=1e-9)
    valid_v = np.ones(gt.shape[:-1], bool)
    valid_v[:, 0] = False
    missing = translated.copy()
    missing[:, 0] = np.nan
    np.testing.assert_allclose(pve(missing, gt, valid_v), np.sqrt(0.14), atol=1e-12)
    fps, T, J = 30.0, 21, 8
    time = np.arange(T) / fps
    joints_gt = np.broadcast_to(rng.normal(size=(1, J, 3)), (T, J, 3)).copy()
    joints_gt += time[:, None, None] * np.array([0.1, 0.2, 0.0])
    A = np.array([0.2, -0.1, 0.3])  # error trajectory A*t^2; acceleration error = 2*||A||
    joints_pred = joints_gt + time[:, None, None] ** 2 * A
    ae, count, per_joint = acceleration_error(joints_pred, joints_gt, fps)
    np.testing.assert_allclose(ae, 2 * np.linalg.norm(A), atol=1e-10)
    constant_error, _, _ = acceleration_error(joints_gt + 0.5, joints_gt, fps)
    np.testing.assert_allclose(constant_error, 0, atol=1e-10)
    mask = np.ones((T, J), bool)
    mask[10] = False
    broken = joints_pred.copy()
    broken[10] = np.nan
    masked_ae, n, _ = acceleration_error(broken, joints_gt, fps, mask)
    if n != (T - 5) * J:
        raise AssertionError("Missing frame must invalidate all three touching windows")
    np.testing.assert_allclose(masked_ae, ae, atol=1e-10)
    ids = np.arange(T)
    ids[10:] += 1
    _, gap_count, _ = acceleration_error(joints_pred, joints_gt, fps, frame_ids=ids)
    if gap_count != (T - 4) * J:
        raise AssertionError("Frame gap windows were joined incorrectly")
    rejected = []
    for name, operation in [
        ("empty_vertices", lambda: pve(gt, gt, np.zeros(gt.shape[:-1], bool))),
        ("nan_vertices", lambda: pve(missing, gt)),
        ("topology_shape", lambda: pve(gt[:, :-1], gt)),
        ("empty_time", lambda: acceleration_error(joints_pred, joints_gt, fps, np.zeros((T, J), bool))),
        ("bad_fps", lambda: acceleration_error(joints_pred, joints_gt, 0)),
    ]:
        try:
            operation()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("Failed to reject " + name)
    report = {"data": "synthetic known GT; not HMR2 benchmark", "seed": 42,
              "pve_translation_m": pve(translated, gt), "accel_error_m_s2": ae,
              "accel_error_m_frame2": ae / fps ** 2, "valid_joint_windows": count,
              "missing_frame_joint_windows": n, "gap_joint_windows": gap_count,
              "constant_wrong_pose_accel_error_m_s2": constant_error,
              "rejected": rejected, "official_reference": "not_run"}
    if a.vibe_ref:
        ns, digest = reference_functions(a.vibe_ref)
        reference_pve = float(ns["compute_error_verts"](translated, target_verts=gt).mean())
        reference_accel = float(ns["compute_error_accel"](joints_gt, joints_pred).mean())
        np.testing.assert_allclose(reference_pve, pve(translated, gt), atol=1e-12)
        np.testing.assert_allclose(reference_accel * fps ** 2, ae, atol=1e-10)
        report.update(official_reference="passed_same_protocol", reference_sha256=digest)
    np.savez_compressed(a.out / "metric_inputs.npz", vertices_gt=gt, vertices_pred=translated,
                        joints_gt=joints_gt, joints_pred=joints_pred, fps=fps,
                        accel_error_per_joint=per_joint)
    (a.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(a.out / "metrics.json")


if __name__ == "__main__":
    main()
```

### d5_failure_check.py

```python
"""Run declared stress cases. Output drift has no accuracy meaning without GT."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from d3_trace_check import analyze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--detector", choices=["vitdet", "regnety"], default="vitdet")
    a = p.parse_args()
    cases_file = a.cases.resolve()
    cases = json.loads(cases_file.read_text())
    if not isinstance(cases, list) or len(cases) < 3:
        p.error("Declare baseline and at least two stress cases")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        p.error("Case IDs must be unique safe directory names")
    if "baseline" not in ids:
        p.error("Need a baseline case")
    prepared = []
    for c in cases:
        if not c.get("condition") or not c.get("change"):
            p.error("Each case needs condition and change")
        image = Path(c["image"])
        image = image if image.is_absolute() else cases_file.parent / image
        if not image.is_file():
            p.error("Missing image: " + str(image))
        prepared.append((c, image.resolve()))
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for c, image in prepared:
        run = out / c["id"]
        cmd = [sys.executable, str(Path(__file__).with_name("d1_demo_check.py")),
               "--repo", str(a.repo.resolve()), "--checkpoint", str(a.checkpoint.resolve()),
               "--image", str(image), "--out", str(run), "--detector", a.detector]
        result = subprocess.run(cmd, capture_output=True, text=True)
        row = dict(c, run=str(run), returncode=result.returncode, stderr=result.stderr[-2000:])
        manifest_file = run / "manifest.json"
        if not manifest_file.is_file():
            row["status"] = "launcher_failed"
        else:
            m = json.loads(manifest_file.read_text())
            row["status"] = m["status"]
            people = []
            for name in m["captures"]:
                with np.load(run / name, allow_pickle=False) as z:
                    people.extend(z["personid"].tolist())
            row["detected_people"] = len(people)
            if m["status"] == "demo_ok":
                try:
                    row["camera_contract_ok"] = analyze(run, run / "projection")
                except (ValueError, KeyError) as error:
                    row["camera_contract_ok"] = False
                    row["camera_error"] = str(error)
        row["human_observation"] = "待本人查看原图与渲染后填写"
        row["accuracy_3d"] = "unavailable_without_GT"
        rows.append(row)
    (out / "failure_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(out / "failure_summary.json")
    # A no-detection stress case is a recorded outcome; baseline must still work.
    base = next(r for r in rows if r["id"] == "baseline")
    return 0 if base["status"] == "demo_ok" and base.get("camera_contract_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### visualization.py

```python
"""Visualize saved evidence; never invent a skeleton topology or an accuracy score."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def overlay(run, projection, output):
    m = json.loads((run / "manifest.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 8))
    with Image.open(m["image"]) as im:
        ax.imshow(im.convert("RGB"))
        w, h = im.size
    for fn in sorted(projection.glob("capture_*.npz")):
        with np.load(fn, allow_pickle=False) as z:
            for i, pid in enumerate(z["personid"]):
                uv = z["uv"][i, z["valid"][i]]
                ax.scatter(uv[:, 0], uv[:, 1], s=12, label=f"detection {int(pid)}")
    if not list(projection.glob("capture_*.npz")):
        raise ValueError("No projection data; run Day 3 first")
    ax.set(xlim=(0, w), ylim=(h, 0), title="Full-image projected joints (not GT)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def metric_plot(folder, output):
    with np.load(folder / "metric_inputs.npz", allow_pickle=False) as z:
        fps = float(z["fps"])
        times = np.arange(1, len(z["joints_gt"]) - 1) / fps
        error = z["accel_error_per_joint"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, error, marker="o")
    ax.set(xlabel="Time (s)", ylabel="Acceleration error (m/s^2)",
           title="Synthetic known-GT test; not HMR2 benchmark", ylim=(0, max(error) * 1.2 + 1e-9))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def failure_board(summary, output):
    rows = json.loads(summary.read_text())
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), squeeze=False)
    for pair, row in zip(axes, rows):
        run = Path(row["run"])
        mf = run / "manifest.json"
        m = json.loads(mf.read_text()) if mf.exists() else {}
        source = m.get("image")
        rendered = run / "renders" / "source_all.png"
        for ax, fn, label in zip(pair, [Path(source) if source else None, rendered], ["input", "render"]):
            if fn is not None and fn.is_file():
                with Image.open(fn) as im:
                    ax.imshow(im.convert("RGB"))
            else:
                ax.text(0.5, 0.5, row["status"], ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{row['id']} / {row['condition']} / {label}", fontsize=10, wrap=True)
            ax.axis("off")
    fig.suptitle("Declared stress cases: inspect detection, crop and body reconstruction", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["overlay", "metrics", "failures"])
    p.add_argument("--run", type=Path)
    p.add_argument("--projection", type=Path)
    p.add_argument("--metrics", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("Choose a new output filename")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "overlay":
        if not a.run or not a.projection:
            p.error("overlay needs --run and --projection")
        overlay(a.run, a.projection, a.out)
    elif a.mode == "metrics":
        if not a.metrics:
            p.error("metrics needs --metrics")
        metric_plot(a.metrics, a.out)
    else:
        if not a.summary:
            p.error("failures needs --summary")
        failure_board(a.summary, a.out)
    print(a.out)


if __name__ == "__main__":
    main()
```
