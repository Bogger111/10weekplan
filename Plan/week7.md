# Week 7 — Camera Geometry & SfM: From 3D Points to World / Camera / Image

> 3D/4D Human 10 周科研训练 · Week 7 执行计划 · 2026-11-02  
> 5 天，合计 **19h 专注学习**；Day 1/2/4/5 各 3h45m，Day 3 为 4h，休息另计。  
> 参照 [Week 6](week6.md) 的结构、独立实验与验收方式，落实总表第七周；不改写总表或已有周计划。  
> 文档状态：总表与官方资料已核对；以下代码、手机采集、COLMAP 重建与学习 Gate 均待本人执行。

## 1. Week 7 Overview

### 目标与总表对应关系

从“读懂同一帧的 camera-space 人体输出”推进到“能判断坐标变换方向，用多视图恢复几何，并解释相机位姿、重投影与尺度的边界”。本周的核心不是再运行一次人体重建 Demo，而是亲自建立 Week 8 所需的几何接口和证据。

依据 [10 周总计划 Excel](00_plan/10周研究计划_曾奕博.xlsx) 的 **`10周计划!B16:D16`**。编写时已将 `B4:D4、B6:D7、B15:D18` 共 21 个单元格与 [Numbers 原表](00_plan/10周研究计划_曾奕博.numbers) 比对，一致。下列四项来自原表；五天拆分、实验规模、文件名和候选容差是本计划的执行建议。

| 总表 Week 7 要求 | 本周落实 | 验收证据 |
| --- | --- | --- |
| 实现 project / unproject / SE(3) inverse，做 round-trip | Day 1 完成 E1，明确深度、变换方向和有效点策略 | 本人代码、手算点、非单位旋转、正常与反例数值 |
| 实现两视图 triangulation，分析 baseline / focal length / 2D noise | Day 2 建立 E/F 与两视图约定；Day 3 完成 E3 和三组单因素实验 | DLT、独立对照、原始 CSV、误差曲线及无效样本数 |
| 手机拍 20–40 张图，完成一次 COLMAP sparse reconstruction | Day 1 采集与筛图；Day 4 在自己的数据上运行稀疏管线 | 原始照片清单、配置、数据库、稀疏模型与日志 |
| 导出 camera poses / sparse points，可视化 camera trajectory | Day 5 完成 E5，检查重投影并画相机中心、朝向和点云 | 位姿表、点表、带图像顺序的可视化与抽样残差 |
| 给定 point / pose / human root 能判断坐标系并变换 | 五天均使用显式 frame；Day 5 闭卷综合 Gate | 从定义推导变换、判断缺失信息，不凭变量名猜含义 |

主题覆盖 pinhole、K/R/t、SE(3)、projection/unprojection、E/F、triangulation、reprojection error、SfM、bundle adjustment（BA）。COLMAP 只做 **sparse**；本周不展开 dense MVS、完整 SLAM、TRAM/Human3R 配置或训练，也不额外复现一个新的人体模型。

### 复现层级与完成边界

| 层级 | 本周要求 | 完成边界 |
| --- | --- | --- |
| Algorithm Implementation | 本人实现 E1、E2 的几何核心、E3 的两视图 DLT | 不能只调用 OpenCV 后声称自己实现了三角化 |
| Pipeline Reproduction | 在本人手机 20–40 张照片上完成一次 COLMAP sparse | 官方样例只作排错对照，不能替代手机数据交付 |
| Controlled Experiment | 合成 GT 下分别改变基线、焦距、像素噪声 | 不能同时改变多个变量，再把差异归因于其中一个 |
| Method Understanding | 解释 SfM 的初始化、注册、三角化与 BA；审核导出协议 | 不要求从零实现完整 SfM 或 BA 求解器 |
| Benchmark / Training | 本周不要求 | 自选小场景没有 GT 时，不报告相机轨迹真值误差或训练复现 |

### 已有基础及证据等级

[Week 4](week4.md)、[Week 5](week5.md)、[Week 6](week6.md) 已安排投影、坐标、人体参数与 camera-space 输出。Week 6 还记录了本人此前提供的 Week 5 进度；这些历史记录本次没有重新运行。**当前未收到 Week 6 的新实验记录，不能把计划中的 Gate 当成已通过。**

| 已有材料 / 当前未知 | 本周处理 |
| --- | --- |
| Week 3–5 已安排 SO(3)/SE(3)、SMPL、投影与指标 | Day 1 用短诊断和新反例承接，不重新讲整天旋转或训练 |
| Week 6 采用 Multi-HMR 2 / Anny；K、pelvis 与 mesh 是其相机协议 | 有本人实测结果就取一项作接口对照；无结果也不阻塞本周合成几何 |
| Week 6 的 K/跨帧身份/有效点测试是否完成，当前无法确认 | 开工列“已有证据 / 待补 / 不适用”，不预填完成状态 |
| 是否已有 COLMAP / PyCOLMAP、能否在当前 WSL 运行，尚未实测 | 设安装排查预算；CPU 稀疏路线可用作候选，不承诺 GPU 兼容 |
| 手机照片、相机标定、已知尺度参照尚未提供 | 本人本周采集；默认重建单位未知，不要求提前下载数据集 |

### 教学与协作约定

沿用 **Understand → Run → Trace → Implement → Debug → Controlled Experiment → Evaluate → Research Log**。本人先预测、写函数、构造反例并解释结果；Research Agent 提供 Hint、检查定义、定位错误和审核证据。默认不一次性交付整套参考实现。

总表的“能完成变换”也包含识别信息不足：只有像素没有深度、只有 root-relative 人体没有全局平移、两个系统尺度不一致时，应说明还缺什么，不输出看起来合理的世界轨迹。

### 五天预算

| Day | Understand / Trace | Run（实际运行与测量） | Independent Experiment / Debug | Analyze / Log | 合计 |
| --- | --- | --- | --- | --- | --- |
| 1：坐标协议、投影与采集 | 45m | 60m | 85m | 35m | 225m |
| 2：E/F 与两视图几何 | 55m | 35m | 100m | 35m | 225m |
| 3：三角化与误差实验 | 45m | 45m | 110m | 40m | 240m |
| 4：手机照片与 COLMAP sparse | 45m | 45m | 100m | 35m | 225m |
| 5：导出、重投影与 BA | 25m | 50m | 110m | 40m | 225m |
| 总计 | 215m | 235m | 505m | 185m | **1140m = 19h** |

Run + 独立实验/调试共 **740m，约 65%**。这是预算，不是已投入工时；采集、数据整理和环境排查均包含在相应时段内。较长的重建任务保存日志后可等待机器运行；不把等待时长当成专注学习时长。超过预算先减可选任务，硬任务受阻则如实结转。

## 2. Method Understanding

本周以课程几何和 SfM 方法为主，不强行增加一篇必须完整复现的论文。主读 [CS231A 课程目录](https://web.stanford.edu/class/cs231a/course_notes.html) 中的 [Camera Models](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)、[Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)、[Stereo Systems and Structure from Motion](https://web.stanford.edu/class/cs231a/course_notes/04-stereo-systems.pdf)。定向读投影、E/F、线性三角化、重投影最小化与 SfM；affine factorization 等推导作为选读，不挤占实验。

| 阅读维度 | 本人需要回答的问题 | 本周判断 / 证据 |
| --- | --- | --- |
| Problem / Previous limitation | 单张图为何不能仅靠投影确定每点深度？多视图新增了什么约束？ | 射线、视差和退化例；区别 learned prior 与几何可辨识性 |
| Core insight / Method | E/F 怎样约束对应点？三角化与相机位姿估计谁已知、谁未知？ | 自己画两视图图，写清依赖顺序 |
| Why it works | 基线和观测角怎样影响深度敏感性？为何匹配点多仍可能失败？ | E3 数值曲线与纯旋转/错配反例 |
| Training | 经典几何链是否在训练神经网络？ | 本周数值优化是几何参数估计，不称模型训练 |
| Inference / Optimization | COLMAP 从照片到相机和点的步骤是什么？BA 调整什么？ | 实际阶段日志、数据文件与目标函数 |
| Metrics / Results | 像素重投影误差、三维误差、注册率分别需要什么？ | 合成 GT 和手机无 GT 结果分开 |
| Ablation | 怎样分别检验 baseline、focal、noise？ | 固定点集与随机数，三组单因素实验 |
| Limitations / Connections | 动态人体、纯旋转、尺度不定怎样影响后续 world-space HMR？ | 失败案例、尺度反例和 Week 8 待解决接口 |

这里没有新增方法贡献需要宣称。**本周真正要掌握的能力**是把“定义 → 实现 → 可检查的实验 → 结论边界”接起来。课程推导、工具输出和本人提出的解释分开记录；COLMAP 能生成好看的点云不等于相机位姿或真实尺度已被验证。

本人完成 `notes/method_map.md`：每一维写“自己的解释 + 资源位置 + 一项代码/数值证据或待验证项 + 一个局限”。不抄整份讲义，不以阅读页数过 Gate。

## 3. Prerequisites & Environment

### 资源和工作区边界

沿用项目资料中的执行设备信息：WSL2 Ubuntu 24.04、RTX 5070 Laptop 8 GB；**本次未连接 WSL，未运行 COLMAP，也未验证版本、速度或设备兼容性**。Mac 保存资料，WSL 保存本人代码与实验；独立环境不改动已有 HMR2 / Multi-HMR 2 环境。

```text
Mac: /Users/mac/CCFA_maker/01_10week_plan/week7.md

WSL（本人按需创建，不表示目录已经存在）:
~/10weekplan/week7/
├── exp/                         # 本人编写的 E1–E5
├── inputs/
│   ├── phone_originals/         # 原始照片或指向原文件的记录
│   ├── phone_working/           # 统一方向/格式后的工作副本
│   └── manifest.csv
├── results/<run_id>/
│   ├── config/                 # 版本、参数、照片清单、seed
│   ├── colmap/                 # 独立 database.db、sparse/<component>
│   ├── export/                 # 原生模型导出及本人 CSV/图
│   └── logs/
├── notes/                      # protocol、method_map、failures、weekly_review
└── research_log.md
```

不在同步 Project 的 `sources/` 写文件。官方依赖源码若确需下载，单独放 `~/research-repos/`；本周不要求为了读代码而编译 COLMAP。提交前检查 Git 根目录和暂存文件，只提交本人代码、轻量配置/记录与必要图表，不把数据库、全量照片或大型重建缓存随手加入代码仓库。

### Day 1 开工检查

1. 先用 10 分钟闭卷判断：`K`、`T_cw`、camera center、camera-space root 各是什么；`unproject(pixel)` 能返回什么；root-relative 人体还缺什么。错误项进入 E1，不重复所有旧内容。
2. 记录当前解释器、NumPy / OpenCV 版本及是否已有 COLMAP / PyCOLMAP。轻量实验需要 NumPy、matplotlib、OpenCV；PyCOLMAP 主要用于读取模型与按真实相机模型投影。不要为本周默认安装 PyTorch3D、Open3D 或完整训练栈。
3. 选择独立 Week 7 环境，先确认官方安装说明与本机实际版本；安装或 GPU/图形兼容性连续排查 15m 未推进就整理证据，累计最多 45m 后转环境专项。剩余时间先完成合成几何；真实 SfM Gate 保留待补。
4. 核对 COLMAP CLI 与 Python 包是两个入口，版本分别记录；安装 PyCOLMAP 不等于一定有 `colmap` 可执行文件，也不证明 CUDA 可用。当前 [PyCOLMAP 官方说明](https://colmap.github.io/pycolmap/) 提供预编译包入口，实际设备支持以运行结果为准。
5. 本周从 SIFT + 小图集 sparse 开始。保留一个可靠的 CPU 路线作为候选，不为了用满 GPU 引入额外 learned matcher。不同版本的 GPU 参数名可能变化，先查本机帮助再使用。
6. Day 1 完成 20–40 张手机照片的采集与清单；Day 2 检查格式、尺寸和方向，给 Day 4 留出重拍余量。照片不存在时不能预先勾选采集完成。

### 版本核对与最小入口

以下是本人执行时的检查项，本次编写文档没有运行它们：

```bash
python -c "import sys, numpy, cv2; print(sys.executable); print(numpy.__version__); print(cv2.__version__)"
python -c "import pycolmap; print(pycolmap.__version__)"
colmap -h
colmap feature_extractor -h
colmap exhaustive_matcher -h
colmap mapper -h
colmap model_converter -h
```

保存帮助输出和实际版本；安装失败也保存原始报错。当前在线 CLI 的 CPU 开关为 `FeatureExtraction.use_gpu` / `FeatureMatching.use_gpu`，旧版可见 `SiftExtraction` / `SiftMatching` 命名；**本机 `-h` 是执行依据**。不照抄一个版本的选项去修另一个版本。[官方 CLI](https://colmap.github.io/cli.html)

### 采集协议与 8 GB 推进顺序

默认拍 **约 24–30 张，最终输入 20–40 张**：固定同一手机镜头、焦距/变焦与图像模式，围绕静止且有纹理、有前后层次的场景移动拍摄。相邻图保持明显重叠，确保关键区域在至少三个视角出现；移动相机位置，避免只站原地转动。先选桌面物体/室内静景，不把行走人体作为 SfM 结构点。[COLMAP 采集建议](https://colmap.github.io/tutorial.html#structure-from-motion)

原始照片保留；方向修正、HEIC 转换、缩图均输出到工作副本，并登记原始到工作图像的关系。统一分辨率不是校准保证；若切镜头、变焦或裁切不同，先分组检查相机模型，不能强制所有图共享 K。默认候选用 `SIMPLE_RADIAL`，只有图像确已去畸变/有标定依据时才相应考虑 pinhole；根据实际镜头和残差检查，不盲目增加参数。[相机模型](https://colmap.github.io/cameras.html)

| 阶段 | 最小动作 | 何时调整 |
| --- | --- | --- |
| 合成几何 | float64、100 个点、CPU | 不依赖 GPU 或人体资产；先验证数学定义 |
| 照片输入 | 工作副本可先设最长边约 1600 px，记录实际尺寸 | 这是建议预算；只缩副本，不能混用旧 K / 旧数据库 |
| Sparse | 特征 → 匹配 → mapper，不启动 dense / renderer | 小图集可先试 CPU；实际耗时和内存待测 |
| 显存/图形失败 | 保存失败阶段和日志，尝试版本支持的 CPU 特征/匹配 | 不承诺某个开关一定修复；不改已有 HMR 环境 |
| 重建失败 | 先查输入/匹配/初始化/注册，而非连续换一堆参数 | 一次只改一个原因；补拍优先于扩大工具栈 |
| 可视化 | matplotlib 画稀疏点和相机中心/朝向即可 | GUI 失败不妨碍数值检查；Open3D 只作可选 |

## 4. Geometry & Pipeline Map

### 坐标协议：本周唯一约定

使用列向量，**`T_ab` 把 b 坐标变到 a 坐标**；`T_cw` 是 world→camera，`T_wc=T_cw⁻¹`。文件必须写出含义，不能只写“pose”。若本人代码采用行向量，明确 `Xc = Xw @ R_cw.T + t_cw`，数学约定保持一致。

```text
X_world --T_cw--> X_camera --K + perspective divide--> pixel
X_world <--T_wc-- X_camera <--pixel + known Z_camera-- pixel

T_cw = [R_cw, t_cw; 0, 1]
Xc = R_cw Xw + t_cw
T_wc = [R_cw^T, -R_cw^T t_cw; 0, 1]
C_world = -R_cw^T t_cw                     # camera center，不是 t_cw
T_c2c1 = T_c2w T_wc1                     # camera 1 → camera 2

ideal pinhole: [u,v,1]^T ∝ K Xc
known optical-axis depth: Xc = Z_camera K^-1 [u,v,1]^T
```

上述最后一式针对无畸变 pinhole、K 最后一行为 `[0,0,1]`。没有 Z 只能得到一条射线；若给的是沿单位射线的欧氏距离 ρ，应先归一化方向再乘 ρ。不要把 range 和 optical-axis depth 混用。真实手机图需按实际 distortion model 处理。[Camera Models 讲义](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)

### 两视图与 SfM 数据流

对 `X_c2=R_21 X_c1+t_21`，令 `x_i=K_i⁻¹ ũ_i` 为无畸变归一化像点：

```text
E = [t_21]_x R_21
F = K_2^-T E K_1^-1
x_2^T E x_1 = 0;  u_2^T F u_1 = 0
l_2 = F u_1

已知 K 和两相机 pose + 对应点 → triangulation → 3D point
只有对应点 + K → robust E estimation → relative pose（平移仅到尺度）

phone images → features → matches / geometric verification
  → initial pair → incremental image registration / PnP
  → triangulation / filtering → local & global BA
  → sparse model → camera poses / points / observations
```

先由本周约定推导矩阵方向，再对照讲义；讲义的相机编号、左右乘顺序不必与本人一致。E 的非零整体尺度不改变约束；F 的代数残差也随矩阵尺度变化，不直接当像素距离。[Epipolar Geometry 讲义](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

### 资源与接口导航

| 编号 | 官方资源 / 入口 | 本人定位与记录 |
| --- | --- | --- |
| R1 | [CS231A Camera Models](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf) | projection、intrinsics/extrinsics；与 E1 对上 |
| R2 | [CS231A Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf) | E/F、epipolar line；自己核对编号方向 |
| R3 | [CS231A Stereo / SfM](https://web.stanford.edu/class/cs231a/course_notes/04-stereo-systems.pdf) | 线性三角化、重投影最小化、SfM；不通读所有扩展 |
| R4 | [OpenCV calib3d](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html) | `undistortPoints`、`findEssentialMat`、`recoverPose`、`triangulatePoints`；记录本机重载与 mask |
| R5 | [COLMAP Tutorial](https://colmap.github.io/tutorial.html) / [CLI](https://colmap.github.io/cli.html) | extraction → matching → mapper；实际参数与日志 |
| R6 | [COLMAP Output Format](https://colmap.github.io/format.html) | world→camera、qw-first quaternion、camera center、非连续 ID |
| R7 | [PyCOLMAP API](https://colmap.github.io/pycolmap/pycolmap.html) | `Reconstruction`、`Image.cam_from_world()`、`projection_center()`、`Camera.img_from_cam()` |
| R8 | [Camera Models](https://colmap.github.io/cameras.html) / [FAQ](https://colmap.github.io/faq.html) | distortion、共享内参、像素约定、BA 与 gauge 检查 |

这些是阅读导航，不是本机已安装接口的保证。当前 PyCOLMAP 文档把 `Image.cam_from_world()` 定义为方法；若安装版本不同，先查实际帮助。不要把旧示例中的属性访问原样当成当前事实。

### Shape / 协议契约

| 对象 | 本周练习契约 | 必查语义 |
| --- | --- | --- |
| 点与有效性 | `X[N,3]`，`mask[N]` | frame、unit、finite；N=0/1 不丢轴 |
| 相机 | `K[3,3]`，`T_cw[4,4]` | 内参与外参分离；合成无畸变与真实镜头分离 |
| 像点与深度 | `uv[N,2]`，`Z[N]` | pixel / normalized；Z 是相机光轴深度 |
| 两视图 | `uv1/uv2[N,2]`，`P1/P2[3,4]` | 两组点同序，P 与像点坐标类型一致 |
| DLT | `A[N,4,4]` → `Xh[N,4]` → `X[N,3]` | SVD 最后右奇异向量、齐次 w、cheirality、视线夹角 |
| COLMAP | image/camera/point ID 为字典键 | ID 不是数组下标；image 与 camera 数不同 |
| observations | `(image_id, point2D_idx, point3D_id, uv)` | 负/无效 point ID 不作匹配；按关联找点，不按行号猜 |
| human root | `root[N,3]` + frame / unit / origin | root-relative、body placement、camera translation 不可互换 |

真实导出中的畸变参数不能扔掉后只用 K；本人的 pinhole `project()` 验证合成几何，真实手机重投影调用对应 camera model 或明确去畸变后再比。COLMAP 文本四元数为 `(qw,qx,qy,qz)`，与默认 xyzw 接口对接要显式重排；camera center 用 `-Rᵀt`。这些约定以 [输出格式](https://colmap.github.io/format.html#images-txt) 为准。

## 5. Day 1–5 Detailed Tasks

五天同级，每天采用相同九个栏目。E1–E5 的输入输出、反例与验收详见第 6 节；只建立目录或写完模板不算运行证据。

### Day 1 — Coordinate Contracts, Projection & Capture

#### 当日目标

本人写出 project / unproject / SE(3) inverse，用非平凡相机检验 round-trip；完成手机 20–40 张图的采集和输入清单。

#### 前置条件

能解释矩阵乘法、旋转与齐次坐标；用开工短测检查旧知识。Week 6 实验未完成时，以自造 point / root 测试，不等待人体模型。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 45 | 10m 短测，定向读 R1，写坐标协议；核对轻量环境 |
| Run | 60 | 本人运行 E1 手算例、已有函数诊断与 round-trip；记录正常/无效点 |
| Independent Experiment / Debug | 85 | 补 E1 非单位旋转/反例约 50m，采集和登记手机照片约 35m |
| Analyze / Log | 35 | 整理协议、误差与照片质量，标出待重拍项 |

#### 核心资源

R1 的 intrinsics / extrinsics 与第 4 节协议；对照既有 Week 4/5 投影代码，只复用自己能解释的部分。手机采集按 R5 的重叠、纹理与移动视点建议。

理解检查：t_cw 是相机在世界的位置吗？`unproject` 的 Z 属于哪个 frame？无深度时答案是什么？`R=I` 的测试为何可能放过错误逆变换？缩图后 K 还能原样用吗？

#### 执行步骤

1. **Understand / Trace**：本人先写 `T_ab` 定义与四项函数契约；用一个 camera center 非零、R 非单位的手算/可视化场景画出 axes。暂不调用 COLMAP。
2. **Run**：完成 E1 基础例：world→camera→world、camera→pixel→camera。只有对可见、finite、正深度点才计算投影；将像素、三维和矩阵误差分别记录。
3. **Implement / Debug**：加入负深度、Z=0、空 mask、错误 K 和把 `-t` 当逆平移的反例。拍摄静态有层次场景，保留原片；清单登记 20–40 张最终拟输入，模糊/重复图明确标记。
4. **Analyze**：本人说明哪些测试只验证代码自洽、哪些有外部手算答案；检查照片是否只有原地旋转或单张平面。质量明显不足时安排 Day 2 的小额重拍，不等到 Day 4。

Hint：变换互逆双方若犯同一种错，round-trip 仍可能“通过”；用已知相机中心、轴向和手算点独立检查。不要只测试单位矩阵。

正常测试：非单位 R、非零 t，世界中的相机中心变到相机原点。反例：仅用 `T=[Rᵀ,-t]` 取逆，应被相机中心/点变换测试识别。

#### 交付物

`exp/geometry.py`、`notes/coordinate_protocol.md`、`results/<run_id>/roundtrip.csv`、一张本人画的坐标图、20–40 张手机输入的 `inputs/manifest.csv`。若实际仅拍到不足 20 张，采集 Gate 保持未完成。

#### 验收标准（Gate）

- [ ] 本人实现 project、带 Z 的 unproject、SE(3) inverse；约定与 shape 明确。
- [ ] 非单位旋转/平移及独立手算例有数值，不能只凭 round-trip 通过。
- [ ] 无深度返回射线或明确缺少 Z；无效/全空点不返回健康的零误差。
- [ ] 手机 20–40 张输入已采集并有清单；静态场景、纹理、重叠和视点移动已检查。

#### 受阻处理

轻量环境阻塞先保存错误，用纸上手算继续核对定义；真实执行 Gate 不提前通过。采集失败先修采集，不用他人照片冒充本人数据；后续可用官方数据诊断软件，但必须另标。

#### 下一步

Research Log：哪个约定最可能被自己写反？有什么独立证据能抓到共同错误？Day 2 带上 K、两组 pose 和同一批 synthetic points。

### Day 2 — Epipolar Geometry & Relative Pose

#### 当日目标

本人由已知 R/t 构造 E/F，解释 normalized 与 pixel 坐标的区别；在合成对应点上检查对极线与相对位姿方向。

#### 前置条件

E1 的变换和投影已验证。默认使用两个已知、无畸变的 pinhole camera 与非共面合成点，不先依赖真实照片特征匹配。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 55 | 读 R2，推导 T_c2c1、E/F 与 l2；定位 R4 相对 pose 接口 |
| Run | 35 | 用同批合成点验证已知 E/F，运行一次 OpenCV pose 对照 |
| Independent Experiment / Debug | 100 | 本人完成 E2、错配/纯旋转反例；抽约 15m 检查手机工作副本 |
| Analyze / Log | 35 | 画对极线，解释 R/t 方向、scale、mask 与退化 |

#### 核心资源

R2 的 E/F 与对极约束；R4 的 `findEssentialMat`、`recoverPose`。先读输入点类型、K、阈值单位、输出 mask 和平移尺度；本周不要求从零实现五点求解器。

理解检查：为什么 F 可作用于像素、E 需要校准坐标？`l2=F u1` 与 `Fᵀ u2` 分别在哪张图？E 乘 10 后“代数误差”会怎样？recoverPose 的 t 是米吗？纯旋转为什么不能给出可靠深度？

#### 执行步骤

1. **Understand / Trace**：从本人 `T_c2w T_wc1` 算 R_21/t_21，写 `skew(t)`、E、F；先推导再对照资源，避免把讲义编号照搬错。
2. **Run**：由 E1 产生已知对应点，检查双视图投影、epipolar residual 和 E 的 rank。用适合本机版本的 OpenCV 接口做一次 pose 对照，明确这是库实现而非本人独立求解。
3. **Implement / Debug**：本人完成 E2，画少量对极线并检查目标点到线距离；打乱对应、混用 pixel/normalized 或反转 pose 应产生可检测问题。检查手机文件能解码、方向/尺寸一致，必要时补拍并更新清单。
4. **Analyze**：将 OpenCV 的相对 R 与合成 GT 比较；平移只比较正确方向/尺度约定，不将单位方向与真实米制 t 直接相减。记录 robust estimation 与 cheirality mask 的区别。

Hint：E/F 是齐次对象，比较矩阵时需消除整体尺度/符号，不能只用裸 `allclose`。合成已知 R/t 路径与由 noisy points 估计 E 的路径分开报告。

正常测试：非零基线、非共面点，已知 E/F 约束成立。反例：零基线/纯旋转不能把 `E=0` 的零残差算作好模型；错误对应有时仍接近对极线，约束小也不保证匹配正确。

#### 交付物

`exp/epipolar.py`、`notes/two_view_protocol.md`、E/F 与 mask 数值、对极线图、pose 对照结果、手机工作副本与原图的映射清单。

#### 验收标准（Gate）

- [ ] 本人独立写出相对变换、skew、E/F 构造，并指出两个 frame。
- [ ] pixel/normalized 使用正确；对极线方向与目标图一致，残差有明确定义。
- [ ] 错配、pose 方向错误与纯旋转反例有记录；零 E 被标退化。
- [ ] 能解释 recoverPose 只恢复到尺度的平移及 mask；手机输入在 Day 4 前已可解码。

#### 受阻处理

OpenCV 重载或估计不稳定时，先在无噪声、非共面合成点上缩小问题；已知 R/t 的 E/F 检查继续。真实图的未知 K 不拿 EXIF 焦距毫米值直接当像素焦距。

#### 下一步

Research Log：约束成立到底排除了什么，没有排除什么？Day 3 用已经解释的 P1/P2，而非未核实的估计 pose，隔离三角化误差。

### Day 3 — Triangulation & Controlled Error Study

#### 当日目标

本人实现两视图 DLT，建立 cheirality / 退化检查；以合成 GT 分别分析 baseline、focal length、2D noise 对三维误差的影响。

#### 前置条件

E1 投影和 E2 坐标约定可靠；相机真值、点真值和对应关系全部已知。相机估计误差暂不混入主实验。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 45 | 读 R3 线性三角化，手推 A；写变量、预期和失败政策 |
| Run | 45 | 无噪声恢复、OpenCV 独立对照、齐次坐标与正深度检查 |
| Independent Experiment / Debug | 110 | 本人完成 E3，运行三组单因素扫描、随机重复和退化反例 |
| Analyze / Log | 40 | 画误差曲线，比较三维误差/重投影，写替代解释 |

#### 核心资源

R3 的 linear triangulation 和 nonlinear reprojection minimization；R4 的 `triangulatePoints` 仅用于独立对照。主实现使用已去畸变 normalized points 与 `P=[R|t]`，不把 K 乘两遍。

理解检查：A 的每行怎么来？最后一个右奇异向量为什么可用？齐次 w 近零意味着什么？低重投影误差为何仍可能有很大深度误差？焦距变大在“固定像素噪声”下的结论能否推广到所有相机？

#### 执行步骤

1. **Understand / Trace**：本人从 `x × PX=0` 写出每视图两行约束。给出噪声假设、共同点集和三个单因素扫描表，先写预期再运行。
2. **Run**：完成 E3 的无噪声恢复与 OpenCV 对照，保存逐点误差；先确认同一 frame 和尺度，再汇总。检查正深度、finite、w 和视线夹角，不只看数值有没有输出。
3. **Implement / Debug**：按第 7 节预算扫描 baseline/focal/noise，复用点集和底层随机噪声。每个条件保存有效数/无效原因；加入零基线、极小视差、错配等反例，不为画曲线静默删掉困难点。
4. **Analyze**：同时看三维 GT 误差和像素重投影误差；用受限 rectified 模型中的深度敏感性解释趋势，指出真实图还有匹配、标定、畸变和姿态误差。

Hint：不要用 `inv(A)` 解齐次系统；先用 SVD。噪声条件的 DLT 最小化的是代数目标，不保证等于最优重投影解；固定相机只优化点叫 nonlinear triangulation refinement，不叫完整 BA。

正常测试：已知基线、正深度、无噪声的非共面点恢复到 GT。反例：基线=0 时即使库返回有限坐标也应标退化；只有重投影小不能通过三维准确性 Gate。

#### 交付物

`exp/triangulate.py`、`exp/triangulation_sweep.py`、实验卡、`conditions.csv`、逐 seed/逐点结果、三组曲线、`notes/triangulation_analysis.md`。主实验所有数值均待本人实跑，本文不预填。

#### 验收标准（Gate）

- [ ] DLT 由本人实现；无噪声 GT 与独立库对照有真实残差和容差。
- [ ] baseline / focal / noise 三组单因素扫描齐全，点集、seed、尺度与相机协议清楚。
- [ ] 图和 CSV 可互相追溯，报告无效点数与原因，没有只报告幸存样本。
- [ ] 能用证据解释“小 baseline + noisy 2D”的深度不稳定，并说出结论的固定噪声前提。
- [ ] 零基线/纯旋转、近无穷远或错配有反例；不将重投影小等同于三维真值正确。

#### 受阻处理

曲线异常先关闭噪声查坐标/齐次除法；再固定一个点检查 disparity 与深度，最后恢复批量。不先增加样本量掩盖 bug；可减少可选曲线美化，但三类变量分析保留。

#### 下一步

Research Log：实测是否符合运行前预测？哪项结果更支持“几何退化”而非“实现错误”？Day 4 将视差与重叠判断迁移到本人手机照片。

### Day 4 — Phone Images & COLMAP Sparse Reconstruction

#### 当日目标

在本人 20–40 张照片上完成一次 COLMAP 稀疏重建，沿日志说明特征、匹配、初始化、注册和 BA 的作用，保留失败图与组件信息。

#### 前置条件

Day 1–2 的照片与工作副本可用；COLMAP 入口已检查或在预算内完成配置。使用静态场景，不用动态人体图替代。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 45 | 读 R5 的 sparse 步骤，核对照片、镜头分组和选项 |
| Run | 45 | 执行小图集 extraction / matching / mapper，保存阶段日志 |
| Independent Experiment / Debug | 100 | 本人完成 E4 的审计与包装，检查注册/组件；必要时做一次定因重试 |
| Analyze / Log | 35 | 看主组件及失败帧，解释实际管线和受限结论 |

#### 核心资源

R5 Tutorial / CLI；R8 的 camera model、shared intrinsics 和 FAQ。用默认 SIFT 稀疏路线，20–40 张小图集可采用 exhaustive matching；不额外引入 dense、learned features 或大规模优化。

理解检查：一张图提取到很多 keypoints 是否说明对应可靠？matching / geometric verification / registration 有何区别？为什么 image 数不等于 camera 数？多组件为何不能直接画成同一个世界？BA 出现在哪个阶段？

#### 执行步骤

1. **Understand / Trace**：本人核对图像清单与工作副本 hash，决定实际 camera model 和是否共享内参，写出依据。每个新条件使用新的数据库/输出目录，避免把旧特征混入新图像。
2. **Run**：按下方最小命令骨架执行 sparse，逐阶段保存命令、日志、耗时和退出状态；不是只保留 GUI 截图。可先处理排错小子集，但最终仍要在完整 20–40 张清单上运行。
3. **Implement / Debug**：本人完成 E4，列输入、已注册/未注册图、各组件图数/点数和异常。失败时定位到采集、解码、匹配、初始化或注册中的一层；只针对证据做一次参数/采集调整，并保留初次结果。
4. **Analyze**：选主组件看相机分布、稀疏结构和重投影统计。若只是少量图被注册或模型明显破碎，记“部分完成”，不能把目录非空当成成功。

以下在**新 run 目录已经建立**后使用；路径是本人按第 3 节约定填写的计划路径。完整参数由本人根据本机帮助和采集协议补充并保存，不预设已经适配：

```bash
# 每次新运行选择新的 run_id；此处仅展示首次命名。
w7_run="$HOME/10weekplan/week7/results/d4_sparse_01"
w7_images="$HOME/10weekplan/week7/inputs/phone_working"
mkdir -p "$w7_run/colmap/sparse"

colmap feature_extractor \
  --database_path "$w7_run/colmap/database.db" \
  --image_path "$w7_images"

colmap exhaustive_matcher \
  --database_path "$w7_run/colmap/database.db"

colmap mapper \
  --database_path "$w7_run/colmap/database.db" \
  --image_path "$w7_images" \
  --output_path "$w7_run/colmap/sparse"
```

这是官方稀疏阶段的最小入口，镜头模型、共享内参、CPU/GPU 等有效设置须随 run 落盘。重跑时换新 run_id；不复用有不同图像/参数历史的数据库。[COLMAP CLI](https://colmap.github.io/cli.html)

Hint：`sparse/0` 不一定是应选的主组件；按实际注册图数与几何检查选组件并记录 ID。两个不连通组件各有自己的坐标与尺度，不能简单拼接。

正常测试：同一静态有纹理区域从不同位置观察，形成可检查的多视角模型。反例：原地旋转、重复纹理、模糊、切镜头或主要点来自移动物体；不要仅靠“点数更多”判断更好。

#### 交付物

`exp/run_sfm.sh` 或本人等价入口、`notes/colmap_protocol.md`、完整输入清单、独立数据库与 sparse 模型、阶段日志、`components.csv`、`registered_images.csv`、初始可视化和失败记录。文件名是本人练习建议，不是官方必定生成的 CSV。

#### 验收标准（Gate）

- [ ] 本人 20–40 张照片的完整清单实际参与 sparse 运行，输入/配置/日志可复查。
- [ ] 存在跨多个不同相机位置、非退化且可检查的稀疏模型；明显破碎或仅两帧初始化列部分完成。
- [ ] 报告注册率、未注册图和所有组件；解释主组件选择，不把多组件强行合并。
- [ ] 本人能从日志定位 extraction / matching / registration / BA，说明数据库与模型目录各存什么。
- [ ] 至少一个失败/边界样例有候选原因与最小排除实验；未将低重投影当真实几何精度证明。

#### 受阻处理

安装/GPU 问题超过 45m 累计预算转环境专项；科研部分继续 E1–E3。用官方数据跑通只证明软件路径可用；手机数据重建仍登记受阻。场景退化优先检查/补拍，不无止境换 matcher。

#### 下一步

Research Log：失败发生在有匹配之前还是之后？最可能推翻当前原因的证据是什么？Day 5 带上选定组件路径、注册清单、相机模型和 observations。

### Day 5 — Pose Export, Reprojection Audit & Bundle Adjustment

#### 当日目标

导出相机位姿和稀疏点，画相机中心/朝向，按真实相机模型检查重投影；解释 BA 目标与单目尺度，完成通往 Week 8 的接口审查。

#### 前置条件

Day 4 有可读 sparse 模型；若只有合成结果，可完成 E5 的数学反例，但真实导出 Gate 不通过。camera model、图像尺寸、组件和图像顺序必须明确。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 25 | 读 R6/R7 输出定义，写 BA 目标、变量与 gauge |
| Run | 50 | 导出原生文本/读取模型，保存 camera/points 表和相机分布图 |
| Independent Experiment / Debug | 110 | 本人完成 E5，做重投影审计、尺度反例和 root 变换检查 |
| Analyze / Log | 40 | 完成 Final Gate、闭卷解释、一页周总结与 Week 8 接口卡 |

#### 核心资源

R6 的 `images.txt` / `cameras.txt` / `points3D.txt`；R7 的模型与 camera projection API；R3 的非线性重投影及 R8 的 BA。只读与本次导出相关的代码/接口，不要求读完整 COLMAP C++ 仓库。

理解检查：为什么 camera center 不能用 t 连线？相机 ID 能排序表示时间吗？两套独立单目系统为何不能直接拼接？BA 优化什么，固定了哪些自由度？已有 mapper 结果再次 BA 必然显著改善吗？

#### 执行步骤

1. **Understand / Trace**：本人写 BA 的 observations、残差、优化变量和 gauge；从 Day 4 实际日志定位 BA 证据，不能只因文档说有 BA 就声称观察到了。本周不要求从零求解完整 BA。
2. **Run**：用 PyCOLMAP 读取选定组件；保留原生模型，并按需用 `model_converter` 导出 TXT。本人生成带 image_id/name、camera_id/model、T_cw/T_wc/C_world 的 pose 表，稀疏点另表导出。
3. **Implement / Debug**：完成 E5：由 observations 的关联选点、投影并比较实测像素；画 C_world、朝向与 sparse points。按 manifest 的拍摄顺序连已注册图，并标记缺图；无可靠顺序时只画相机分布，不假造时间轨迹。
4. **Analyze**：在合成副本上将点与相机平移同时乘同一正尺度，验证图像投影不变。再用同 frame/同尺度的合成 human root 做 camera↔world；实际 HMR/Anny 与 SfM 没有对齐信息就写清缺口，保留到 Week 8。

BA 的本周理解目标是 `sum_(i,j in observations) rho(norm(project_camera_i(R_i X_j+t_i)-u_ij)^2)`：camera poses、3D points，及按配置允许调整的 intrinsics/distortion 共同参与。实际权重、鲁棒损失、固定参数以本次配置为准。只固定相机优化点是三角化 refinement；完整 BA 还需处理坐标/尺度自由度。本周不强制额外运行第二次 BA 来凑实验。

Hint：COLMAP 的图像/点 ID 可能不连续，`points2D` 中未三角化项不能用作有效关联。用实际 camera model 重投影，避免把手机 distortion 丢掉后把偏差误判成 pose 错误。[Output Format](https://colmap.github.io/format.html)、[PyCOLMAP API](https://colmap.github.io/pycolmap/pycolmap.html)

正常测试：先选 3 张已注册图，按确定规则抽样 observations，复算重投影并输出 residual / valid_count。反例：把 t 当 C、qw-first 当 xyzw、或混用相机 ID，应被 camera center / 图像重投影检查暴露。

#### 交付物

`exp/audit_reconstruction.py`、`export/camera_poses.csv`、`sparse_points.csv`、`observations_audit.csv`、相机中心/朝向/点云图、尺度不定反例、`notes/ba_and_scale.md`、`notes/weekly_review.md` 与更新后的 research_log。

#### 验收标准（Gate）

- [ ] 相机 poses 与 sparse points 实际导出，图像/相机/点 ID、frame、单位或 arbitrary scale 明确。
- [ ] 本人绘制 camera centers 与朝向；有拍摄顺序才连线，未把 t 或 ID 排序当轨迹。
- [ ] 真实 observations 的重投影有逐项数值、有效数和相机模型，不仅比较图片外观。
- [ ] 能解释 BA 目标、优化变量与 gauge；不将 fixed-pose refinement 或一次库调用称完整方法复现。
- [ ] 尺度反例与合成 root 变换有证据；能说明为什么当前人体与 SfM 输出不能直接合成米制世界运动。

#### 受阻处理

PyCOLMAP 版本接口不符先查本机帮助；也可用官方 converter 的 TXT 按文档读取，仍需保留原生模型。不要忽略新版本 rigs/frames 文件或私改数据库。若真实模型没有可用对应观测，导出/审计保持未完成，合成检查不替代它。

#### 下一步

Research Log：哪些结果证明实现自洽，哪些涉及真实精度仍无法判断？写出 Week 8 的 camera pose / human root / scale 三项接口与缺失证据，不立即搭完整 SLAM 工程。

## 6. Independent Coding Exercises

以下是**本人要编写的练习契约**，不表示这些文件或函数已实现，也不表示官方有同名脚本。每项先写预期，再实现、运行、交证据；Agent 默认只给 Hint 和最小修改。

### E1 — Projection / Unprojection / SE(3) Contract

- **定义**：`Xc=R Xw+t`；`uv=dehom(K Xc)`；`Xc=Z K⁻¹ uv_h`；逆变换平移为 `-Rᵀt`。接口名可采用 `project / unproject / invert_se3 / transform_points`，由本人设计。
- **输入/输出**：点 `[N,3]`、K、T、mask；输出 uv/Z/valid 或恢复的点，另给 invalid 原因；point transform 和 camera projection 分开。
- **Hint**：只给 Z>0/finite 点做可见投影；先验证 RᵀR、det(R)、齐次最后一行；无条件 squeeze 会破坏 N=1。
- **正常测试**：`K=[[500,0,320],[0,500,240],[0,0,1]]`，相机点 `[1,0,5]` 应投影 `[420,240]`；再用非单位 R 检查 C_world→0 与 point round-trip。前者是合成手算答案，不是已运行结果。
- **反例**：无 Z、把 range 当 Z、Z=0/负值、NaN、singular K、N=0、全无效 mask；R 非法时显式拒绝，不强行修成“通过”。
- **验收**：float64 正常例候选 `atol=1e-8`（几何单位）、像素 `1e-6 px`；实际误差与容差落盘，未通过先查根因。inverse 结果与独立通用矩阵逆/手算作对照，但主实现不能只包一层通用 inv。

可选：在明确像素坐标变换 A 下检查 `K'=A K`；涉及 resize 的像素中心约定写清。不以简化缩放公式替代真实 image pipeline 的核对。

### E2 — Epipolar Constraints & Pose Direction

- **定义**：按第 4 节构造 T_c2c1、E/F；非退化 E 应 rank 2 且两个非零奇异值相等（数值容差内）。F 作用于无畸变 pixel homogeneous points。
- **输入/输出**：K1/K2、T_c1w/T_c2w、对应点 `[N,2]`；输出 E/F、每点代数残差、点到对极线距离及有效 mask，不把所有残差统称像素误差。
- **Hint**：`distance=abs(u2_hᵀ l2)/sqrt(l2_x²+l2_y²)`，分母接近零要标无效；比较 E/F 时规范化尺度并容许符号翻转。
- **正常测试**：已知合成 geometry 的约束接近零、`l2` 过正确对应点；增加一项 OpenCV relative-pose 对照，记录实际输入和返回 mask。
- **反例**：像素/normalized 混用、F 转置方向错误、点顺序打乱、零 baseline。E=0 的 residual=0 是退化信号，不是正确模型。
- **验收**：本人实现已知 pose→E/F 与残差；能解释库估计 R/t 的方向与任意尺度。八点 F 求解器仅作可选扩展，不以未实现它判本周失败。

### E3 — Two-view DLT & Sensitivity Sweep

- **定义**：对 normalized `(x_i,y_i)` 和 `P_i=[R_i|t_i]`，每视图取 `x_i P_i[2]-P_i[0]`、`y_i P_i[2]-P_i[1]`，组成 A；SVD 解最小右奇异向量，再除齐次 w。
- **输入/输出**：P1/P2、同序像点；输出 X、valid、每视图 Z、triangulation angle、w 和残差。主实现用 float64；库对照转成其实际所需 shape，不混用 `[N,2]` 和 `[2,N]`。
- **Hint**：OpenCV 返回齐次点，先检查 w；射线夹角与正深度是不同检查。数值有效不等于几何稳定，保留 near-degenerate 标签和筛选规则。
- **正常测试**：共同非共面 GT、无噪声、非零 baseline，逐点恢复；与 OpenCV 在相同输入/坐标下比较，不只比均值。
- **反例**：P 相同、纯旋转、极小 baseline、平行射线、错误对应、全无效。不能 clamp w 到 epsilon 后宣称得到可靠有限深度。
- **验收**：默认无噪声例候选三维误差 `1e-8 m`、像素 `1e-6 px`；小基线/病态例另看条件与容差，不为通关随意放宽。第 7 节三组 sweep、每 seed 结果和 invalid 计数齐全。

### E4 — Sparse Pipeline Manifest & Run Audit

- **定义**：输入数 N；所选组件注册率 `N_registered/N`；组件分别计数，不把重复注册到多个组件的图相加后当单一完整率。
- **输入/输出**：工作照片 manifest、真实 CLI/配置、run 目录与各组件；输出运行记录、各阶段状态、组件表、未注册图清单。路径与文件 hash 对应。
- **Hint**：主要训练本人组织可复现运行和阅读日志；调用官方 extraction/matching/mapper 不算自写 SfM。新配置单独数据库，旧结果保留。
- **正常测试**：从一个非空真实 sparse 组件列出图像名，与本次 manifest 对齐；分母来自完整输入，不从注册图反推输入数。
- **反例**：重复/缺失文件、HEIC 不能解码、改图后复用旧数据库、未注册图被静默删除、把 component 0 当永远最大组件。
- **验收**：本人手机 20–40 张真实运行有原始模型与阶段证据；失败项不消失。图像采集质量与工具兼容性分别记录。

### E5 — Reconstruction Export & Geometry Audit

- **定义**：`C=-Rᵀt`；真实观测残差为 `project_actual_camera(RX+t)-uv_observed`。无 GT 的 residual 检查自洽，不提供轨迹绝对误差。
- **输入/输出**：一个明确组件的 reconstruction、对应工作图像及顺序；输出所有 registered poses / sparse points 与确定规则抽样的 observations audit。点数大时可抽样检查，但导出数量与检查数量分别报告。
- **Hint**：先查 ID 关联，再查 quaternion/pose/frame，再查畸变与像素协议。COLMAP 与 OpenCV 标定交接涉及 0.5 px 原点差时，按官方协议明确转换，不能重复加偏移。[FAQ 的 calibration 说明](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools)
- **正常测试**：先验证一个点/一图，再扩到至少 3 张已注册图；相机中心与 API 结果对照，真实投影使用匹配 camera model。
- **反例**：非连续 ID、未三角化点、t 当相机中心、quaternion 排序错误、pinhole 假装适用于含 distortion 模型、没有时间顺序仍按 image_id 连线。
- **尺度测试**：合成同一场景分别令 `X'=sX、t'=st、R'=R、K'=K`，取正 s=0.5/2；投影不变而坐标/基线尺度改变。这验证 Sim(3) gauge 的尺度分量，不宣称实现完整 Sim(3) 对齐。
- **验收**：实际导出、逐观测残差与图可核对；尺度反例成立；合成 human root 在同 frame/scale 中可正确 round-trip，真实跨系统融合仍需外参、同步、内参/图像处理与尺度对齐证据。

## 7. Experimental Design & Metrics

### 数据与预算

| 输入 / 实验 | 建议最小规模 | 使用规则 |
| --- | --- | --- |
| 合成非共面点 | 100 点，X/Y 在约 ±0.25 m，Z 在 4–8 m | 共同点集、正深度；合成单位 m，不套到真实 SfM |
| 两视图相机 | R1=R2=I；C1=(0,0,0)，C2=(b,0,0) | 基线沿 x，t2=(-b,0,0)；主实验 pose 已知 |
| 合成像平面 | 1280×960，principal point=(640,480) | 这里人为定义像素约定；检查所有条件是否可见，避免条件间换点 |
| 随机噪声 | seed 0–9；两视图各自独立 standard normal | 不同条件复用底层随机数再乘 sigma，保存 seed 和点集 hash |
| 手机数据 | 一处静态场景，最终 20–40 张 | 原片与工作副本分开；完整 manifest、拍摄顺序、镜头/尺寸 |
| 真实重投影审计 | 至少 3 张已注册图，按固定规则抽样 | 建议每图最多约 200 个有效关联；报告覆盖量，不按误差挑点 |
| 可选扩展 | 只选一项：小型 point refinement 或有尺度参照的检查 | 主线完成后再做；不增加完整新仓库复现 |

`manifest.csv` 至少含 `input_id、original_name/hash、working_name/hash、尺寸、拍摄顺序/时间证据、镜头/焦距模式、处理变换、纳入/排除与原因`。若裁切/旋转/缩图，更改后的像素坐标与相机参数须对应；原始 EXIF 和原片保留。

### 主实验：三组单因素扫描

默认固定 `b=0.1 m、f=800 px、sigma=1 px`，`fx=fy=f`。noise 是给双视图像点各轴加独立高斯噪声的标准差；不是给 GT 3D 点加噪声。

| 分组 | 唯一改变的因素 | 其余固定 | 运行前的可反驳预期 |
| --- | --- | --- | --- |
| Baseline | b = 0.02 / 0.1 / 0.5 / 1 m | f=800 px，sigma=1 px，点/pose 规则固定 | 固定像素噪声和可见对应下，小 b 的深度更敏感 |
| Focal length | f = 400 / 800 / 1600 px | b=0.1 m，sigma=1 px，同分辨率/点集 | 固定像素噪声下增大 f 通常降低角度噪声；不等同真实相机总是更准 |
| 2D noise | sigma = 0 / 0.5 / 1 / 2 px | b=0.1 m，f=800 px，同 GT | 无噪声接近数值基线；噪声增加时误差分布通常变宽 |

三组公共 baseline 可复用，共 **9 个唯一条件 × 10 seeds × 100 points**，规模用于 CPU 小实验，不是论文 benchmark。每个 seed 使用同一 GT 点集；sigma=0 时重复 seed 不产生新信息，允许复用同一无噪声结果并注明。无须做 4×3×4 全因子网格。

rectified 理想模型可用 `Z=f b/disparity` 和一阶敏感性 `abs(delta_Z)≈Z² abs(delta_disparity)/(f b)` 辅助解释；其假设包含已校准、平行相机和小扰动。不能用这条式子替代一般 pose 的三角化或推断真实场景精度。近退化条件单列，不只画均值。

每组同时保存：三维欧氏 GT 误差、abs 深度误差、相对深度误差、pixel reprojection residual、cheirality/finite/angle 状态与有效点数。统计先按 seed 汇总，再看跨 seed 分布；不把每个点当成独立重建试验来夸大样本量。曲线应显示离散程度，不能只挑一个最好 seed。

### 实验前填写的卡片

| 字段 | 本人运行前填写 |
| --- | --- |
| Observation / Question | 要检验哪一项几何敏感性或实际失败现象？ |
| Hypothesis | 预期模式是什么，什么结果能反驳它？另列一个替代解释 |
| Independent Variable | 本次只改变 b、f 或 sigma 之一 |
| Controlled Variables | K 其余项、pose 规则、GT 点集、noise draws、实现版本、mask/退化阈值 |
| Metric / Validity | GT 来源、frame、unit、有效政策、分母、近退化/全空如何报告 |
| Failure Criteria | 坐标错误、实现不一致、无有效点、严重退化分别怎样识别？ |
| Reproduction | 命令、解释器、代码版本、seed、run_id、原始结果和图路径 |

### 指标与允许结论

| 指标 | 最少需要什么 | 能说明什么 / 不能说明什么 |
| --- | --- | --- |
| round-trip / inverse residual | 明确协议、独立手算例 | 实现检查；双方共享错误时自洽不充分 |
| 点到对极线距离（px） | 非退化 F、无畸变像素对应 | 约束相容性；不单独证明对应正确或深度准确 |
| 三维/深度误差（m） | 同 frame/scale 的 synthetic GT | 该合成设置下的重建误差；不外推真实手机精度 |
| 相对深度误差 | 正深度 GT、明确分母 | 不同距离敏感性；不能掩盖失效点 |
| 重投影 residual（px） | 正确 pose、camera model、observation 对应 | 图像拟合/实现自洽；不证明米制 scale 或轨迹准确 |
| 注册率 / 组件图数 | 完整输入清单、组件选择 | 当前数据的可重建覆盖；点多/注册多不自动代表更准 |
| camera center / 朝向分布 | world→camera 方向明确 | 几何布局；无可靠时间顺序时不是时间轨迹 |
| ATE / RPE / 绝对尺度误差 | pose GT、时间同步、坐标/尺度对齐与正式协议 | 本周手机数据默认不具备，不输出伪真值误差 |
| 耗时 / 内存 | 同设备/版本/分辨率/阶段与计时边界 | 本次资源消耗；不能照抄官方或异机速度 |

建议结果表头：`run_id,condition_id,seed,point_id,metric,value,unit,frame,valid,invalid_reason,evidence_path`。无有效样本时 value 留空并注明原因；保留 total / valid / invalid，不能让筛掉大多数点后的低均值看起来更好。图示和实际数值在本人运行前保持待测。

## 8. Debugging Guide

沿用 **症状 → 假设 → 最小验证 → 根因 → 最小修复 → 复测**。先检查定义和合成小例，再改真实管线；一次只改变一个因素。环境安装、驱动与图形依赖交环境专项，科研 Agent 继续检查数学、代码接口和结论。

| 症状 | 优先假设 | 最小验证 / 下一步 |
| --- | --- | --- |
| round-trip 过但相机方向怪 | 两端共享错误、t 当 camera center | 用 C_world→0、非单位 R 与手算点独立检查 |
| 投影上下/前后反 | 轴向、pose 方向、渲染坐标混用 | 画三个基轴和一个正 Z 点，明确 world→camera |
| unproject“恢复”出任意尺度 | 没有深度却人为补值、range/Z 混淆 | 固定一个像素换两种 Z；解释同一射线 |
| E/F residual 很大 | 编号/转置、pixel/normalized 或 distortion 混用 | 关闭 noise，已知 pose/point 逐项手算 |
| E=0 但 residual=0 | 零 baseline/纯旋转退化 | 检查基线与 E 范数，不把零模型验收 |
| DLT 输出巨大值/NaN | 齐次 w、near-parallel rays、错配 | 单点检查 w、视差、cheirality 与 angle；不强行 clamp 后报告 |
| 像素误差小、深度误差大 | 几何病态/尺度不定或错误对应 | 有 GT 时查 3D；无 GT 时只报告不能确认真实精度 |
| focal 扫描结果解释不通 | 不小心同时缩图、改噪声/点集 | 查条件表与 hash，保持像素噪声假设明确 |
| COLMAP CLI/包找不到 | 独立环境入口或二者只装其一 | 记录实际路径/版本，按预算转环境专项 |
| 手机图解码/特征失败 | 格式、方向、模糊、无纹理 | 检查工作副本，不覆盖原片 |
| 匹配多但初始化失败 | 纯旋转、平面/重复纹理、低视差 | 看几何验证与重叠，优先补有平移和层次的照片 |
| 很多图未注册/多个组件 | 匹配图不连通、重叠不足、内参错误 | 列失败图及连接，不只看总点数 |
| 所有图共享 K 后变差 | 切镜头、变焦、尺寸/裁切变化 | 查 manifest / camera_id 分组，一次验证一个假设 |
| 导出“轨迹”乱序 | image_id 不连续且非拍摄顺序 | 按文件映射/manifest 排序，缺图显式标记 |
| 重投影出现系统偏差 | distortion、K 与图像不配、像素原点差 | 单图单点查实际 camera model；不机械加 0.5 px |
| BA 后像素误差降但看着更歪 | 错匹配、内参退化、gauge/可视化问题 | 查相同 observations、参数与固定项；低 cost 不等于真实精度 |
| 接入人体后世界 root 跳变 | 图像/时间/内参不匹配，尺度或 frame 未对齐 | 先用已知尺度 synthetic root；不把另一个模型输出直接套进 SfM pose |

失败记录至少保存：最小输入、预期与实测、日志/版本、已排除项、仍未验证的原因、下一项最小实验。没有对照证据时写候选原因，不把推测写成根因。

## 9. Daily Research Log Template

以下复制到 WSL `week7/research_log.md`，每天一份；不要预填 Gate 或实验结果。

```markdown
## YYYY-MM-DD / Day X

- 今日目标与 Gate：
- 实际专注时间 / 环境与采集耗时 / 机器等待：
- 设备 / 解释器 / NumPy / OpenCV / COLMAP / PyCOLMAP：
- 本人代码版本与 diff / 官方工具版本 / 配置路径：
- 输入 ID / 原片与工作副本 hash / 尺寸与处理变换：
- run_id / 命令 / seed / GT 点集与噪声协议：
- 阅读位置与定位到的函数/阶段：
- 运行前预测或假设 / 一个替代解释：
- 本人独立实现的部分 / 使用的库 / 收到的 Hint：
- K / distortion model / T_ab 定义 / frame / unit / scale：
- shape / dtype / valid 与 invalid policy：
- 正常测试 / 手算或库对照 / 反例 / 实际残差与容差：
- 数值结果 / total-valid-invalid / 图与原始数据路径：
- 手机输入数 / 注册图数 / 组件与未注册图（相关日填写）：
- 已知事实（直接观察）：
- 证据支持的推断：
- 未验证的假设 / 当前无法判断：
- Debug：症状 → 假设 → 验证 → 根因 → 修复 → 复测：
- 最大漏洞 / 哪个关键事实改变会推翻结论：
- 当日 Gate：通过 / 部分完成 / 受阻；逐项证据：
- 下一项最小验证实验与预计时长：
```

“理解了相机”不构成证据。至少写一个可检查的变换解释、一个失败边界、一个数值或文件；安装成功和目录创建只记准备工作。

## 10. Week 7 Final Gate

所有项目初始均待执行。本人实现、数值和解释共同支持时再勾选；未完成项保留原因与最小下一步，不用计划/模板或他人结果充数。

- [ ] **坐标变换**：本人解释 T_ab、K/R/t、C_world；E1 非平凡变换、独立对照与 round-trip 有证据。
- [ ] **Projection / Unprojection**：project、带深度的 unproject、SE(3) inverse 自己实现；知道像素单独只能给射线，valid/empty 行为明确。
- [ ] **E/F**：本人由 known pose 构造 E/F，解释点/线方向与尺度；正常和退化例可复查。
- [ ] **Triangulation**：本人实现 DLT；GT、库对照、w/cheirality/angle 检查和反例齐全。
- [ ] **三类误差实验**：baseline、focal、2D noise 分别控制，其余条件/点集/seed 固定；结果含无效样本与离散程度。
- [ ] **手机 Sparse**：本人 20–40 张照片完成真实 COLMAP sparse；有多视角非退化模型、配置、数据库、日志和未注册图说明。
- [ ] **Pose / Point Export**：相机 poses 与 sparse points 真实导出，正确画 C_world/朝向；可靠拍摄顺序下才连相机路径。
- [ ] **重投影审计**：至少 3 张 registered images 的有效对应被复算；真实 camera model / distortion / pixel convention 清楚。
- [ ] **BA 与尺度**：能解释目标、变量、固定项和 gauge；完成尺度不定反例，不宣称自动得到米制尺度或重力方向。
- [ ] **Human Root 综合判断**：同 frame/scale 的 synthetic root 可以 camera↔world；root-relative / 跨系统 scale 等缺信息时能指出不可直接计算。
- [ ] **科研记录**：E1–E5、实验卡、数值、图、日志和周总结可互相追溯；事实/推断/无法判断分开。
- [ ] **衔接 Week 8**：明确哪些能力已过、哪些仅静态理解或受阻；没有为赶进度提前扩成完整 TRAM/DROID-SLAM 复现。

周末 8–10 分钟闭卷讲解：一个 3D point 怎样到 pixel；两张图如何产生 E/F 和三角化；COLMAP 怎样得到 camera/points；为什么低重投影与真实尺度不同；一项本人实验支持了什么、没支持什么。随后回到自己的代码/日志定位证据。

## 11. Open Research Questions

以下是后续候选问题，**不是已经确认的 research gap 或 novelty**。本周先完成主线；只从已观察到的现象提出一个小问题，避免扩成模块拼接。

| 候选问题 | Baseline / 最小验证 | 最大漏洞与失败时的收获 |
| --- | --- | --- |
| 低视差下，哪些检查比单看重投影更能提示深度不稳定？ | E3 DLT baseline；比较 angle / disparity 与 GT depth error | 噪声模型过于理想；失败可定位指标或阈值不适用的范围 |
| 共享内参错误会怎样影响同一小场景的重建？ | 只有实际出现相机模式变化时，比较合理分组与错误共享 | 两次 reconstruction gauge 不同，不能裸比 poses；先审查输入与 pixel residual |
| 背景 SfM 与人体 root 不一致时，怎样区分尺度、相机和人体误差？ | 先用已知 GT 的 synthetic camera/root 分别注入一个错误 | 同时变动多个系统无法归因；失败促使接口与同步协议更严格 |

每个 idea 至少记录：问题是否真实存在、已有方法是否已解决、baseline、改善来源、是否只是模块组合、最小实验、失败能学什么，以及 novelty 候选属于 problem / method / representation / training / data / evaluation 哪一项。**目前不能仅凭本周小实验证明 novelty。**若要作为研究方向，下一步先定向查相关原始论文，再做最小对照。

## 12. Next Step: Week 8–9

按总表推进，Week 8 才将本周几何用于 **moving-camera / static-human** toy、四类 camera×human motion 场景与 TRAM 方法理解；Week 9 再连接 Human3R 的 human/camera/scene world-frame 检查。本周只把接口准备好，不提前要求配置完整 SLAM 或训练系统。

带入 Week 8 的最小材料：`coordinate_protocol.md`、E1/E3 可复用函数、一个明确组件的 K/distortion/poses、手机 manifest 和图像顺序、尺度不定反例、一个实测失败案例。Week 6 若已有 camera-space 人体/ID 结果也保留原始协议；若尚无，不影响先做已知 GT 的相机—人体 toy。

关键待解决项是：HMR 与 SfM 是否使用同一图像及时间、内参与 crop/resize 是否对应、root 是否已放入 camera frame、坐标轴与尺度是否一致。**不能把 `T_wc × human_root_camera` 的公式正确，等同于任意两套输出可直接组合。**

下一次讨论优先提交 E1 的非单位旋转反例或 E3 的第一组真实 CSV，并按五项开始：结论、当前证据、最大不确定性、最小验证实验、下一步行动。先审核本人实现和证据，再决定是否进入 world-space motion。

---

文档边界：本文件交付的是第七周执行计划；官方资料核对与结构检查不代表实验已完成。手机采集、WSL 环境、COLMAP/PyCOLMAP 兼容性、真实重建和所有学习 Gate 仍待本人运行验收。
