# Week 6 — Multi-HMR 2: Multi-Person Camera-Centric Human Detection, Mesh Recovery and Tracking

> 3D/4D Human 10 周科研训练 · Week 6 执行计划 · 2026-10-26  
> 5 天，合计 **19h 专注学习**；Day 1/2/4/5 各 3h45m，Day 3 为 4h，休息另计。  
> 本周使用 **Multi-HMR 2（2026）**替换总表中的 Multi-HMR（ECCV 2024）。仅新增本文件；总表和已有周计划保持原样。  
> 文档状态：总表、论文和官方核心源码已核对；以下环境兼容性、模型推理、独立实验与学习 Gate 均待本人执行。

## 1. Week 6 Overview

### 目标与总表对应关系

从“能够解释 HMR2 单人重建”推进到“能够定位并检验多人检测、相机坐标定位、Anny 人体恢复和跨帧关联”。完成一次官方推理只是起点；核心交付是自己写出的小实验、失败反例和证据支持的解释。

本周依据 [10 周总计划 Excel](00_plan/10周研究计划_曾奕博.xlsx) 的 `10周计划!B15:D15`；已将 `B4:D4、B6:D7、B14:D18` 与同目录 [Numbers 原表](00_plan/10周研究计划_曾奕博.numbers) 的相应内容核对，一致。保持原表能力主线，替换模型相关实现：

| 总表 Week 6 要求 | 换成 Multi-HMR 2 后的落实 | 验收证据 |
| --- | --- | --- |
| 必跑官方单人/多人 Demo，保存输入和可视化 | Day 1 跑官方 API/CLI，后续复用同一会话逐张增加样例 | 输入清单、配置、输出结构、代表性叠图 |
| image → localization → person query → HPH / cross-attention → 人体模型 | Day 2 按实际顺序追踪 image → encoder → learned queries → HPH → 多个 prediction heads → Anny | 源码位置、真实 shape、Q/K/V 来源及筛选步骤 |
| SMPL-X + 3D location | 改成当前仓库支持的 Anny；保留表示与定位的接口检查 | shape/pose 协议、几何检查、拓扑记录 |
| 多人统一 camera coordinate，理解 relative depth / scale / localization | Day 3 检查共享 K、camera-space joints/vertices 和相对深度 | 两人布局表、投影一致性与失败例 |
| **≥10 张复杂图像 stress test** | Day 1–5 累积 10 张复杂图像；简单对照图另计 | 每图配置、输出、failure taxonomy；不能只交视频 |
| 解释一张 RGB 如何产生多人 mesh 和 camera-space layout | Day 5 综合 Gate，增加短视频 tracking 理解 | 闭卷讲解、一个控制变量实验、周研究总结 |

原表的 SMPL-X 路线不机械继承到 Anny；本周不要求再复现旧 Multi-HMR，也不因总表提到 MMPose/PyTorch3D 就额外安装整套工具。单图定位与多人恢复仍是主线，tracking 用最短片段建立方法理解，Week 7 的 SfM 与 Week 8 的 world-space motion 不提前展开。

### 复现层级与完成边界

| 层级 | 本周要求 | 什么不能算完成 |
| --- | --- | --- |
| Demo Reproduction | 官方 checkpoint 成功推理，保存配置与输出 | 只有安装成功、截图或他人的结果 |
| Method Reproduction | **主要目标**：读懂关键 forward，独立验证输出、attention 数据流、几何与身份关联 | 只会复制命令；只背论文流程图 |
| Experimental / Benchmark Reproduction | 必做一个小型受控实验；正式 benchmark 仅在数据、映射、官方评测协议具备时开展 | 10 张自选图的现象统计不能称论文 benchmark 复现 |
| Training Reproduction | 本周不要求；只理解训练设计 | 加载预训练权重不等于复现训练 |

### 已有基础及证据等级

已核对现用资料目录中的 [Week 5 计划](week5.md) 主体、验收规则和代码附录使用说明。该文件是执行计划，其初始待办不代表最新完成状态；下表采用本次附件提供的更新进度，属于**本人提供的已有记录，本次未重新运行核验**。不能因 Mac 没有 WSL 实验文件就判断这些工作未完成。

| Week 5 记录 | Week 6 的使用方式 |
| --- | --- |
| 官方 HMR2 单图、SMPL 导出、forward trace、同图复跑已完成 | 不重新安排整天 HMR2 Demo；只比较单人/多人接口 |
| RGB crop `[1,3,256,256]` → backbone `[1,3,256,192]` → features `[1,1280,16,12]` → tokens `[1,192,1280]` → vertices `[1,6890,3]` | 作为已知对照，不能照搬到 Multi-HMR 2 |
| 24 个旋转检查、SMPL replay、Beta0 干预；replay 最大绝对差约 `2.98e-7` | 迁移“参数协议 → 独立回放 → 一致性验证”的思路 |
| camera 转换与 44 joints 投影；整体 Frobenius 差约 `1.15e-7` | 本周重点检查新的 K、位置定义及预处理坐标，不重讲基础投影 |
| PVE、Acceleration Error、FPS、frame ID 检查已有实现；`valid_mask` 完整测试未确认 | 只补与本周相关的缺失值/空集合反例，不能写成全部通过 |
| Fixed-BBox Occlusion 对照已完成；四组平均顶点位移约 `3.5139e-7 / 0.0184 / 0.0019 / 0.0179`，顺序依次为官方 vs 自定义原图、原图 vs 固定框遮挡、固定框 vs 重检框遮挡、原图 vs 重检框遮挡 | 继承控制混杂变量的方法；这些数是预测之间的位移，不是 GT 误差或本周结果 |

### 教学与协作约定

采用 **Understand → Run Official → Trace Source → Implement Tiny Experiments → Debug → Controlled Ablation → Evaluate → Research Log**。

本人先阅读、定位函数、预测 shape、实现小实验、提出解释。Research Agent 提供阅读顺序、检查题、Hint、反例与最小修改，审核证据后再过 Gate。默认不一次性给出完整脚本；只有本人明确要求、需要标准参考，或工程环境严重阻塞时才调整。解释错误先指出定义和根因，不直接重写整份实验。

### 五天预算

| Day | Understand / Trace | Run（实际运行与测量） | Independent Experiment / Debug | Analyze / Log | 合计 |
| --- | --- | --- | --- | --- | --- |
| 1：官方推理与输出 | 45m | 60m | 85m | 35m | 225m |
| 2：多人 Query 与架构 | 55m | 35m | 100m | 35m | 225m |
| 3：Camera-Centric 与 Anny | 45m | 45m | 110m | 40m | 240m |
| 4：Tracking 与身份关联 | 45m | 45m | 100m | 35m | 225m |
| 5：控制实验与失败分析 | 25m | 50m | 110m | 40m | 225m |
| 总计 | 215m | 235m | 505m | 185m | **1140m = 19h** |

Run + 独立实验/调试共 740m，约 65%，符合总表至少一半时间用于代码、实验和调试的要求。时间是执行预算，不是已经投入的工时。若本人尚不能解释关键接口，就缩减扩展任务并保留未过 Gate，不机械赶进度。

## 2. Paper Understanding

主论文：[arXiv:2606.14841v1](https://arxiv.org/html/2606.14841v1)，对照 [官方 README](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/README.md)。以下是阅读问题与代码核对任务，不代替本人精读。

| 阅读维度 | 本人需要回答的问题 | 本周应形成的判断 |
| --- | --- | --- |
| Problem / Previous limitation | 为什么 pelvis-centered 指标不足以衡量多人场景定位？固定相机会遗漏什么？ | 区分人体形状姿态、检测覆盖和相机坐标布局 |
| Core insight / Method | 为什么采用 DETR queries、共享相机内参和分解的位置预测？ | 用方法图与一条真实 forward 相互解释 |
| Why it works | Query 间交互、图像 cross-attention、位置参数化各试图解决什么？ | 给出机制假设及能反驳它的样例，不能只说“attention 更好” |
| Training | §3 的 matching、heterogeneous supervision 和 SAM2 feature distillation 分别监督什么？ | 区分论文训练机制与公开推理代码能验证的部分 |
| Inference | 哪些输出逐图独立？哪些历史信息只用于 track association？ | 区分 learned model、几何后处理、状态化 tracker |
| Metrics / Results | §4.1 表 1/2/4/6 中哪些指标需要 GT、多人匹配或相机协议？ | 不跨人体模型/相机设置直接对比数字 |
| Ablation | §4.2 哪个实验最直接支持一个核心机制？替代解释是什么？ | 只选择一条论证链，不为本周下载全部数据 |
| Limitations / Connections | HMR2、旧 Multi-HMR、Multi-HMR 2、后续 TRAM/Human3R 的能力边界怎样变化？ | camera-space 结果和 track ID 不自动构成 world-space 4D 重建 |

**作者声称的贡献**：面向相机坐标定位的多人检测与恢复，以及通过 SAM2 特征蒸馏支持身份关联；论文还讨论高效 matching 与多类监督。**本周优先检查的贡献**：共享相机和人体定位是否形成可解释的几何链，身份特征与历史几何如何实际参与关联。这是本周研究取舍，不是额外的新贡献结论。[论文 §3–4](https://arxiv.org/html/2606.14841v1)

本人完成 `notes/paper_map.md`，每个维度写“自己的解释 + 一处论文位置 + 一处代码或待验证项 + 一个局限”。训练损失、训练匹配与蒸馏训练不能仅凭推理成功验收。

## 3. Prerequisites & Environment

### 资源和工作区边界

执行设备以本人提供的信息为起点：WSL2 Ubuntu 24.04、RTX 5070 Laptop 8 GB、已有成功运行 HMR2 的 Conda 环境。**本次没有连接 WSL，也没有实测 Multi-HMR 2 的显存、速度或渲染兼容性。** Mac 保管计划与文献，WSL 放代码与实验。

```text
Mac: /Users/mac/CCFA_maker/01_10week_plan/week6.md

WSL（本人执行时按需创建）:
~/research-repos/multi-hmr2/       # 独立官方仓库，记录 commit 和本地 diff
~/10weekplan/week6/
├── exp/                         # 本人编写的 E1–E5 入口
├── inputs/                      # 输入清单、小图、短视频或指向原文件的记录
├── results/                     # 按 run_id 隔离，不复用同一输出/抽帧目录
├── notes/                       # environment、paper_map、shape_trace、camera、tracking
└── research_log.md
```

编写期间发现原 Mac 资料目录已迁至 `/Users/mac/CCFA_maker/01_10week_plan/`；总表和 Week 1–4 的内容哈希与读取时一致，本文件随现用资料目录保存。上述 WSL 路径是计划约定，尚未创建。官方仓库不放进 `~/10weekplan/week6/`；提交前检查 Git 根目录、暂存清单和嵌套 `.git`，只提交自己的小代码、轻量记录及必要图表。权重、大数据、授权资产不随手加入训练记录仓库。

### Day 1 开工检查

1. 先记录原 HMR2 环境名称、解释器、Python/PyTorch/CUDA 版本，保留现状；为本项目新建独立环境，不在原环境里升级 torch 或安装本项目。
2. 核对官方 `pyproject.toml`：Python `>=3.10`、`anny==0.6.0`；README 的示例使用 Python 3.10。按官方最小依赖开始，成功后记录实际解析出的版本，避免把“未固定版本”当成稳定环境。[依赖](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/pyproject.toml)
3. 检查 NVIDIA 驱动、可用显存及当前解释器对 GPU 的实际支持。已有 HMR2 成功不证明新环境对这张卡、Anny/Warp 或渲染一定兼容。
4. 查看已有 `~/.local/bin/researchdl` 的帮助/说明或实现，确认目前支持的操作；**本计划不臆造其子命令**。权重下载和 Git 下载分别确认。官方自动权重下载调用 `wget`；视频抽帧调用系统 `ffmpeg`，两者不是 pip 安装成功的必然结果。
5. 核对 checkpoint 文件名 `multihmr2.pt`、来源与 SHA-256；第一次 Anny 资产解析/缓存可能较慢，区分正常初始化、下载卡住和 forward 失败。[官方安装与用法](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/README.md)
6. 本次源码核对基线为 commit **`f066532d096034700e5ba8b7b1ca655729d3ea6e`**。本人执行时记录自己的实际 commit；有变化就先检查本计划所依赖的函数和输出协议，不盲目认定完全一致。

本计划只提供少量经源码确认的入口命令。以下由本人在独立环境中、官方仓库根目录执行，并把真实绝对路径写入日志；不会自动执行安装：

```bash
git rev-parse HEAD
git status --short
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
multihmr2 --help

# 首次先用默认人体拓扑，不开启 compile 和渲染；保存参数以保留实物输出。
multihmr2 --checkpoint checkpoints/multihmr2.pt \
  --image demo_data/sample_image.jpg \
  --out "$HOME/10weekplan/week6/results/d1_official_default" --save_anny_params
```

### 8 GB 推进顺序与停止条件

| 阶段 | 做什么 | 必须记录 / 何时调整 |
| --- | --- | --- |
| 默认单图 | batch=1，官方默认预处理，`compile=False`，不渲染 | 人数、输入分辨率、dtype、耗时和 GPU 峰值；成功才记 Demo PASS |
| 保存与可视化 | 推理成功后再安装独立环境内所需 render extras，做一张叠图/mesh | OpenGL 失败与预测失败分开；数值几何任务可先继续，可视化 Gate 待补 |
| OOM 定位 | 保存完整报错并定位 encoder / decoder / Anny / renderer 阶段 | 先确认实际空闲显存；不能承诺释放缓存或 `--lowres` 一定有效 |
| 低分辨率人体 | 只有需要时测试官方 `--lowres` | README 标为 613 顶点；这是人体拓扑变化，**不等于降低网络输入图像分辨率** |
| 网络降分辨率 | 仅作明确标注的自定义路径，先 trace 预处理再试 | 当前 CLI 没有 `--input_resolution`；字符串图像路径另走固定 768 预处理，不能只改一个属性就假定生效 |
| 视频 | 单图稳定后处理约 30–60 帧的小片段，不做整段 mesh 渲染 | 输入 FPS、抽帧数、输出列表内存、耗时、独立 tmp_dir；需要时只渲染代表帧 |
| 仍阻塞 | 单项先排查 15m，累计安装/兼容性排查预算 45m | 保存版本、报错、最小触发与已试方案，交环境专项；可继续只读 trace，GPU Gate 保持未完成 |

扩展顺序：默认推理 → 保存输出 → 小实验 → 一张渲染 → 短视频 → 可选优化。暂不从头训练，不默认启用 `torch.compile`，不为本周安装 COLMAP 或完整视频训练栈。迁移到其他 GPU 时记录设备和配置差异，不把异机结果写成本地 8 GB 实测。

## 4. Architecture Map

### 官方源码导航（已静态核对）

下列链接固定到上述 commit，路径相对 `~/research-repos/multi-hmr2/`。函数和配置已核对，**静态核对不等于在本设备执行成功**。

| 编号 | 真实源码 | 从哪里读、检查什么 |
| --- | --- | --- |
| S1 | [api.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/api.py) | `InferenceSession.__init__/__call__`、`init_hmr_session`、`infer_image`、`infer_video`；再看 `save_results_image/video` |
| S2 | [multi_hmr.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/models/detr_root_relative/multi_hmr.py) | `DETRConfig`、`DETR_Root_Relative.forward`；encoder → full_body_decoder |
| S3 | [encoder.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/models/detr_root_relative/encoder.py) | `Encoder.forward`、`EncoderOutput`；patch features、CLS → FOV → K |
| S4 | [decoder.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/models/detr_root_relative/decoder.py) | `DecoderConfig`、`Decoder.forward`、`PersonOutput`、`DecoderOutput`、`nms_3d_pelvis` |
| S5 | [hph.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/models/detr_root_relative/hph.py) | `HPH.forward`、`TransformerCrossAttn.forward`、`Attention.forward`、`CrossAttention.forward` |
| S6 | [trackfeat.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/models/trackfeat.py) | `TrackFeatRegressor.forward` 与特征维度映射 |
| S7 | [featpelvis.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/tracker/featpelvis.py) | `FeatPelvisTracker.reset/init_tracker/track_next_frame/_compute_assignment` |
| S8 | [featbank.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/tracker/featbank.py) | `FeatureBank.similarity/update/update_without_detection`；历史特征与过期规则 |
| S9 | [pelvis.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/tracker/pelvis.py) | `PelvisPredictorBank`、`RidgePredictor`；位置/方向历史与外推 |
| S10 | [tracker/tools.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/tracker/tools.py) | `hungarian_matching`、`get_combine_function`；阈值、匹配和加权 |
| S11 | [utils/camera.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/utils/camera.py) | `perspective_projection`、`inverse_perspective_projection` |
| S12 | [datasets/itw_image.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/datasets/itw_image.py) / [utils/image.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/utils/image.py) | `preprocess_image`、`ImagePreproc.forward`、`nonsquare_and_resize_image_without_K`；两条预处理路径 |
| S13 | [datasets/itw_video.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/datasets/itw_video.py) / [cli.py](https://github.com/naver/multi-hmr2/blob/f066532d096034700e5ba8b7b1ca655729d3ea6e/src/multihmr2/cli.py) | `extract_video`、`main`；抽帧命名、目录、真实支持的命令行参数 |

### 单图与视频的计算路径

```text
init_hmr_session(checkpoint) → InferenceSession
infer_image(session, path) → session.__call__
  → RGB 预处理 [B,3,H,W]
  → DETR_Root_Relative.forward
      → Encoder.forward
          ├─ image patch features → 映射/位置编码 → visual context
          └─ CLS → FOV head → 共享 K
      → Decoder.forward
          learned person queries + visual context
            → HPH [self-attention → cross-attention → MLP] × 8
            → loc / distance / confidence heads
            → confidence 筛选
            → pose / shape / tracking-feature heads → Anny
            → camera-space vertices / joints → 3D pelvis NMS
  → 单个 DecoderOutput（转到 CPU）

infer_video(session, video, tmp_dir)
  → extract_video → 按帧 infer_image
  → tracking feature + pelvis 位置/方向
  → FeatPelvisTracker（历史 feature/pelvis banks + Hungarian）
  → 每帧 persons.track_id → list[DecoderOutput]
```

注意这里的“distance head”是源码命名，不直接断言其输出是米或欧氏距离；Day 3 要沿运算确认。模型用固定 learned queries，不把 HMR2 的 bbox/crop 或旧 Multi-HMR 的 localization-to-query 流程原样套入。证据见 S1–S5。

### 输出契约和 Shape：已核对的定义与运行时待办

记 `B` 为 batch、`Q` 为 query 数、`N` 为筛选后的单图人数、`V` 为当前拓扑顶点数、`J` 为骨架数、`Hp=H/patch_size`、`Wp=W/patch_size`、`D_enc` 为真实 backbone 宽度。

| 对象/阶段 | 当前源码事实 | 本人运行时需要验证 |
| --- | --- | --- |
| encoder 输入/输出 | `[B,3,H,W]` → `feat[B,Hp,Wp,D_enc]`、`K[B,3,3]` | 实际 H/W、patch_size、D_enc、dtype/device；不固定成正方形 |
| HPH 输入 | 默认 `Q=100`、宽度 1024，queries `[B,100,1024]`；context `[B,Hp×Wp,1024]` | 用 hook 核对真实会话配置与 shape |
| attention | 默认 16 heads、每 head 32，投影内宽 512，输出恢复 1024 | self-attn 的 Q/K/V 与 cross-attn 的来源；不能把 32 当成 1024/16 |
| raw heads | `loc[B,Q,2]`、`_dist[B,Q,1]`、`conf[B,Q,1]`；保留 `conf > conf_thresh` | 大于而非大于等于；记录筛选前、阈值后、NMS 后数量 |
| `DecoderOutput` | `K[3,3]` 和 `persons: PersonOutput`；`len(pred)` 返回人数 | `pred` 不是普通 person 列表；字段位于 `pred.persons` |
| 人体输出 | `conf[N,1]`、`shape[N,11]`、`v3d[N,V,3]`、`j3d[N,163,3]`、`j2d[N,163,2]`，默认配置如此 | `phenotype_labels`、`bone_labels`、完整拓扑实际 V 与 faces；README 的约 10k 不作精确 V |
| pose 输出 | `bone_poses`、`rest_bone_poses` 为 Anny 骨骼变换；empty 分支建立 `[0,163,4,4]` | 非空实际 `[N,J,4,4]` 及局部/整体语义，不能当作 SMPL `body_pose` 或输入 6D 向量 |
| pelvis 输出 | `transl_pelvis` 来自已放置后的 `j3d` 第 0 骨骼；非空路径预期 `[N,3]` | 与原始 inverse-projection 的 `transl` 区分；全空 early-return 源码使用 `[0,1,3]`，适配器须检查 |
| tracking 输出 | 默认 feature head 宽度 4096；`trackfeat` 可选；`track_id` 默认 None | 单图不生成 track ID；视频分配 ID 后为 `[N]`；全空 feature 可能为 `[0,0]` |
| 视频返回 | 一个每帧 `DecoderOutput` 组成的 list | 每帧人数可变；空帧必须保留，不能把时间轴压缩 |

以上数值来自 S3–S6 的默认配置/运算，尚非本设备实测。当前代码含 empty 分支形状差异；测试应暴露并记录它，不能通过无条件 `squeeze()` 抹掉 batch/person 维。

### 三项容易误读的论文—实现差异

1. **Anny shape 维数**：论文介绍六个可解释体型参数；当前 `DecoderConfig.num_betas=11`，以 `phenotype_labels` 逐项建字典。实际配置与标签必须记录，不能把六维写成本仓库输出事实，也不能把 SMPL betas 的语义搬过来。
2. **FOV 的方向**：论文 §3 描述 vertical FOV；当前 `Encoder.forward` 用图像宽度计算 `f=(W/2)/tan(FOV/2)`，因此实现处应按 horizontal FOV 追踪。用非方形图核对派生的 horizontal/vertical FOV，不自行改源码“统一”文字。
3. **Tracking 特征距离**：论文描述 L1 检索；当前 `FeatPelvisTracker` 默认对特征做 L2 normalization，并给 FeatureBank 传 `pdist=2`。同时结合 pelvis 历史外推，组合权重为 1:8。读论文、类自身默认值、API 实际实例化参数需要分开记录。

另一个边界：camera-centric K 与人体平移不提供 camera-to-world 外参。本周不存在已验证的全局相机轨迹；track association 的 ridge 外推也不等于对 mesh 进行 temporal smoothing。当前 decoder 对部分骨骼姿态置为单位变换，因此“whole-body”不代表所有面部/脚部关节均独立回归，相关能力判断须查 `_indices_to_discard`（S4）。

## 5. Day 1–5 Detailed Tasks

下面五天是同级任务；每天采用同样九个栏目。论文部分、源码、检查题、独立实验、Hint、反例与 Research Log 都落在各自栏目中。E1–E5 的详细输入输出契约见第 6 节。

### Day 1 — Official Demo & Output Exploration

#### 当日目标

跑通一次默认官方单图推理，自己解释 `DecoderOutput → PersonOutput`。建立环境与输入清单；区别推理、导出和渲染成功。

#### 前置条件

复用 Week 5 的 output parsing 与 shape 习惯。确认独立环境，不重复基础 PyTorch/SO(3) 训练；记下仍未确认的 valid_mask 测试。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 45 | 阅读摘要、§3 Overview / Inference on images；核对依赖与 S1/S4 |
| Run | 60 | 开工检查、一次官方默认推理、记录内存与保存输出；环境排查包含在内 |
| Independent Experiment / Debug | 85 | 本人完成 E1，检查简单对照图与 C01/C02，保存一张代表性可视化 |
| Analyze / Log | 35 | 回答检查题、整理输出 schema 与阻塞 |

#### 核心资源

论文摘要、图 1、§3 Overview、Inference on images；README Installation / Python API。源码按 S1 的 session 初始化与 `infer_image` → S2 forward → S4 两个输出 dataclass 读，暂不追全 tracker。

理解检查：为什么 `len(pred)` 能返回人数？单图有 `trackfeat` 是否就有 `track_id`？保存的参数 `.pkl` 是否含完整身份记录？默认命令未开任何输出 flag 时，文件夹存在能证明预测已保存吗？

#### 执行步骤

1. **Understand / Trace**：先写出预期输入输出对象和不知道的字段；核对真实构造函数，不凭名字猜 camera/pose 含义。
2. **Run Official**：先执行默认单图，记录 commit、权重 hash、真实命令、首跑/稳态时间。若再做一次复跑，只用于建立本环境数值基线，不无限重复成功 Demo。
3. **Implement / Debug**：本人写 E1，打印字段的 type/shape/dtype/device/finite、人数与置信度；没有字段或值为 None 就如实记录。保存可重载的原始 tensor/元数据，而非只有终端截图。
4. **Analyze**：本日用 S01 简单单人、S02 简单多人和 C01/C02 复杂图。若短视频所需依赖已就绪，可只检查 API 入口；正式 tracking 留 Day 4。

Hint：先看 dataclass，再看 `__len__` 与 `.to()`。官方 `--save_anny_params` 保存 K/shape/pose_parameters，**未保存 track_id/trackfeat**；E1/E4 需要另行序列化自己关心的字段（S1）。

正常测试：一张能检出至少两人的图。反例：用 `conf_thresh=1.0` 检查空输出（先核对有限 sigmoid 输出和 strict `>`）；再用合成 schema 检查 N=1 的 singleton 维，不能假定背景图一定无人检出。

#### 交付物

`notes/environment.md`、`inputs/manifest.csv`、`exp/inspect_outputs.py`（本人编写）、`results/<run_id>/schema.json`、可重载预测、至少一张可视化。输入清单先登记全部 10 张复杂图和两个简单对照，不要求今天全部推理。

#### 验收标准（Gate）

- [ ] 默认单图成功，保存配置与至少一个实际预测输出；是否渲染成功单独记录。
- [ ] 本人不看 Agent 成稿，解释 K、persons、conf、v3d、j3d、shape、bone_poses、trackfeat/track_id 的位置与已知/未知语义。
- [ ] E1 通过 N=0/1/多人测试；shape 差异可见，缺失项未冒充零值。
- [ ] C01/C02 的输出和失败观察已保存；全部本周 Gate 仍按实况勾选。

#### 受阻处理

环境超过预算则保留错误并转环境专项；可继续读 dataclass、用自造 schema 练习，但它不替代官方 Demo。渲染失败时先完成 tensor 检查和简单 2D joints 叠图，mesh 可视化仍登记待补。

#### 下一步

Research Log：我今天亲自证实了什么？哪个字段的名称最容易误导我？还有哪项只来自源码、没有运行证据？Day 2 带上自己的 schema 与三个最不确定的 tensor。

### Day 2 — Multi-Person Queries & Architecture

#### 当日目标

定位从图像到多人体预测的真实 forward，解释 learned queries、有效检测与最终人体实例的区别，并指出 query 交互的代码证据。

#### 前置条件

Day 1 输出协议可解释；复用 Week 2 的 attention/QKV 与 Week 5 的 hook 经验。若 Day 1 推理未过，可先读代码，但运行 shape Gate 不能提前通过。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 55 | 定向阅读 §3 Image Encoder / Decoder / Detection score / Matching，定位 S2–S5 |
| Run | 35 | 在一次推理中捕获 encoder、HPH、heads 的真实输入输出 |
| Independent Experiment / Debug | 100 | 本人完成 E2，测试非方形输入与筛选边界；处理 C03/C04 |
| Analyze / Log | 35 | 画带 shape 的数据流，回答 QKV 与实例数问题 |

#### 核心资源

先读 `DETR_Root_Relative.forward`（S2），再读 `Encoder.forward`（S3）、`Decoder.forward`（S4），最后展开 `TransformerCrossAttn.forward`、`Attention.forward`、`CrossAttention.forward`（S5）。训练 Hungarian matching 的论文段落用于理解，不虚构当前仓库不存在的训练入口。

理解检查：100 个 queries 为什么不是 100 个人？self-attention 的 Q/K/V 来自哪里？cross-attention 的 Q 与 K/V 为什么序列长度可以不同？模型在哪一步才调用 Anny？NMS 会不会让近距离真人相互抑制？Query slot 是跨帧身份吗？

#### 执行步骤

1. **Understand / Trace**：本人先画路径，标出 CLS 分支、context、query、各 head、筛选与 NMS；对照源码纠正自己画错的顺序。
2. **Run**：写最少 hook 捕获 `encoder`、`full_body_decoder.decoder`、`mlp_conf` 等实际模块；先用真实对象确认位置。记录输入输出结构，不假定 hook 收到的都是单一 tensor。
3. **Implement / Debug**：完成 E2，记录 B、H/W、Hp/Wp、Q、D、注意力头数与 head width。新增 C03/C04；若已保存同配置预测，复用而非为截图重复推理。需要 pre-NMS 信息时采用局部插桩，保存 diff。
4. **Analyze**：给出阈值前/后/NMS 后的人数流；指出 query 与 query 通过 self-attention 发生交互的实际调用。若没读取注意力值，不声称已经解释某个 query “关注”了哪位真人。

Hint：先抓 HPH 整体输入输出，再看其内部；默认 cross-attention 的 score 轴由 Q 与 context 长度共同决定。Hook 只保存 detach 后的必要 shape/小样本，使用后移除，避免把整个计算图或注意力矩阵长期留在显存。

正常测试：多人横图，验证 context token 数等于 Hp×Wp。反例：竖图或 H≠W，暴露把 tokens 当成固定平方网格的代码；置信度恰等阈值的合成例检查 strict `>`；Q 保持 100 但 N=0。

#### 交付物

`exp/trace_shapes.py`、`notes/shape_trace.md`、一张本人画的架构图、hook 配置/局部 diff、C03/C04 输出、阈值与 NMS 数量记录。记录实际值，不能直接复制第 4 节表格充当运行证据。

#### 验收标准（Gate）

- [ ] 能独立打开关键 forward 并从图像追到 PersonOutput。
- [ ] 解释全部 Q/K/V 来源、query 交互、100 queries 与可变 N 的关系。
- [ ] 有一次真实 shape trace，含非方形输入与筛选阶段；E2 反例能暴露错误假设。
- [ ] 能说明与 HMR2 单人 token/crop 流程的两项实质区别，未把 query index 当 track ID。

#### 受阻处理

Hook 遇 dataclass/tuple 时先打印类型并写小适配，不重写模型。拿不到中间筛选数就明确注明只测到最终 N，补一次局部插桩；不能反推未观察到的中间结果。

#### 下一步

Research Log：我原先把哪两个轴混淆？何处证明存在 query 交互？哪项论文训练机制目前无法从公开推理代码验证？Day 3 复用本人的 K、j3d、j2d 与两个人体输出。

### Day 3 — Camera-Centric Localization & Anny

#### 当日目标

解释两个人如何共享同一个相机坐标系，自己完成投影与相对位置检查，并读懂 Anny 表示接口和完整/低分辨率拓扑的边界。

#### 前置条件

复用 Week 5 的投影与 SMPL 参数实践。本日重点是坐标与表示迁移；不能直接沿用 SMPL joints、6890 顶点或 betas 定义。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 45 | 阅读 §3 Camera intrinsics / Location prediction / HMR prediction，定位 K 与人体放置 |
| Run | 45 | 复用已有预测，检查标签、faces、骨骼与双人布局；新增 C05/C06 |
| Independent Experiment / Debug | 110 | 本人完成 E3；验证投影、逆投影、pelvis 与拓扑；按余量做单人 Anny replay |
| Analyze / Log | 40 | 解释几何差异、记录单位/frame 与 monocular 不确定性 |

#### 核心资源

S3 的 `fov → focal_length → K`；S4 的 `mlp_loc/mlp_dist → inverse_perspective_projection → Anny → person_center → j3d/v3d → j2d`；S11 的两个几何函数；S1 的参数导出逻辑。需要 replay 时查安装的 Anny 0.6.0 实际接口，不虚构 `smplx` 参数名。

理解检查：K 是内参还是外参？源码 `dist=f/exp(_dist)` 中的 dist 是 optical-axis depth 还是射线长度？`transl` 与 `transl_pelvis` 为何不能盲目相等？bone_poses 的平移能直接和输出 j3d 比吗？全图深度顺序预测正确是否证明米制深度准确？

#### 执行步骤

1. **Understand / Trace**：本人解释公式与代码对应：对 `u=(u,v)`、`ũ=(u,v,1)`，`X=Z K⁻¹ũ`；代码用 focal 参与由 `_dist` 恢复 Z，不把 `_dist` 当米制深度。人体放置为 `X_camera=X_body−person_center+transl`。
2. **Run**：选同图两人记录 pelvis XYZ、Z 排序、共同 K 与相对向量；输出 bone/phenotype labels 和 topology。**同帧 camera-space 可比较，跨帧相机运动时不能直接串成 world trajectory。**
3. **Implement / Debug**：本人投影 `j3d` 与官方 `j2d` 比较，完成 E3 的已知 GT 合成 round-trip；同时检查 `transl_pelvis` 对应的 root。针对低分辨率仅做一次必要对照，检查 V/faces 变化，不逐顶点比较不同拓扑。
4. **Analyze**：追出 `bone_poses` 所在 Anny frame 与 camera-space j3d 的平移关系；记录还未确认的单位/轴向。若做 replay，先选一人，按官方 world 参数导出方式与 `root_relative_world` 接口对照；解释 pose/bone transforms 与 root translation 的处理，再比较同拓扑结果。未成功就保留待补，不伪称 method 已完全回放。

Hint：K 对应预处理后的图；原图叠图必须追 resize/padding 变换。当前代码 `person_center` 用全分辨率顶点 5054 或 lowres 顶点 9；它不是直接取骨骼第 0 点。不要再给已在 camera-space 的 v3d 加一次 pelvis 平移（S4）。

正常测试：两个正深度的人体与非方形图，先保存每点残差再汇总。反例：Z=0/负深度、把原图像素与预处理 K 混用、给全部人体重复加平移、full/lowres faces 混配；应明确失败或无效，不能填成误差 0。

#### 交付物

`exp/check_geometry.py`、`notes/camera_anny_protocol.md`、`results/<run_id>/projection_checks.csv`、双人布局表/图、实际 V/J/标签/单位/frame 记录、C05/C06 输出。可选 replay 要单独保存参数化设置、数值差与失败定位。

#### 验收标准（Gate）

- [ ] 本人能解释同帧两人的相对 XYZ、共享 K 与相机坐标边界。
- [ ] E3 合成正常/反例通过，真实 j3d→j2d 一致性有残差及容差说明。
- [ ] 区分 Anny shape/pose、SMPL betas/body_pose、camera placement 与 world extrinsics。
- [ ] 完整和 lowres 拓扑差异有记录；没有跨拓扑逐顶点误差结论。
- [ ] 能指出 root / center / bone transform 各自来自哪里；Anny replay 若未完成，如实登记。

#### 受阻处理

投影不一致时依次查图像坐标变换、K、轴向/单位、Z、是否重复平移、dtype；先缩成单人单点。Anny replay 接口阻塞不影响先做输出几何检查，但不能把几何自洽写成 Anny 参数回放成功。

#### 下一步

Research Log：投影一致性证明了哪段实现，不能证明什么？哪一个坐标约定若改变会推翻深度解释？Day 4 只将已解释的 pelvis 与 feature 交给 tracker，不携带未经确认的“世界位置”。

### Day 4 — Video Tracking & Identity Association

#### 当日目标

追踪真实跨帧状态与关联规则，自己记录每帧实例和 track ID，区分失检、ID 异常、轨迹终止与 mesh 抖动。

#### 前置条件

Day 1–3 的单图输出可解释；一个约 30–60 帧、至少两人有交互或短遮挡的片段。若视频 FPS 与抽帧策略不明，先记录再做时间统计。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 45 | 阅读 §3 Regressing track features / Tracking in videos 与 §4 tracking 指标 |
| Run | 45 | 运行短视频 API，保留逐帧输出与独立抽帧目录 |
| Independent Experiment / Debug | 100 | 本人完成 E4；手工可核对的 ID 序列测试；定位空帧/过期状态分支 |
| Analyze / Log | 35 | 看代表帧，解释候选异常，写 tracking 状态图 |

#### 核心资源

按 S1 `infer_video` → S7 `FeatPelvisTracker.track_next_frame` → S8 `FeatureBank.similarity` → S9 `PelvisPredictorBank.similarity` → S10 `hungarian_matching` 阅读。再回看 S6 解释 trackfeat 从哪里产生；不要把 SAM2 的完整视频系统当成本模型推理依赖。

理解检查：谁持有跨帧 feature/pelvis banks？新视频在哪 reset？新 ID 在哪里创建？为什么历史位置外推不等于输出平滑？`max_tskip=50` 是帧还是秒？同一个 query slot 连续出现能证明同一真人吗？视频参数 pkl 是否足够还原关联？

#### 执行步骤

1. **Understand / Trace**：本人画“当前检测 → 两类 similarity → 组合 → Hungarian → 沿用/新建 ID → 更新 banks”的状态图，记录实际默认配置和各函数调用。
2. **Run**：本人调用 `infer_video`，将每帧的 frame_id、输入时间戳、N、track_id、conf、pelvis 与所需 feature 写到自己的结果文件；空帧也写入。只渲染少量代表帧，不重复跑全片以换不同颜色。
3. **Implement / Debug**：完成 E4；数每帧检测人数、每条 ID 的观测帧数/跨度/内部缺口。对两人交叉、短遮挡和错误复用 ID 设计可手算反例。统计代码先在合成序列验证，再处理视频输出。
4. **Analyze**：逐帧检查一个候选关联异常，先区分检测没产生人、阈值/NMS 丢弃、feature 混淆、pelvis 外推偏差或历史状态问题；有证据再缩小原因。

Hint：当前 `infer_video` 只在非空帧调用 `track_next_frame`，空帧直接给空 ID；而 tracker 本身有空检测时推进/清理 bank 的分支。**两条路径不是自动等价。**连续空帧与超过历史窗口后的重现，应作为边界测试；这里只提出源码支持的检查点，不预判真实视频一定会触发错误（S1/S7–S9）。

正常测试：两条连续 track 的已知 ID 序列。反例：中间空帧、ID 先消失后出现、不同帧人数改变；可选用 CPU 合成 tracker 输入检查第 49/50/51 帧附近的历史窗口，先核对代码的不等式。每次新实验 reset，不能让两段视频共享旧状态。

#### 交付物

`exp/analyze_tracks.py`、`results/<run_id>/frames.csv`、`tracks.csv`、带 frame_id 的原始输出、代表帧图、`notes/tracking_state.md`。记录视频 FPS、帧编号从 0 还是 1 开始；API 时间索引与导出六位文件名相差一位，显式映射（S1/S13）。

#### 验收标准（Gate）

- [ ] 能定位 tracking 入口、持续状态、新建 ID、匹配和过期规则。
- [ ] 逐帧输出长度与抽帧清单一致，空帧未丢失；E4 人工序列结果正确。
- [ ] 能给出一个异常的候选原因与下一项排除实验。
- [ ] 未将启发式异常计数称为标准 IDSW/IDF1/MOTA；没有 GT identity 时只报告候选异常。
- [ ] 明确本系统的历史关联、pelvis 外推和 mesh temporal smoothing 是不同环节。

#### 受阻处理

视频推理 OOM 时先缩短片段并避免渲染；不要通过丢掉所有困难帧伪造稳定跟踪。缺 ffmpeg/视频依赖时可做 CPU tracker 与统计练习，但真实视频 Gate 保留待补。看到空帧行为差异，先复现和记录，不立即重写官方 tracker。

#### 下一步

Research Log：哪条状态真正影响了下一帧？我认为的“ID switch”有没有人工身份依据？能否用更小的序列反驳当前原因假设？Day 5 选择一个已观察到的问题，其他候选暂存。

### Day 5 — Controlled Experiment & Failure Analysis

#### 当日目标

完成至少一个本人独立实现、可重复、有数值与反例的受控实验，交齐 ≥10 张复杂图像压力测试，提出一个下一阶段值得验证的问题。

#### 前置条件

复用 Day 1–4 的已有预测、schema 与几何/统计函数；不先做 benchmark、不一次改变多个参数。环境 Gate 未完成就将本日实验标为受阻，不能用计划代替结果。

#### 时间安排

| 阶段 | 分钟 | 动作 |
| --- | --- | --- |
| Understand / Trace | 25 | 回查与选题直接相关的论文/代码，写实验预注册卡 |
| Run | 50 | 完成 C07–C10 的默认 baseline，汇总 10 张复杂图像 |
| Independent Experiment / Debug | 110 | 本人完成 E5，做最小对照、匹配检查与重复性验证 |
| Analyze / Log | 40 | 写事实/推断/不确定性，完成 Final Gate 与一页周总结 |

#### 核心资源

论文 §4.1 的检测/场景定位指标与 §4.2 对应 ablation；选题若为遮挡，回查 S3/S4 的 K 与人体定位；若为阈值，回查 S4 的 strict comparison 和 NMS；若为 tracking 配置，回查 S7，确认需要自己显式实例化 tracker，当前 CLI 并无通用 tracking 超参入口。

理解检查：自变量是什么？匹配失败的人有没有被静默排除？预测 mesh 改变是否就是 error 增加？K 的变化是设置被改动还是模型对遮挡的响应？十张图的结果可以推广到完整 benchmark 吗？

#### 执行步骤

1. **Understand / Trace**：本人先填第 7 节实验卡。默认选择 **固定图像尺寸与预处理的局部遮挡敏感性实验**；如果基础接口仍不稳，选 confidence threshold 后处理实验，明确它检验的是筛选机制，研究结论范围更小。
2. **Run**：先用同一默认配置补齐 C07–C10，不重跑已有 baseline。将 10 图按 detection/localization/depth/occlusion/truncation/crowd 分类；允许多标签，不把分类原因当成已证明根因。
3. **Implement / Debug**：在两张已跑过的多人图上，针对一个事先选定的人，保持分辨率、文件读入路径、权重、threshold/NMS、topology 和编译设置一致；只改变遮挡面积（0%、预定小值、预定大值），共至多 6 个条件，原图结果可复用。遮挡位置、填充值和 seed 先写定，不按结果挑样例。
4. **Analyze**：比较同一被匹配人体的检测保留、confidence、pelvis 位移、同拓扑 mesh displacement、K/焦距变化；未匹配实例单独报告。若主张某个机制，写下一项可以区分它与替代解释的实验，不把观察直接提升为因果定论。

Hint：本模型不依赖外部 person bbox detector/crop；Week 5 的 Fixed-BBox 方法迁移成“固定整图预处理与后处理配置”。预测 K 随遮挡变化可以是响应路径，记录它；不要悄悄将固定 K 当成实验控制，因为当前高层 API 没有该开关。

正常测试：0% 遮挡生成图与原图的像素/预处理 tensor 一致，预测落在本环境重复性基线内。反例：把人物输出顺序打乱仍应匹配正确；目标消失必须计漏检/未匹配；遮挡区完全在画面外应拒绝无效条件，而非当作成功干预。

#### 交付物

`exp/controlled_experiment.py`、实验卡、完整 conditions/配置、`stress_test.csv`（至少 10 个复杂图像 ID）、`ablation.csv`、原图/变体/叠图、`notes/failure_analysis.md`、`notes/weekly_review.md`、更新后的 research_log。结果栏在本人实跑前保持“待测”。

#### 验收标准（Gate）

- [ ] ≥10 张不同复杂图像有默认配置输出及观察；两个简单对照不计入数量。
- [ ] 一个受控实验由本人实现，有固定条件、可复跑命令、数值、失败例和匹配规则。
- [ ] 至少一组条件复跑核对可重复性，没有只保留成功样本。
- [ ] 结论分成事实、证据支持的推断、当前无法判断、局限和下一项验证。
- [ ] 能解释完整方法链，提出一个范围可控的 video-based HMR 问题；未宣称 novelty 已成立。

#### 受阻处理

如果遮挡实验被匹配/几何错误阻塞，先修最小测试；可切换为同一预测上的 threshold 实验，明确改题与结论边界。不得拿人工挑选的好看可视化或另一个模型输出替代当前模型的实验 Gate。

#### 下一步

Research Log：哪项假设被证据否定？哪个关键事实改变会推翻结论？再做哪一个最小实验最有信息量？按第 12 节衔接 Week 7–9，不直接扩成视频训练工程。

## 6. Independent Coding Exercises

以下文件名与接口均为**本人要编写的练习建议**，不是声称官方存在同名函数，也不是已交付的参考实现。每项先写输入输出/预期，再实现、测试、给 Agent 审查；默认只获得 Hint 和局部修改。

### E1 — 输出审计与序列化

- **数学定义/行为**：输出人数 N 应与所有逐人非空字段的第一维相符；schema 保留缺失、None、空 tensor 的区别。
- **输入/输出契约**：输入单个真实 `DecoderOutput`；输出 JSON schema 与可重新加载的原始 tensor/必要元数据。不将 `str(pred)` 当成数据保存。
- **Shape**：K `[3,3]`、conf `[N,1]`、shape `[N,11]`（默认），joints/vertices 和可选字段按第 4 节；N=0 分支存在差异，先识别再决定适配。
- **Hint**：从 dataclass fields 遍历；保存 detach/CPU tensor，不以无条件 squeeze 消除维度。
- **正常测试**：多人输出保存后重载，人数、字段 shape、数值与 None 状态一致。
- **反例**：N=1 时保留 `[1,1]` confidence；N=0 时不对空 tensor 求 min/max；track_id=None 不伪造 ID 0。
- **验收**：读写一致、有 finite 检查和显式 empty policy；人工能解释每类字段，而非只展示打印结果。

### E2 — Query / Attention Shape Trace

- **数学定义/行为**：`Attention(Q,K,V)=softmax(QKᵀ/√d, dim=key_axis)V`；self-attention 的输入为同一 query 序列，cross-attention 的 K/V 来自图像 context。
- **输入/输出契约**：一次真实图像 forward 和少量已定位模块；输出 `module、input/output type、shape、dtype、device` 表，附 hook 注册/移除清单与源码位置。
- **Shape**：queries `[B,Q,1024]`、context `[B,Hp×Wp,1024]`；默认 cross score 若展开为 `[B,16,Q,Hp×Wp]`，self score 为 `[B,16,Q,Q]`，这是运算推导，实测时标明是否真的捕获。各 head 的 q/k/v 宽度默认 32。
- **Hint**：先捕获整体 HPH，再检查 `to_q/to_kv/to_qkv`；只记录实际需要的数据，避免存全量 attention。
- **正常测试**：多人非方形图 token 数与 H/W、patch size 对上；筛选前 Q 固定，最终 N 可变。
- **反例**：转成竖图，固定 48×48 假设应被检测；合成 score 的 0.4 与阈值 0.4 相等时应剔除；不能沿 query 轴做 key softmax。
- **验收**：真实运行与代码推导明确区分，能指出 query 交互与筛选顺序。未修改 query 数量以迁就测试。

### E3 — 共享 K 的几何检查

- **数学定义**：`u=π_K(X)=([K(X/Z)]_x,[K(X/Z)]_y)`；`X=Z K⁻¹[u,v,1]ᵀ`。这里 Z 是光轴深度；`ΔP=P_b−P_a` 表示同帧相对位置。
- **输入/输出契约**：相机坐标 points `[N,J,3]`、共享 K `[3,3]`、有效点 mask `[N,J]`；输出 pixels `[N,J,2]`、逐点 residual、有效数、invalid 原因与汇总。批处理时显式扩为 `[N,3,3]`。
- **Shape**：空人数、单人和多人均保持 person 维；full/lowres 的 V 不同，不能广播成同一 mesh。
- **Hint**：先用手算点验证，再比较官方 j2d；`X_body−center+transl` 与已经放置的 X_camera 分开。只比较有效正深度/finite 点，不能让 NaN 乘 0 后继续污染均值。
- **正常测试**：取 `K=[[500,0,320],[0,500,240],[0,0,1]]`、`X=[1,0,5]`，投影应为 `[420,240]`；逆投影再恢复 X。它是预设合成答案，不是模型结果。
- **反例**：Z=0/负值、mask 全 false、K 不可逆、把 `[320,240]` 原图中心错误用于另一预处理尺寸、米/毫米混用。
- **验收**：合成 float64 round-trip 可先用 `atol=1e-8`，像素用 `atol=1e-6`；真实混合精度比较先声明候选像素容差（如 `1e-2 px`）并报告实际 max/mean/有效数，超限先定位，不为过关任意放宽。全无效返回明确 unavailable，而不是 0。

可选 **E3b Anny 单人 replay**：输入本人从 S1 导出协议理解后的 shape/pose 与准确模型配置，输出同拓扑 mesh；先确认 Anny 0.6.0 参数化接口和 phenotype 标签，记录 root 平移注入。正常例为同参数回放；反例为重复平移、root_relative/world 混用或 faces 不匹配。不预设误差一定达到 Week 5 的 `2.98e-7`。

### E4 — Track 统计与人工序列反例

- **数学定义**：每帧人数 `N_t`；ID k 的观测集合 `T_k={t | k 在 t 被观察}`；长度为 `|T_k|`，跨度为 `max(T_k)−min(T_k)+1`，内部缺口为该区间内不属于 T_k 的帧。长度与跨度不同。
- **输入/输出契约**：逐帧 frame_id/timestamp + ID tensor `[N_t]`，可选 pelvis `[N_t,3]`；输出 frames/tracks 表与候选异常列表。固定步长帧和不规则时间戳分别处理。
- **Hint**：先保留完整时间轴，再 group by ID；同帧同 ID 出现两次要直接报异常，不能去重后掩盖。
- **正常测试**：四帧 IDs 为 `[0,1]、[0,1]、[1]、[0,1]`；ID 0 长度 3、跨度 4、内部缺口 1，ID 1 长度 4。该例只验证统计，未模拟真实关联。
- **反例**：`[0,1]、[]、[0,2]` 不能删除中间空帧；同帧 `[3,3]` 应报错；输入 frame_id 不连续时不按列表下标伪造相邻帧；无 GT 时不计算“真实 IDSW”。
- **验收**：合成答案正确，真实 CSV 与抽帧数相符，异常可回到原帧人工核对。若只观察到高速度/位置跳变，命名为候选异常，不断言身份已切换。

### E5 — 最小控制实验运行器

- **数学定义**：同一被匹配人体、同一拓扑下的预测变化 `D_V=(1/V)Σ_v ||V'_v−V_v||₂`；pelvis 变化 `D_P=||P'−P||₂`；还可计算 pelvis-centered displacement，并与绝对变化分开。
- **输入/输出契约**：固定输入 manifest、baseline config、一组只改变单个因子的条件、显式人体匹配表；输出每个条件的 raw predictions、匹配/未匹配、数值、配置与图。不从输出顺序假定同一个人。
- **Shape**：每次 `[N_c,V,3]`，N_c 可变化；只对双方有效匹配且同 topology 的人体计算位移，未匹配率/数量同时报告。
- **Hint**：先验证 0% 遮挡是 identity transform；匹配可用人工身份标签和图像位置辅助，小样本优先人工复核。不要仅凭 learned query slot 作跨条件身份标签。
- **正常测试**：无变化输入与一次必要复跑，得到环境自身的数值差基线；遮挡只改变预定区域，图像尺寸/预处理保持一致。
- **反例**：打乱 person 顺序，统计仍应一致；目标消失应记 unmatched；不同 V 应拒绝逐顶点计算；全部无匹配应返回 unavailable。
- **验收**：本人可复跑，图、CSV 与结论对应；所有单位、frame、匹配策略清楚，未把 D_V 称作 PVE、提升或 GT error。

## 7. Experimental Design & Metrics

### 数据与预算

| 输入 | 最小数量 / 分配 | 使用规则 |
| --- | --- | --- |
| 简单对照 S01/S02 | 1 张单人、1 张简单多人，Day 1 | 检查接口，不计入复杂图数量 |
| 复杂图 C01–C10 | 至少 10 张不同图；遮挡 3、截断 2、拥挤 3、深度/尺度差 2，可多标签 | Day 1 跑 2，Day 2 跑 2，Day 3 跑 2，Day 5 补 4；保存每图 baseline |
| 视频 V01 | 约 30–60 帧，至少两人 | 使用有权限的小片段，固定 FPS/抽帧与 tmp_dir；不预先大规模渲染 |
| 遮挡变体 | 两张已用图 × 3 个等级，baseline 可复用 | 先写 ROI、面积定义、颜色和 seed，避免按结果挑图 |
| 合成数据 | E1–E4 手工可算的最小 tensor/序列 | 检验实现正确性，不替代真实模型或 benchmark |

`inputs/manifest.csv` 至少含 `input_id、来源/许可、文件hash、原始尺寸、场景标签、运行日、配置ID、输出路径`。人为粗计“可见人数”要写规则（例如头/躯干可辨认，镜像/海报是否计入）和 uncertain 标记；没有完整标注时不称标准 detection GT。

### 实验前填写的卡片

| 字段 | 本人运行前填写 |
| --- | --- |
| Observation | 哪张图/哪帧观察到了什么？证据文件在哪里？ |
| Hypothesis | 可以被哪种结果反驳？同时列一个替代解释 |
| Independent Variable | 只选遮挡比例、置信度阈值、或一个实际支持的 tracker 设置 |
| Controlled Variables | 输入/变体生成、权重、commit、预处理、dtype、阈值/NMS、topology、seed、reset 规则 |
| Metric | 精确定义、单位、坐标系、mask、匹配、分母及无效值政策 |
| Expected Outcome | 运行前写方向/模式；不要事后把任何结果都解释为支持 |
| Failure Criteria | 实现错误、身份匹配失败、无有效样本、数值不稳分别怎样处理 |
| Reproduction | 实际命令、解释器、配置、run_id、原始结果文件与代码版本 |

推荐主实验是遮挡敏感性。备用 threshold 实验可选 `0.2/0.4/0.6`（建议值），固定 NMS 与同一图像，预期观察筛选数随阈值变化；这主要检验后处理和检测取舍，不能单凭人数更多就声称更好。改变 tracking settings 需要自己的显式调用和配置记录；降低网络输入分辨率需要独立验证预处理，不存在已核对的通用 CLI 开关。

### 本周可报告与有条件报告的指标

| 指标 | 最少需要什么 | 允许的结论 |
| --- | --- | --- |
| schema/finite/shape、projection consistency | 本模型输出与实现契约 | 检查实现/几何自洽；不能证明人体恢复准确 |
| 检测人数、confidence、候选漏检/重复 | 输出；人工观察要明确覆盖与计数规则 | 小样本现象；没有完整 GT 不报告标准 precision/recall/AP |
| 相对深度顺序、pelvis/mesh displacement | 相同 camera-space 约定、有效匹配；mesh 还需同拓扑 | 预测布局/敏感性；不等于真实深度误差或 PVE |
| track 长度、跨度、缺口、候选异常 | 完整 frame_id 与预测 ID | 身份输出的描述统计；不是标准 IDSW/IDF1/MOTA |
| 推理耗时 / FPS / 峰值显存 | 同设备和设置；CUDA 计时同步；注明热身、IO/renderer/初始化是否计入 | 指定协议下的资源与速度，不照抄论文数字 |
| PVE / MPJPE / PA-MPJPE / Abs-PVE 等 | 对应 GT、人体拓扑/关节映射、匹配、单位与对齐协议 | 具备完整协议时才报真实误差；Anny 与 SMPL 不能直接按索引相减 |
| IDF1 / MOTA / IDSW | GT identity、完整时间轴、ignore regions 和正式匹配/评测器 | 具备条件才进入 benchmark；十个候选异常不等于十次 IDSW |
| Acceleration Error | 同身份、正确时间步长、连续有效三帧、预测与 GT 的相同语义坐标 | 无 GT 只能描述预测加速度/抖动，不能叫 acceleration error；camera motion 可能混入 |

正式指标定义以论文 §4 及其实际评测协议为准；当前公开树主要是推理与跟踪实现，本次未找到可直接运行的完整训练/benchmark 管线。因此不把 benchmark 复现安排为 19h 硬门槛。论文数值、本人实验数值和执行建议阈值分开保存。

结果表头可先建立，**数据待测，不预填通过或数字**：`run_id,input_id,condition,config_id,N,matched_N,unmatched_N,metric,unit,value,valid_count,status,evidence_path`。`value` 无有效样本时留空并注明原因。

## 8. Debugging Guide

始终按 **症状 → 假设 → 最小验证 → 根因 → 最小修复 → 复测** 记录。一次只改一个因素；自己的代码先构造小反例，官方代码先保留 commit/diff。环境安装/驱动/代理交给环境专项，科研 Agent 继续检查接口、数学定义和结论。

| 症状 | 优先假设 | 最小验证 / 下一步 |
| --- | --- | --- |
| 安装成功但初始化失败 | 独立解释器选错、权重/资产缺失、硬件依赖不兼容 | 打印解释器/版本、失败阶段、checkpoint hash；不在原 HMR2 环境试升级 |
| 首跑很久无输出 | 下载、Anny 资产缓存或模型初始化 | 看当前阶段日志/下载与进程状态；未 forward 前不报告推理 FPS |
| 单图 OOM | backbone/context、Anny 或渲染阶段显存不足 | batch=1 默认不渲染，记录报错位置与峰值；lowres 只改变人体分支，不能保证修复 encoder OOM |
| 人数为 0 | 图像/颜色/归一化错误、阈值过高或真实失败 | 检查 preprocess tensor、conf 分布和官方样例；先解释再调阈值 |
| shape 打印像少了一维 | singleton squeeze 或 empty 分支差异 | 使用 N=0/1/2 对照；保留 person 维，显式 empty policy |
| 不同图像 token 数对不上 | 假设固定方图/固定 patch grid | 记录真实 H/W、patch_size、Hp/Wp；回查两条预处理路径 |
| 人体投影偏移或重复平移 | 原图与预处理 K 混用、camera/body frame 混用 | 单点 E3 + K/resize/padding 记录；不要先修改模型权重 |
| mesh/faces 异常 | full/lowres 拓扑混用、旧缓存或错误模型参数 | 比较 V、faces.max、拓扑和 Anny 版本；禁止跨拓扑逐顶点误差 |
| 视频 ID 异常 | 失检、feature/geometry 混淆、reset 或过期行为 | 检查该帧 N/匹配、两类 similarity 和 banks；用短序列隔离，未验证前只记候选原因 |
| 连续空帧后重现异常 | 高层 API 与 tracker 的空帧推进路径不同 | 对照 S1 与 S7/S8/S9，验证 49/50/51 附近边界；不预设 bug 已复现 |
| 轨迹看起来很平滑但身份错了 | 可视化/平滑掩盖 ID 错配 | 逐帧核对身份与原始输出；平滑不能作为关联正确的证据 |
| 加速度异常大 | frame_id/FPS 错、缺帧、ID 切换或相机运动 | 先检查时间轴、valid triples、frame 和 GT；不直接归因模型动态不稳 |
| 视频帧数/画面混入旧实验 | 抽帧目录复用，遗留 PNG 被再次列出 | 每个 run 使用新 tmp_dir；不删除其他实验目录来“清干净” |

每条失败记录保存 `error、最小输入、预期/实测、版本/配置、已排除项、下一项验证`，并让数值、图和配置可以通过 run_id 相互定位。

## 9. Daily Research Log Template

以下复制到 WSL `week6/research_log.md`，每天一份；字段不预填实验结果。

```markdown
## YYYY-MM-DD / Day X

- 今日目标与 Gate：
- 实际投入 / 环境阻塞时长：
- 设备 / 解释器 / torch / CUDA / Anny：
- 官方 commit / 本人代码版本与 diff / checkpoint hash：
- 输入 ID、hash、原尺寸、预处理尺寸 / seed：
- 命令 / 配置 / conf、NMS、lowres、compile / run_id：
- 读到的论文位置与源码路径、函数：
- 运行前预测或假设：
- 本人独立编写的部分、收到的 Hint：
- 输入输出契约与真实 shape（含 dtype/device/empty）：
- 正常测试 / 反例 / 数值容差：
- 数值结果、单位、frame、mask、匹配、有效样本数与文件：
- 已知事实（直接观察）：
- 证据支持的推断：
- 尚未验证的假设 / 当前无法判断：
- 最大漏洞或替代解释：
- Debug：症状 → 假设 → 验证 → 根因 → 修复 → 复测：
- 哪个关键事实改变会推翻结论：
- 当日 Gate：通过 / 部分完成 / 受阻；逐项证据：
- 下一项最小验证实验与预计时长：
```

解释不能只写“理解了”；至少写一句能被检查的机制解释及一个反例。Agent 审核时优先检查本人实现、定义、反例和结论范围，再决定推进。

## 10. Week 6 Final Gate

所有项目初始均为待执行；只有本人结果与口头/书面解释共同支持时才勾选。未完成项带证据和阻塞原因进入 backlog，不能把预留文件夹、模板或 Agent 文档算作实验成果。

- [ ] **官方复现**：默认官方单图在记录的设备/环境成功，输入、checkpoint、commit、配置、输出和代表性可视化齐全。
- [ ] **输出理解**：本人解释 DecoderOutput/PersonOutput，能处理 N=0/1/多人、None 与可选字段，不把 Anny 当 SMPL 协议。
- [ ] **架构 trace**：独立定位关键 forward，解释 learned queries、self/cross attention、多个 heads、阈值和 NMS，附真实 shape。
- [ ] **相机与表示**：说明 K、深度参数化、body-to-camera placement、root/center、两人相对位置和拓扑差异；E3 正常与反例有记录。
- [ ] **Tracking**：短视频逐帧输出与 E4 统计齐全，解释 feature/pelvis 历史、Hungarian、新建/过期与 reset；未混同 smoothing。
- [ ] **Stress test**：≥10 张不同复杂图的结果和 failure taxonomy 完整，至少覆盖遮挡、拥挤、截断与定位/深度问题。
- [ ] **独立实验**：E1–E5 的本人实现及必要反例可复查；至少一个受控实验可复跑，结果含未匹配/无效样本处理。
- [ ] **科研解释**：周总结明确事实、推断、不确定性、最大替代解释、下一项最小实验；未将 displacement 当 GT error。
- [ ] **范围与复现记录**：明确 Demo/Method/Benchmark/Training 各自状态；保留 environment、输入、版本、原始数据和命令。
- [ ] **下一步问题**：提出一个 video-based HMR 小问题，写 baseline、最小验证、失败能学到什么；novelty 未经文献验证就标待验证。

周末用 8–10 分钟闭卷讲解：一张图如何变成人体 queries、相机 K 和多人 Anny meshes；一段视频如何保持 ID；一个自己的失败案例说明了什么、没有说明什么。回到代码定位关键证据，不能只展示结果图。

## 11. Open Research Questions

以下是候选问题，**不是已成立的 research gaps 或 novelty 声明**。Week 6 只选一个延伸，不额外做全套实验。

| 候选问题 | 首个 baseline 与最小验证 | 最大漏洞 / 若失败能学到什么 | novelty 状态 |
| --- | --- | --- | --- |
| 遮挡导致关联异常，主要来自失检还是身份特征？ | 官方默认 tracker；先固定已保存检测，只分析关联，再与原始完整路径对照 | 检测和关联耦合，单张图观察不足；失败可定位问题属于哪一层 | evaluation/method 候选，待查文献 |
| 相机运动时 pelvis 先验何时反而伤害关联？ | 官方默认组合与只改变一个组合权重的自定义 tracker 对照；复用相同 per-frame 预测 | 无真实相机/ID GT 很难判因果；先用人工可核对的小片段 | 问题已有相关研究，不能预称新颖 |
| 两人近距离时漏检来自 confidence 还是 3D NMS？ | 固定同一次网络预测，只改变 NMS 距离；回看被抑制的实例 | 预测深度本身可错；失败也能区分后处理与恢复错误 | 本周为诊断实验，未证明 novelty |
| 自洽投影为何仍可能有错误的多人深度关系？ | E3 作为实现检查，另用有人工深度排序的小样本核对 | 单目尺度/焦距歧义与模型先验仍在；投影一致不能当准确性证据 | representation/evaluation 候选，待验证 |
| 预测 mesh 抖动和 ID 错配应怎样分开评估？ | 先按正确身份/连续帧统计，再看未平滑输出；有 GT 才谈 error | camera motion 和姿态变化混入，平滑可掩盖错配 | 不因增加平滑模块就有 method novelty |

每个 idea 继续前先检查：问题是否真实存在、是否已有人解决、baseline 是什么、改善来自哪里、是否只是模块拼接、最小验证是什么、失败能学到什么、novelty 属于 problem/method/representation/training/data/evaluation 哪一种。证据不足就写“目前无法判断”。

## 12. Next Step: Video-Based Multi-Person HMR

本周得到的是 per-frame camera-space 人体预测与跨帧身份关联。下一阶段先区分三个需求：**同一人是否关联正确、相机坐标中的人体是否稳定、人体在世界坐标中的运动是否正确**。这三者需要不同证据，不能用一条看起来平滑的轨迹一并回答。

按总表继续：Week 7 完成相机几何、坐标变换与 SfM；Week 8 用 TRAM 理解 moving-camera / static-human、camera-to-world 组合及尺度问题；Week 9 再连接 Human3R 的 camera/human/scene 状态。保留本周短视频、frame_id、K、pelvis、identity features 与失败案例，作为后续对照输入；不在 Week 6 扩成训练时序生成/平滑模型。

下一次科研讨论以五项开始：**结论、当前证据、最大不确定性、最小验证实验、下一步行动**。优先拿本人写的 E1 或一个真实失败案例继续，而不是再运行同一个成功 Demo。

---

文档编写检查：总表相关 Excel/Numbers 单元格已比对；官方 README、论文及 S1–S13 核心路径已静态阅读；继承 Week 5 自主实现方法；五天均有统一栏目、独立练习与 Gate；未提供整套参考脚本、未预填实验结果。本计划已考虑 8 GB 的执行顺序与退路，**本地显存适配、Anny replay、真实视频和 benchmark 结果仍待本人运行验证**。
