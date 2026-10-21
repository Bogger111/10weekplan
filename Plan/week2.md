# Week 2 — Transformer → Spatio-Temporal Transformer

> 3D/4D Human 10 周科研计划 · Week 2 执行细化 · 创建于 2026-10-11  
> 总投入：5 天 × 约 3.5–4h，合计 **19h 专注学习**；休息另计。  
> 当前状态：任务已布置，代码、学习与实验结果待本人完成、记录和验收。  
> 结构：Day 1–Day 5 为五个并列章节，沿用 `week1.md` 相同的九个栏目与四个执行阶段。

## Week Goal

本周结束时，你应能从 Q/K/V 的含义讲到一个 skeleton 时空 Transformer 的实现：自己写出 attention、MHA 与 Encoder Block，闭卷推导主要 tensor shape，知道 spatial attention 和 temporal attention 分别在哪个轴上交换信息，并用一个小实验检查模型是否实际利用了跨帧上下文。

形成四项能力：

- **Understand / Trace**：解释 scaled dot-product attention、multi-head attention、positional encoding、Pre-LN / Post-LN；沿源码追踪 token、head、joint 与 time 轴。
- **Run / Manipulate**：不用 `nn.MultiheadAttention`，独立实现 Attention → MHA → Encoder Block，并把它们复用于 `[BT,J,C]` 与 `[BJ,T,C]`。
- **Experiment / Analyze**：完成 skeleton spatio-temporal Transformer toy，保存 spatial / temporal attention 图和一次有控制组的最小实验。
- **Research Log**：留下 shape 推导、预测、实测、代码位置、失败原因及限制，为 Week 3 的人体表示和 Week 4 的方法桥接提供可复用证据。

**总计划依据（原计划内容）**

本次已读取本地 [总计划表](00_plan/10周研究计划_曾奕博.xlsx)，以工作表 `10周计划!B11:D11` 为本周事实依据。Week 2 的正式安排如下，保留原表措辞：

| 字段 | 原计划内容 |
| --- | --- |
| Focus（B11） | Transformer → Spatio-Temporal Transformer。重点：Q/K/V、scaled dot-product attention、multi-head attention、positional encoding、Pre-LN / Post-LN、tensor shape、spatial / temporal attention。 |
| Core Resources（C11） | Stanford CS231n · Attention & Transformers；D2L · Transformer；Annotated Transformer；后续源码：PoseFormer / MotionBERT。 |
| Deliverable 1（D11） | 不调用 nn.MultiheadAttention，从零实现 Scaled Dot-Product Attention → MHA → Encoder Block。 |
| Deliverable 2（D11） | 完成 tensor-shape gym；给定 [B,T,J,C] 能推导 Q/K/V 与 attention map shape。 |
| Deliverable 3（D11） | 实现 Spatial Attention [BT,J,C] 与 Temporal Attention [BJ,T,C] 两种 block。 |
| Deliverable 4（D11） | 完成一个 skeleton spatio-temporal Transformer toy，并可视化 spatial / temporal attention。 |
| Gate（D11） | 能闭卷推导 MHA 与时空 Transformer 的主要 shape，并独立实现。 |

衔接依据为 `10周计划!B10:D13` 与 `资源清单!B22:G26`：Week 1 提供训练与排错基础；Week 3 是 **Skeleton → Rotation → FK → SMPL**；Week 4 再用 2–4h 快速回顾 PoseFormer / MotionBERT，连接人体表示与重建。后两者在本周只用于短源码桥接，不增加完整 Repo 复现、预训练权重或 Human3.6M 下载任务。前 9 周三个必跑 Demo 的优先级仍是 HMR 2.0 / 4DHumans、Multi-HMR、Human3R（`10周计划!B4、B6`）。

**执行建议与版本边界**

以下 Prerequisites、19h 时间分配、每日步骤、合成数据、数值容差、文件名和细化 Gate 均为**执行建议**，用于落实上述四项交付，不是原表新增事实。19h 是对原表“≈20h / 周、至少 50% 时间用于代码、实验和调试”（B7）的执行拆分。

`week1.md` 标称 v2.2，但当前本地 Excel 文件名及已读取表格未明确标注版本；不能仅凭内容相近断言它就是 v2.2。原表 Week 1 的 optimizer / bug 数量要求也比现有 `week1.md` 更广，本文件不修改两者，不据此认定 Week 1 已验收。本周目标以当前 Excel 的 Week 2 行为准。当前目录未找到 `week.md`，结构范例采用已存在的 [week1.md](week1.md)。

## Prerequisites

**执行建议**：先确认真正依赖的能力，不用把所有 Week 1 扩展任务重做一遍。

- 数学：矩阵乘法、转置、softmax、均值 / 方差；能解释 `(N,d) @ (d,M)` 的结果维度。
- Python / PyTorch：会写 `nn.Module`、`Linear` 和基本训练循环；理解 `reshape / permute / transpose`，区分 `eval()` 与 `no_grad()`。
- Week 1 衔接：能说明 zero_grad → forward → loss → backward → step；会检查参数梯度、输入 / 标签 shape 和非有限值。已有训练器可参考，但本周代码放在 `week02/`，不修改 Week 1。
- 环境：沿用本人已有的 PyTorch 环境，CPU 足够完成本周 toy；画图使用已有 matplotlib。总计划的 Python / GPU 配置是规划信息，不能当作本机已验证状态。
- 数据：本周使用自造 skeleton 序列，不依赖人体模型授权、真实数据集或旧 Repo 环境。
- 记录：执行时新建 `week02/environment.md`，注明解释器、Python / PyTorch 版本、设备、dtype、seed 与源码访问日期。

先用你实际学习用的解释器检查：

```bash
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

环境排查先限时 15m；没有可用 PyTorch 时先做手算和 shape 推导，代码实验记为待补。需要新环境时交由环境专项处理，不把本周变成旧依赖安装任务。Week 1 尚缺的能力按实际证据列入待补，不把计划文件存在视作已完成。

## 统一科研资料结构

**执行建议**：当前资料根目录是 `/Users/mac/顶会刊收割机/01_10week_plan/`。本次只生成 `week2.md`；下方 `week02/` 是本人执行任务时自行新建的建议结构，尚未提供练习骨架、实验脚本或结果。

```text
01_10week_plan/
├── 00_plan/                       # 原计划表，只读使用
├── week1.md                       # 原 Week 1，不改动
├── week2.md                       # Day 1–Day 5 并列执行计划
└── week02/                        # 执行时按需创建
    ├── environment.md
    ├── src/                       # attention.py / transformer.py / skeleton_toy.py
    ├── experiments/               # 各日入口，自行实现
    ├── notes/                     # 手算、shape、源码桥接
    ├── results/                   # 原始输出、CSV、图与实验报告
    ├── research_log.md
    ├── fail_log.md
    └── weekly_review.md
```

| 日程 | 代码与资料准备状态 | 执行状态 |
| --- | --- | --- |
| Day 1 | Attention 实现与手算对照按当日任务新增 | 本人实现、实验与验收待完成 |
| Day 2 | MHA、shape gym 与多 head 对照按当日任务新增 | 实验与验收待完成 |
| Day 3 | 位置编码、Encoder Block 与置换实验按当日任务新增 | 实验与验收待完成 |
| Day 4 | 时空 block、合成 skeleton 与训练入口按当日任务新增 | 实验与验收待完成 |
| Day 5 | 上下文实验、attention 图、短源码 trace 与周总结按当日任务新增 | 实验与验收待完成 |

所有建议命令从上述资料根目录运行，实际参数由你实现入口后写入日志。计划、空目录与未填写日志均不计为实验成果。实验导出、临时文件与新增代码也留在本目录内，不向同级 `02_papers/` 或其他目录写入。

## 五天安排总览

**执行建议**：先保证实现和 shape 正确，再做小训练与解释；阅读、调试、复测都包含在预算内。

| 日程 | 专注时间 | 模块与时间分配 | 当日交付与验收证据 |
| --- | --- | --- | --- |
| Day 1 | 3h45m | 概念与手算 50m；Attention 实现 70m；数值 / mask 对照 70m；分析日志 35m | `src/attention.py`；手算与实测；softmax / mask 检查；Research / Fail Log |
| Day 2 | 3h45m | MHA 与 shape trace 45m；MHA 实现 75m；shape gym / 多 head 对照 70m；分析日志 35m | 手写 MHA；各 head 对照；`notes/tensor_shapes.md`；shape 与重排错误证据 |
| Day 3 | 3h45m | PE / LN / block 阅读与 trace 45m；Encoder 实现 75m；置换 / LN 实验 70m；分析日志 35m | `src/transformer.py`；位置编码；Pre-LN / Post-LN 流程；置换实验与梯度检查 |
| Day 4 | 4h | 时空数据流与任务定义 45m；两种 block 与 toy 75m；sanity check / 小训练 80m；分析日志 40m | 两种时空 block；skeleton toy；single-batch 检查；训练曲线、配置与权重 |
| Day 5 | 3h45m | 短源码桥接 / 实验设计 40m；上下文实验 / attention 图 75m；复测与解释 45m；最终验收 / 周总结 65m | attention 两类图；最小实验报告；DSTFormer 短 trace；最终 Gate 与 Week 3 交接 |
| **合计** | **19h** | 每天沿用 Understand → Trace → Run → Manipulate → Experiment → Analyze → Research Log | Run / Experiment 阶段共 **11h45m**，不以读完资源数量计算进度 |

## 共同执行约定

**执行建议**：五天均使用相同的九个栏目：当日目标、前置条件、时间安排、核心资源、执行步骤、交付物、验收标准、受阻处理、下一步。执行步骤统一为四个阶段，按当日主题分配时间。时间表是累计专注时间；每 60–90m 可休息 10–15m，休息不计入 19h。

**独立实现与参考边界**

1. 可以使用 PyTorch 的 tensor 运算、autograd、`Linear / LayerNorm / Dropout / GELU` 和 optimizer。Attention 的打分、缩放、mask、softmax、加权求和及拆 / 合 head 自己写。
2. 自写主路径不调用 `nn.MultiheadAttention`、`nn.TransformerEncoderLayer`、`nn.TransformerEncoder` 或 `F.scaled_dot_product_attention` 来代替实现。参考资料用于解释和对照，读后关掉参考写核心逻辑，并记录参考程度。
3. 数值对照先用 CPU、float64、dropout=0；训练用 CPU、float32。检查全部输出有限；数值对照默认 `atol=1e-6、rtol=1e-5`，记录最大绝对差。容差是本次实现检查口径，不是原表提供的阈值。
4. 固定 seed=42；相同输入、初始化和模式下比较。训练结果与预期分开记录，不规定某种结构必然胜出。

**统一 shape 与 mask 约定**

- `B`：batch；`T`：帧数；`J`：关节数；`D_in`：原始坐标 / 标志通道；`C`：embedding 宽度；`H`：head 数；`d_h=C/H`。原始坐标维数不等于 embedding 宽度。
- 通用 MHA 输入为 `[N,L,C]`；拆 head 后 Q/K/V 为 `[N,H,L,d_h]`，self-attention map 为 `[N,H,L,L]`。这里 `N` 是有效 batch，不能默认等于人体序列 batch `B`。
- 本周布尔 attention mask 统一约定 **True=屏蔽**，在 softmax 前把对应 logit 置为 `-inf`。这是自写接口约定，使用其他 API 时重新核对语义。
- 每个 query 至少保留一个可见 key；遇到整行屏蔽，主动报错。softmax 沿最后的 key 轴，dropout 前每行和约为 1；dropout 后不要求行和仍等于 1。
- toy 中“坐标被遮挡”通过输入中的 observed flag 表达，默认不等于 attention mask。被遮挡的 query 仍要读取其他帧；本周不叠加 causal / padding 任务。

**Day 4–5 共用最小任务：遮挡中心帧的 skeleton reconstruction**

以下均为执行建议，具体生成、训练和评估步骤在 Day 4 落实。

- 64 条训练序列、16 条独立验证序列；每条 `T=9、J=5`。关节名为 root、left_hand、right_hand、left_foot、right_foot，固定次序；只用一棵 root 连接四个端点的自定义示意骨架，不冒充标准人体 keypoint convention。
- 干净坐标 `Y[n,t,j]=p_j+a_n+v_n·τ_t`，`τ_t` 为从 -1 到 1 的等间隔帧位置；`p_j` 是固定骨架偏移。每条序列独立采样 `a_n∈[-0.3,0.3]^3、v_n∈[-0.15,0.15]^3`。坐标为任意单位，不是人体物理测量。
- 可见帧输入加标准差 0.03 的独立噪声；将中心帧 `t=4` 的所有关节坐标置 0、observed flag 置 0，其他帧 flag=1。输入 `[B,9,5,4]`；标签是加噪和遮挡前独立保留的干净坐标 `[B,9,5,3]`。
- 模型：`Linear(4,32)` → joint / temporal PE → 一个 Spatial Encoder → 一个 Temporal Encoder → `LayerNorm(32)` → `Linear(32,3)`；`H=4、d_h=8、FFN=32→64→32、Pre-LN、dropout=0`。
- 输出 `[B,9,5,3]`，**loss 和本周重建指标仅计算中心帧**：`mean((pred[:,4]-Y[:,4])²)`。其他帧输出未经该目标直接监督，不将它们展示为已验证的完整 motion reconstruction。
- Adam、lr=1e-3、batch=8、固定 batch 序列、300 次参数更新；不加 scheduler / AMP。single-batch sanity check 使用另一份独立模型，正式训练从保留的初始权重重新开始。
- Day 5 用同一训练后模型做“原上下文 vs 跨序列交换可见帧”的评估对照，并比较零坐标与邻帧线性插值。目的是检查信息路径，单种子 toy 不支持真实人体 benchmark 或方法优越性的结论。

## Day 1：Q/K/V + Scaled Dot-Product Attention｜3h45m

### 1. 当日目标

解释 attention 如何按 query 为 keys 分配权重，再聚合 values；自己实现最小 attention，并用手算、数值输出和 mask 实验验证。

### 2. 前置条件

会矩阵乘法、softmax 和基本反传；用本人环境完成检查。**执行建议**：本日新增 `week02/src/attention.py` 与 `week02/experiments/attention_check.py`，先不加 MHA、PE 或完整训练。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:50 | 50m | Understand → Trace | 环境检查、Q/K/V、手算与 shape | 手算表、输入输出轴 |
| 00:50–02:00 | 70m | Run | 实现打分、缩放、mask、softmax、加权和 | 可反传的 Attention 函数 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 手算对照、mask、缩放和梯度检查 | 预测 / 实测 / 解释表 |
| 03:10–03:45 | 35m | Analyze → Research Log | 整理证据、日志与口述验收 | 结果、失败记录、Gate 状态 |

### 4. 核心资源

- [Stanford CS231n · Attention and Transformers](https://cs231n.stanford.edu/schedule)：只看 attention 的输入、打分与聚合；从课程入口定位对应讲义。
- [D2L · Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)：总计划指定的中文入口，按需回查 attention 评分部分。
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)：只定位 `attention`；若网页无法访问，读其[官方源码](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)。

阅读包含在当日预算，合计约 20m，不要求跑教材的翻译训练。

### 5. 执行步骤

#### A. Understand → Trace

1. **10m 环境**：记录解释器、版本、设备和 seed；确认能做一次 tensor backward。
2. **20m 阅读**：回答 query “要找什么”、key “如何匹配”、value “被聚合什么”；说明 self-attention 的输入来源相同，但投影后的 Q/K/V 不必相等。
3. **20m 手算**：从 `S=QKᵀ/√d_k、A=softmax(S, key_axis)、O=AV` 推导各步 shape。一般 Q `[N,Lq,d_k]`、K `[N,Lk,d_k]`、V `[N,Lk,d_v]`，A `[N,Lq,Lk]`、O `[N,Lq,d_v]`。本周主体采用 self-attention，先保留一般记法以免把两个 token 轴混淆。

主例固定无 batch 的 `d_k=1`：

```text
Q = [[1], [0]]
K = [[0], [ln(3)]]
V = [[2], [6]]
S = [[0, ln(3)], [0, 0]]
A = [[1/4, 3/4], [1/2, 1/2]]
O = [[5], [4]]
```

逐行解释为什么第一个 query 的结果是 5，而不是选出某一个 value。随后给 Q/K/V 加一个 batch 轴，预测输出 shape。

#### B. Run

1. **20m 实现主路径**：在 `scaled_dot_product_attention(q,k,v,blocked_mask=None)` 中手写矩阵乘法、`1/sqrt(d_k)`、softmax 与 `A@V`；返回 output 和 dropout 前的 attention map。运算针对最后两个轴，允许一致的前导 batch / head 轴，以便 Day 2 直接复用。
2. **15m 加检查**：验证 Q/K 的末维相同，K/V 的 token 数相同，输入与输出有限；`softmax(dim=-1)`，不要对 feature 轴求 softmax。
3. **20m 加 mask**：用布尔 True 表示屏蔽，先广播到 logits shape，再在 softmax 前处理。被屏蔽位置的概率应为 0；检查每行至少有一个未屏蔽 key。
4. **15m 初步运行**：float64 运行主例；再用 `[2,3,4]` 的 Q/K/V 检查 batch 输出、行和、一次 backward。打印实际最大误差和梯度是否有限。

保留函数与实验入口分离；不要导入文件就开始实验。只在输出存盘时 detach，不在参与 loss 的主路径上 detach。

#### C. Manipulate → Experiment

| 实验 | 时间 | 唯一改变量 / 操作 | 要记录 / 要解释 |
| --- | --- | --- | --- |
| 主例与 batch 对照 | 20m | 将同一手算主例扩成 batch；不换公式 | O 是否为 5 / 4；A 的行和与输出 shape |
| 屏蔽一个 key | 20m | 两个 query 都屏蔽第 2 个 key，其他不变 | A 应为 `[[1,0],[1,0]]`、O 均为 2；整行屏蔽须得到明确错误 |
| 缩放与梯度 probe | 20m | 固定随机 Q/K/V `[2,4,16]`，仅切换有 / 无 `1/√16` | logits 标准差、A 的每行最大概率、output 与 `O.square().mean()` 的梯度是否有限；不预设饱和程度 |
| 故障与复测 | 10m | 临时将 softmax 轴改错，再恢复 | 何种行和 / 手算输出检查抓住错误；同输入修复结果 |

缩放的作用用统计与实测解释，不把一次随机输入的现象概括成所有 head 的行为。梯度检查是确认计算图连通，不要求每个梯度元素都非零。

#### D. Analyze → Research Log

15m 整理 `results/d1_attention.md`；10m 填 Research / Fail Log；10m 关掉资料口述从 Q/K/V 到 O 的每一步。报告区分手算真值、预测和实测，并记下本日真正遇到或受控触发的错误。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week02/src/attention.py` | 本人写出 attention 主路径与 mask 约定 |
| 实验 | `week02/experiments/attention_check.py` | 主例、batch、mask、缩放与梯度入口 |
| 结果 | `week02/results/d1_attention.md` | 实测输出、误差、行和、梯度与修复证据 |
| 推导 | `week02/notes/attention.md` | Q/K/V 含义、公式与各步 shape |
| 日志 | `week02/research_log.md`、`week02/fail_log.md` | Day 1 的问题、预测、观察、解释与下一步 |

### 7. 验收标准（Gate）

- [ ] 主例、batch 与单 key mask 输出有数值证据，和手算对齐。
- [ ] 能闭卷推导 logits / attention map / output shape，指出 softmax 的 key 轴。
- [ ] 能解释 `√d_k`、为什么 Q/K 不必相等、为什么 value 是被加权的对象。
- [ ] 正常输入 forward / backward 有限；整行 mask 明确拒绝，不用 NaN 继续训练。
- [ ] 有一条真实或受控的 softmax / mask 困惑及修复复测记录。

### 8. 受阻处理

先退回无 batch、无 mask 的 2-token 主例；逐项打印 S、A、O。超时删去额外可视化和扩展阅读；达到 3h45m 后登记待补，不把未跑的结果写成通过。

### 9. 下一步

把同一个 attention 函数用于多个 head，并先推导拆 head 与合 head 的顺序。

## Day 2：Multi-Head Attention + Tensor-Shape Gym｜3h45m

### 1. 当日目标

自己实现 MHA；给定输入尺寸能闭卷写出 Q/K/V、attention map 和输出 shape；用数据对应关系确认“shape 对了”也没有把 head 或 token 混排。

### 2. 前置条件

Day 1 attention 的手算与 mask 通过。**执行建议**：在 `attention.py` 新增 MHA 类，不重写另一套 attention；本日先不用 PE、LN 或训练任务。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | Q/K/V 投影、拆 / 合 head 与 shape | MHA 数据流、手推表 |
| 00:45–02:00 | 75m | Run | 自写 MHA、返回各 head 的 A | 可运行 MHA 与前向输出 |
| 02:00–03:10 | 70m | Manipulate → Experiment | 逐 head 对照、编号重排、时空 shape gym | 数值差、元素对应与 shape 表 |
| 03:10–03:45 | 35m | Analyze → Research Log | 整理证据、日志与闭卷验收 | 报告、失败记录、Gate 状态 |

### 4. 核心资源

- [Annotated Transformer 官方源码](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)：只追 `MultiHeadedAttention` 的投影、拆 head 和 concat / output projection。
- [D2L · 多头注意力](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html)：原计划中文入口的相邻章节，核对并行 head 的意义。

阅读合计约 20m。参考资料可能采用不同张量排列，先写出自己的接口约定再对照。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 阅读**：说明多个 head 有各自的投影子空间；concat 后为何还需要 `W_O`。不要把 MHA 描述成同一个 attention output 简单复制 H 份。
2. **15m 手推**：固定 `[N,L,C]=[2,6,32]、H=4、d_h=8`，逐步写下下表，不运行代码先预测。
3. **10m Trace**：用纸上编号说明 `reshape` 只改变分组，`permute` 才交换轴；逆变换必须恢复 token 与 head 的对应关系。

| 步骤 | 通用 shape | 固定例子 |
| --- | --- | --- |
| 输入 / 三个 Linear 投影 | `[N,L,C]` | `[2,6,32]` |
| 每个投影拆 head | `[N,L,H,d_h]` | `[2,6,4,8]` |
| 换轴后 Q/K/V | `[N,H,L,d_h]` | `[2,4,6,8]` |
| 每个 head 的 S / A | `[N,H,L,L]` | `[2,4,6,6]` |
| 加权 V | `[N,H,L,d_h]` | `[2,4,6,8]` |
| 换轴、concat、W_O | `[N,L,C]` | `[2,6,32]` |

#### B. Run

1. **20m 建模块**：`W_Q / W_K / W_V / W_O` 均为 `Linear(C,C)`，先令 bias=False；检查 `C % H == 0`，不满足就报错。
2. **25m 实现拆 / 合 head**：线性投影 → reshape → permute → 调用 Day 1 attention → permute 回 token-first → reshape 合并 → output projection。返回 O 和 `[N,H,L,L]` 的 A。
3. **15m 接 mask**：把 `[L,L]` 或 `[N,1,L,L]` 的 True=屏蔽 mask 广播到各 head，不让 mask 意外对齐到 feature 轴。
4. **15m Run**：在 `[2,6,32]` 上完成 forward / backward，记录 input / output shape、参数梯度与 A。保持 dropout=0，给后续 block 提供稳定基线。

若 permute 后直接 `view` 报错，检查 stride / contiguous，或用适当的 `reshape`。能运行不代表轴语义正确，下面用编号与逐 head 对照验证。

#### C. Manipulate → Experiment

| 子任务 | 时间 | 固定项与改变量 | 要记录 / 要回答 |
| --- | --- | --- | --- |
| 逐 head 参考 | 25m | 使用同一组投影权重与输入；参考路径显式循环 H 次，分别取每个 head 的通道，再 concat / W_O | 与批量 MHA 的 output / A 最大差；H=1 与 H=4 两例。参考路径不用同一个拆轴 helper |
| 编号重排与故障 | 15m | 用连续整数创建 `[B,T,J,C]`，走 spatial / temporal 重排再逆变换；临时漏一次 permute | 原张量与恢复张量应逐元素完全相等；错误版本即使 shape 合法也可能失败 |
| Tensor-shape gym | 30m | 先闭卷写表，再用代码打印；例子取 B=2、T=9、J=5、C=32、H=4 | Q/K/V、A、output、有效 batch 和 token 的含义；空间 / 时间 / 全时空三种排列 |

本周必须能推导以下形状。括号表示相乘，不是新增轴：

| 模式 | 输入 | Q/K/V | Attention map | 还原后输出 |
| --- | --- | --- | --- | --- |
| Spatial | `[B*T,J,C]` | `[B*T,H,J,d_h]` | `[B*T,H,J,J]` | `[B,T,J,C]` |
| Temporal | `[B*J,T,C]` | `[B*J,H,T,d_h]` | `[B*J,H,T,T]` | `[B,T,J,C]` |
| Flattened joint-time（只推导） | `[B,T*J,C]` | `[B,H,T*J,d_h]` | `[B,H,T*J,T*J]` | `[B,T,J,C]` |

固定例子的 Spatial A=`[18,4,5,5]`、Temporal A=`[10,4,9,9]`、flattened A=`[2,4,45,45]`。只比较 attention score 元素数量：分别为 `B*H*T*J²`、`B*H*J*T²` 和 `B*H*(T*J)²`；这是打分矩阵规模，不是完整模型的显存 / 耗时结论。

#### D. Analyze → Research Log

15m 整理 `results/d2_mha.md` 与 `notes/tensor_shapes.md`；10m 写日志；10m 闭卷解释有效 batch、token 与 head。把漏 permute 的错误定位到元素对应关系，不能只写“维度问题”。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week02/src/attention.py` | 自写 MHA，复用 Day 1 主路径 |
| 实验 | `week02/experiments/mha_shape_check.py` | 逐 head 参考、编号重排、mask 与 backward |
| 推导 | `week02/notes/tensor_shapes.md` | 通用 / 数值 shape、每个轴含义、打分规模 |
| 结果 | `week02/results/d2_mha.md` | 数值误差、重排证据、实际 shape 与修复记录 |
| 日志 | `week02/research_log.md`、`week02/fail_log.md` | Day 2 的预测、实测、解释与下一步 |

### 7. 验收标准（Gate）

- [ ] MHA 核心由本人实现，未用封装 attention / Transformer 层替代。
- [ ] H=1 / H=4 与逐 head 参考对齐；mask、输出和梯度有限。
- [ ] 编号重排与逆变换逐元素一致，能指出错误版混掉哪个轴。
- [ ] 能闭卷推导普通、Spatial、Temporal、flattened 模式的 Q/K/V 与 A shape。
- [ ] 能说明多 head、concat、W_O 的作用及 `C % H` 限制。

### 8. 受阻处理

先用 H=1 对齐，再拆成 H=2 / H=4；遇到数值不同先查权重、bias、scale、模式，再查轴。超时不做 flattened attention 实现，保留其手推结果；本周只必做 factorized 时空 block。

### 9. 下一步

把已验证的 MHA 放进残差 block，并用置换实验检查位置编码实际改变了什么。

## Day 3：Positional Encoding + Pre-LN / Post-LN Encoder Block｜3h45m

### 1. 当日目标

自己组装可反传的 Encoder Block；解释 PE 为何需要、LN 归一化哪个轴，以及 Pre-LN / Post-LN 的残差路径有何不同。

### 2. 前置条件

Day 2 MHA 已通过数值与 shape 检查；回顾 Week 1 的 LN 和 train / eval。**执行建议**：本日新增 `week02/src/transformer.py`，复用自写 MHA，先不增加 decoder、cross-attention 或语言任务。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | PE、LN、残差和 FFN 数据流 | 两种 LN 顺序与 PE shape |
| 00:45–02:00 | 75m | Run | 实现 PE、FFN 与 Encoder Block | 可反传的 block |
| 02:00–03:10 | 70m | Manipulate → Experiment | 无 / 有 PE 置换、零子层、LN 轴检查 | 机制对照与最大差值 |
| 03:10–03:45 | 35m | Analyze → Research Log | 整理证据、日志与口述验收 | 报告、失败记录、Gate 状态 |

### 4. 核心资源

- [D2L · 自注意力和位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)：从原计划中文入口回查正弦位置编码。
- [D2L · Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)：只看 position-wise FFN、AddNorm 与 Encoder Block。
- [Annotated Transformer 官方源码](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)：定位 `SublayerConnection / EncoderLayer / PositionalEncoding`，以实际运算顺序判断 LN 位置。

阅读合计约 25m。不同教材可以采用不同 LN 顺序，不能因为都叫 Encoder 就直接混用实现。

### 5. 执行步骤

#### A. Understand → Trace

1. **25m 阅读**：写出 attention、position-wise FFN、残差、LN 和 PE 分别改变什么。FFN 在每个 token 上独立作用，权重共享；它本身不交换 token 信息。
2. **10m 手画**：`LayerNorm(C)` 在每个 token 的 C 个 feature 上统计，输出 shape 保持不变；不要把时间 / joint 轴包含进 normalized shape。
3. **10m Trace**：用下面流程区分两种 Encoder；本周 dropout=0，训练主路径选 Pre-LN，Post-LN 只作机制对照。

```text
Pre-LN:
  U = X + MHA(LN1(X))
  Y = U + FFN(LN2(U))

Post-LN:
  U = LN1(X + MHA(X))
  Y = LN2(U + FFN(U))

FFN: Linear(C,2C) → GELU → Linear(2C,C)
```

这是 block 内部的 LN 顺序；模型末端是否另加 final LN 是另一项设计。本周 toy 在输出 head 前另加 final LN，不因此把 block 变成 Post-LN。解释不同残差路径即可，不用一个浅 toy 宣称某种 LN 普遍更稳定。

#### B. Run

1. **20m 位置编码**：实现偶数 `C=32` 的正弦 PE，`PE(pos,2i)=sin(pos/10000^(2i/C))`、`PE(pos,2i+1)=cos(pos/10000^(2i/C))`。缓存 shape 为 `[1,L,C]`；检查长度、设备、dtype，位置 0 的偶数通道为 0、奇数通道为 1。
2. **25m Encoder**：实现 `EncoderBlock(C,H,ffn_dim,norm_first=True)`，使用本人 MHA、两套 LN 与 FFN；返回输出和 attention map。`norm_first=False` 走上面的 Post-LN 流程，不额外加一套共享 LN。
3. **15m 未来时空 PE**：为 Day 4 准备独立 learned joint embedding `[1,1,J,C]` 和固定 temporal sinusoidal PE `[1,T,1,C]`，加到 `[B,T,J,C]` 上；保留 J / T 的广播含义，joint embedding 不代表骨骼拓扑约束。
4. **15m Run**：在 `[2,6,32]` 上运行两种 block；检查输出 shape、参数梯度和有限值。若返回用于可视化的 A，训练输出仍保留完整计算图。

先实现一种固定 temporal PE 即可；不展开 RoPE、relative bias 或各种新位置编码，避免偏离原计划预算。

#### C. Manipulate → Experiment

**PE 置换实验（30m）**

固定同一组 Encoder 权重、X `[2,6,32]`、dropout=0、eval 模式；不加 mask。令 `π` 是一个非平凡 token 置换：

| 对照 | 操作 | 检查 / 解释 |
| --- | --- | --- |
| 无 PE | 比较 `f(πX)` 与 `πf(X)` | 最大差应在容差内；同参数、逐 token LN / FFN、无位置约束时的置换等变性 |
| 固定槽位 PE | 比较 `f(πX+PE)` 与 `πf(X+PE)` | 内容换位但 PE 留在原槽位，通常不再对齐；记录实测差，若差很小检查输入 / 权重是否退化 |
| 内容与 PE 一起移动 | 比较 `f(π(X+PE))` 与 `πf(X+PE)` | 应恢复对齐；说明不是矩阵运算失效，而是内容与位置的配对改变 |

“等变”表示输出跟随输入一起换位，不是所有输出都不变。内容与 PE 同时移动不等于让模型重新识别一个固定位置；不能用这个对照得出“PE 没用”。

**LN / 残差机制（25m）**

用独立副本将 MHA 和 FFN 输出置零；输入取均值明显不为 0 的张量，例如 `X=3+0.7*randn(...)`。Pre-LN block 的输出应等于 X；Post-LN 输出为 `LN2(LN1(X))`，一般不同于 X。这里不含模型末端 final LN。记录最大差与样本 feature 均值，不能把“零子层对照”当作正常训练。

**故障与复测（15m）**

先保持 `LayerNorm(C)`，再临时把归一化改为跨 `[L,C]`；复做无 PE 置换 / token probe，解释哪些检查可能仍通过、为什么 LN 轴仍与本周约定不同。用一个更直接的 probe：只改另一个 token 的内容，检查当前 token 在 LN 后的结果。恢复 `LayerNorm(C)` 后，本 token 的 LN 不依赖其他 token；整个 attention block 仍可以依赖其他 token，比较时不要把两层混在一起。

#### D. Analyze → Research Log

15m 整理 `results/d3_encoder.md`；10m 写日志；10m 口述 PE、LN 轴与残差路径。至少记录三组置换误差及零子层对照，区分数学性质检查、实测现象和未经训练验证的推测。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week02/src/transformer.py` | 正弦 PE、joint embedding、FFN、Pre-LN / Post-LN block |
| 实验 | `week02/experiments/encoder_probe.py` | 三组置换、零子层、LN 轴与 backward |
| 结果 | `week02/results/d3_encoder.md` | 实际差值、模式、配置和机制解释 |
| 推导 | `week02/notes/encoder.md` | 两种残差路径、PE 与 LN 的轴 |
| 日志 | `week02/research_log.md`、`week02/fail_log.md` | Day 3 的预测、实测、解释与下一步 |

### 7. 验收标准（Gate）

- [ ] Encoder Block 复用本人 MHA，forward / backward 正常，shape 保持不变。
- [ ] 正弦 PE 的位置 0 对照、广播和 device / dtype 检查通过。
- [ ] 三组置换对照有数值输出，能区分等变、固定槽位 PE 与一起移动 PE。
- [ ] 能闭卷画出 Pre-LN / Post-LN，解释零子层实验和 final LN 的区别。
- [ ] 能说明 LN(C) 的统计轴，给出一个错误 LN 轴的定位 / 复测证据。

### 8. 受阻处理

先保留 dropout=0 的单层 Pre-LN；把 PE 缩为 L=3、C=4 手算。若置换实验不对齐，先查模式、mask、LN 轴与位置是否随内容移动；不急着加深模型。Post-LN 正常训练对照不在本周必做范围。

### 9. 下一步

复用同一个 Encoder 接口，分别把 joint 和 time 作为 token 轴，先验证逆变换再接 skeleton toy。

## Day 4：Spatial / Temporal Blocks + Skeleton Toy｜4h

### 1. 当日目标

实现两种独立的时空 block，正确恢复 `[B,T,J,C]`；训练一个中心帧遮挡重建 toy，留下可复现的训练、梯度与数据证据。

### 2. 前置条件

Day 2–3 的 MHA、Encoder 与重排检查可用；能执行 Week 1 的训练循环和 single-batch 检查。**执行建议**：新增 `week02/src/skeleton_toy.py` 与 `week02/experiments/train_skeleton.py`，全程使用本周自写模块和合成数据。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:45 | 45m | Understand → Trace | 时空轴、重排、toy 输入 / 标签 / loss | 数据流和无泄漏任务定义 |
| 00:45–02:00 | 75m | Run | 两种 block、toy 模型与合成数据 | 可运行模型、shape / 编号检查 |
| 02:00–03:20 | 80m | Manipulate → Experiment | single-batch sanity check、正式小训练、证据检查 | loss、梯度、验证 MSE 与权重 |
| 03:20–04:00 | 40m | Analyze → Research Log | 整理报告、日志与口述验收 | 训练结论、失败记录、Gate 状态 |

### 4. 核心资源

- [Annotated Transformer 官方源码](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)：按需回查 Encoder 数据流；不运行完整教材模型。
- 本周 `notes/tensor_shapes.md` 与 `notes/encoder.md`：优先用自己的推导定位问题。
- [MotionBERT · DSTformer.py](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py)：仅在重排理解受阻时回查 `forward_spatial / forward_temporal`；正式短 trace 放在 Day 5。

本日外部阅读限时约 15m；时间重点给独立实现、训练与排错。

### 5. 执行步骤

#### A. Understand → Trace

1. **15m 回查与手推**：空间 block 在同一帧内让 J 个 joint 交换信息；时间 block 在同一个 joint 的 T 个时刻间交换信息。两者都不把不同序列当作 token 混在一起。
2. **15m 重排 trace**：写出并检查下表，用连续编号张量验证每个入口与逆变换。临时把 Encoder 换成 identity，输出必须逐元素恢复原张量。
3. **15m 定义任务**：把“共同执行约定”中的输入、干净标签、observed flag、中心帧 MSE 和任意坐标单位写进报告，先写可验证假设再训练。

| Block | 入口变换 | 有效 batch / token | 逆变换 |
| --- | --- | --- | --- |
| Spatial | `[B,T,J,C] → reshape(B*T,J,C)` | batch 为序列与帧，token 为 joint | `[B*T,J,C] → reshape(B,T,J,C)` |
| Temporal | `[B,T,J,C] → permute(0,2,1,3) → reshape(B*J,T,C)` | batch 为序列与 joint，token 为 time | `[B*J,T,C] → reshape(B,J,T,C) → permute(0,2,1,3)` |

本周 toy 是 **Spatial Encoder → Temporal Encoder** 的单路径串联，两套 Encoder 参数独立；不把它称作完整 PoseFormer 或 DSTFormer。joint / time PE 在输入 embedding 后加一次，位置 shape 分别为 `[1,1,J,32]` 与 `[1,T,1,32]`。

#### B. Run

1. **20m Spatial**：复用 Day 3 Pre-LN Encoder；保存 dropout 前 A，形状 `[B*T,H,J,J]`，还原索引为 `[B,T,H,J,J]`。
2. **20m Temporal**：同一接口另建 Encoder；A 为 `[B*J,H,T,T]`，还原索引为 `[B,J,H,T,T]`。用 identity / 编号检查逆变换，尤其检查 batch 与 joint 是否混排。
3. **20m Toy 模型**：按共同配置连接 `Linear(4,32)`、PE、两个 Encoder、final LN 与坐标 head；输出 `[B,9,5,3]`。先 forward / backward 一次，确认两种 attention 模块都在 loss 路径上。
4. **15m 数据**：写生成函数，固定偏移 `p_j` 为 root `(0,0,0)`、left_hand `(-0.5,0.6,0)`、right_hand `(0.5,0.6,0)`、left_foot `(-0.3,-0.8,0)`、right_foot `(0.3,-0.8,0)`。训练数据 generator seed=42，验证数据 generator seed=43；生成一次后保存 / 固定复用。

保留干净 Y 的独立副本，再构造加噪 X 和 mask，不能用共享存储原地清零后把标签也清零。训练 / 验证的 `a_n、v_n` 独立采样，但使用相同生成规律与关节定义。打印输入、标签、中心帧与 flag，确认中心帧所有坐标为 0，干净标签未被覆盖。

实验入口具体参数由你实现，再把实际命令写入日志。所有路径都从资料根目录出发，例如运行自己完成的 `week02/experiments/train_skeleton.py`；本计划没有声称该入口已经存在。

#### C. Manipulate → Experiment

**Single-batch sanity check（25m）**

1. 用独立调试模型固定 8 条训练序列，Adam lr=1e-3 运行 100 步；记录初始 / 最终中心帧 MSE、梯度范数和参数变化。
2. 可把最终 MSE 低于初始的 20% 作为排查目标；这不是理论保证或原表新增阈值。未达到先查数据泄漏、loss 轴、图连通和更新，再单独改变 lr / 步数并记录。
3. 所有输入 / loss / 梯度 / 更新后参数均应有限；有受控故障也保留真实症状，修复后同条件复测。

**正式训练（35m）**

1. 新建正式模型，从保留的初始状态开始；不得把调试模型已见过 batch 的权重当作正式初始状态。model seed=42，记录初始 state、数据、batch 顺序与配置。
2. 全部 64 条训练序列按固定顺序 batch=8，循环至 300 次参数更新；每 25 步记录 loss。只用中心帧 MSE，不让大量可见帧把遮挡错误平均掉。
3. 验证时使用全部 16 条独立序列，eval + no_grad；记录初始 / 最终验证 MSE 和权重路径。数据固定，不在每次评估重新采样噪声。
4. 达到本段时间上限就保存实际完成的 steps 与状态。没有明显下降也先如实记录，不能为了通过而改标签或挑选最好的一条样本。

**信息路径与证据检查（20m）**

- 一次正常 backward 后，检查 Spatial / Temporal 的投影参数是否获得梯度；`grad=None` 与某个元素恰为 0 是不同问题。
- 用独立输入副本取中心帧输出的一个标量，对可见帧坐标求梯度，检查跨帧依赖是否存在；打印对应 frame / joint 索引。数值很小先查图和权重，不能单凭梯度大小解释人体运动。
- 保存干净标签、遮挡输入、权重、配置和训练曲线；不要把用于绘图的 detach 放回 loss 主路径。

#### D. Analyze → Research Log

20m 整理 `results/d4_training.md`；10m 填日志；10m 口述一条 joint 从原始坐标到 Spatial、Temporal、中心帧 loss 的路径。解释此 toy 只做已有 3D 坐标的遮挡补全，不是 2D→3D lifting，不输出旋转、SMPL 参数或 mesh。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实现 | `week02/src/transformer.py`、`skeleton_toy.py` | 两种时空 block、数据生成、toy 模型 |
| 实验 | `week02/experiments/train_skeleton.py` | 独立 sanity check、正式训练、验证与梯度 probe |
| 数据 / 权重 | `week02/results/toy_data.pt`、`toy_model.pt` | 固定 X/Y、joint 顺序、配置与实际更新次数 |
| 结果 | `week02/results/d4_training.csv`、`d4_training.md` | 训练记录、验证 MSE、shape、梯度与限制 |
| 日志 | `week02/research_log.md`、`week02/fail_log.md` | Day 4 的假设、实测、排错与下一步 |

### 7. 验收标准（Gate）

- [ ] Spatial `[BT,J,C]` 与 Temporal `[BJ,T,C]` 都由本人 Encoder 实现；identity 重排与逆变换逐元素一致。
- [ ] 输入、输出、两类 A 的 shape 与手推一致，joint / frame / batch 没有混排。
- [ ] 干净标签与遮挡输入分开保存，中心帧 loss 口径、joint 顺序和任意单位明确。
- [ ] 有实际 single-batch / 正式训练记录，finite 检查及梯度 / 参数更新证据；异常与未完成步数如实登记。
- [ ] 能解释信息如何跨帧到中心 query，以及 toy 与真实人体表示任务的边界。

### 8. 受阻处理

先做 identity 重排，再做 B=1、T=3、J=2 的 forward / backward，最后恢复统一配置。loss 不降依次查 **Data → Forward → Loss → Gradient → Update → Evaluation**。CPU 太慢时先缩调试规模；正式结果若改变 T、C 或 steps，要记录并在 Day 5 所有组保持同口径。超时删去 animation、美化和新增网络，不增加真实数据下载。

### 9. 下一步

冻结同一训练后权重与验证输入，准备上下文对照与两类 attention 图；先写预测，再运行。

## Day 5：Attention Visualization + Minimal Experiment + Final Gate｜3h45m

### 1. 当日目标

可视化实际模型的 spatial / temporal attention；完成一次上下文干预实验；把自己的 shape 与后续源码对应起来，整理本周可复现成果和 Week 3 交接。

### 2. 前置条件

Day 4 的固定验证输入、干净标签与训练后权重可定位；模型能返回各 head 的 dropout 前 A。**执行建议**：新增 `week02/experiments/context_probe.py`，同一入口完成评估与绘图，不再训练第二套大模型。

### 3. 时间安排

| 累计专注时间 | 时长 | 阶段 | 具体动作 | 当段产物 |
| --- | --- | --- | --- | --- |
| 00:00–00:40 | 40m | Understand → Trace | DSTFormer 短 trace、对照假设与变量冻结 | 源码 shape、实验预测 |
| 00:40–01:55 | 75m | Run | 上下文对照、插值基线、两种 attention 图 | MSE、原始 A、标注完整的图 |
| 01:55–02:40 | 45m | Manipulate → Experiment | 受控轴错误、修复复测、结果解释 | 索引检查与结论限制 |
| 02:40–03:45 | 65m | Analyze → Research Log | 最终验收 45m；日志与周总结 20m | 最终 Gate、周总结、Week 3 交接 |

### 4. 核心资源

- [MotionBERT · DSTformer.py](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py)：只追 `DSTformer.forward → Block.forward → Attention.forward_spatial / forward_temporal`。
- [PoseFormer · model_poseformer.py](https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py)：作为下方桥接区别的查证入口；本日主 trace 选择 DSTFormer，不要求再完整读第二个模型。
- 本周 `notes/tensor_shapes.md` 和 Day 4 训练报告：优先检查自己已经实现的数据流。

本日主源码 trace 限时 20m。上述源码路径已于 2026-10-11 只读核对；执行时记录访问日期与 commit / 版本，若目录变化先查函数名。PoseFormer / MotionBERT 的论文深入阅读仍放在原计划 Week 4 的 2–4h 桥接预算。

### 5. 执行步骤

#### A. Understand → Trace

1. **20m 源码桥接**：在 `notes/repo_bridge.md` 标出 DSTFormer 中 joint embedding、position / temporal embedding、`stage_st / stage_ts`、两种 attention 的位置。选择一个 temporal 路径逐步写 shape，不安装仓库环境、不下载 checkpoint。
2. **10m 写假设**：例如：“中心帧全遮挡后，模型应利用同一序列可见帧的位移信息；把可见帧换成其他序列、标签不变，MSE 可能升高。”把“可能”留在预测栏，运行后再判断。
3. **10m 冻结变量**：固定权重、16 条验证样本、噪声、中心位置、joint 顺序、PE、eval 模式和指标。用固定循环移位 `π(n)=(n+1) mod 16`，保存交换对应表。

短桥接要能区分以下结构，不能因为都有时空 attention 就把 toy 当成论文复现：

| 结构 | 本周需要确认的区别 | 证据入口 |
| --- | --- | --- |
| 自写 toy | 空间 token 是 joint，时间 token 是每个 joint 的各帧；单条 S→T 路径 | 本人实现与 shape 输出 |
| PoseFormer | 空间后把 joint 特征拼成每帧向量，再以帧为 temporal token；此处不是逐 joint 的 `[BJ,T,C]` temporal 结构 | `Spatial_forward_features / forward_features`；[官方源码](https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py) |
| DSTFormer / MotionBERT | 区分 S→T 与 T→S 两种顺序及其融合；temporal 打分保留 joint 维，可写成 `[B,H,J,T,T]` | `Block.forward / Attention.forward_temporal / DSTformer.forward`；[官方源码](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py) |

MotionBERT 是完整方法 / 训练框架，DSTFormer 是其中的结构；本周不复现预训练、双路径融合或论文评测。对实际输入通道、坐标约定和输出定义，后续读所选配置与数据处理代码确认，不从 `dim_in=3` 之类默认值推断任意任务。

#### B. Run

**最小实验（25m）**

新入口加载 Day 4 固定数据和同一 checkpoint；不重新训练，不因某个组表现不好换权重。

| 组别 | 输入 / 方法 | 固定项 | 指标 |
| --- | --- | --- | --- |
| A 正常上下文 | 原验证输入：本序列可见帧，中心仍遮挡 | 权重、标签、模式与样本 | 每条序列中心帧 MSE 与均值 |
| B 交换上下文 | 把 n 的全部可见帧坐标换成 π(n) 的对应可见帧；中心坐标 / flag 保持原遮挡设置 | 同一模型；Y[n] 不交换，帧次序、PE 与 joint 次序不变 | 同口径 MSE；`ΔMSE=B−A` |
| C 零坐标基线 | 中心所有关节预测为 `(0,0,0)` | 同一干净中心标签 | 中心帧 MSE |
| D 邻帧插值 | 中心预测为 `(X[:,3,:,:3]+X[:,5,:,:3])/2` | 使用原输入，仅可见邻帧，不能读取中心干净标签 | 中心帧 MSE |

B 是主动破坏上下文与标签对应关系的机制 probe，不能当作另一组正常测试性能；可见帧整体来自同一条替换序列，避免把逐帧随机拼接额外引入。若 B 比 A 差，支持“输出依赖序列上下文”的推断，但不证明模型已学到时间先后、人体动力学或某个 attention head 的因果作用。

**Attention 可视化（30m）**

对 A 组固定样本 n=0，保存原始 `A_s / A_t` 与索引。先看未平均的 head，再按需补 head 均值；默认 eval、dropout=0。

| 图 | 取法 | 行 / 列 / 注释 |
| --- | --- | --- |
| Spatial heatmap | `A_s.reshape(B,T,H,J,J)[0,3,0]`，选一个可见帧 | 行=query joint；列=key joint；标出 5 个 joint 名、frame=3、head=0、checkpoint |
| Temporal heatmap | `A_t.reshape(B,J,H,T,T)[0,0,0]`，选 root | 行=query time；列=key time；帧 0–8；标出遮挡中心 frame=4、joint=root、head=0 |
| 中心帧 skeleton 对照 | clean Y、插值和 toy 的中心预测 | 相同坐标范围；骨架连线一致；任意单位，清楚标明输入中心已遮挡 |

原始 A 有限、非负，最后一维的行和约为 1；图的行 / 列与张量索引一致。空间与时间图可以不同，但不要把跨不同 softmax 大小的单个格子值直接当作相同量级的“重要性”。attention 图不能单独证明解剖连接或机制正确。

**整理数值（20m）**

`results/d5_context.csv` 至少记录 `sample_id, source_sample_id, condition, masked_mse`；同时记录 group 均值、模型 / 数据路径、dtype、steps、seed、joint 顺序与中心帧索引。将两类原始 A 存为 `attention_maps.pt`，图片使用同一组权重和输入，避免用另一个运行的图解释本次数字。

#### C. Manipulate → Experiment

1. **20m 受控故障**：在独立实验副本里临时漏掉 Temporal 入口的 joint / time permute，运行编号逆变换与固定输入 probe。不要覆盖正常权重或旧结果；记录 shape 可能合法但样本索引错误的证据，再恢复。
2. **15m 复测**：从同一 checkpoint 重新加载，恢复正确重排，重复 A/B 评估与 A 行和检查。相同 CPU / dtype / 输入下结果应在数值容差内复现；定位不同进程中的首次分歧。
3. **10m 分析**：解释 A/B 差值、插值结果和图中能观察到的事实。若交换上下文没有明显影响，先查是否真的交换、中心是否泄漏、Temporal 是否参与 loss、模型是否只输出固定骨架；若模型不如插值，保留这一结果，不增加新模型来制造优势。

当前 toy 的 motion 是共享平移，邻帧插值天然是强 baseline；它能检查跨帧信息路径，**不足以证明 spatial attention 的必要性、时间顺序理解或真实动作泛化**。这些是后续实验问题，不在本周强加新消融。失败结果的价值是定位实现 / 训练问题，或发现任务本身太简单。

#### D. Analyze → Research Log

**整体验收（45m）**

1. **15m 材料检查**：代码、数据、权重、命令、环境、结果和图可定位；只把实测写进 Observation，清除把预期误写为结果的记录。
2. **20m 复现关键证据**：重跑 Attention 手算 / mask、MHA 逐 head、PE 置换、时空 identity 重排及固定权重 A/B 评估；已有入口共用，不再另建重复实现。
3. **10m 闭卷验收**：给一个没事先背过的例子，如 `B=3、T=7、J=6、C=24、H=3`，推导 Spatial / Temporal Q/K/V 和 A；讲清 PE、LN、loss 与两种源码结构的区别。

**日志与周总结（20m）**

新增 `week02/weekly_review.md`，用 15m 回答下列问题，再用 5m 更新日志与最终 Gate：

- 本周能独立写出的三个模块是什么？分别链接哪份代码与正确性证据？
- 最容易混淆的两个轴是什么？哪次编号 / 数值检查抓住了错误？
- 最小实验支持了哪个具体推断？什么仍只是预测或解释？
- 模型与插值的差异说明什么？本 toy 为什么无法证明方法 novelty？
- 哪些 Gate 待补？根因、下一步动作与通过证据是什么？
- Week 3 中，joint positions、joint rotations、SMPL parameters、mesh 分别是什么？目前不能解释的地方列成问题，不把本周坐标 toy 误当成已掌握人体表示。

### 6. 交付物

| 类型 | 文件或位置 | 完成要求 |
| --- | --- | --- |
| 实验 | `week02/experiments/context_probe.py` | 固定权重 A/B、零预测 / 插值、attention 导出与复测 |
| 结果 | `week02/results/d5_context.csv`、`d5_experiment.md` | 原始逐样本指标、汇总、实际命令、解释与限制 |
| Attention | `week02/results/attention_maps.pt`、`spatial_attention.png`、`temporal_attention.png` | 原始 A 可定位，query / key、joint / frame / head 标注正确 |
| 可视化 | `week02/results/center_reconstruction.png` | 同一中心帧的 clean / 插值 / toy 对照，坐标范围一致 |
| 源码桥接 | `week02/notes/repo_bridge.md` | 一个 DSTFormer 路径的函数位置、shape 与 toy 区别 |
| 总结 / 日志 | `week02/weekly_review.md`、`research_log.md`、`fail_log.md` | 本周结论、证据、最大不确定性、待补和 Week 3 第一步 |

### 7. 验收标准（Gate）

- [ ] 两类 attention 图来自实际 toy，原始 A、轴标注与 checkpoint / 输入一致。
- [ ] A/B 与两种简单 baseline 有同口径实测；复测可重现，不预设 Transformer 胜出。
- [ ] 有一次时空轴故障的定位、修复与同输入复测，能解释合法 shape 不等于语义正确。
- [ ] 能闭卷推导新尺寸的主要 MHA / 时空 shape，并指出 PE 和 LN 的作用位置。
- [ ] 能区分自己的 S→T toy、PoseFormer 的每帧向量、DSTFormer 双顺序结构；短 trace 有源码证据。
- [ ] 已更新 Week 2 最终 Gate，所有待补项都有具体行动与证据要求。

### 8. 受阻处理

若没有可用训练后权重，可以先对随机初始化模型检查 shape / 行和并画图，但必须标为“未训练模型”，训练、机制实验和完整 toy Gate 保持待补。绘图受阻时先保存原始 A 与带标签的数值表，图形交付继续待补；不把数值表冒充已完成可视化。优先删去美化和第二仓库阅读，保留正确证据、日志与周总结。

### 9. 下一步

按最终 Gate 优先补实现 / 数据 / shape 问题。进入 Week 3 时，先区分坐标、旋转和局部 / 全局坐标系，再依原表做 rotation playground、FK、SMPL；本周不提前启动 SMPL 下载或改写后续周路线。

## Fail Log / Research Log 模板

**执行建议**：执行时自行新建 `week02/research_log.md` 和 `week02/fail_log.md`。沿用 Week 1 的字段与证据要求，额外记录当前 tensor layout、mask 语义和 frame / joint 索引。

### Research Log

```markdown
## YYYY-MM-DD / Week 02 Day N / 主题
状态：未开始 / 进行中 / PASS / 待补
计划时间 / 实际时间：
环境、seed、代码版本或文件快照：
Question：本日要回答的具体问题？
Hypothesis：改变什么，预期出现什么，为什么？
Understand / Trace：概念、函数位置、token / head / joint / time 轴。
Run：实际命令、配置、输入、权重和结果文件路径。
Manipulate / Experiment：控制组、实验组、唯一改变的变量。
Prediction / Observation：预测与实测分开写，附误差 / MSE / shape。
Analyze：结论、机制解释、替代解释、适用条件与限制。
Implementation provenance：哪些自己写，哪些参考了教材、源码或助手？
One thing I can now explain：不看资料能讲清楚的一件事。
Bugs / confusion：关联 Fail Log ID。
Gate：逐项状态，未通过项的补做动作与证据。
Tomorrow first step：一个可直接执行的动作。
```

### Fail Log

```markdown
## W02-DN-F01 / 简短标题
时间 / 状态：Open / Explained / Fixed / Retested
Context：环境、代码版本、输入、shape、mask 语义与复现命令。
Expected：预期值、轴含义或行为。
Observed：真实输出、报错、误差或梯度；附文件路径。
Minimal reproduction：缩到哪个 B / T / J / C / H 与操作？
Hypotheses：候选原因与排除证据。
Root cause：有证据支持的原因，未确认则写待验证。
Fix / explanation：改动或对机制的解释。
Retest：同输入、同权重复测的命令、结果与判据。
Lesson：下次在哪里加什么 shape / index / finite 检查。
```

**常见失败点 / Debug 提示（执行建议）**

| 症状 | 优先假设 | 最小验证 / 修复方向 |
| --- | --- | --- |
| A 的 key 轴行和不为 1 | softmax 轴错，或检查的是 dropout 后 A | dropout=0，打印 logits / A 的轴，检查 `sum(-1)` |
| mask 后 NaN 或禁止位置仍有概率 | True 语义反了、mask 广播错、整行被屏蔽 | 2-token 主例，逐格查看 mask / logits；拒绝整行 mask |
| MHA shape 对但数值不对 | 拆 / 合 head 缺 permute，或 scale 用了 C 而非 d_h | 逐 head 参考与连续编号重排；确认 `1/sqrt(d_h)` |
| permute 后 view 报错 | 张量 stride 不满足 view 条件 | 查看布局，正确换轴后使用 reshape 或 contiguous + view |
| 无 PE 置换对照失败 | dropout / mask / 轴 / norm 约定不同 | 固定 eval、dropout=0、无 mask、LN(C)，查每一步 |
| PE 加法报错或位置错位 | T / J 广播轴错、长度 / dtype / device 不匹配 | 分别打印 joint `[1,1,J,C]`、time `[1,T,1,C]` |
| 中心帧 loss 很低却不使用上下文 | 标签被清零、读取了干净标签、只看了可见帧均值 | 打印干净中心 Y 与遮挡 X；只算中心 MSE；做 A/B probe |
| Temporal 参数 grad=None | 输入走错分支、detach、Temporal 输出没进入 head | loss 反查到 head、temporal、spatial、embedding |
| 热图好看但不是预期样本 | 把 BT / BJ 当成 B，head / joint / frame 取错 | 先还原 `[B,T,H,J,J]` / `[B,J,H,T,T]`，记录索引 |
| 交换上下文后结果完全没变 | 没有真正交换、模型常数输出、上下文路径失效 | 保存 π 与两组输入差、逐样本输出差，再查梯度与训练 |
| Toy 不如线性插值 | 模型未训好，也可能任务本来很适合插值 | 同口径复测，检查训练后保留失败结果；不改 benchmark 口径 |

## Week 2 最终 Gate

下面以原表四项交付为主，将本周细化证据映射回 `10周计划!D11`。所有状态初始均为待完成；数值结果必须来自本人实际执行。

| 项目 | 对应原计划 | 通过证据 | 当前状态 |
| --- | --- | --- | --- |
| Attention → MHA → Encoder | 交付 1 | 本人核心实现；手算、mask、逐 head 与 block forward / backward 检查齐全 | 待完成 |
| Tensor-shape gym | 交付 2 / 原 Gate | 闭卷推导普通、Spatial、Temporal Q/K/V 与 A；新尺寸口述正确；编号重排通过 | 待完成 |
| Spatial / Temporal blocks | 交付 3 | `[BT,J,C]` / `[BJ,T,C]` 两种 block 能运行、还原正确、进入同一 toy 的 loss 路径 | 待完成 |
| Skeleton toy | 交付 4 | 固定数据、无标签泄漏、训练 / 验证 / 权重 / 命令有证据；异常已定位或明确待补 | 待完成 |
| Attention visualization | 交付 4 | 两类图与原始 A 可定位，索引 / 行列 / 模式 / 权重标注正确 | 待完成 |
| PE / Pre-LN / Post-LN | Focus 的执行细化 | PE 置换、LN 轴与零子层对照；闭卷解释作用及残差顺序 | 待完成 |
| Minimal experiment / Debug | 执行建议 | 同权重 A/B、零预测 / 插值、复测和失败记录；结论不超出证据 | 待完成 |
| Research habit / Bridge | 执行建议 / 后续源码 | 五日日志、来源记录、短 trace、周总结与 Week 3 问题可定位 | 待完成 |

工程证据和口述解释都通过才记完整 `PASS`。代码没写完、图未完成、数据或轴错误未排除时登记待补；每项写清一个下一步动作和预期证据。训练性能不优于插值、attention 不呈现直观人体关系，均不自动否定实现，但必须如实解释；单靠 loss 下降或热图不能替代正确性验收。

超出 19h 的必做待补项先登记，不擅自挤占或重排 Week 3–10；优先补原表四项交付，再考虑扩展阅读。额外 seeds、真实 pose lifting、完整 MotionBERT / PoseFormer 环境、预训练、旋转 / FK / SMPL 实现都不计入本周必做。

## 与 research-journey 衔接

沿用 Week 1 的“计划、代码、实验与日志只有一份维护位置”的约定，当前实际目录为 `/Users/mac/顶会刊收割机/01_10week_plan/`。本周的新增学习产物按建议放入其 `week02/`；不依据 Week 1 中的旧路径再次迁移目录，不改 `week1.md`、原计划表或目录外文件。

**Week 3 交接（依据 `10周计划!B12:D12`，具体第一步为执行建议）**

- 保留本周的 joint 名称 / 顺序、坐标维数、任意单位、输入 / 标签定义与时空 shape；注意 `C=32` 是网络特征，不是人体自由度或 rotation representation。
- 写清“joint positions ≠ joint rotations ≠ SMPL parameters ≠ mesh”。本周 toy 只有自造 joint 坐标，未实现 SO(3)、FK、LBS 或人体模型。
- 下一模块先手画一个最小 kinematic tree，标出父子关系和 local / global frame，再按总计划实现 rotation playground、FK、smplx 参数控制与可视化。AMASS 不可用时依原表采用自造 SMPL motion sequence，不让数据权限替代学习目标。
- Week 4 再按原计划预算回顾 PoseFormer / MotionBERT 的任务、representation 与关键 shape，连接 camera / HMR；本周短 trace 不等于已完成 Week 4 的论文阅读。

**尚未确认的源文件事项**

- 当前目录未找到 `week.md` 或独立命名的“导师反馈修订版”Excel；已采用存在的 `week1.md` 与 `00_plan/10周研究计划_曾奕博.xlsx`，未把历史路径中的其他版本带入本周。
- Excel 的 Week 2 行完整支持上述主线与四项交付，但没有规定五天日期、具体 toy 任务、19h 拆分和数值阈值；这些已明确标为执行建议。
- 同目录存在 `.numbers` 原生文档，本次按任务优先读取 `.xlsx`，未核对两者是否逐单元格一致；本文件不能证明该 Excel 是所有文档中的最新版本。
- Week 1 的实际学习 / 实验完成状态未获得证据；本周只列必要入口能力，不替本人勾选已通过，也不修改其计划。
