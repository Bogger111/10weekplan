# 实验 D：SORT Tracking + Local Mesh Smoothing + Original Root

## 实测结论

本次使用现有 50 帧、10 FPS、两人推理缓存，实际完成 D0、D1、四路 Overlay、Windows Blender 动画及指标验证，未训练、未调用 Multi-HMR 的 `infer_image` / `infer_video`。只加载原 checkpoint 中的同一 body model，重建平滑前后的局部几何。

**D 没有同时降低两人平均的 3D Relative Mesh 与 2D Projected Mesh Jitter。** D 的 3D Relative Mesh 二阶差分下降 54.26%，2D Mesh 二阶差分增加 6.13%。ID 0 的二维指标改善，ID 1 变差，因此不能把两人结果合并成“局部平滑必然同时改善二维和三维”。

**Root 位置平滑是该片段中 B 二维恶化的主要原因。** B 与 D 的局部几何相同，改回 Original Root 后 2D Mesh 二阶差分从 7.8274 降到 2.6133 px/frame²，明显接近 A 的 2.4624。局部平滑仍有残余的二维代价，不能据此认为平滑强度和动作保真已经理想。

## 统一五项指标

全部是逐点向量二阶差分的模长均值，按两个人、48 个三帧窗口等权平均。3D 单位 m/frame²，2D 单位 px/frame²；10 FPS 下乘 100 才是每秒平方单位。没有按各组分别重标定或归一化；A/B/C/D 共用米制与同一预处理图像的 768×432 像素坐标。

| 指标 | A Original | B 3D Smooth | C SORT | D1 Local + Original Root | D 相对 A |
|---|---:|---:|---:|---:|---:|
| 3D Center | 0.384803 | 0.072775 | 0.155061 | 0.384803 | 0.00% |
| 3D Root-relative Mesh | 0.025028 | 0.011448 | 0.025028 | 0.011448 | **−54.26%** |
| 3D Absolute Mesh | 0.385635 | 0.075495 | 0.157617 | 0.385374 | −0.07% |
| 2D Projected Mesh | 2.462403 | 7.827413 | 5.875593 | 2.613263 | **+6.13%** |
| 2D Projected Center | 1.802556 | 6.344847 | 4.707813 | 1.802556 | 0.00% |

Center 定义始终是官方 person-center 顶点 **5054**，不是 Mesh AABB 中心，也不是骨盆骨骼位置。投影共用原始逐帧 K。坐标是原推理的相机空间，不是经外参标定的稳定世界坐标；因此不能将这些值当成真人动作误差。

### 每个 Track 的五项指标

| ID | 组别 | 3D Center | 3D Relative | 3D Absolute | 2D Mesh | 2D Center |
|---|---|---:|---:|---:|---:|---:|
| 0 | A | 0.396253 | 0.024909 | 0.398021 | 2.962686 | 1.890638 |
| 0 | B | 0.081231 | 0.010278 | 0.083098 | 6.827847 | 5.057795 |
| 0 | C | 0.157677 | 0.024909 | 0.160639 | 5.476783 | 4.144045 |
| 0 | D | 0.396253 | 0.010278 | 0.396055 | 2.739105 | 1.890638 |
| 1 | A | 0.373354 | 0.025146 | 0.373248 | 1.962120 | 1.714475 |
| 1 | B | 0.064319 | 0.012618 | 0.067892 | 8.826979 | 7.631900 |
| 1 | C | 0.152445 | 0.025146 | 0.154595 | 6.274402 | 5.271580 |
| 1 | D | 0.373354 | 0.012618 | 0.374693 | 2.487421 | 1.714475 |

ID 0：D 的相对 3D 指标下降 58.74%，2D Mesh 下降 7.55%。ID 1：相对 3D 指标下降 49.82%，2D Mesh 增加 26.77%。本次未做姿态与 shape 的进一步单独消融，因此不能把 ID 1 的退化具体归于其中一个参数通道。

## D0 / D1 的实际构造和一致性

**身份关联：** 从已审计 `experiment.py` 提取原 `Track` / `iou` 定义，重跑同一 3D IoU、匈牙利匹配、阈值与 Kalman 状态。检测顺序同样逐帧反转。重放 ID 和位移与旧 C 缓存完全一致；D 只使用 ID，重放位移没有参与 D 的顶点计算。两条轨迹各覆盖 1–50 帧，SORT ID 0/1 与缓存 A/B 的 official ID 0/1 逐帧相同，无需靠 X 排序重新猜测对应关系。

**D0：** 在验证帧和人物对应后，直接计算 `D0 = A_center + (B - B_center)`。

**正式 D1：** 按 SORT ID 分组并在时间缺口处拆段，独立重跑 B 的滤波：163 个局部关节旋转的双向 SO(3)，alpha=0.3；shape 二阶 Butterworth 1 Hz、双向零相位，裁剪至 [0,1]。不运行 B 的中心 2 Hz 滤波。用同一 Anny body model、LBS 得到 `base` 和 `smoothed`，构造：

`delta_local = (smoothed - smoothed_center) - (base - base_center)`

`D1 = A + delta_local`

该差分转移与原 B 相同，避免把 Anny 往返重建残差带入官方网格。D1 的顶点 5054 的 `delta_local` 为零。

Root 的“原始”在本实验指 **Root 位置**：沿用 B 对全部 163 个关节旋转的平滑，其中也包括根骨骼朝向。本次没有单独冻结根朝向；若冻结它，会成为另一个与本 D0 不同的消融条件。

| 验证项 | 结果 |
|---|---:|
| D0/D1 网格差的平均向量模长 | 4.7220e-8 m |
| D0/D1 网格差的最大向量模长 | 9.8348e-7 m |
| D0/D1 最大单坐标差 | 9.5367e-7 m |
| D0/D1 最大投影单坐标差 | 5.1596e-5 px |
| 重算 B 与原保存 B 的最大差 | **0，逐元素完全一致** |
| D1 smoothed pose 与旧 B sm_pose 最大单元素差 | **0** |

D0 与 D1 在 float32 精度内一致。非零微差来自 B 已执行并存储 `A + root_delta + local_delta`，D0 再减 B_center、加 A_center，而 D1 直接 `A + local_delta`，浮点加减顺序不同。重算 B 完全一致排除了人物对应错误和全局旋转处理差异。

D0 的五项指标分别为 0.384803422、0.011447880、0.385373868、2.613263452、1.802556177，与 D1 一致到显示精度。详细见 `validation.json` 和 `qa_additional.json`。

## 不变量、异常与动作边界

- D0、D1 的顶点 5054 均与 A **逐帧 bitwise 完全相等**，XYZ 最大差为 0；中心深度差也为 0，没有 Kalman Z 修正。
- SORT ID、official ID、50 帧集合、拓扑、逐帧 K 全部一致。A/B/C 直接复用原缓存，不重新计算或修改它们的输出。
- D 相对 A 的局部 Mesh 确有变化：平均顶点位移 0.02217 m、最大 0.22554 m。这不是“只改了标签”；同时也提示强平滑可能改变动作，缺少真值时不能把稳定性下降幅度等同于动作精度提升。
- 全部输出有限、无 NaN/Inf；D 顶点最小 Z 为 7.28158 m，无零/负深度投影异常。相机坐标在评价阶段没有做轴转换。Blender 只在显示时共用 `(x,y,z)→(x,z,-y)` 和列间平移。
- 原始中心轨迹的跳动被 D 有意保留。因此 D 绝对网格的帧间整体位移并不会消失：ID 0/1 最大平均绝对帧间位移约 0.63834/0.67544 m，都在第 29→30 帧附近，A 对应为 0.63779/0.67643 m。该跳变已存在于 A；不能把“保留 Original Root”误称为没有整体跳变。
- D 的最大平均相对网格帧间位移为 ID 0=0.01970 m（46→47 帧）、ID 1=0.02147 m（22→23 帧），小于 A 的 0.06725/0.05057 m，未发现新增的局部帧间爆跳。

异常候选使用 **每方法、每 ID、每指标的 median + 5×MAD** 阈值，并同时列出前三大值。前三大值不一定超过阈值，也不是人工标注错误。`anomaly_frames.csv` 分别标出 `above_threshold` 和 `top3`，`jitter_by_frame.csv` 保留全部 48 个窗口。帧号均是三帧二阶差分窗口的中间帧。

| D 的指标 | ID 0 三个最大值所在帧 | ID 1 三个最大值所在帧 |
|---|---|---|
| 3D Center | 29、26、11 | 49、29、48 |
| 3D Relative Mesh | 47、46、22 | 22、21、43 |
| 3D Absolute Mesh | 29、26、11 | 29、49、48 |
| 2D Mesh | 20、19、21 | 3、2、48 |
| 2D Center | 20、19、21 | 3、2、19 |

D 的 ID 0 二维 Mesh 超过本组 MAD 阈值的中间帧为 19、20、21、28、47；ID 1 没有超过本组阈值的二维 Mesh 帧，表中的 3、2、48 只是 top3。ID 0 第 20 帧 D=22.2945 px/frame²，A 本身为约 30.9935；该段保留了原中心投影的明显变化。

## 三个关键比较如何解释

**A vs D：** 固定同一 Original Center，局部旋转/shape 平滑显著降低 3D Relative Mesh 的二阶差分；两人平均的 2D Mesh 并未改善。3D Absolute 指标主要受原中心轨迹支配，整体只下降约 0.07%。

**B vs D：** 局部几何保持一致，仅恢复 A 的中心位置。2D Mesh 从 7.8274 降到 2.6133；2D Center 从 6.3448 完全恢复为 A 的 1.8026。该控制变量结果支持 Root 位置平滑为 B 二维恶化的主要因素。指标是非线性均值，不将差额简单解释为可相加的误差贡献率。

**C vs D：** C 只改整体中心，保留原局部姿态，Relative Mesh=0.02503、2D Mesh=5.87559；D 只改局部旋转/shape、保留原中心，Relative Mesh=0.01145、2D Mesh=2.61326。它们分别测试位置更新与局部平滑，不能把 D 的局部改善归功于 SORT 跟踪。

本片段只有 5 秒、两条完整且未交叉的轨迹；相机内参逐帧变化。没有真人 2D/3D、接触或身份真值，所有 jitter 均包含真实动作和重建变化，数值更低不能证明重建更准确。B 和 D 为双向离线滤波，使用未来帧，不应作为实时滤波性能结果。

## 文件、展示与复现

WSL 全部结果目录：`/home/bogger/10weekplan/week6/experiment_D/`。Windows 可打开副本：当前 `outputs/experiment_D/`。

- `overlay_ABCD.mp4`：四列同一原视频叠加，3072×488、50 帧、10 FPS、5 秒、H.264。ID 0 蓝、ID 1 橙，各列均标出人物 ID 与帧号；底层单列图仍为 768×432。
- `human_motion_ABCD.blend`：8 个人体、50 帧、10 FPS，含完整顶点形态键、8 个人物 ID 标签和共同固定/跟随观察相机。`打开四路对比.cmd` 使用本机 Blender 5.2.3 打开它。
- `blender_ABCD_preview.mp4`：实际三维场景完整时间轴渲染，2400×500、50 帧、10 FPS。三维观察相机共用于四列，延续旧工程的正交视角与材质/光照；仅为容纳第四列扩大共同视野，没有给每组单独找视角。它不是原视频的透视重投影画面。
- `jitter_curves.png`：五项指标、每人独立的逐帧曲线；`metric_summary.png`：相对 3D 与投影 2D 的总量比较。
- `metrics.csv`、`metrics_by_track.csv`、`metrics.json`、`jitter_by_frame.csv`、`anomaly_frames.csv`：汇总、逐 ID、逐帧和异常候选。
- `data/geometry_ABCD.npz`：A/B/C/D1 实际网格、D0、faces、共享 K、SORT IDs、official IDs、center index。`data/D_parameters.npz` 保存原/平滑 pose、shape 与 ID。
- `validation.json`、`qa_additional.json`、`blender_verify.json` 与 `logs/`：不变量、独立指标复核、Windows Blender 重开验证和运行日志。

Windows 原生 Blender 重新打开工程，逐帧比较 **8 网格×50 帧=400 个网格帧**，最大坐标误差 1.49e-8 m，完整渲染 50 帧。两部 MP4 都已完整解码并核对帧数、帧率和时长；未测试 GUI 能否实时达到 10 FPS。

### 关键代码与行号

均为本目录 `code/` 下的实际运行副本：

| 文件 / 函数 | 行号 | 作用 |
|---|---|---|
| `experiment_D.py` → `replay_sort()` | 43 | 原 C 类定义的精确复用与关联重放 |
| `experiment_D.py` → `local_smooth_by_id()` | 66 | 按 SORT ID 与连续时间段运行 B 的旋转/shape 滤波 |
| `experiment_D.py` → `main()` | 141 | 验证 SORT 重放结果与旧 C 一致，并验证 official ID 映射 |
| `experiment_D.py` → `main()` | 146 | D0 根相对几何移植 |
| `experiment_D.py` → `main()` | 149 | D1 独立运行局部滤波 |
| `experiment_D.py` → `main()` | 155–160 | 使用同一 body model 生成局部形变差；D1=A+delta，无 Kalman 位移 |
| `experiment_D.py` → `evaluate()` | 78 | 五项统一指标、逐帧曲线和异常候选 |
| `experiment_D.py` → `render_overlay()` | 113 | 四路同相机叠加及 Tracking ID 标注 |
| `build_blender_D.py` | 39 | 每个人的 Tracking ID 标签 |
| `build_blender_D.py` | 64 | 保存 Windows 可播放 .blend |
| `verify_render_blender_D.py` | 1 | 重新打开后的全部顶点验证和时间轴渲染 |

WSL 复现数值和 Overlay：

```bash
cd ~/10weekplan/week6/experiment_D
export PATH="$HOME/miniforge3/envs/multihmr2/bin:$PATH"
python -u -X faulthandler code/experiment_D.py > logs/rerun.log 2>&1
python code/qa_experiment_D.py "$PWD"
```

代码保留 `torch → triton → torch._dynamo → multihmr2` 的导入顺序与 OSMesa 设置。复现需要本机原 checkpoint、缓存和帧图；不会重新运行 Multi-HMR 图像/视频推理。

原 A/B/C 网格、参数缓存、原脚本与原三路工程的 SHA-256 在本次前后一致，记录在 `logs/source_preservation.json`。本次仅新增 `experiment_D/` 与其 Windows 副本。
