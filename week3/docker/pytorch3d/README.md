# Week3 的 PyTorch3D Docker 环境

上次 Docker 承担的是 Python 2.7 / Chumpy 模型清理。清理后的 SMPL 交给原有 WSL `.venv` 运行；那次操作并没有给这个 `.venv` 安装 PyTorch3D。

本目录新增独立的日常研究环境：软件保存在 Docker 镜像中，代码和结果保存在 WSL 的 week3 目录中。每次命令启动一个临时容器，用完移除容器；镜像、代码、模型和已保存的结果都继续保留。

## 日常运行

先打开 Docker Desktop，然后在 **WSL Ubuntu-24.04** 终端执行：

```bash
cd /home/bogger/10weekplan/week3

# 验证 GPU、PyTorch3D 扩展和 SMPL
./docker/pytorch3d/p3d.sh verify

# 运行已有的旋转表示实验
./docker/pytorch3d/p3d.sh python -m experiments.d6_check

# 运行已有的 SMPL 实验
./docker/pytorch3d/p3d.sh python -m experiments.smpl_check

# 运行自己写的脚本：把下面文件名换成实际脚本
./docker/pytorch3d/p3d.sh python your_script.py

# 需要连续交互时进入容器，输入 exit 退出
./docker/pytorch3d/p3d.sh shell
```

自检默认只在终端显示结果，不生成留档目录。如需保存 JSON，可手动指定保存位置，例如：

~~~bash
./docker/pytorch3d/p3d.sh verify --json /workspace/week3/results/verification-cuda.json
~~~

进入容器后，当前目录是 `/workspace/week3`，可以直接运行 `python -m experiments.d6_check`。无需激活原 `.venv`。在普通 Ubuntu 终端直接输入 `python` 或从原 VS Code 解释器运行，仍会使用原来的环境；要使用这套环境，请通过以上入口执行。

代码里的 `device` 决定计算位置。`verify` 显式执行 CUDA 测试；原有 `d6_check`、`smpl_check` 保持各自默认的 CPU 设置。

也可以在 Windows PowerShell 中调用交付目录里的 `run-pytorch3d.ps1 verify`，或 `run-pytorch3d.ps1 python -m experiments.d6_check`。这个入口会转到已安装的 WSL 环境执行。

## 文件怎样保存

| 内容 | 保存方式 |
| --- | --- |
| PyTorch、PyTorch3D、编译工具和其他依赖 | 本地镜像 `bogger-pytorch3d:0.7.9-torch2.7.1-cu128-sm120` |
| week3 的代码和 results | 原 WSL 目录挂载到容器 `/workspace/week3`，编辑与输出持久保存 |
| SMPL 模型及其他 models 内容 | 挂载为只读；没有复制进镜像或上传 |
| 原 `/home/bogger/10weekplan/.venv` | 继续单独保留，不挂载给容器 |
| 容器内临时目录、临时安装的包 | 退出后丢弃；新增依赖应修改构建文件后重建 |

运行时默认关闭容器网络。构建使用本地保存且校验哈希的源码与依赖文件。没有发布镜像，也不需要外传模型。SMPL 的使用仍受你获得的模型授权约束。

## 版本选择和复现

主环境实测为 Python 3.12.3、PyTorch 2.14.0+cu130。容器选择独立的 PyTorch 2.7.1 / CUDA 12.8，针对本机 RTX 5070 Laptop 的 `sm_120` 编译 PyTorch3D 0.7.9。

实测版本为 Python 3.11.13、PyTorch 2.7.1+cu128、PyTorch3D 0.7.9、CUDA 编译器 12.8.61、GCC/G++ 11.4.0。GPU 与 CPU 算子、SMPL 前向、原有实验、Windows 入口及容器间文件持久化均已通过。完整结果见同目录《验证记录.md》。

选择依据是 [PyTorch 2.7 官方说明](https://pytorch.org/blog/pytorch-2-7/) 中的 Blackwell / CUDA 12.8 支持。PyTorch3D 的 [安装文档](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 列出的 PyTorch 范围仍是 2.1–2.4，旧 wheel 示例不能证明支持本机的新组合。因此采用 [0.7.9 官方源码](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.7.9) 编译，并用本机功能检查验收；不把这个组合描述为官方认证组合。

`nvidia-smi` 中的 CUDA 13.2 表示驱动支持能力；主 `.venv` 的 CUDA 13.0 和容器的 CUDA 12.8 是各自 PyTorch 的运行库版本。无需把三个数字改成一样。

复现材料：

- `Dockerfile`：固定官方基础镜像的 SHA256 摘要；编译目标固定为 `12.0`；默认两个编译任务。
- `source.lock.json`、`pytorch3d-source.tar.gz`：源码提交 `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` 与归档校验值。
- `requirements.in`：显式新增依赖；`requirements.lock`：安装版本和 wheel 的 SHA256。
- `wheelhouse/`、`downloads.lock.json`：保存的依赖包及 PyPI 下载来源和哈希。
- `installed.lock.txt`：最终镜像中全部 Python 包的实际版本，构建完成后导出。
- `image.lock.json`：本次本地镜像、基础镜像、源码和已编译扩展的身份与校验值。
- `verify.py`：实际运算检查；请求 CUDA 时失败就报错，不会偷偷改用 CPU。
- `compose.yaml`、`compose.cpu.yaml`、`p3d.sh`：日常运行入口和挂载方式。

在构建目录中执行重建：

```bash
cd /home/bogger/10weekplan/week3/docker/pytorch3d
./p3d.sh build
./p3d.sh verify
```

保留整个构建目录即可重复构建；在一台新机器上，仍需能够取得 Dockerfile 固定的官方基础镜像。不要将 10weekplan 根目录的旧 `requirements.txt` 直接安装进此镜像，那份清单属于原 PyTorch 2.14 环境。依赖变化时应更新 `requirements.in`、依赖包和哈希锁，再重建及重新验证。

## 验证范围与 CPU 备用入口

检查内容包括导入及 C++/CUDA 扩展载入、轴角和 6D 旋转转换、已知 90° 旋转、6D 往返与梯度、KNN 对照数值和梯度、最远点采样、网格面积和采样、三角形光栅化及梯度、SMPL 前向与模型哈希。

```bash
cd /home/bogger/10weekplan/week3
./docker/pytorch3d/p3d.sh verify-cpu
```

CPU 入口不申请 GPU，适合 CUDA 暂不可用时检查有 CPU 实现的功能。它不等于全部 PyTorch3D 功能均可在 CPU 上执行。GPU 镜像针对本机 `sm_120` 编译，移到其他架构显卡前需要重新选择编译目标。

2026-10-20 的环境配置验证说明见同目录《验证记录.md》。2026-10-21 已按要求清理 results 下的两次配置留档目录；日常自检默认仅在终端显示结果。检查不代表训练精度、动作重建质量或所有 PyTorch3D 接口均已覆盖。
