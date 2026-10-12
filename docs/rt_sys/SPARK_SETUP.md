# DGX Spark 配置与小型 GPU 验证

验收日期：2026-10-10。通过 SSH `njupt@192.168.1.201` 部署到 `/home/njupt/oran_slot_sim_sionna`，使用项目独立 `.venv-spark`。详细摘要见 [spark_validation.json](spark_validation.json)，准确依赖见 [requirements-spark.lock.txt](../../requirements-spark.lock.txt)。

## 实际环境

| 项目 | 实际值 |
|---|---|
| 系统 | Ubuntu 24.04.4 LTS，Linux aarch64 |
| GPU | NVIDIA GB10，Blackwell，compute capability 12.1 |
| 驱动 | 580.173.02 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu130，CUDA 13.0 |
| Sionna / Sionna RT | 2.2.0 / 2.2.0 |
| Mitsuba / Dr.Jit | 3.9.1 / 1.5.0 |
| RT 后端 | cuda_ad_mono_polarized |
| SYS | CUDA 上实际执行 PHYAbstraction / EESM / TB |

原计划优先 Python 3.11；设备现有 Python 3.12.3 满足 Sionna 要求，实际采用该版本并验证了对应 ARM64 CUDA wheel。没有更改系统 Python、网络设置或其他项目。项目虚拟环境未安装 TensorFlow。

官方 CUDA wheel 下载经过 SHA256 校验。此局域网中 IPv6 连接和 PyTorch 的 download-r2 CDN 有问题，安装脚本仅在自身进程优先 IPv4，并使用同一官方 wheel 的 download.pytorch.org 地址，全程保持 TLS 校验。依赖锁中的 Torch 使用官方 URL 和 SHA256，不包含机器上的 wheel 缓存绝对路径。

PyTorch 会提示其列出的最大架构为 12.0，而 GB10 为 12.1；保留了该警告。下表中的实际 CUDA 运算、SYS 与 RT 验证均成功。本记录只确认这些已执行的工作负载。

## 已完成验证

| 检查 | 结果 |
|---|---|
| ARM64 / GB10 / CUDA 预检 | 真实 RT 信道与 SYS 调用成功 |
| 数值、缓存、真实 SYS、HARQ 专项 | 20 项通过，1 项上述架构警告；专项中 SYS 单元测试使用 CPU |
| street 闭环 | 2 小区、4 UE、40 时隙；160 行 UE 记录；交付 42885 字节 |
| munich 闭环 | 2 小区、4 UE、40 时隙；160 行 UE 记录；交付 17221 字节 |
| street 固定种子重放 | 160 行逐 UE CSV 完全一致；使用同一信道缓存 |
| 两地图真实 MU 接口 | 各 2 小区、4 UE；每小区 2 流；各 28 个 MU 资源组 |
| RT 缓存与重新生成 | 两地图 CFR 最大绝对差均为 0；容差 rtol=1e-4、atol=1e-10 |

两个闭环以及重放均 `completed: true, spark_verified: true`。MU 接口通过强制重叠 grant 验证，包含功率预算和 TB 交付约束；这些短闭环没有证明 MAC 会自主选择 MU，也不构成地图上的 MU 复用收益结论。

street / munich 的本次进程耗时约 8.20 / 8.73 秒，重放约 4.89 秒。这些是短样例的进程耗时，包含初始化和缓存差异，不能外推为稳定时隙速度。进程峰值 RSS 约 2.27 / 2.32 GB；Torch CUDA 分配峰值约 48.9 / 48.8 MB。该 CUDA 计数不包含全部 Dr.Jit/Mitsuba 分配，不能当作整体 GPU 内存峰值。

原始结果已同步到 Mac 副本的 `outputs/rt_sys/spark_street_smoke/`、`spark_munich_smoke/`、`spark_street_replay/`、`preflight/`，以及两个 `mu_validation_*_cuda.json`。远端保留安装、预检、专项测试和各次运行日志。

## 继续运行

```bash
ssh njupt@192.168.1.201
cd ~/oran_slot_sim_sionna
source scripts/spark/activate.sh
python experiments/run_rt_sys.py --cells 2 --ues 4 --slots 40 --device cuda
python experiments/validate_rt_sys_mu.py --device cuda
```

新设备安装入口为 `bash scripts/spark/bootstrap.sh`；可设置 `ORAN_PYTHON` 指定解释器。准确依赖锁针对本次 Python 3.12 / ARM64 / Linux 环境，重建时使用相同解释器并在独立虚拟环境执行 `python -m pip install -r requirements-spark.lock.txt`，随后重跑 GPU 预检。

## 尚未完成

完整 12 场景矩阵已完成并通过物理记账审计。汇总发现资源网格同步问题，修复后的控制 GPU 探针已通过，完整矩阵正在独立目录重跑；当前状态和内存记录见 [FULL_ACCEPTANCE.md](FULL_ACCEPTANCE.md)。完整 PHY 校准、正式训练/测试地图隔离及算法泛化结论仍属后续阶段。
