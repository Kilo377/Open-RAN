# Sionna RT + SYS 独立副本

第一阶段接入三维传播信道驱动的多小区下行闭环。入口为 `experiments/run_rt_sys.py`，后端为 `sionna_rt_sys`。原目录没有修改；`source_manifest.json` 保存复制时的源码 SHA256，独立 Git 的首个提交保存原始快照。

## 已实现

- Sionna 2.2 RT：内置 `simple_street_canyon` / `munich`，固定三维部署、可变 UE 数量、预设轨迹、全 BS→UE 的绝对多天线 CFR。启动时检查基站、UE 位置与轨迹线段是否被建筑遮盖或穿过几何。
- 分块传播缓存：保留复数系数、时延、角度与 Doppler；默认 10 ms 求解几何路径，块内由 RT CFR 接口演化相位。按时隙加载，不缓存控制后的 SINR 或吞吐量。
- 集群 PHY：全部小区 grant 汇总后，用 PyTorch 批量计算 RZF 预编码与 LMMSE SINR，包含同小区其他空间流和邻小区重叠资源干扰。数据及导频共享小区平均总功率预算，空闲资源不发射，深度休眠不发射。
- SU/MU-MIMO：码本与波束 ID 对应归一化 DFT 天线权重；MU 组在各用户波束子空间的并集内预编码。移除新路径中的经验波束/MU 增益与经验传播损耗。
- 真实 Sionna SYS：BLER/TB 容量映射，按 seed/slot/UE 索引抽样 ACK。交付受队列、grant 与 SYS TB 容量限制；逻辑信道/HARQ 登记实际 TB 字节，保持原生命周期，不模拟软合并增益。
- 保留原 rApp/xApp、业务、MAC、切换状态机与延迟观测。候选小区测量来自同一 RT 信道，并经过延迟和噪声；物理计算使用理想 CSI。

## 本地运行

当前副本的 `.venv-local` 已安装 PyTorch/Sionna；本地验证使用 Python 3.12、CPU，不能据此确认 Spark GPU 可用。新后端无需 TensorFlow，旧 TensorFlow 后端应在独立环境运行。

```bash
cd '/Users/mihutao/IBN-ORAN论文/oran_slot_sim_sionna'
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export DRJIT_CACHE_DIR="$PWD/.cache/drjit"
# 当前 Mac 的局部 LLVM；部署到 Spark 时无需传输此目录
export DRJIT_LIBLLVM_PATH="$PWD/.local-llvm/lib/libLLVM.20.1.dylib"
.venv-local/bin/python scripts/spark/preflight.py --cpu-local
.venv-local/bin/python -m pytest -q tests/test_rt_sys.py tests/test_rt_sys_actual.py
.venv-local/bin/python experiments/validate_rt_sys_mu.py --device cpu
.venv-local/bin/python experiments/run_rt_sys.py --device cpu --cells 2 --ues 4 --slots 40
.venv-local/bin/python experiments/run_rt_sys.py --scene munich --device cpu --cells 3 --ues 24 --slots 40
```

配置入口：`configs/rt_sys/default.json`、`configs/rt_sys/scene_catalog.json`。轨迹按固定速度（默认 1.5 m/s）运行，避免改变仿真时长同时改变移动速度。不同 UE 的起点沿合法路段分散；业务模板交错包含 eMBB/URLLC 等已有业务。

`--dry-run` 检查场景清单而不启动 RT；`--generate-only` 只生成传播缓存。默认设备为 CUDA，缺依赖、缺 CUDA、缺信道、越界 grant、未映射码本或跨频/不同 numerology 会明确失败。通用 factory 入口可提供 `rt_sys.manifest_path` 与副本内 `cache_root`；专用入口自动构造 provider。

## Spark 部署与验收

Spark 的当前配置及远端实测见 `docs/rt_sys/SPARK_SETUP.md`。以下为重新部署流程：

```bash
# 在 Mac 上先检查命令，再传输；替换为实际 SSH 目标
.venv-local/bin/python scripts/spark/deploy.py --host USER@SPARK_IP --dry-run
.venv-local/bin/python scripts/spark/deploy.py --host USER@SPARK_IP
# 后续更新同一项目增加 --update；不会覆盖其他项目目录
```

在 Spark 的 `~/oran_slot_sim_sionna/` 内：

```bash
bash scripts/spark/bootstrap.sh
# 若已准备其他官方兼容 Python，显式设置 ORAN_PYTHON；目标为 Python 3.11
# ORAN_PYTHON=/path/to/python3.11 bash scripts/spark/bootstrap.sh
.venv-spark/bin/python experiments/run_rt_sys_matrix.py --dry-run
.venv-spark/bin/python experiments/run_rt_sys_matrix.py --skip-completed
```

bootstrap 创建隔离的 `.venv-spark`，优先 Python 3.11，否则使用系统 Python ≥3.11；安装官方 PyTorch 2.9.1+cu130 ARM64 wheel（SHA256 校验）和 Sionna 2.2.0。安装进程采用 IPv4，保留 TLS 校验，不改系统网络。若官方 ARM64 CUDA wheel 或已有镜像不兼容，预检明确失败，届时根据 Spark 驱动和镜像配置解决。不会把 Mac 的 LLVM、venv、结果或缓存同步到 Spark。

预检要求 ARM64 Linux、GB10 Blackwell、可用 PyTorch CUDA、RT CUDA variant，且实际执行 RT 与 SYS；记录 GPU/驱动/版本。仅 GPU 预检成功才能生成 `requirements-spark.lock.txt`。随后运行 2 小区/4 UE 冒烟测试和两地图的重叠 MU grant 实测，再跑 2 地图 × 3 小区 × 12/24 UE × 3 种子 × 1000 时隙，共 12 次接入验证。矩阵启动也执行 GPU 预检；CPU 结果不能被 CUDA 的 `--skip-completed` 当作已验收。

10 ms 路径更新的敏感性检查：对同一 scene/cells/UE/seed/slots 分别运行 `--path-update-ms 10` 与 `--path-update-ms 1`，使用不同输出目录。首批本地 40 时隙检查已执行，尚不能证明长期误差可忽略。正式泛化实验、训练/测试地图隔离、完整 PHY 校准留到下一阶段。

## 输出与物理约定

输出全部位于副本 `outputs/`，传播缓存位于 `channel_cache/`。每次运行保存配置快照、缓存 ID、后端版本、逐 UE/小区 CSV、逐时隙 JSONL、原有 KPI/编排报告，以及耗时、CPU 峰值 RSS、PyTorch CUDA 峰值分配内存。CUDA 内存数值不包含 Mitsuba/Dr.Jit 的全部分配；Spark 的整体 GPU 内存监测仍需远端补充。

`rt_sys_trace.jsonl` 中的 precoders 保存每个资源活动组的代表子载波复数权重、功率、用户/层与子空间维度；物理计算本身逐资源批量求解。CSV 未调度用户的 SINR 值是占位零，应根据 `scheduled` 字段过滤。缓存 key 包含地图资产/材料、部署、天线、轨迹、射线参数、seed、软件版本和时长；不含算法名或运行时控制功率。不同算法须使用同一场景/业务清单。

功率约定是**每时隙平均总辐射功率**：每个占用 RE/空间流分配 `P_cell × N_symbols / N_RE-streams`，各预编码列单位范数。导频用于资源开销与干扰计算，未模拟导频接收误差。候选小区的 RSRP 来自 RT 参考波束；候选质量评分沿用负载估计，实际传输干扰来自当时全部发射计划。

SYS 默认 EESM beta 只覆盖 NR 表 1/2，因此 RT 专用 scheduler 沿用 MAC 策略，但把表 3 请求转换为表 1；直接请求表 3 会失败。默认 BLER 表 1 从 MCS 3、表 2 从 MCS 2 起覆盖；低于覆盖范围的请求映射到最小已校准项，逐 UE 写入 `mcs_mapping`，若超出 MCS cap 则失败。默认表的码块大小边界及 EESM 标定仍有局限；当前不宣称完整 PHY 校准。

控制能力映射见 `phy/rt_sys/control.py`：功率/波束/休眠影响 PHY，策略/权重影响 MAC，切换参数影响关联。`control_audit` 记录发出的配置字段及路由；它不是“动作已成功生效”的证据，需结合原有 actuation_outcomes 与 PHY trace。无消费者字段明确记录 unsupported 原因。后续新增 xApp 动作应同时补消费者、审计和物理响应测试。

本地证据与未完成项见 `docs/rt_sys/VALIDATION.md`。

官方依据：[安装要求](https://nvlabs.github.io/sionna/v2.2.0/installation.html)、[SYS PHYAbstraction](https://nvlabs.github.io/sionna/sys/api/sionna.sys.PHYAbstraction.html)。
