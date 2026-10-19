# Sionna RT + SYS 独立副本

第一阶段接入三维传播信道驱动的多小区下行闭环。入口为 `experiments/run_rt_sys.py`，后端为 `sionna_rt_sys`。原目录没有修改；`source_manifest.json` 保存复制时的源码 SHA256，独立 Git 的首个提交保存原始快照。

## 平台开关：DGX / AGX

两台机器使用同一份代码和场景，通过 `--platform` 选择执行后端：

| 参数 | Sionna RT | NR / SYS | CFR、RZF、LMMSE |
| --- | --- | --- | --- |
| `--platform dgx` | CUDA / OptiX | PyTorch CUDA | PyTorch CUDA（CFR 保留原有 NumPy 重建） |
| `--platform agx` | LLVM CPU | CPU | CuPy CUDA；启动检查失败时改用 NumPy / Torch CPU |

DGX 合作者在已有、支持本机 GPU 的 Sionna 2.2 / PyTorch CUDA 环境内运行：

```bash
python experiments/run_rt_sys.py --platform dgx --scene munich --cells 3 --ues 12 --slots 40
```

DGX 模式启动时加载驱动提供的 `libnvoptix.so.1`，并实际执行 Torch CUDA
solve/SVD 检查；仿真仍必须实际执行 CUDA Sionna RT 调用才能成功。
缺少 OptiX 驱动库、容器未挂载驱动库、GPU wheel 不兼容或后续 CUDA/RT 错误均明确失败，
不会自动转 CPU。此模式不限制为某个 CPU 架构或 GB10 型号；原有 Spark 专项验收仍单独保留。
该参数不会安装或切换 Python 环境。DGX Spark 可沿用 `scripts/spark/activate.sh`；
其他 DGX 机器请使用其已有的 CUDA 环境，不要套用 Orin 的 CPU wheel 或 Spark 专用 ARM64 lock。

本次配置好的 Jetson AGX 上运行：

```bash
source scripts/orin/activate-gpu.sh
python experiments/run_rt_sys.py --platform agx --scene munich --cells 3 --ues 12 --slots 40
# 等价的环境激活与启动封装：
bash scripts/orin/run.sh --scene munich --cells 3 --ues 12 --slots 40
# 明确要求全部计算使用 CPU：
bash scripts/orin/run-cpu.sh --slots 40
```

AGX 的默认 CuPy 检查实际执行 CUDA solve/SVD；检查失败时，仅在仿真开始前选择 CPU
后端并记录原因。运行过程中不隐藏异常、不重启为 CPU 重跑。
`--platform agx --algebra-backend cupy` 是强制 GPU 请求，检查失败会报错；
`--platform agx --algebra-backend torch` 则直接使用 CPU，不要求安装 CuPy。

`--dry-run` 只显示请求的配置，不加载 OptiX 或执行 GPU 检查，不代表硬件验收通过：

```bash
python experiments/run_rt_sys.py --platform dgx --dry-run
python experiments/run_rt_sys.py --platform agx --dry-run
```

`config_snapshot.json`、`run_status.json` 和最终环境记录保存平台选择、实际执行设备和
回退原因。不指定 `--platform` 时保留旧的 `--device`、`--rt-device`、
`--algebra-backend` 行为，默认仍为 CUDA；与所选平台冲突的设备参数会报错。
两个后端保持同一套物理模型，但 CPU/GPU 数值和抽样结果不保证逐位相同。

## 实时网络监测与三维统一工作区

AGX 远程桌面中运行 `bash scripts/orin/live.sh`。窗口包含同一次仿真的 3D 场景、
网络监测页签和 KPI 曲线，并增加一个初始关闭的注册 xApp 测试页。
`bash scripts/orin/live.sh --read-only` 保留纯监测；普通 `--live-state` 导出也仍然只读。

- **Overview**：全网吞吐量、MAC 队列、packet ledger 积压、溢出字节、已完成包平均时延、
  连接 UE 数、真实 NR 解码尝试/NACK 样本，以及实际 CPU/GPU 后端。
- **UEs**：逐 UE 的 serving cell、SINR、MCS、ACK/NACK、PRB/符号、rank、切换状态、
  HOL delay 和 HARQ 积压；选择 UE 后可点击 Focus in 3D 定位。
- **Cells**：逐小区吞吐量、队列、关联/调度 UE 数、实际功率预算/平均发射功率、
  资源格占用率、空间层/MU 组，以及运行态调度参数。
- **Config**：本次运行经平台和模式覆盖后的频率、带宽、SCS、slot 时长、天线、
  ray solver、CSI/HARQ、观测/执行延迟和 rApp/xApp 周期。
- **Control**：来自当前小区状态的实际生效值、待执行流水线及计划生效 slot、
  请求来源/目标/参数、applied/rejected/delayed 回执、约束、字段仲裁和 xApp bindings。

顶部 View 可切换 Integrated / Network only / Scene only；也可通过
`bash scripts/orin/live.sh --view telemetry` 或 `--view scene` 选择启动视图。
曲线包含每个已完成时隙的样本，保留最近 300 个时隙；控制历史保留最近 60 条回执。
UI 轮询不会丢掉轮询间完成的 slot 样本。界面同时显示仿真时间、实际耗时、数据年龄和运行状态。
计算下一 slot 时保留上一个真实状态；不伪造动画、不自动重播结束的运行。

baseline 模式明确显示 rApp/xApp 未启用。baseline 仍可能产生空控制 envelope 的执行回执；
“applied 回执数”不等于“参数修改次数”，当前实际生效值独立显示。
请求值、待执行值和当前生效值分别来自不同状态对象，不能将请求值当作已经执行。
NACK fraction 是当前实际解码样本比率，不代表已收敛 BLER；平均完成时延不包含未完成包。
图中连线表示 serving association，不表示绕过建筑的真实传播射线。

工作区仅依赖 Polyscope/NumPy，直接显示主仿真 Sionna RT 2.2 导出的变换后网格；
不导入旧的官方 GUI Sionna 环境，不另外计算 PHY。官方 GUI 的场景/路径/radio-map 编辑功能见
[Sionna RT GUI 手册](https://nvlabs.github.io/sionna/rt/rt_gui.html)。
项目的 O-RAN 调度、HARQ、队列及控制回执由本项目监测层提供。

DGX 合作者也可在有 OpenGL 图形桌面的环境内安装 `requirements-viewer.txt`，
继续使用 DGX 原有的 Sionna/CUDA 环境。仿真与 UI 可分别运行：

```bash
python experiments/run_rt_sys.py --platform dgx --scene munich --cells 3 --ues 12 --slots 400 \
  --output outputs/dgx_live --live-state outputs/dgx_live/live.json
# 另一个终端，读取相同的 state，不启动第二次仿真：
python scripts/orin/project_live_viewer.py --state outputs/dgx_live/live.json --view integrated
# 如需两个独立窗口，可另外启动 --view telemetry 或 --view scene 读取同一个文件。
```

运行文件位于 `outputs/live_project/<timestamp>/`，旧版不含新增字段的状态仍可查看；
没有数据的字段显示 `--`。监测导出采用字段白名单，不导出 API key 等无关设置。

2026-10-12：Spark 上修复后完整接入矩阵 12/12 通过，覆盖两地图、3 小区、12/24 UE、3 种子、各 1000 时隙；真实 RT/SYS GPU 控制探针、资源/功率记账和固定种子重放通过。全量回归 193 项通过、4 项原有失败。当前调度矩阵没有自然 MU 组，MU 由专项重叠资源及已知信道测试验证。详见 [完整接入报告](docs/rt_sys/FULL_ACCEPTANCE.md)；这些结果不作为算法跨场景泛化结论。

## UI 中的 xApp 测试与在线启停

`live.sh` 默认加入 `--live-xapp-test`，控制模式标为 `xapp_test`，不启用自动编排或学习。
测试页从现有注册表检测 46 个库内实现及下面的测试实例，显示 **Select xApp** 下拉列表。
无命令副作用的库内实例预注册且初始全部关闭；根据真实 KPI/action 契约、SLA 声明、
波束报告和连接事件模型检查可用条件。不满足条件的项显示 unavailable 和原因，不能启用。
切换/连接命令类实现只列出，不在这个可恢复参数的测试入口中注册。
该模式开启扩展测量报告，默认关闭额外连接事件模型和波束扫描报告；显式场景配置优先。
测试实例 `live_scheduler_test:queue_hol` 只消费正常延迟、噪声处理后的队列和 HOL 测量：
某小区队列超过 4000 B 或 HOL 达到时延预算的 50% 时，请求 `latency_aware`；否则请求
`round_robin`。它只写 `scheduler_mode` 和 `scheduler_policy_hint`，不改变信道、业务、功率、CSI 或 HARQ。
这是本地确定性功能测试规则，不是训练完成的优化算法，也不保证启用后 KPI 改善。

在 **xApp test** 页选择一个可用算法，再选择 All cells 或一个小区，点击 **Enable xApp**。
一次只测试一个实例。**Disable & restore** 撤回当前实例，并通过原控制通道恢复它控制的参数；
恢复完成后才能启用另一个实例。`--xapp-test-app IMPLEMENTATION_ID` 也可指定启动算法，
配合 `--xapp-test-start on` 使用；默认仍是队列/HOL 测试实例且关闭。
在线启停复用实例绑定、选择版本、字段仲裁和撤回机制；停用时使旧版本待执行动作失效。
新控制仍等待 20 ms 控制周期和原有执行延迟，不重置队列、UE、信道、HARQ 或仿真时间。
恢复未完成时拒绝重新启用，防止把仍在恢复的状态当成基线。回执中的 accepted 表示绑定准入，
实际生效值以 Effective controls 和 Control 中的执行回执为准。停用恢复配置参数，不撤销已经发生的业务交付、移动或网络历史。

分段 KPI 跳过一个控制周期加执行延迟，统计实际交付、平均队列、HOL 和已完成包时延。
同一次运行的启停前后仍受到移动、业务和队列历史影响，只能作为观察。
**Run matched A/B** 对下拉列表选中的算法启动两个独立的 OFF/ON 运行，固定场景、seed、业务与初始队列，
并检查环境和执行源码指纹相同。默认每组最多 200 个时隙，可通过
`--xapp-test-compare-slots` 调整；进度和结果写到当前运行旁的 `live.comparison.json`，
完整数据在 `comparison_*/off` 与 `comparison_*/on`。它可以在主演示完成后启动，
使用主仿真的 Python 环境，不使用旧 GUI Sionna 环境。多次配对和多种子才适合评价泛化效果。

命令使用当前运行独有的 ID 和本机文件队列，只允许已注册实例的启停；结束的运行不接受启停。
这里的热插拔是预注册 xApp 的逻辑启停，不加载任意 Python 文件，也不是 Kubernetes 部署或真实 E2 互通。
不加 `--live-xapp-test` 的运行保持原行为；该模式与 orchestrated/train/test 明确隔离。
DGX 同样可在普通 RT/SYS 命令中加入 `--live-xapp-test --live-state ...` 后连接现有 viewer。

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
