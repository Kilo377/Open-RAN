# Sionna RT + SYS 独立副本


## Full / fast and automatic offline HTML

Select the PHY feedback model independently of the DGX/AGX hardware profile:

```bash
# New AGX with working CUDA NR kernels
bash scripts/orin/run-six-ue-cuda-nr.sh --model full --duration-s 10
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --duration-s 10
# Historical full-grid fast, for controlled comparisons
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --fast-profile reference --duration-s 10

# Older AGX CPU profile; the same model switch is supported
bash scripts/orin/run-six-ue.sh --model fast --duration-s 1
```

`full` is the default model and retains actual NR transport-block decoding.
`fast` retains MAC-selected MCS, exact NR TBS, queues, HARQ identities, OLLA and
control delays, and uses official Sionna SYS EESM/BLER feedback. Missing table/beta
entries use explicitly counted actual NR fallback. Fast has no waveform soft
combining; select full for Chase.

The fast profiles are:

- `system` (default): four samples/PRB, block-updated RZF spatial fields and
  power-conditioned LMMSE coupling gains, with a 2 ms update bound. Each 0.5 ms
  slot uses its actual scheduled streams, per-PRB powers and receiver noise.
  Reference gains use the same update bound; filtering and E2 observation queues
  still advance every slot. Power, scheduling, pairing, beam/RET and sleep changes
  affect the current slot. New RT geometry invalidates older gain epochs.
- `efficient`: the previous four-sample/PRB, per-slot channel calculation and
  batched CPU SYS feedback. Use this for paired precision comparisons.
- `reference`: the historical per-RE/per-symbol calculation and scalar SYS on the
  selected device.

```bash
# New default fast, with 2 ms gain updates
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --duration-s 10
# Previous per-slot fast
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --fast-profile efficient --duration-s 10
# Optional 5 ms system gain updates
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --fast-gain-update-ms 5 --duration-s 10
```

`--fast-gain-update-ms` overrides both spatial and reference intervals in the
system profile and must align with whole slots. It does not change the MAC slot
or RT path-update interval. JSON uses `rt_sys.fast_profile`,
`system_gain_update_ms` and `reference_gain_update_ms`; CLI takes precedence.
System currently requires ideal CSI; select efficient/full for CSI-error studies.
Its spatial calculations use NumPy CPU, official SYS feedback uses batched CPU,
and actual NR fallback uses the selected PHY device. Hardware profiles still
select OptiX versus LLVM and the channel reconstruction backend independently.

In experiment recording, system also uses detached JSON report copies, vectorized
subband CQI quantization and diagnostic snapshots at window boundaries plus the
final slot. Per-slot packet/HARQ records, observations, controls, RF accounting
and KPI traces remain available. Audit recording retains per-slot state snapshots.

The paired new-AGX 400-slot sweep measured 70.46 s for efficient, 39.84 s for system
at 2 ms and 37.37 s at 5 ms. Main-loop time fell from 59.75 s to 29.00 s at 2 ms.
The 2 ms default was selected for smaller latency-tail deviations. This is a
finite scenario comparison, not packet-level equivalence or guaranteed accuracy
for every xApp/channel. See [system profile validation](docs/rt_sys/FAST_SYSTEM_ACCEPTANCE.md)
and the [earlier efficient-profile study](docs/rt_sys/FAST_EFFICIENT_ACCEPTANCE.md).

The existing `agx_munich_6ue_depth2_fast.json` filename describes the execution
optimizations; it does not select the PHY model. `--model full|fast` overrides
`rt_sys.model` in the JSON configuration. The legacy `--feedback-mode` remains
an advanced alternative and cannot be combined with `--model` on the CLI.
`--duration-s` and `--slots` are alternatives; durations must align to whole slots.

Every successful finite RT run now writes `network-replay.html` in its output
folder and prints its absolute path. Default output folders are unique per run.
The single HTML embeds the real scene mesh, trajectories, Three.js, network
configuration, cell/UE metrics, throughput, delay and RF energy; it needs no
Sionna installation or internet connection on the viewing computer.

The replay adapts its time units, playback speed and KPI windows to the run
length. The default targets at most 500 windows plus the initial frame. The
last partial window uses its actual duration, and all cumulative byte/error/energy
counters remain exact. Raw per-slot records are retained. Override resolution
with `--replay-window-ms 20` or `--replay-max-frames 1000`.

Use `--no-html` for timing-only runs; the mesh and raw data remain available.
An existing completed run can be exported again without rerunning physics:

```bash
python experiments/export_replay.py --run outputs/rt_sys/<run-folder>
# Optional: --output /path/to/report.html --window-ms 20 --max-frames 1000
```

A failed export reports the error and preserves the simulation files. Automatic
exports use a packaged MIT-licensed Three.js r170 bundle, not a CDN. The HTML
shows the selected model, and BLER denominators include both actual and sampled
transmitted-TB outcomes while actual-decode counters remain separate.
Fast exports also show the selected profile and actual PHY sampling. Historical
fast results without profile metadata are labelled reference.


第一阶段接入三维传播信道驱动的多小区下行闭环。入口为 `experiments/run_rt_sys.py`，后端为 `sionna_rt_sys`。原目录没有修改；`source_manifest.json` 保存复制时的源码 SHA256，独立 Git 的首个提交保存原始快照。

原生网络 MU/SU 对照、独立 Sionna 公式复验，以及首传/重传和队列时延的统计口径，见
[Native MU 验证记录](docs/rt_sys/NATIVE_MU_VALIDATION.md)。这些修复保留原有调度、
RZF 和 OLLA 参数，不以提高基线吞吐或让 BLER 看起来达标为验收条件。

## 平台开关：DGX / AGX

JetPack 7.2 / L4T R39 设备的独立环境与可选 CUDA NR 入口见
[JetPack 7 setup](docs/rt_sys/AGX_JETPACK7.md)；不要使用 JetPack 6 的 CUDA 12 依赖锁。

AGX 六 UE 默认预设（2026-10-30）：`configs/rt_sys/agx_munich_6ue_depth2_fast.json`。
保留原 12 UE 部署中的 UE0、UE1、UE2、UE4、UE6、UE8，全部保留原位置、轨迹、业务模板、速率和 1.5 m/s 移动。
按用户选择接受 UE0 的较差表现，撤销 UE0 的替代位置。小区归属仍由物理参考测量决定。
深度 2 下的短测分布为 cell_0: UE0/UE6/UE8、cell_1: UE1、cell_2: UE2/UE4；
保留原 UE 优先于每站恰好两台。此前换位布局的 100-slot 验证不代表本布局的 KPI。
`ray_solver.max_depth` 为 2，保留 LoS、镜面反射及每源 100000 个采样。
几何更新为 5 ms，位移上限为 1 cm，Doppler 仍逐 0.5 ms slot 演化。
NR 使用 6 个独立进程 × 每进程 2 线程，主进程和 RT 各一个线程；
小矩阵代数使用 Torch CPU，并启用参考测量复用、理想 CSI 去重及 RT 路径预取。
原 0.5 ms 几何更新预设仍保存在 `agx_munich_6ue_depth2.json` 中供历史复现。
400-slot 对比结果、正确性检查与 10 秒运行命令见
[AGX execution efficiency](docs/rt_sys/AGX_EFFICIENCY.md)。
预设中的 `rt_sys.rt_cpu_threads: 1` 仅限制本次 Dr.Jit LLVM 线程池，避开现场复现的 nanothread queue 断言；不修改全局环境。
`scripts/orin/live.sh` 默认使用此预设；无 GUI 运行：

```bash
bash scripts/orin/run-six-ue.sh --slots 400
```

`ue_subset.source_num_ues` 和 `ue_subset.ue_ids` 用于从原部署选择 UE，避免直接 `--ues 6` 重排位置和业务。
可选 `ue_route_overrides` 显式记录替代位置的 `route_index`（从 0 开始）与 `start_fraction`。
原 `default.json` 保留 12 UE 历史运行使用的射线配置，便于复现。

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

AGX 的执行效率开关配置在 JSON 的 `rt_sys` 中，六 UE 默认预设采用已验证的组合：

| 配置 | 用途 |
| --- | --- |
| `torch_num_threads`, `torch_num_interop_threads` | 当前进程的 Torch CPU 线程数，与 Dr.Jit 线程池独立 |
| `reuse_reference_within_slot` | 同一时隙复用物理参考测量；RF 参数或报告波束变化时失效 |
| `ideal_csi_fast_path` | 理想 CSI 下省略未使用的重复接收信号计算；非理想 CSI 保留原计算 |
| `packed_phy_diagnostics` | 合并 GPU 诊断传输，仅读取实际导出的预编码器样本 |
| `retain_channel_on_gpu` | 保留 CuPy CFR，供 GPU 代数直接使用；CPU 模式忽略 |
| `nr_execution: "parallel"`, `nr_parallel_workers` | 同时解码同一时隙的独立 UE，限定 CPU `nr_direct`；默认每 worker 一个 Torch 线程 |
| `nr_worker_threads` | 每个解码 worker 的 Torch 线程数，避免与 worker 数相乘后超额分配 |
| `nr_pin_workers` | Linux 上把每个解码 worker 约束在独立 CPU 核组，作用于本轮子进程；要求核数足够 |
| `rt_prefetch` | 独立 CPU 进程提前生成固定轨迹的 compact RT 块，继续使用一个 Dr.Jit 线程；缺块时前台正常计算 |

并行解码沿用 Sionna 的 20 次 BP 迭代，由主进程持有 HARQ 状态，保留按 UE/TB
确定的码字和噪声序列。预取只使用已配置的场景、轨迹、种子和几何更新周期，
不读取未来业务、ACK 或控制决策，也不改变 Doppler 的逐 slot 演化。
上述实现开关不减少射线样本数、不改变 MCS、调度或解码算法。`path_update_ms: 5`
是另外的几何近似选项，不能与这些实现优化混称为完全等价。
`performance.json` 记录线程、参考测量复用、预取统计；并行解码耗时查看
`batch_feedback_inclusive`，其内部的 `actual_tb_feedback` 只计已完成结果的读取。
各阶段为包含子阶段的墙钟时间，不能直接相加；首次启动 worker 和 codec 的成本计入本轮。

`configs/rt_sys/agx_munich_6ue_depth2_fast.json` 是选定的默认文件。
`run-six-ue.sh` 和 `live.sh` 均显式选择该配置与 Torch CPU 代数；原候选文件及入口
`agx_munich_6ue_depth2_fast_candidate.json` / `run-six-ue-fast-candidate.sh` 保留用于实验复现。

```bash
bash scripts/orin/run-six-ue.sh --slots 400
```

单独调用 Python 时需同时指定 `--platform agx --algebra-backend torch` 和该配置文件；
平台选择先于配置加载，`--algebra-backend` 决定实际执行位置。

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

2026-10-22：xApp 按开发者分为两个 pool。RT/SYS 默认配置选择 `chongyu`；CLI
`--xapp-pool chongyu|yisong` 优先于 JSON 顶层的 `xapp_pool`。pool 在初始化时确定，
切换 pool 或新增清单条目后需要重启；同一 pool 内仍可使用现有按钮启停和切换。
现有库和测试实例全部归入 `yisong`，保留原导入路径；`chongyu` 当前为空，空池正常运行原生网络，
不会补载旧库、生成占位控制器或启动 ON。UI、运行状态、配置快照和 A/B 重放参数记录所选 pool。
新开发目录和接口见 [Chongyu pool](ric/near_rt/pools/chongyu/README.md)，
旧实现位置见 [Yisong pool](ric/near_rt/pools/yisong/README.md)。

```bash
bash scripts/orin/live.sh --xapp-pool chongyu
# 显式查看旧池：
bash scripts/orin/live.sh --xapp-pool yisong
# DGX 同样支持：
python experiments/run_rt_sys.py --platform dgx --xapp-pool chongyu --live-xapp-test --slots 400
```

`live.sh` 默认加入 `--live-xapp-test`，控制模式标为 `xapp_test`，不启用自动编排或学习。
选择 `yisong` 时，测试页从现有注册表检测 46 个库内实现及下面的测试实例，为当前可用的每个 xApp 直接显示一个开关按钮。
无命令副作用的库内实例预注册且初始全部关闭；根据真实 KPI/action 契约、SLA 声明、
波束报告和连接事件模型检查可用条件。不满足条件的项显示 unavailable 和原因，不能启用。
切换/连接命令类实现只列出，不在这个可恢复参数的测试入口中注册。
该模式开启扩展测量报告，默认关闭额外连接事件模型和波束扫描报告；显式场景配置优先。
测试实例 `live_scheduler_test:queue_hol` 只消费正常延迟、噪声处理后的队列和 HOL 测量：
某小区队列超过 4000 B 或 HOL 达到时延预算的 50% 时，请求 `latency_aware`；否则请求
`round_robin`。它只写 `scheduler_mode` 和 `scheduler_policy_hint`，不改变信道、业务、功率、CSI 或 HARQ。
这是本地确定性功能测试规则，不是训练完成的优化算法，也不保证启用后 KPI 改善。

在 **xApp test** 页直接点击一个 OFF 按钮启用，点击绿色 ON 按钮停用。默认应用到全部小区，
也可通过上方小区按钮指定下一次启用的范围。点另一个 xApp 会自动撤回旧绑定、等待原参数实际恢复，
然后启用新实例，不再要求手动先停后开。琥珀色 SWITCHING 按钮显示切换进度，点击可取消；
点其他算法会替换尚未完成的切换请求。无效/不可用的目标不会中断当前实例。
不可用项目折叠在 **Unavailable xApps** 中，操作区仅展示可用按钮；悬停可查看算法说明。
顶部显示真正的绑定状态 `xApps ON: ...`、`ALL OFF` 或切换中的目标。
ON 表示绑定已准入；高级执行明细按实例、小区、选择版本匹配
真实执行回执，显示 applied、rejected 或等待/保持原因，旧版本回执不会被当成新启用的执行证明。
结束的运行明确显示 RUN ENDED / HISTORY，不能把结束时的 ON 当成仍在运行。
一次只测试一个实例。停用和切换通过原控制通道恢复它控制的参数，切换超时会停止并显示失败原因，
不会强行在旧参数上启用新实例。`--xapp-test-app IMPLEMENTATION_ID` 也可指定启动算法，
配合 `--xapp-test-start on` 使用；`yisong` 默认选择关闭的队列/HOL 测试实例，`chongyu` 则按自身清单选择实例，空清单保持无 xApp。
在线启停复用实例绑定、选择版本、字段仲裁和撤回机制；停用时使旧版本待执行动作失效。
新控制仍等待 20 ms 控制周期和原有执行延迟，不重置队列、UE、信道、HARQ 或仿真时间。
直接调用旧 enable 接口仍会在恢复未完成时拒绝重新启用；按钮操作使用自动等待的 activate 接口。
回执中的 accepted 表示绑定准入，
实际生效值以 Effective controls 和 Control 中的执行回执为准。停用恢复配置参数，不撤销已经发生的业务交付、移动或网络历史。

分段 KPI 跳过一个控制周期加执行延迟，统计实际交付、平均队列、HOL 和已完成包时延。
同一次运行的启停前后仍受到移动、业务和队列历史影响，只能作为观察。
**Run matched A/B** 对最后点击的算法（首次打开时为当前启用实例）启动两个独立的 OFF/ON 运行，页面显示 Replay target。固定场景、seed、业务与初始队列，
并检查环境和执行源码指纹相同。默认每组最多 200 个时隙，可通过
`--xapp-test-compare-slots` 调整；进度和结果写到当前运行旁的 `live.comparison.json`，
完整数据在 `comparison_*/off` 与 `comparison_*/on`。它可以在主演示完成后启动，
使用主仿真的 Python 环境，不使用旧 GUI Sionna 环境。多次配对和多种子才适合评价泛化效果。

命令使用当前运行独有的 ID 和本机文件队列，只允许已注册实例的启停；结束的运行不接受启停。
这里的热插拔是预注册 xApp 的逻辑启停，不加载任意 Python 文件，也不是 Kubernetes 部署或真实 E2 互通。
不加 `--live-xapp-test` 的运行保持原行为；该模式与 orchestrated/train/test 明确隔离。
DGX 同样可在普通 RT/SYS 命令中加入 `--live-xapp-test --live-state ...` 后连接现有 viewer。

**KPIs** 页汇总当前已导出的实时 KPI：网络 goodput、队列、HOL、包交付/溢出/截止时限计数、
已完成包平均/最大时延和观测 NACK 比例，以及每小区资源与 RF 功率、每 UE 无线/HARQ/包明细。
它不是模型内全部指标。P95/P99 包时延、单独首传 BLER、指定窗口的 UE goodput 公平性
尚未在实时导出中汇总；控制器扩展报告仍在实验文件中。整机能耗也不等同于发射 RF 功率。

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
