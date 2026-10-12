# 完整接入测试

本记录对应两个地图、3 小区、12/24 UE、3 个种子、每次 1000 时隙。结果以实际矩阵审计文件为准，不作为算法泛化结论。

## 2026-10-12 状态

修复后矩阵已完成，12/12 场景通过严格审计：共 12000 时隙、216000 条 UE 记录、17667224 字节交付，三个并行进程总耗时 3055.43 秒（50.92 分钟）。实际 Sionna RT CUDA 与 SYS/PyTorch CUDA 执行、控制物理响应、多小区物理记账及资源统计接入验证通过。摘要见 [full_acceptance.json](full_acceptance.json)，最终详细输出在 `outputs/rt_sys_matrix_grid/` 与 `outputs/rt_sys_acceptance_grid/`，已从 Spark 同步回本地副本。

修复前首轮输出保留在 `outputs/rt_sys_matrix/` 与 `outputs/rt_sys_acceptance/`，该轮物理审计通过，但旧资源汇总及其控制观测存在下述缺口。它共交付 17829847 字节、耗时 76.58 分钟，属于排查证据；当前结论以修复后复验为准。

| 地图 | UE | 三种子平均交付 MB | 平均单场景耗时 秒 |
| --- | ---: | ---: | ---: |
| simple_street_canyon | 12 | 1.754 | 1119.1 |
| simple_street_canyon | 24 | 2.476 | 898.3 |
| munich | 12 | 0.794 | 481.6 |
| munich | 24 | 0.865 | 489.2 |

矩阵的原调度策略没有产生自然 MU 组，因此不能据此证明 MU 调度收益。两地图的强制重叠资源 GPU 探针各通过 28 个双用户组；正交与相关已知信道测试另检查复用与干扰响应。

最终汇总时发现，首轮旧 `slot_metrics.csv` 的资源利用率及总层数为零：StateUpdater 替换每小区局部状态的资源网格后，集群视图仍引用初始化对象。物理独立 trace 与交付记账仍通过，但该轮资源汇总及相关控制观测不应使用。

现已在新 RT/SYS 路径同步更新 managed cell 和主小区资源网格。修复后的本地闭环测试、Spark GPU 控制探针均通过；探针明确检查非零利用率进入原有延迟观测链路。Spark 全量回归为 193 passed、4 个原有失败、2 subtests passed，没有新增失败。由于控制输入发生变化，独立目录重跑了完整 12 场景，并通过逐时隙层数与资源利用率精确记账审计，缺口已解决。最大功率记账误差 6.66e-16 W，最大预编码列归一化误差 3.68e-7。

复验记录 8351 次逐 UE PHY 服务小区变化，这是关联轨迹计数，不能直接解释成完成切换次数。当前矩阵用于接入验证；正式实验前仍需校准移动性参数、完整 PHY 抽象和训练/测试地图隔离协议。

## 执行与审计

```bash
source scripts/spark/activate.sh
python experiments/run_rt_sys_matrix.py --workers 3 --output-root outputs/rt_sys_matrix_grid
python experiments/audit_rt_sys_matrix.py --root outputs/rt_sys_matrix_grid --require-resource-metrics --output outputs/rt_sys_acceptance_grid/matrix_audit.json
python experiments/validate_rt_sys_controls.py --device cuda
python -m pytest -q tests --ignore=tests/test_xapp_experiment_utils.py --disable-warnings
python experiments/audit_rt_sys_path_update.py
```

矩阵支持每个场景单独日志、失败汇总和续跑。默认 workers=1；本次采用三个进程并行。进程耗时受到并行竞争和缓存状态影响，不能当成单独占用设备时的性能基准。

逐时隙审计检查完整 GPU 执行、12/24 个 UE 记录、实际预编码列归一化、发射能量与资源掩码一致、功率预算、TB 与队列预算、ACK/NACK 交付记账、控制观测延迟、未支持动作的明确理由。控制探针另外在真实 RT/SYS GPU 上验证邻区功率、资源分离、静默、波束和休眠，并通过原控制延迟链路核对功率、波束、休眠和调度策略。闭环探针直接检查每个 UE 的队列扣除及 HARQ 进程字节与汇总一致性。

控制探针使用脚本签发确定动作，以隔离因果；不同变体共享传播信道缓存、业务到达和随机种子。它不替代真实 xApp 的算法性能比较。已有专项的已知信道张量继续验证正交/相关用户及切换服务链路。

## 本轮发现与修复

首轮 street / 12 UE 在第 143 时隙拒绝了 target MCS 大于 cap 的重传 grant。旧 MAC 的目标与上限会在重传以及 slice-aware 提升后冲突；新 RT/SYS 适配要求显式一致。已在专用 RtSysScheduler 中将目标限制到原策略 cap，并将最低目标限定到 SYS 表 1 的校准入口 MCS 3，然后重新计算 grant/TB。旧调度器保持原状。新增 3 个回归用例覆盖重传、低时延目标及最低校准入口。

控制对照还发现：同一进程反复加载地图时，Mitsuba 为匿名对象分配的 `no-name-N` 名称递增，对象字典顺序也会变化，导致相同场景的缓存键不同。现按实际变换后的 mesh 顶点/面哈希绑定材料，再排序，并继续哈希完整 XML/mesh 资产和材料属性。新增用例确认运行时名称及对象顺序变化不影响缓存键，材料变化仍使缓存失效。早期矩阵进程使用旧键完成的有效物理结果保留其原键；后续输出使用稳定键。该修复不改变 RT 信道计算、轨迹或物理参数。

无服务传播路径的已调度流原本会产生零 RZF 列，但仍记为分配了发射功率。现为这种零列使用确定的归一化波束：服务链路仍零交付，但发射能量和对其他接收机的干扰得到一致计算。补充已知信道的干扰因果检查，并在 CPU/CUDA 上检查零列处理。矩阵审计逐列检查权重范数；对于早期版本生成且通过该检查的场景，这个零列分支未被触发。

每小区局部资源网格替换后现同步回运行状态与主小区视图；旧近似后端的流程保持原状。矩阵新增 `--output-root`，审计新增 `--require-resource-metrics` 及 `--output`，以保留修复前证据并独立验证新版本。

## 更新间隔与重放

Spark 上额外运行 street / 2 小区 / 4 UE / 40 时隙的 1 ms 和 10 ms 更新，以及独立固定种子 10 ms 重放。重放的全部 trace 完全一致；三次均交付 42885 字节。两更新间隔原始 CFR 相对 L2 差异为 0.00088336，非零链路平均增益最大差异为 0.006933 dB。报告 SINR 处于 SYS 标定范围限制，不能用相同 SINR 证明物理信道一致；原始张量比较见 [path_update_validation.json](path_update_validation.json)。这段 20 ms 短对照不构成长时轨迹误差保证。

## 回归范围

原有缺失模块 tests/test_xapp_experiment_utils.py 显式排除。此前复现的四项原有失败单独记录：场景注册表数量断言、Windows 配置路径、两个 xApp 选择断言。不得把它们记为本轮新增失败，也不得把完整测试套件记为全部通过。

## 内存解释

矩阵每秒采样 cudaMemGetInfo 与 Linux MemAvailable。CUDA 计数覆盖设备总体可用内存，能够观察 Torch 和 RT/Dr.Jit 之外的设备内存变化，但也包含并行任务、其他进程及 GB10 共享内存效应，不能用于精确归因单个项目。另保留每个进程峰值 RSS 和 Torch 分配峰值；Torch 计数不包含全部 RT 分配。

首轮最大进程 RSS 4.99 GB，Torch CUDA 分配峰值 77.85 MB；4593 次设备整体采样中，最小系统 MemAvailable 84.88 GB、最小 CUDA 全局 free 0.849 GB，未发生 OOM。该共享内存设备的全局 free 包含其他使用者且不等于可回收文件缓存后的容量，不能把它解释为本项目独占峰值。

修复后复验最大进程 RSS 3.51 GB，Torch CUDA 分配峰值 77.85 MB；3054 次采样中最小系统 MemAvailable 101.65 GB，最小 CUDA 全局 free 0.855 GB，未发生 OOM。两轮缓存状态与并行负载不同，耗时及 RSS 差异不代表同步修复带来了性能优化。
