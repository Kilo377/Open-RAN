# 六项问题修复与复验

仅修改独立副本 `oran_slot_sim_sionna`。原目录的 1113 项源码校验仍一致。

## 修复内容

1. **关联稳定性**：RT 测量增加 20 ms 指数滤波、50 ms 负载滤波、40 ms 连续触发时间和 100 ms 最小驻留。候选目标改变会清除计时；深度休眠链路允许紧急离开。保留原切换延迟和中断状态机。逐时隙分别记录服务关系变化、完成切换事件和 HARQ 状态，避免将二者混为一谈。
2. **自然 MU 调度**：修复运行时 scheduler 状态覆盖已启用的 MU 配置的问题；单小区和多小区均提供 CSI 快照。配对依据当前物理波束子空间内的信道相关性、空间层数和子空间维数。资源利用率按物理占用 RE 统计，共享 RE 不重复计数。自然 MU 审计要求至少两个不同 UE，单 UE 多层另行计数。
3. **PHY 假设和校准边界**：增加实际 NR CRC/LDPC/QAM/AWGN 校准工具、置信区间与失败报告；支持自定义 SYS BLER/EESM beta 表，缺失文件明确失败。增加按种子、时隙、UE/小区索引的 CSI NMSE 扰动以及可选 Chase 抽象。Chase 强制 RV=0，只有 NDI、MCS、表、编码资源数、载荷和服务小区均相符时合并；其余情况明确重置。默认仍不使用软合并。记录物理 TB 容量和填充字节；已调度的零 SINR RE 按擦除计入编码资源，完整无路径保持零交付。
4. **实验协议**：增加双折留地图协议，每折训练/验证使用同一地图、测试使用另一地图。训练/验证/测试种子隔离，验证部署编号与训练隔离，测试还包含更大 UE 数量和不同业务负载。提供 10 秒轨迹、2 秒预热和两种方法的配对输入；长轨迹到路段端点后沿合法路段折返。反思方法训练产生记忆文件，评估冻结学习并核对文件 SHA256。两个折使用独立模型状态。协议是执行和隔离基础，不代表论文算法泛化结论。
5. **旧测试问题**：补全实验配置辅助模块，清除测试中的 Windows 绝对路径；场景注册表测试覆盖现有九个场景；xApp 测试按依赖投影后的实际组合检查，保留 URLLC 的调度依赖及 steering 的移动性依赖。
6. **缓存和耗时**：生产默认 `paths_v2`，只保存 CIR、绝对时延、角度、Doppler 与路径增益，逐时隙重建 CFR。旧 dense 缓存保留。新路径缓存有跨进程锁和 5 GiB 全局限额，超过限额明确失败；清理工具拒绝删除任何被输出引用或标记 PINNED 的缓存。RZF/LMMSE 按相同发射集合跨 OFDM 符号批处理，保留每个 RE 信道和每符号正则化。新入口默认 BLAS/OMP/MKL 线程数为 1，保留显式用户设置，并记录实际预算，避免 Spark 多案例线程争用。

## PHY 校准尚未通过的部分

不能把第 3 项的工具接入等同于默认物理抽象已校准。代表性的 24 个粗 AWGN 点通过，但扩大至 312 个默认 MCS 过渡区点后有 114 个未满足验收准则；两级频率选择性信道的 24 个点也有 2 个未通过。例如 qam64/MCS10、324 个数据 RE、5 dB、0.1/1.9 两级功率：实测 BLER 为 0.9648，而默认 SYS 为 0.0809。qam64/MCS28 的平坦 AWGN 19 dB 点实测 0.9277，SYS 为 0.1020。

因此正式协议执行默认被拦截。`--allow-uncalibrated` 只能用于明确标记的接入/隔离探针；`full_phy_calibrated` 始终为 false。自定义表接入和生成接口已经准备好，但重新拟合 BLER、频率选择性 EESM beta 并独立验证整个论文使用域仍待完成。当前不提供 URLLC 尾概率保证，不实现 IR，也不将 Chase 抽象探针称作波形级 HARQ 校准。

## 复现命令

```bash
source scripts/spark/activate.sh
python -m pytest -q
python experiments/run_rt_sys_matrix.py --workers 3 --output-root outputs/rt_sys_matrix_repairs
python experiments/audit_rt_sys_matrix.py --root outputs/rt_sys_matrix_repairs --require-resource-metrics --output outputs/rt_sys_repairs/matrix_audit.json
python experiments/validate_rt_sys_cache.py
python experiments/validate_rt_sys_fidelity.py
python experiments/validate_rt_sys_controls.py --device cuda
python experiments/validate_rt_sys_mu.py --device cuda
python experiments/rt_sys_protocol.py --output outputs/rt_sys_protocol_plan
python experiments/rt_sys_protocol.py --proof --execute --allow-uncalibrated --output outputs/rt_sys_protocol_proof
python experiments/rt_sys_cache_inventory.py --output outputs/rt_sys_repairs/cache_inventory.json
```

校准工具默认使用真实 NR PHY，不使用经验 BLER 后端：

```bash
python experiments/calibrate_rt_sys_phy.py --full-mcs --output outputs/rt_sys_repairs/phy_all_mcs.json
python experiments/calibrate_rt_sys_phy.py --frequency-selective --output outputs/rt_sys_repairs/phy_selective.json
python experiments/generate_rt_sys_bler_tables.py --output outputs/rt_sys_calibration/custom_table1.json
python experiments/calibrate_rt_sys_phy.py --full-mcs --bler-table outputs/rt_sys_calibration/custom_table1.json --output outputs/rt_sys_calibration/independent_validation.json
```

表生成的数据属于拟合数据，须以独立 NR TB 仿真种子和保留信道剖面验证。每项证据的时间、设备、执行源码 SHA256 和校准边界以 `outputs/rt_sys_repairs/` 的报告为准。

## 已通过的专项验证

本地最终完整套件：210 passed、1 个 CUDA skip、2 subtests passed；Spark 最终完整套件：211 passed、1 warning、2 subtests passed。Spark 的 PyTorch 仍提示编译支持列表最高为 sm_120、设备为 sm_121；实际 RT CUDA 和 Torch/SYS CUDA 运算均通过，保留该提示。

两个地图的路径缓存与 Sionna 原生 `paths.cfr()` 比较，7 个时隙跨越路径更新边界：最大相对 L2 误差分别为 6.9242e-7 和 6.8241e-7，缓存重放完全一致。40 时隙缓存分别为 20713 和 16124 字节；对照的 7 张 dense CFR 共 25690112 字节。这个比值是该探针的存储对照，不是整个生产矩阵的压缩比。

两个地图的实际 RT/SYS GPU CSI/Chase 探针通过。Chase 探针为了保留待重传 TB，只在第一轮人为覆盖 ACK 为 NACK；第二轮仍执行真实 SYS，输入 SINR 恰好相加为两倍（3.0103 dB）。它验证接口和合并记账，不把强制 NACK 称为真实 NR 解码结果。

跨地图隔离协议的 12 个 GPU 探针全部完成，训练记忆非空，验证/测试模型 SHA256 保持不变。完整协议计划包含 516 个运行任务，但尚未作为正式实验执行；该探针显式标为使用未校准 PHY。

## 导出修正的版本范围

完整修复矩阵的物理源码 SHA256 为 `1cb10ce2bc13de48c6dae568197e121daea7cb981d1056d00c15b7b0646a7f32`。该版在运行中正确计算完成切换总量并保存 `evaluation_summary.json`，但 `rt_sys_trace.jsonl` 来自添加移动性字段之前的浅复制，缺少逐 UE 事件和累积计数。保留这些原始记录，不补造事件。

最终导出修正在 Simulator 复制 trace 之前加入事件、计数与连接状态，并用闭环单元测试和独立 GPU 探针验证。完整矩阵仍可按 PHY 服务关系变化检查最小驻留，并与实时统计的完成切换总量核对。审计报告显式标明计数来源及事件字段是否存在；该导出修正不改变信道、调度、SINR、BLER 或交付。

## 最终矩阵和资源开销

2026-10-12 完成并同步的复验：两个地图 × 3 小区 × 12/24 UE × 3 种子 × 1000 时隙，12/12 场景通过，合计 12000 时隙、216000 条 UE 记录、14243452 字节交付。三个进程并行总耗时 2510.36 秒（41.84 分钟）；并行负载及缓存状态不同，不能把与前轮的差异归因为单个优化。

物理服务关系变化和运行中完成切换计数均为 247，逐 UE 驻留检查通过。前轮为 8351 次物理服务关系变化；这是两轮关联变化的比较，不把前轮数值称为完成切换。自然多用户资源组共 74754，全部来自 street 场景；本部署的 Munich 场景没有自然 MU 配对，不能据此宣称该地图上的 MU 收益。两地图的专项已知信道/重叠资源探针另行验证 MU 物理计算。

最大功率记账误差 9.99e-16 W，预编码列范数误差 3.77e-7。详细逐 UE/小区审计位于 `outputs/rt_sys_repairs/matrix_audit_release.json`，汇总为 `matrix_summary_release.json`。完整矩阵的事件导出缺口如上一节所述；修正后独立 street/Munich 各 400 时隙 GPU 探针通过，事件、逐 UE 累积计数和运行中最终计数一致，各完成 2 次切换，证据为 `export_audit.json`。

矩阵的 12 个唯一路径缓存合计 24739713 字节（约 23.6 MiB）。Spark 上全部新 `paths_v2` 缓存合计 35965636 字节；历史 dense 缓存 279703345720 字节仍保留，没有删除旧证据。5 GiB 限额针对新路径缓存，不表示历史占用已清除。盘点证据为 `cache_inventory.json` 与 `cache_summary.json`。

## 1 秒路径更新对照

street、2 小区、4 UE 的 2000 时隙（1 秒）对照已完成。10 ms 固定种子重放 trace 完全一致；1 ms、10 ms 和重放均交付 322132 字节。原始 CFR 相对 L2 差异为 0.00266356，最不利链路平均增益差异为 7.71582 dB。因此不能以交付相同或整体误差较小推断每条链路均准确，10 ms 更新仍需按链路、场景和轨迹评估近似误差。

1 ms 运行使用完整矩阵源码版本，10 ms 与重放使用事件导出修正版；两版的信道、调度和 SINR 算法一致，导出字段不同。对照只在同版的 10 ms 与重放之间要求 trace 完全一致。原始证据位于 `outputs/rt_sys_path_repairs/`；它不等于长时泛化实验或 10 ms 普适精度验收。

## 当前交付边界与连接

代码、专项验证、完整接入矩阵和跨地图协议隔离探针已交付。默认 SYS 的 BLER/EESM 校准仍未通过，自定义表生成未执行验收；当前不能标记“六项物理问题全部解决”，也不能进入论文正式性能实验。下一阶段先拟合并独立验证论文所用 MCS、TB/资源大小及频率选择性信道域，再执行 516 项正式协议。

Spark 地址已变为 `njupt@192.168.1.202`，通过保存的原 Spark 主机公钥验证同一设备。当前 SSH 可用 `-o HostKeyAlias=192.168.1.201` 指定已有可信密钥；没有关闭身份校验，也没有替换旧 IP 的记录。机器可读交付汇总见 [six_issue_repairs.json](six_issue_repairs.json)。
