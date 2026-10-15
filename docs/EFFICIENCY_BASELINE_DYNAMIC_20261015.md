# 真实 NR 的效率、无故障基线与动态 OLLA

本轮保留原地图、基站、UE、覆盖门槛、0.5 ms RT 更新、控制周期和真实 NR 解码。新模式不执行在线 Monte Carlo 试验，也不生成 BLER 查表。弱覆盖、合法等待、过载和样本不足分别报告，不能以所有 UE 接入、零积压或所有动态链路 BLER=0.1 作为结构验收条件。

## 实现与配置

- `configs/rt_sys/efficiency_audit.json`：完整 RT/预编码 trace、每时隙 UE/小区 CSV、流式状态快照，以及分组/HARQ/控制/OLLA 事件。
- `configs/rt_sys/efficiency_experiment.json`：精简 RT trace 保留每时隙真实 TB、原码字资源契约、分组账本和功率信息；UE/小区 CSV 为窗口边界的瞬时采样，不冒充窗口平均值。窗口诊断另存在 `runtime_diagnostics.json`。
- `--recording-mode audit|experiment` 和 `--nr-execution scalar|batched` 可显式覆盖配置。默认仍为 audit/scalar。缓存的是编码器结构、精确 SYS 预算和同一时隙/同一 RF 配置的参考测量，不是 BLER、SINR 结果表或控制结果。
- 向量化资源分组保留原行优先坐标顺序和首遇模式顺序。批处理只合并同一时隙、相同 table/MCS/编码 RE 的独立 UE；逐 TB 独立生成原业务位和噪声，保留原码字、Chase LLR、反馈时序与 20 次译码迭代。
- experiment 省略完整预编码权重和重复 paper reports，保留独立控制回执与 OLLA 决策/消费事件。非法发送、拒绝动作和无线掉线/切换失败触发后续完整诊断窗口；普通 NACK 不触发持续全量记录。不能恢复此前未保存的完整权重。
- 完整 trace 和状态快照流式写出，历史标量统计保留用于旧汇总；内存不再保留所有历史 CFR/预编码及状态快照。TB 去重集合和标量历史仍随运行增长，不宣称整个进程恒定内存。
- `slot_metrics.csv` 不再重复嵌入全部 RT 复数权重；完整信息在 audit 的 `rt_sys_trace.jsonl`。原验证工具读取逐 UE CSV 的接口只适用于 audit，新的权威审计直接读取逐时隙 trace。

## 诊断与验收

`performance.json` 是无额外 CUDA fence 的嵌套主机墙钟计时，不能把父子阶段相加或解释为独占 GPU 内核耗时。独立 `profile_exact_nr.py` 使用编码/映射/解调/译码 CUDA Event 和一次末尾同步，结果仅用于诊断，不进入速度比较。

`validate_efficiency.py` 比较两地图、4/24 UE 的 8/128-slot 门禁，以及 512-slot、64 ms 预热、每组合串行重复三次的性能。冻结参考源码独立执行；两种记录与两种解码路径均比较精确离散结果，SINR 容差为绝对 1e-4 dB、相对 1e-6，功率绝对容差 1e-10 W。分组和 HARQ 事件流也逐项比较。数据解码、无探针、资源、功率和 ACK/字节守恒单独审计。

无故障矩阵：两地图 × 3 小区 × (4 UE 正常负载、24 UE 0.1/1 倍负载) × 种子 512/513/514；每例 4000 slots，空队列、400 ms 预热，不注册 xApp/rApp、不注入动作。参考链路状态机和正常关联/恢复继续工作。全体及曾连接 UE 分别报告，不能剔除 Munich ue_18 的零交付。

动态 OLLA：平坦/选择性真实业务 TB 的 0→8→0 dB 阶跃、短时恶化与恢复，测量延迟 2/8 slots，种子 512/513。每稳态段 1500 首传预热、3000 首传评估；只有不少于 2000 首传、饱和率低于 5% 才评估 0.07–0.13 目标。短暂阶段报告过渡错误和恢复时间。Wilson 区间为描述性区间，不消除自适应样本的时间相关性。

真实移动 RT：两地图 × 1.5/10 m/s × 种子 512/513，3 小区/4 UE，先 128-slot 门禁，再按耗时和缓存估算扩大到 4000 slots。动态结果按 UE/服务小区/table/切换分段，不合并后宣称每条链路达标；不足 2000 首传明确标记。独立切换夹具使用受控重叠覆盖，只验证协议消费者和干扰事件，不作为原部署容量。

## Spark 运行与数据保存

原始数据仅保存在 `~/oran_slot_sim_sionna/outputs/efficiency_baseline_20261015/`。参考源码与原始 profile 已冻结；完整长验收由 `scripts/validation/run_efficiency_acceptance.py` 顺序执行，阶段之间检查源码指纹，失败保留现场并停止扩展。历史结果不覆盖。

当前状态：实现和本地回归已完成，Spark 分级验收进行中。未完成的性能复测、无故障矩阵和动态专项不能标为通过。最终状态以 Spark 的 `pipeline_status.json`、各阶段 `validation.json`、源文件/验证脚本归档与 SHA256 清单为准。Mac 不接收完整日志、信道缓存和实验数据。
