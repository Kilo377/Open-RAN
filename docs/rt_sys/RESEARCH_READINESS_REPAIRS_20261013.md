# 2026-10-13：两层算法之前的环境修复

本轮保持 DU、xApp、rApp 的既有执行周期：slot = 0.5 ms、xApp = 5 slots、rApp = 10 slots。用户明确要求暂不调整两层时间尺度。本报告不宣称这两个控制周期已符合真实 near-RT/non-RT 时间范围。A1/E2 仍为语义生命周期仿真；PHY 仍为理想 CSI 下的联合多小区后均衡等效高斯资源，不是完整 OFDM 波形。

## 初始条件与业务

`baseline` 和 `orchestrated` 统一使用实际 RT 参考接收功率最强的初始关联，考虑各小区发射功率并排除休眠小区。默认初始队列为空；可用 `--initial-queue-bytes` 显式配置积压。部署、轨迹、业务、种子及初始关联/波束/队列写入 `initial_conditions.json`，附环境指纹。控制模式和算法名称不进入物理信道缓存标识。

新码字由原始 TB 的 UE、进程、NDI、slot、种子确定；实际噪声和统计 ACK 抽样由种子、slot、UE 标识索引，不依赖遍历顺序。不同控制方法可以改变资源和码字，但不能通过改变调用顺序改变外生业务或同一 UE/slot 的噪声流。

`--warmup-ms` 分离预热和统计区间。`steady_state.json` 保存预热后的到达、溢出、交付、积压变化、100 ms 窗口、调度/无路径/容量不足次数。完成时延仅统计预热之后到达且完成的分组；未完成和超期分组另行记账，不能以只统计已完成分组的平均时延替代可靠性指标。字节守恒在逐 UE 账本与稳态区间分别检查。有限运行的队列斜率只能用于诊断，不能证明渐近稳定。

## 三种明确的链路反馈模式

| 配置 | BLER/AMC | 传输 ACK/NACK | HARQ |
|---|---|---|---|
| `nr_tb_reference`（默认） | 独立实际 NR 探针 | 实际 CRC、比特检查 | 同码字尝试，可选 Chase |
| `nr_cached_reference` | 经独立校验的局部 BLER 缓存；失败则实际探针 | 每次仍实际 NR 解码 | 同码字尝试，可选 Chase |
| `nr_bler_bank` | 同一经校验的局部缓存 | 显式 Bernoulli TB 抽样；缓存拒绝则明确实际 NR 反馈 | 保留 MAC 生命周期，拒绝宣称波形 Chase |

失败的 EESM 表没有启用。局部缓存保存完整、有序的逐 RE log-SINR 分箱向量，以及 NR 表、MCS、资源和编码后端；分箱宽度最多 0.01 dB。不能把另一个频率选择性形状映射成相同均值后复用。

每个条目先用中心及上下分箱边界各 64 个独立 NR 样本试探。只有三组共同全成功或全失败才提前结束；内部概率点仍使用至少 256 个拟合及两组至少 256 个独立边界样本。边界与拟合概率的绝对差必须不超过 0.08。保存实际样本量和 Wilson 95% 区间；AMC 使用拟合与验证上界中的最大值。这个准入规则和单独的 GPU 抽查属于有限样本验收，**不保证 URLLC 极低错误尾概率**。未知组合需重新准入，失败组合明确使用实际 NR 参考，不外推。

缓存命中时统计模式可以只恢复已验证的波形元数据，无须重新构造 LDPC 解码器。实际参考模式仍执行每个 TB 的解码。冷校准成本单独计入实际解码次数和耗时，不能把热缓存速度宣称成所有移动场景的速度。

在当前等效独立高斯 RE 假设下，AMC 可先用信息论必要条件排除不可能达到目标的候选：总信息上界为 `sum(min(Qm, log2(1 + SINR)))`，由 Fano 不等式得到 TB 错误概率下界 `max(0, 1 - (信息上界 + 1) / TB_bits)`。只有下界大于目标才拒绝；通过筛选仍须 NR 校验。固定 MCS 重传不经过该筛选。全部候选不可能时明确记录 `capacity_outage_no_tb`，不能产生有效载荷。

## HARQ 与控制闭环

实际 NR 模式保存每个 UE/进程的原始 TB，比特不因 NACK 改变。`--harq-combining chase` 使用同码字、RV0，逐轮 demap 后累加 LLR 并实际解码。MCS、资源数、NDI 或服务小区不兼容时明确无 TB，保留 MAC 载荷等待后续处理；不会替换成另一随机码字。ACK 后清理物理上下文。**IR 增量冗余尚未实现**，不要把 Chase 或统计反馈称作 IR。

RT 控制写入遵守既有执行延迟与 `effective_from_slot` 的较晚者。未知小区、无法映射的字段、过期动作和概率拒绝分别记录原因。结果保存实际生效 slot、小区范围、字段路由以及运行时消费者的前后状态。有效期是持久目标状态的准入窗口，不是到期自动回滚。

字段仲裁同时记录被更高优先级覆盖的低优先级写入；获胜字段携带自身有效窗口，不再因合并而丢失。不同窗口拆分成不同配置动作。验收覆盖 rApp 策略 → xApp 消费 → 多 xApp 冲突 → 指定小区延迟生效 → 调度及关联参数消费者；不以控制器发出动作替代实际生效证据。

## 复现入口

```bash
.venv-spark/bin/python -m pytest -q tests
.venv-spark/bin/python scripts/validation/validate_research_links.py
.venv-spark/bin/python scripts/validation/validate_control_lifecycle.py
.venv-spark/bin/python scripts/validation/validate_initial_fairness.py
.venv-spark/bin/python scripts/validation/run_long_baselines.py
```

输出目录都在独立副本 `outputs/rt_research_readiness_20261013/`。矩阵使用新目录并拒绝覆盖已保存场景；重新执行应修改脚本输出目录。较长矩阵为两地图 × 3 小区 × 4/24 UE，静止 UE、空初始队列、200 ms 预热，1–2 秒仿真时长；它属于有限承载检查，不是移动信道泛化实验。

执行结果与源码指纹、各版本是否完成、数据校验记录将在相邻 JSON 验收报告中保存。历史报告与停止后的部分数据保留，不把未完成运行计为验收通过。
