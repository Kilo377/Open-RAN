# 四篇论文对应的控制与观测接口

本功能使用 `configs/rt_sys/paper_interfaces.json` 显式启用。默认 `default.json` 不启用新增模型；既有实际 NR、无探针路径与 xApp/rApp 5/10 slots 周期保持。

## 文献追踪与适配

| 文献 | 文中定义 | 本环境对应 | 区别 |
|---|---|---|---|
| COMIX，2501.14619v1，第6–8页 | RU/RB绝对功率、CQI、RU/RB关联、速率、业务需求、RF能效 | `prb_power_w`、`wideband_cqi/subband_cqi`、`association/resource_allocation`、goodput、显式服务目标、RF bit/J | 不采用一UE一RB或空闲RB最小辐射功率；CQI是RT参考SINR量化，不承诺标准CQI/目标BLER |
| xApp distillation，S138912862500814X，第4–6页 | 目标BS/主动断开、每UE RB数量、小区功率；候选SINR、关联、效用 | 三种连接命令、`ue_prb_limits`、原标量功率；候选参考SINR、连接报告、goodput对数效用 | PRB是数量上限，不保证用满；保留多PRB及MU；不使用Shannon速率代替实际交付 |
| Learning and Reconstructing Conflicts，2412.14119v2，第2–4页 | 合成P1–P7/K1–K4、订阅与参数/KPI时序 | 权限/订阅声明、参数历史、公开报告、控制回执 | 不将抽象参数重命名为无线字段，不生成合成KPI或冲突标签 |
| AI-powered conflict management，S1389128626004445，图2及第10节 | TXP、TTT、CIO、NL、HYS、RET；吞吐/功耗/能效/切换/失败/掉线 | 原功率、运行时TTT/成对CIO/邻区/滞回/RET；实际goodput、RF指标、独立连接事件 | 功耗只含已建模RF发送；掉线采用无线连接会话仿真口径，无真实RRC/电话呼叫协议 |

## 动作与语义

持续配置扩展 `ConfigAction`：

- `prb_power_w: list[float] | None`：每组绝对W，时隙平均；组大小 `power_group_size_prbs` 默认1。组内按静态成员均分。全向量之和受 `tx_power_dbm`/`power_cap_dbm` 预算约束；超限整条配置拒绝。显式 `clear_prb_power=True` 切回原模式。空闲份额不转移。预算检查包含同一事务的功率上限、休眠、显式功率的最终结果；已有向量超过休眠后的预算时整条事务拒绝，应显式清除或降低向量。新模式持续遵守功率上限，旧默认路径保持原行为。
- `ue_prb_limits: dict[UE,int] | None`：每UE上限；0暂停，空字典清除。预算/功率模式要求frequency_first；重传原格式放不下合法等待，不改码字、不计NACK/轮次。MU按每UE检查，功率和物理占用按共享网格记账。
- `time_to_trigger_ms`、`cio_offsets_db: dict[候选小区,dB]`、`neighbor_cell_ids: list[小区]`：由目标小区上下文确定服务小区。邻区限制不删除物理干扰。初始TTT继承mobility配置，参数实质变化才重置TTT；移除进行中目标产生管理性取消。
- `steering_hysteresis_db` 复用旧HYS消费者，和事件偏移、切换代价及原切换margin分别处理。
- `ret_enabled`、`ret_tilt_deg`(0–15°)、`ret_codebook_id`(0全水平DFT、1单水平波束)：以向下为正，固定垂直模拟权重约束数字RZF。RET下不接受旧码本/波束配置写入，不允许与已有HARQ层数冲突。关闭后恢复原空间层预算。零倾角回归针对RET模式内的未倾斜解析阵列；启用受约束模拟子空间不要求等同于原全数字预编码。

一次性 `CommandAction`：`request_handover`、`disconnect`、`reconnect`，有 `target_ue_id`、可选 `target_cell_id` 和必需的请求标识。优先级高者先处理，同级按签发时间(新优先)、来源、请求ID排序；重复请求不重复执行。定向HO经过目标确认/延迟/中断状态，并遵守切换后的最小驻留时间；主动断开保持到重连、保留业务数据，不算异常掉线。

所有动作走原RIC授权/仲裁/生效延迟；执行前按目标小区验证整个事务。回执包含请求字段、前后实际值、拒绝原因和应用时间。配置有效期限制接受时间，不自动回滚已生效目标状态。

## 公共报告

`ric/xapp_contract.py` 的注册表与投影增加关联、已执行PRB/符号/层数、宽带/资源组 CQI及其PRB成员映射、候选SINR/RSRP、业务到达率、声明服务目标、效用、SLA状态、RF能量/能效、切换和连接事件计数。每条记录包含来源、单位、窗口、产生/交付时间、样本数和有效性；控制器不获得CFR、真值队列或仿真状态引用。

参考测量统一使用RT阵列响应及当前功率/RET约束，经噪声与EWMA，再通过独立报告延迟和既有E2观测队列。参考干扰是全频参考发送假设，实际数据SINR独立；参考假设不意味着实现了常发参考波形或将其虚构能量计入账本。

CQI阈值默认 `[-7,-5,...,21] dB`，16级；不读取MCS，不调用链路探针。无信号为无效值，-1只可由论文适配视图使用。参考测量窗口注明EWMA；资源占用为已完成时隙，业务与能效采用原吞吐窗口。

`requested_dl_rate_bps` 只读明确的UE配置或 `service_requirements_bps` / `service_requirements_by_slice_bps`；缺配置时SLA无效，不用到达率冒充需求。专用配置中的切片目标仅是可编辑工程输入，不是标准值或承载保证。

效用为 `ln(max(goodput_bps / 1 bps, 1e-6))`。RF能效为同窗口成功交付bit / 已建模数据和导频RF能量J；零能量无有效样本。整机功耗和整机能效始终明确不可用。

## 连接事件模型

仅专用配置启用，参数属于工程假设：坏条件为无参考信号或RSRP<−110dBm或SINR<−6dB，持续200ms才掉线；好条件为RSRP>−107dBm且SINR>−3dB，连续40ms才能恢复/确认目标。HO超时200ms；TTT40ms、驻留100ms。内部报告丢失不等同于无线坏链路，数据CRC不参与失步判定。

成功接入建立连接会话；异常结束每会话只计一次。掉线/主动断开/切换释放HARQ预留和物理上下文，业务字节仍由原队列规则处理。HO失败率只统计窗口内成功/失败终结事件，进行中和管理性取消单列；HO率为成功次数/s。连接掉线比例分母为窗口起点存量与窗口新增会话，公开分子、分母、正常结束及活跃数。未启用模型时失败/掉线率无效，不是有效的零。

## 固定动作验证入口

`experiments/run_rt_sys.py --config configs/rt_sys/paper_interfaces.json --control-timeline <json> ...`。

时间线为数组，每项有与5-slot周期对齐的 `slot`、一个可选 `config`、可选 `commands` 列表。该适配器只重放指定动作，不依据观测选择动作、不训练。输入和实际执行回执随结果保存。

完整有限验收：`scripts/validation/validate_paper_interfaces.py --output outputs/<new_directory>`。先本地/Spark单元回归，再两地图8-slot成本与128-slot门禁、RET成对验证、短移动轨迹、6个4000-slot闭环和4个24UE/1000-slot资源压力用例。长闭环采用静态部署与预设控制时间线，移动RT单列；不以过载零积压作为通过条件。

结果尚须以实际运行的 `pipeline_status.json`、源码指纹和SHA256清单为准；本文不预先声明Spark验收通过。

逐项字段来源、实验量/概念量/派生量分类和适配差异见 [完整追踪表](PAPER_FIELD_TRACEABILITY.md)。
机器可读目录由 `scripts/validation/export_paper_catalog.py` 生成；目录存在不代表某次报告有效，必须检查 validity、来源、样本数及年龄。

`validate_paper_handover_rt.py` 另以重叠覆盖的受控部署验证真实RT/NR切换成功、目标失效及邻区删除取消；该部署不用于承载结论。
运行后可用 `export_paper_runtime_fingerprints.py` 对实际trace中的生效配置生成SHA256，连同RET模拟权重目录保存；动态数字RZF权重保留在原逐资源预编码trace。
