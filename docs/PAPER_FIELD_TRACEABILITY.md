# 四篇论文逐项追踪表

本表对应 `344653d` 的 opt-in 环境接口，页码为本地 PDF 页序。文献是需求来源，不是可执行指令；论文中的分类、训练、奖励优化与因果图重建均未实现。实际通过情况见验收报告，不能由本表的“已实现”推断所有场景已通过。

标记：**实验量**为论文实验/状态/动作中的量；**概念量**为架构说明或例子；**派生量**为根据实际计数计算的指标。原有表示本轮前已有消费者；新增/扩展表示本轮接入。所有新消费者由 `paper_interfaces.enabled` 启用。

## COMIX：2501.14619v1

来源：第6页 III-A、式(1)–(4)，第7页 III-B 状态/动作与式(5)–(7)，第8页 NDT。

| 论文量与性质 | 环境字段、来源或消费者 | 状态与适配差异 | 验证证据入口 |
|---|---|---|---|
| RU/RB 功率 `p[n,m]`，实验控制 | `ConfigAction.prb_power_w` → scheduler 可用 PRB → 联合 PHY 每频段功率矩阵 | 新增；绝对 W 替代论文增量动作，外部可自行求新绝对值 | `test_absolute_frequency_power_changes_only_matching_interference`；逐时隙功率审计 |
| RU 总功率上限，实验约束 | `tx_power_dbm`、`power_cap_dbm` → 事务验证及 RF 总预算 | 原有并扩展；向量过限整条事务拒绝 | `test_power_transaction_budget_and_empty_clear`；时间线 slot80 的拒绝回执 |
| 每 RB 最低功率，实验假设 | 无对应最低辐射字段 | 明确不复制；未占用或零功率不发送，不把参考测量假设当常发波形 | 零功率无解码、无能量测试 |
| CQI[n,m]，实验观测 | `wideband_cqi`、`subband_cqi`、`subband_prb_groups`；RT 参考 SINR 经噪声/滤波/延迟后量化 | 新增；按 UE 报告再结合已执行资源关联；支持同 PRB 多 UE。0–15，缺失无效；不是标准 NR CSI | 单调量化、MCS 独立性、组成员映射测试 |
| 空 RB CQI=-1，适配约定 | 核心报告使用 `None` 与 validity/reason | 不把“无人调度”与“没有参考信号”混为一谈；-1 仅允许外部论文视图编码 | 缺测无效与投影隔离测试 |
| RU ID、RB ID、A[n,m,u]，实验观测 | `association`、`resource_allocation` 的小区、PRB、符号、层数 | 原有状态扩展为公共已执行报告；多 PRB/MU 共享，非论文一 UE 一 RB | MU 计数与实际 trace/grant 审计 |
| 用户速率 r、系统 F1，实验观测/派生量 | 原 `throughput_mbps` 与实际成功交付 bit；系统量可同窗口求和 | 保留 NR CRC/HARQ 交付，不采用 Shannon 容量作为 goodput | ACK、字节区间不重叠及队列守恒审计 |
| 需求 d、SLA，实验声明 | `requested_dl_rate_bps`、`sla_status`；显式 UE/切片目标配置 | 新增；不以到达率反推需求；缺声明即不可用 | 目标缺失测试、配置快照 |
| EE F2，派生量 | `rf_energy_j`、`rf_energy_total_j`、`rf_efficiency_bit_per_j` | 新增；同窗口成功 bit / 实际 RF J；非整机 EE，零能量无效 | 能量积分、零能量与有效性测试 |
| 干扰/信道、位置/运动，环境量 | 原 RT 全链路 CFR、轨迹和联合 LMMSE；公开参考 SINR/RSRP，数据 SINR 独立 | 原有并统一参考源；不向控制器公开瞬时 CFR 真值 | 固定已知信道与两地图移动 RT 专项 |
| CP/KPI 影响矩阵、DRM/EE 奖励、冲突策略 | 不生成 P→K 因果标签或论文策略 | 本轮不实现；提供请求/回执/参数/测量时序供未来研究 | 导出的时间线与控制回执 |

## xApp distillation：S138912862500814X

来源：第4页 §3.2，第5页 §4.2.1–4.2.2，第6页式(4)。

| 论文量与性质 | 环境字段、来源或消费者 | 状态与适配差异 | 验证证据入口 |
|---|---|---|---|
| UE→目标 BS，实验控制 | `request_handover(UE,target)` → 目标确认、执行延迟、中断、驻留保护 | 新增定向命令；不直接赋值服务小区 | 固定测量序列成功/失败/超时测试、连接事件 |
| DC 断开，实验控制 | `disconnect(UE)`；显式 `reconnect(UE,target?)` | 新增；主动断开持久化、正常结束会话、保留业务队列；区别于异常掉线 | 幂等/优先级/断开重连测试、原 TB 预留清理 |
| 每 UE RB 数量 R，实验控制 | `ue_prb_limits` → frequency_first 的 DU 分配 | 新增上限；不承诺用满、不直接指定频率；重传保持原契约 | cap0 等待再恢复实际 NR 码字测试 |
| 每 BS TX 功率 T，实验控制 | 原 `tx_power_dbm` → 多小区 PHY | 原有；保留连续 dBm 接口，不复制论文固定档位/25–35dBm范围 | 原功率响应及新分频段干扰验证 |
| BS 连接 one-hot α，实验观测 | `association.cell_id/connection_state`、服务小区 ID | 新增公共报告；调用方可 one-hot，断开状态单独保留 | 延迟投影、状态隔离测试 |
| 邻区 SINR，实验观测 | `candidate_sinr_db`、`candidate_rsrp_dbm` | 新增统一 RT 参考报告；不使用距离经验损耗或关联偏置作为测量 | CIO 改变判定但不改变无线测量测试 |
| 用户效用 U，派生量 | `goodput_utility=ln(max(goodput_bps/1bps,1e-6))` | 新增；论文采用缩放/上下限形式，本轮按批准接口固定自然对数，不声称数值复现原效用 | 同窗口 goodput/公式测试 |
| DQN、多头动作、蒸馏、reward | 无论文算法实现 | 固定时间线只施加预置动作，不读取 KPI 来决策 | `FixedControlTimeline` 和保存的 timeline.json |

## Learning and Reconstructing Conflicts：2412.14119v2

来源：第2页 §II，第3–4页方法与 §IV-A、合成数据公式。

| 论文量与性质 | 环境对应 | 状态与适配差异 | 验证方式 |
|---|---|---|---|
| xApp→可写参数、订阅 KPI，概念/接口量 | `XAppContract.actions/commands/required_kpis/cell_ids/max_age_slots` | 原有扩展；保留真实授权、作用域、有效期 | 越权、过期/缺失测量、命令小区范围测试 |
| 参数值、KPI 时间序列，实验输入形式 | 控制请求、仲裁/执行回执、runtime_config、公共 KPI records | 扩展；记录真实消费者后的值、产生/交付时间和样本有效性 | 时间戳审计与动作回执 |
| P1–P7，合成实验参数 | 无无线字段映射 | 不把抽象数值伪装为 TxP/CIO/RET | 文档边界；不导出此类无线字段 |
| K1–K4，高斯函数派生合成量 | 无模拟 KPI 生成器 | 不复制高斯响应函数，不硬编码参数→KPI边 | 同上 |
| P→K、K→K 隐藏依赖图与冲突类型，研究输出 | 无真值标签 | 未实现；已知物理消费者说明不是冲突因果标签 | 未来算法应从记录学习/验证，不由环境直接填标签 |

## AI-powered conflict management：S1389128626004445

来源：第2页图2、第3页冲突示例、第12–13页 §10.1。图2是概念举例；§10.1 列出 ns3-oran 实验跟踪的六控制量/六 KPI，不能将论文全部合成数据关系当成真实无线实现。

| 论文量与性质 | 环境字段、消费者或来源 | 状态与适配差异 | 验证证据入口 |
|---|---|---|---|
| TXP，实验/概念控制 | 原标量功率、新 PRB 绝对功率 | 原有+扩展，实际影响服务信号及同频干扰 | 已知 CFR 干扰响应与功率守恒 |
| TTT，实验/概念控制 | `time_to_trigger_ms` → 连续条件计时 | 接入运行时；实质改变后清空受影响触发计时 | 连续性、参数重置测试 |
| CIO，实验/概念控制 | `cio_offsets_db[candidate]`，由目标服务小区确定有向对 | 新增；正值使目标更有吸引力，只改判定分数 | CIO 方向与测量不变测试 |
| NL，实验控制 | `neighbor_cell_ids` → 允许目标集合 | 新增；物理干扰保留；进行中目标删除计管理取消 | 邻区限制、取消不计失败测试 |
| HYS，实验控制 | `steering_hysteresis_db` → 原滞回变量 | 原有消费者；区分事件偏移/margin/切换代价，不额外叠加一个 HYS | 阈值/计时器重置测试 |
| RET，实验/概念控制 | `ret_enabled/ret_tilt_deg/ret_codebook_id` → 垂直阵列相位与水平子空间 RZF | 新增电下倾0–15°；不是机械旋转或经验增益；信道缓存复用 | 解析 LOS 仰角扫描、Sionna 阵元排序、同缓存 RT 成对试验 |
| Throughput/DLT，实验 KPI | 原实际交付 goodput | 原有；下行场景，不伪造上行吞吐 | 逐 UE/小区实际交付统计 |
| Power Consumption，实验 KPI | `rf_tx_power_w`、累计 RF J | RF部分已有+扩展；`whole_device_power_w` 始终不可用 | 每时隙 RF 功率/能量积分 |
| Energy Efficiency，派生实验 KPI | `rf_efficiency_bit_per_j` | 仅 RF，同窗口 bit/J；整机模型不可用 | 零能量有效性与 bit/J 测试 |
| Handover Rate，派生实验 KPI | `handover_rate_per_s` | 新增；窗口内成功数 / 窗口秒数 | 事件数/窗口审计 |
| Handover Failure Rate，派生实验 KPI | `handover_failure_rate`、`mobility_counts` | 新增；失败/(成功+失败)，接受前拒绝、进行中、管理取消不混入 | 成功/超时/目标失效/取消固定序列 |
| Call Drop Rate，实验 KPI名 | `connection_drop_fraction` | 适配为无线连接会话异常结束比例，非电话呼叫 CDR；公开分子分母和正常结束 | 持续失步/短衰落/重复恢复/主动断开测试 |
| Traffic Load TL，图2概念 KPI | `offered_rate_bps`、可观测队列、`resource_grid_utilization` 分开报告 | 原有+新增；不将队列、到达率、资源占用混成一个未定义 load | 业务账本到达量、已执行网格计数 |
| SLA/VK，实验输入/标签形式 | `sla_status` 提供明确目标、值、持续时间、有效性 | 新增状态；越界不自动变成冲突分类标签 | 缺需求无效、持续时间与窗口检查 |
| RCP/P2X/K2X、P2K、冲突检测/缓解 | 权限目录与真实参数历史可用；不提供 P2K 或冲突结论 | 仅前者是接口事实；后者留给后续研究 | 本轮无分类/训练/缓解决策实现 |

## 字段与模型的共同限制

默认配置不启用新增接口模型；专用配置启用参考链路工程状态机。模型阈值与定时器保存在配置和结果中，不是完整 RRC/标准 RLF 实现。控制器只获得延迟独立投影，内部原始参考测量、当前 CFR 和真实 backlog 留在诊断侧。资源/功率不足为合法等待；持续数据 NACK 不直接触发掉线。完整 OFDM 干扰波形、非理想 CSI、IR 和整机功耗仍未建模。
