# 目标驱动的 xApp 与显式并发实验

原 12 类、28 个实现保留，新增 18 个目标反馈基线，共 46 个实现。目标、方法和实例分开登记；新增方法为本地确定性基线，没有上游复现或性能改进声明。源码目录中的 evidence_status 表示需要对应版本证据，不能把注册成功解释为验收通过。

## 目标和方法

| 类别 | 新方法 | 真实目标与决策代理 |
|---|---|---|
| throughput_power | sum_service_power / weak_user_power | 总 goodput / 低吞吐群体；需求×CQI 仅作频段排序 |
| rf_energy | rf_efficiency_search / sla_admission_power | RF bit/J / 满足吞吐需求的 UE 数量；实际执行窗口搜索 / 接近达标及 HOL 保护规则 |
| ue_fairness | sum_service_caps / proportional_utility_caps / starvation_guard_caps | 总 goodput / 对数吞吐和 / 连续未获资源时长；质量贪心 / 质量除以历史 goodput / 实际资源计数反馈 |
| slice_sla | sla_count_weights / worst_slice_weights / deadline_loss_weights | 切片达标比例 / 最差归一化缺口 / 超期；分别优先接近达标、最大缺口和新增超期压力 |
| traffic_steering | coverage_margin_steering / continuity_first_steering | 覆盖裕量 / 连续服务；弱覆盖立即请求更好候选 / 连续弱链路达到 100 ms 才请求 |
| mobility_robustness | pingpong_guard / failure_guard | 乒乓 / 已终结失败及掉线；新增方法的乒乓识别窗口默认 1000 ms（现有最小驻留/接入确认不变）；增加 TTT/HYS / 有失败且参考链路弱时尝试提前触发 |
| ret_coverage | coverage_fraction_grid / sum_goodput_grid | 达到固定参考门槛的比例 / 成功交付量；相同合法倾角，实际执行后评价 |
| beam_management | sum_goodput_beam / sla_count_beam | 总成功交付量 / 达标 UE 数；公开合法候选报告筛选后实际执行评价 |

总吞吐代理与 RF 能效不是实际目标收益的证明。公平资源上限不是硬资源保证，切片权重不是硬隔离；没有可行解时允许 SLA 未满足。RF 能量仅为已建模发射资源的能量，不表示整机能耗。

功率集中规则将 80% 预算投向质量排名前四分之一的有效 PRB，剩余 20% 给其余有效 PRB；资源组按静态成员展开，功率不转移到没有需求的资源。低吞吐功率规则先选择活跃 UE 中吞吐最低的 20%。整数资源的需求上限采用 ceil(queue_bytes/1500) 与物理 PRB 数的较小值，是保守队列启发式，不是预测 NR TBS/容量。比例效用使用 bit/s 下的历史 goodput（决策分母最低 1000 bit/s）及边际分配抑制；评价公式为 ln(max(goodput_bps,1e-6))。

目标型方法只从新的有效公开报告更新；必需 KPI 未交付、过期或无样本时不补零。目录给出 objective_id、方向、评价 KPI、代理分数、保护条件和证据状态。目标不能在实例参数中改名。原方法按所属类别的既有复合目标登记；不把旧参数变体重新计入新增方法。

## 选择接口

默认 exclusive 模式维持原 xapp_bindings 与同类单实例行为。显式启用：

```json
{
  "xapp_library": {
    "enabled": true,
    "selection_mode": "conflict_experiment",
    "instances": [
      {"instance_id": "sum", "implementation_id": "throughput_power:sum_service_power", "priority": 30},
      {"instance_id": "weak", "implementation_id": "throughput_power:weak_user_power", "priority": 30}
    ],
    "active_sets": {"cell_0": {"throughput_power": ["sum", "weak"]}}
  }
}
```

non-RT 使用 PolicyUpdate.xapp_active_sets；形式为 cell → category → instance ID 列表，完整替换当前选择，{} 停用，None 保持。禁止与 xapp_bindings 同时使用或用旧启停列表绕过选择。整笔事务先验证实例、类别、作用域、能力、目标不重复、参数合法性；失败保持原状态。选择列表排序没有优先级含义。

生命周期令牌在并发模式按 cell/category/instance 隔离。移除、重新启用只影响该实例；已开始的 HARQ 和切换继续完成。优先级高者获得字段；同级配置字段按实例 ID 字典序较大者确定性胜出，不依赖注册或选择列表顺序。命令保留现有优先级、签发时间和来源名的确定性规则。功率向量整体仲裁，不混合请求。

退出实例的租约被撤销，余下合法请求重新仲裁；没有请求时走现有基线恢复。默认 exclusive 模式原有替换保持机制不变。未选实例不运行或影子评价。

## 执行后搜索与证据

搜索最小窗口 400 ms，不允许参数缩短。基线、候选的样本起点都晚于已执行配置报告；RF 能效和 goodput 使用累计能量/交付量差值，不叠加重复滑动窗口。RF 搜索比较当前功率及相邻 ±power_step_db；降功率导致吞吐 SLA 满足人数下降时回退初始值。RET 包括当前配置和 0/3/6/9/12/15°；波束仅使用已有合法参考候选。平局优先保持初始配置。

控制器额外接收独立的上一轮仲裁所有权回执（control_ownership，来源为控制面），同时必须核对正常延迟观测里的实际配置。该回执不包含真实信道或内部业务状态，也不单独证明物理生效。延迟配置报告另提供单调 config_revision；实际配置在报告间隔内改动再改回也会增加该计数，使评价窗口失效。动作失权、未执行、相关配置/成员/声明需求变化或窗口不足时，不形成候选得分。搜索没有执行权时不能把其他实例的 KPI 变化当作自身收益。

rule_decisions.jsonl 同时记录 objective_observation、objective_window_evaluated、request_arbitrated 等证据；实际执行以 control_receipts.jsonl 为准。请求分歧、仲裁覆盖、实际响应和优化收益分别报告，不自动生成冲突或因果真值标签。

## 本轮验收

只运行确定性测试及 Spark 短测，不启动长测。普通 128-slot 响应用例；搜索、连接和事件反馈仅延长到覆盖必要真实窗口。原始数据、失败现场、缓存、配置/源码快照和 SHA256 仅在 Spark outputs/objective_xapps_20261016/ 保存。

新增 18 方法两地图分别对照固定控制；并发覆盖功率、RF、资源、切片、引导以及 RET/波束。每组单独运行两目标、共同运行、交换优先级和撤销胜出者。只发请求、被覆盖、条件未触发、样本不足都不算动作响应通过。允许 KPI 不改善，不允许伪造目标收益。资源/功率/分组/HARQ 守恒、在线探针为零、真实解码次数一致为持续门禁。

专用移动性短测为暴露控制消费者，流量引导使用较大的固定自动切换裕量，MRO 使用较小的固定事件裕量；同一用例的参考组与动作组完全相同。失败夹具延长目标确认以产生真实失败；不降低连接判据。初始真实发送后可暂停业务 PRB，以减少与连接事件无关的解码。它们均不是原部署承载结论。

RET 的独立响应夹具复用地图中合法基站位置并共址布置三个站，邻站以较低固定功率工作，保留全部干扰计算，隔离自动重关联对倾角窗口的干扰；各组均使用同一夹具。搜索短测至少需要两个已完成执行窗口，只完成基线窗口不计搜索响应通过。
