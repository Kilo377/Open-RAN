# 目标驱动 xApp 实施与短测验收

本轮为 12 类、46 个实现（保留 28，新增 18），新增目标元数据和显式并发选择接口，不训练算法、不构造冲突标签、不启动长测。

## 版本与资料

- 参考已验收 Git 版本：2f180ca；Spark 参考快照：outputs/objective_xapps_20261016/reference_source.tar.gz，SHA256 f82d0ec9fbe8a126abace628da3d4410f75554cde73ca087285c192cac09e43f。
- 实现提交：14f8385；短测窗口约束提交：668ef6e。
- 最终物理/控制源码指纹：328a3adb7da423f3105f1cd556e5563e1649627694a58bd9d081da01514207e6。
- Mac/Spark 共 1290 个源码、配置、脚本和测试文件已逐一校验一致，见 OBJECTIVE_XAPPS_SOURCE_SHA256.json。
- 接口及约束见 OBJECTIVE_XAPPS.md；机器目录见 xapp_library_catalog.json。

## 本地和探索性结果

本地及最终 Spark 综合回归各 247 项通过，覆盖目标规则、旧实现、公共观测、策略事务、控制面、RT 调度和 HARQ 等。Spark 用时 45.47 秒；日志为 outputs/objective_xapps_20261016/final_regression.log。

Spark startup_v1 的 8-slot 实际 RT/NR 发送与账本通过。早期动作夹具仅控制没有服务 UE 的小区，响应不计通过；改用实际服务小区所在的管理集群验证。

short_rules_v1 中 8 个资源/功率/切片实现 × 两地图，16 组成对响应通过。short_windows_v1 中 RF、RET、波束等探索性响应通过；连续性和 MRO 的专用夹具曾缺少下游触发条件，未计通过。原因及修复：自动切换先于连续性 xApp 动作；MRO 后续候选优势没有跨过固定事件门槛。采用相同参考/动作夹具调整触发条件，并保留旧失败数据。

搜索现在同时检查实际配置、独立仲裁所有权和延迟交付的 config_revision；配置瞬时改动再改回仍使评价失效。功率使用显式 None 判断，0 dBm 上限不会被误当成无限制。新 MRO 的乒乓识别窗口为 1000 ms，不改变最小驻留、目标确认或连接门槛。

## 最终验收状态

最终短测目录：Spark `/home/njupt/oran_slot_sim_sionna/outputs/objective_xapps_20261016/final_short_v1`。进程退出码 0，validation.json 的 completed/passed 均为 true；两者分别核对。

- 18 个新实现 × 两地图：36/36 组成对动作—KPI 响应通过。
- 六类并发专项中的 RET、波束分别测试，共 14 组；每组包含各自单独、共同、交换优先级和撤销，共 70 次运行。14/14 组独立执行、选择、撤销门禁通过。
- 合计 142 次物理运行、138568 slots、448191 次实际 NR 解码，矩阵墙钟 3429.88 秒（57.16 分钟，3 worker）。并发矩阵耗时不能解释为单例性能。
- 数据约 5.77 GB；SHA256 清单中的 3858 个文件逐一复核一致；最终两端 1290 个源码/配置/脚本/测试文件一致。
- 所有发送路径通过既有审计：在线探针为零，不兼容及发送后容量保护为零，功率/资源/分组/HARQ 账本检查通过。
- 搜索的每个方法/地图至少有一个小区独立完成两个执行窗口，没有跨小区合并凑数。真实 RT 短测不声称已遍历全体候选或找到最优点。

另有五种搜索方法的合成公开报告组件检查，验证完整候选集结束和严格同分保留初始配置；RET、波束还通过相同输入序列哈希、不同目标选出不同配置的组件检查。组件数据明确标为 synthetic，不算真实 RT/NR 结果或解码探针。近似相等的浮点分数不等于严格同分，不作数值稳定性/收益保证。

逐项证据见 goal_response_matrix.json、component_search_completion.json、component_goal_difference.json、控制回执和逐时隙记录。

长测未启动，本轮结束后也不启动。原始结果、配置快照、源码归档、日志、失败现场及校验清单仅留在 Spark。Mac 仅保留源码、配置、文档和轻量汇总。

## 并发证据边界

| 专项（各两地图） | 独立执行与撤销 | 同时不同请求 | 交换优先级后 KPI 差异 |
|---|---|---|---|
| 功率、RF、UE 资源、切片，共 8 组 | 通过 | 已观察 | 已观察 |
| RET、波束，共 4 组 | 通过 | 已观察 | 本短窗口未观察 |
| 流量引导，共 2 组 | 通过 | 未观察 | 未观察 |

因此，14 组选择/撤销通过不等于 14 组并发 KPI 响应通过，更不等于真实冲突或目标损害标签。只将 8 组计作已观察到优先级变化的 KPI 响应；另外 6 组保留为对应证据未触发。需要更多此类冲突样本时，应另行设计场景/窗口；本轮不强制制造差异、不运行长测。

## 逐实现验收表

下表的“通过”仅指该实现单独启用时实际请求、实际执行和相对固定控制的 KPI 响应；不表示 KPI 改善。确定性差异用例见 tests/test_objective_xapps.py，RET/波束的完整目标选择差异另见上述组件证据。

| 实现 | 固定目标 | Street 独立响应 | Munich 独立响应 |
|---|---|---|---|
| `throughput_power:sum_service_power` | `sum_goodput` | 通过 | 通过 |
| `throughput_power:weak_user_power` | `lower_tail_goodput` | 通过 | 通过 |
| `rf_energy:rf_efficiency_search` | `rf_efficiency` | 通过 | 通过 |
| `rf_energy:sla_admission_power` | `ue_sla_count` | 通过 | 通过 |
| `ue_fairness:sum_service_caps` | `sum_goodput` | 通过 | 通过 |
| `ue_fairness:proportional_utility_caps` | `log_goodput_utility` | 通过 | 通过 |
| `ue_fairness:starvation_guard_caps` | `resource_starvation_duration` | 通过 | 通过 |
| `slice_sla:sla_count_weights` | `slice_sla_count` | 通过 | 通过 |
| `slice_sla:worst_slice_weights` | `worst_normalized_slice_gap` | 通过 | 通过 |
| `slice_sla:deadline_loss_weights` | `deadline_losses` | 通过 | 通过 |
| `traffic_steering:coverage_margin_steering` | `reference_coverage_margin` | 通过 | 通过 |
| `traffic_steering:continuity_first_steering` | `service_interruptions` | 通过 | 通过 |
| `mobility_robustness:pingpong_guard` | `pingpong_events` | 通过 | 通过 |
| `mobility_robustness:failure_guard` | `terminal_mobility_failures` | 通过 | 通过 |
| `ret_coverage:coverage_fraction_grid` | `reference_coverage_fraction` | 通过 | 通过 |
| `ret_coverage:sum_goodput_grid` | `sum_goodput` | 通过 | 通过 |
| `beam_management:sum_goodput_beam` | `sum_goodput` | 通过 | 通过 |
| `beam_management:sla_count_beam` | `ue_sla_count` | 通过 | 通过 |
