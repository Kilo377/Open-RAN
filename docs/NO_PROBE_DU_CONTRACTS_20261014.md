# 无探针 DU 与 xApp 接口接入（2026-10-14）

本轮在独立副本内实施。Sionna RT、多小区联合 RZF/LMMSE、实际 NR CRC/LDPC 解码继续使用；默认 `nr_direct` 不执行在线 Monte Carlo 探针，也不建立离线 BLER 查表。控制周期仍为 xApp 5 slots、rApp 10 slots，时隙 0.5 ms。A1/E2 仍为语义仿真；理想 CSI、等效高斯 NR 资源和 RV0 none/Chase 边界不变。

## 代码入口

- `phy/rt_sys/tx_contract.py`：冻结的 `TbTxContract`、共用导频掩码、CPU 标量 SYS NR TBS 预算和缓存的编码格式检查。格式检查不编码、不解码、不读取信道。
- `phy/rt_sys/scheduler.py`：先按截止/等待/UE/进程顺序保留 SU 重传矩形，再打包新传 SU/MU。重传不经过通用重打包。资源不足的重传只记录 `retx_deferred_resource`。
- `traffic/packet_ledger.py`：TB 绑定 `(逻辑信道, 分组, byte_start, byte_end)`；预留字节不可重复使用。ACK 支持乱序且幂等；清理释放预留，业务字节不被当作交付或丢弃。
- `phy/rt_sys/olla.py`：固定 DU 的 `OllaState` 和 `HarqFeedbackEvent`。首传 ACK +0.01 dB、NACK −0.09 dB，范围 ±10 dB，目标 0.1，保护 MCS 上下界。按 UE/小区/table 隔离；切换重置。时隙末实际反馈到下一时隙消费；重传、无路径、未发送、资源等待、清理不参与首传 BLER。
- `ric/xapp_contract.py`：公共 KPI 注册表、`XAppContract`、观测投影、动作权限与范围验证。xApp 可用动作是声明与 RT 消费字段的交集。

发送预算由最终资源形状产生，调度与 PHY 使用同一导频计算与 Sionna SYS NR TBS 规则。契约保留原 MCS、table、层数、数据 RE 数、编码长度、NR TBS、业务载荷和逻辑信道分配；PRB 起点可以改变。PHY 在功率分配前检查契约，实际编解码预算再次核对。低码率大 TB 遇到当前单 CB repetition 扩展边界时，先缩小新传资源形状，不在发送后改载荷或改 MCS。

## 参考与迁移差异

固定参考 OCUDU commit `b78d09a77ab060cda752f32483e551f1e9132f66`：

- [OLLA 实现](https://gitlab.com/ocudu/ocudu/-/blob/b78d09a77ab060cda752f32483e551f1e9132f66/lib/scheduler/support/outer_loop_link_adaptation.h)；该文件标示 BSD-3-Clause-Open-MPI。
- [链路自适应测试](https://gitlab.com/ocudu/ocudu/-/blob/b78d09a77ab060cda752f32483e551f1e9132f66/tests/unittests/scheduler/ue_scheduling/ue_link_adaptation_controller_test.cpp)。
- [原 TBS 重传资源分配](https://gitlab.com/ocudu/ocudu/-/blob/b78d09a77ab060cda752f32483e551f1e9132f66/lib/scheduler/ue_scheduling/ue_cell_grid_allocator.cpp)。

本仓库独立编写 Python 适配，没有引入或执行整套 OCUDU C++ RAN。OCUDU 的 DL 控制器采用 CQI 偏置；这里使用已交付参考 SINR 的 dB 偏置与固定初始 MCS 阈值。ACK/NACK 步长比例、限幅和边界保护参考上游；Chase 资源形状约束比上游原 TBS 规则更严格。上游测试存在不代表本实现或 Sionna 链路已经验收。

## xApp 开发接口

调用形式仍为 `run(observation, policy_snapshot)`。`XAppBase(..., contract=...)` 可显式传入冻结的 `XAppContract`，否则根据声明动作和内置类别生成契约。契约包含 required_kpis、actions、entities、max_age_slots、cell_ids；空 cell_ids 表示所有管理小区。默认最大年龄继承报告延迟、窗口与 xApp 周期。

公开的测量包括队列、分组 HOL、参考 SINR/RSRP、数据 SINR、吞吐、真实 CRC 窗口 BLER、独立首传 BLER、重传尝试比例、分组超期比例；小区提供已完成时隙的资源利用率和 RF 发射功率。每项带来源、单位、窗口、产生/交付时间、样本数、有效性和原因。窗口明确区分 slots、transport_blocks、reports 与累计统计；BLER 的 TB 窗口不解释为时隙。超期比例为累计已超期分组/累计接纳分组，不是分位时延。无 BLER 样本表示无效/None；空队列与零吞吐是有效的零业务。

xApp/rApp 接收独立投影和策略副本。投影不含真实 backlog、当前 CFR、未交付测量、MCS 反推 CQI 或内部 precoder 匹配度。兼容别名只能来自已交付测量：demand_bytes 来自报告队列，HOL 来自报告分组时延，参考干扰惩罚由报告 RSRP、参考 SINR 和接收噪声计算，不能解释为实际数据干扰。

必需 KPI 缺失/过期/未来时间戳时不执行并记录原因。独立 CQI 和 precoder 匹配度没有测量来源，因此目前依赖这些输入的 mobility_predictor/massive_mimo 内置算法明确不可用；普通波束、关联参数和其他已有动作消费者仍可通过合法 xApp 控制。内置适配器在 RT 投影下不请求未接入的 TTT/CIO/muting 等参数；能力目录仍明确列出 unavailable_actions。自定义 xApp 写未声明/未实现字段、错误小区或范围时整条动作拒绝；受限小区契约省略目标时，也检查默认主小区是否在允许范围内。non-RT 选择器使用相同能力目录。

OLLA/HARQ 参数不在 xApp 动作面。整机功耗模型尚未建立；`rf_tx_power_w` 只表示 RF 发射功率。

## 验收入口与记录

- `pytest`：契约、精确资源预算、预留乱序 ACK/释放、OLLA 因果性与去重、实际 NR 小闭环、KPI 真值隔离/缺失/过期/权限，以及旧后端回归。
- `scripts/validation/validate_no_probe_environment.py --stage gate|matrix --output <new_directory>`：两地图 8-slot 计时、128-slot 门禁与 18 用例负载矩阵（3 小区，4 UE 正常负载及 24 UE 的 0.1/1 倍，种子 512/513/514，2000 slots，预热 200 ms）。
- `scripts/validation/validate_olla_traffic.py`：固定平坦/选择性信道，真实业务 TB，预热后至少 2000 首传样本，保存 Wilson 区间、饱和比例与目标判定；不生成概率表或调用探针。
- `scripts/validation/validate_xapp_io.py`：固定测试策略逐项检查 KPI → 现有参数消费者。
- `scripts/validation/validate_control_lifecycle.py --feedback-mode nr_direct`：A1 策略、优先级、作用小区、生效/过期窗口与回执。

新版本的承载与 BLER 结论只采用本轮数据，不继承历史 bank/启发式版本。过载、弱覆盖、样本不足、MCS 饱和必须单独报告。实际 GPU 与长期矩阵结果见本轮最终验收报告；此说明不把待运行用例标为已通过。
