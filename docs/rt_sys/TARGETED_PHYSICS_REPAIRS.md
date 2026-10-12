# 三项物理问题修复：NR 参考反馈、槽级 RT 与 Munich 部署

状态：代码和专项验收已完成，12 组 × 1000 时隙的最终矩阵正在 Spark 运行。矩阵结束并严格审计后更新本记录。原目录未修改；本次工作在独立副本内完成。

## 1. 正确性优先的 NR 参考模式

默认入口仍是 `phy_type: sionna_rt_sys`，新增 `rt_sys.feedback_mode: nr_tb_reference`。RT 提供绝对多天线信道，RZF/LMMSE 计算同小区和跨小区干扰；SYS 保留 NR 传输块预算及结果适配，反馈使用实际 Sionna NR 编解码。新模式不使用默认 BLER/EESM 表决定 MCS 或 ACK。

在实际后均衡 SINR 向量上执行 PDSCH TB CRC、LDPC、速率匹配、Gray-QAM、APP 解调和 20 次 BP 迭代。每个 grant 依次执行：

1. 在调度器给出的 MCS 上限内，用每个候选 32 个独立码块探针选择 MCS。原 MAC 调度与业务预算保留；这是有限样本 AMC，不保证最优 MCS 或严格的目标 BLER。
2. MCS 选定后，重新使用独立随机流估计 BLER，默认 64 个码块；记录错误数与 Wilson 95% 区间，避免选择导致的估计偏差。
3. 使用第三个随机流传输一个参考码块，以实际 CRC/比特错误结果决定 ACK/NACK。SYS 的随机 ACK 不参与最终交付。

这些探针是理想 CSI 条件下的内部参考计算，不作为额外空口传输，也不占用 grant 之外的资源或功率。仅缓存编码器/解码器及传播路径，不缓存最终 SINR、BLER、控制结果或吞吐量。随机流按种子、时隙、UE、MCS/表、编码资源数和用途索引，调用顺序及算法名称不会改变同一输入的随机样本。

每次参考解码是后 LMMSE 等效独立高斯链路上的 RV0 码块。保留既有 MAC HARQ 生命周期和完整待重传载荷的容量检查；不宣称实现相同码字的重传、IR/Chase 软合并、实际导频估计或非高斯干扰波形。控制器仍读取既有延迟、噪声和聚合测量。`full_phy_calibrated` 保持 false。

Sionna 编码器的低码率、重复编码或过高有效码率限制会记录在 `unsupported_nr_mcs`，并选择上限内可执行的 NR 码。其它编码器错误直接失败，所有候选均不可执行也明确失败。重传使用固定的可执行 MCS，容量不足不会部分 ACK 待重传码块。参考模式拒绝抽象 Chase/IR 配置。

### 快速 SYS 抽象的状态

已生成 749 个精确资源预算的 NR 平坦曲线，并补充 RT 探针中出现的预算。固定 beta、自适应 EESM、调制互信息候选均进行了开发验收。人工频率剖面的代表性自适应 EESM 测试通过，但真实强选择性 RT 剖面未通过；互信息候选也仍有失败点。因此这些拟合产物保留为诊断数据，未用于默认闭环。用户选择了实际 NR 参考反馈路线。

未通过的默认 SYS 表只能通过 `allow_unvalidated_sys: true` 显式用于诊断。自定义精确资源模型必须有与摘要匹配的独立验收和实际 RT 剖面验收；人工剖面小样例不能启用生产模型。不同编码资源预算不能借用其它资源数的曲线。

### 专项结果

- Spark GPU 的 331 个参考对照点通过：312 个平坦 AWGN 点、16 个强选择性 RT 点和 3 个四码块点。
- 批量计算与单 SNR 解码的错误数和实际码块反馈完全一致，SYS/NR 的 TBS 与概率适配一致，固定种子重放一致。这是参考执行路径验收，不是 EESM 拟合通过。
- 功率、波束、休眠、资源分离、无路径零交付，以及闭环队列/HARQ 记账专项通过。
- 两地图的 40 时隙完整闭环逐 UE/小区 CSV 重放完全一致。
- 待部署源码完整回归：Mac 223 项通过、1 项 GPU 专项跳过；Spark 224 项通过，另有 2 项子测试通过。Spark 的新版 Torch 对 GB10 12.1 仍给出架构范围警告，实际 GPU 算子与闭环已运行；该警告保留在测试日志中。
- 固定相同小区总功率与占用资源的实际 NR 对照通过：8 槽中正交双用户/单用户/相关双用户分别交付 864/704/304 字节。正交 MU 获得复用收益，相关 MU 的 SINR 与交付下降；这是解析信道的物理验证，不是地图吞吐量结论。

证据：`outputs/rt_sys_targeted_repairs/reference_validation.json`、`control_validation.json`、`reference_replay_validation.json`、`reference_mu_validation.json`。拟合与失败证据保留在 `outputs/rt_sys_calibration_repair/` 和 `outputs/rt_sys_targeted_repairs/rt_pilot/`。

## 2. 几何路径每时隙更新

原先用 `seed + block` 生成路径，更新间隔对照同时改变了射线样本。现在一条轨迹固定使用同一 RT 种子，比较时只改变几何更新时间，并更新缓存版本，避免复用旧采样策略的缓存。

1 ms 更新在严格对照中仍出现超过 0.5 dB 的最强链路差异，未通过既定标准。因此默认改为每 0.5 ms 时隙重新追踪几何路径，槽内保持 Sionna 的 Doppler 相位演化。另设 1 cm 位移预算，轨迹速度变化在下一槽强制刷新；超过每槽位移预算时明确拒绝配置。

两地图 × 1.5/10/20 m/s，沿 1 秒轨迹抽样 50 个时间点，默认模式与槽级参考一致；缓存重放精确一致。额外的独立原生 Sionna CFR 对照，street/Munich 最大相对 L2 误差分别为 6.79e-7 / 4.57e-7。

同射线样本的 10 ms 对照仍有明显差异：最强链路最大增益差达到 6.81 dB（Munich、20 m/s），因此未保留为默认近似。每时隙更新解决槽间几何近似；不代表任意速度、地图或槽内遮挡变化均有普适误差保证。

证据：`geometry_validation.json`、`native_cache_validation.json`；未通过的 1 ms 对照另保留为 `geometry_validation_1ms.json`。

## 3. Munich 的部署与自然 MU

旧部署的 UE 集中在三条 3 m 路段，最佳服务链路均来自同一小区；同小区用户方向相关性中位数约 0.997，12/24 UE 均没有低于原 0.8 阈值的候选配对。

新部署沿六条已检查合法的长道路分散 UE，并调整基站朝向和俯角。原相关性阈值、MU 层数预算和调度策略保持不变，没有强制 MU grant。

| UE 数 | 旧部署可配对候选 | 新部署可配对候选 | 新部署同小区相关性中位数 |
|---|---:|---:|---:|
| 12 | 0 | 28 | 0.248 |
| 24 | 0 | 104 | 0.287 |

上述为第 1 槽、全基线波束子空间中的候选诊断；不等于实际被 MAC 调度的资源组或吞吐量收益。早期 200 槽探针已有自然 MU，最终 NR 参考矩阵将复核所有 UE 数与种子。逐预编码组新增实际波束子空间内的用户信道相关性记录。已知正交/相关信道的 MU 数值验证保留。

证据：`munich_deployment_diagnosis.json`、最终矩阵逐槽 `precoders`；部署合法性由端点向上射线及完整轨迹高度上的线段交叉检查保证。部署调整用于接入验证，未按算法吞吐量挑选地图，也不构成论文的正式泛化结果。

## 运行与复现

```bash
ssh -o HostKeyAlias=192.168.1.201 -o StrictHostKeyChecking=yes njupt@192.168.1.202
cd ~/oran_slot_sim_sionna
source scripts/spark/activate.sh
python experiments/run_rt_sys.py --scene munich --cells 3 --ues 12 --slots 1000
python experiments/run_rt_sys_matrix.py --slots 1000 --workers 4 --output-root outputs/rt_sys_targeted_matrix_nr
python experiments/audit_rt_sys_matrix.py --root outputs/rt_sys_targeted_matrix_nr --require-resource-metrics --output outputs/rt_sys_targeted_repairs/matrix_audit.json
```

Spark 是原有可信主机，地址现为 `.202`，使用已有主机公钥别名验证。未关闭身份校验或改动其它项目。环境是 Python 3.12.3、Sionna/RT 2.2.0、Torch 2.9.1+cu130、GB10/Blackwell；精确依赖沿用已验证的 `requirements-spark.lock.txt`。原始源码 1113 项校验一致。

SYS 的自定义有效 SINR 与直接有效 SINR/资源预算入口由[官方 PHYAbstraction 接口](https://nvlabs.github.io/sionna/v2.2.0/sys/api/sionna.sys.PHYAbstraction.html)提供；本参考模式在结果接口中适配实际 NR 测得的 TBLER，最终反馈来自解码。
