# 四篇论文接口 Spark 验收

冻结执行源码：`44d60d126ded597ab10a510ecdd13689cfbd5f30c35bd905e0114fd254154ae3`。有限验收全部完成，18 个闭环用例、29280 个时隙，另有两地图共16个成本门禁时隙。

本次仅验证环境接口，动作由固定时间线产生；未实现论文的 xApp 决策、训练、冲突分类或缓解算法。长闭环含主动断开、功率压低制造失步、恢复和定向切换，吞吐/积压不能据此用作无故障承载结论。

## 验证结果

- Mac 完整测试：`291 passed, 1 skipped, 1 warning, 2 subtests passed in 38.16s`；Spark：`292 passed, 2 warnings, 2 subtests passed in 62.97s (0:01:02)`。Mac 跳过的真实 RT 检查由 Spark 完成。
- 默认功能关闭：1/3小区、各4UE、各32slots，ACK/NACK、MCS、交付、队列、PRB/符号位置与冻结旧版逐项完全一致。
- 8-slot 成本：Street 7.42s，Munich 6.79s，均低于14.64s门禁。该数字是短计时含启动成本，不能当每时隙长期成本。
- 闭环实际 NR 解码 123256 次；在线探针、资源不兼容、发送后容量保护均为0。资源、功率、原重传码字与分组字节审计通过。
- 最大逐时隙 RF 能量积分误差 5.59e-17 J。所有成功业务字节均绑定 ACK，未出现重复交付区间。

| 用例 | slots | 实际解码 | 耗时s | 异常掉线 | 正常结束 | HO成功/失败/管理取消 |
|---|---:|---:|---:|---:|---:|---|
| simple_street_canyon_128 | 128 | 2298 | 53.9 | 0 | 1 | 0/0/0 |
| munich_128 | 128 | 1341 | 40.3 | 0 | 1 | 0/0/0 |
| simple_street_canyon_ret_False | 128 | 357 | 13.7 | 0 | 0 | 0/0/0 |
| simple_street_canyon_ret_True | 128 | 374 | 15.8 | 0 | 0 | 0/0/0 |
| munich_ret_False | 128 | 321 | 14.6 | 0 | 0 | 0/0/0 |
| munich_ret_True | 128 | 321 | 13.5 | 0 | 0 | 0/0/0 |
| simple_street_canyon_moving | 256 | 619 | 19.5 | 0 | 1 | 0/0/0 |
| munich_moving | 256 | 655 | 19.5 | 0 | 1 | 0/0/0 |
| simple_street_canyon_u4_s512_load1.0_4000 | 4000 | 10477 | 424.3 | 4 | 1 | 0/0/0 |
| simple_street_canyon_u4_s513_load1.0_4000 | 4000 | 10557 | 426.1 | 4 | 1 | 0/0/0 |
| simple_street_canyon_u4_s514_load1.0_4000 | 4000 | 10548 | 427.5 | 4 | 1 | 0/0/0 |
| munich_u4_s514_load1.0_4000 | 4000 | 10287 | 432.9 | 4 | 1 | 0/0/0 |
| munich_u4_s512_load1.0_4000 | 4000 | 10453 | 450.1 | 4 | 1 | 0/0/0 |
| munich_u4_s513_load1.0_4000 | 4000 | 10534 | 454.0 | 4 | 1 | 0/0/0 |
| munich_u24_s512_load0.1_1000 | 1000 | 11075 | 462.0 | 1 | 1 | 1/0/0 |
| simple_street_canyon_u24_s512_load0.1_1000 | 1000 | 14570 | 546.4 | 0 | 1 | 3/0/0 |
| simple_street_canyon_u24_s512_load1.0_1000 | 1000 | 16334 | 596.7 | 0 | 1 | 3/0/0 |
| munich_u24_s512_load1.0_1000 | 1000 | 12135 | 355.3 | 1 | 1 | 1/0/0 |

RET 使用相同部署、业务随机流和传播缓存成对检查；阵元布局另由真实 Sionna 阵列验证：

- simple_street_canyon：最大候选/服务参考 RSRP 差 15.8230dB；缓存 `a7afd980bda7db7bdf7ae5a85d45927e76358fc8adaf7c5cd30a677ceef892c6`。
- munich：最大候选/服务参考 RSRP 差 3.3566dB；缓存 `55190f304687e946d31f8c7da164a08f52e00a407db791e1f943040600787d95`。

另有3个受控重叠覆盖的真实RT/NR切换专项通过共同守恒审计，共480slots、450次实际解码：成功切换、执行中目标失效、邻区删除管理取消。成功用例验证了新服务小区实际解码，以及驻留期间反向命令被拒绝。该受控部署不作为真实部署容量结论。

## 资源压力与保留的问题

以下仅为本轮固定动作下200ms预热后的300ms窗口，不是无控制基线容量，也不是长期稳定性或SLA通过结论。

| 24UE用例 | 到达/交付 Mbit/s | 队列增长 byte | 溢出 byte | 新增超期分组 | 从未接入 UE | 合法重传等待 |
|---|---:|---:|---:|---:|---|---:|
| munich_u24_s512_load0.1_1000 | 9.277/8.236 | 39043 | 0 | 435 | ue_18 | 11 |
| simple_street_canyon_u24_s512_load0.1_1000 | 9.277/9.284 | -232 | 0 | 0 | 无 | 0 |
| simple_street_canyon_u24_s512_load1.0_1000 | 93.874/67.666 | 404286 | 578525 | 9441 | 无 | 0 |
| munich_u24_s512_load1.0_1000 | 93.874/31.571 | 696319 | 1640057 | 12076 | ue_18 | 15 |

- munich_u24_s512_load0.1_1000：ue_18全程未建立连接；最终最佳候选RSRP -113.05dBm，工程恢复门槛 -107.00dBm。其零交付、未接入及排队数据均保留。
- simple_street_canyon_u24_s512_load0.1_1000：窗口交付略高于同期到达，同时队列增长为-232byte，表示在清理预热期遗留队列，字节账本仍守恒。
- munich_u24_s512_load1.0_1000：ue_18全程未建立连接；最终最佳候选RSRP -113.05dBm，工程恢复门槛 -107.00dBm。其零交付、未接入及排队数据均保留。

正常负载下的积压、溢出和超期，以及低负载下仍有的业务时延问题，均如实保留。合法资源等待不计NACK或资源不兼容。上述结果不生成冲突标签，也不对联合时间线做单变量因果归因。RF能效只反映射频，不能推导整机节能。内存记录为进程RSS及PyTorch CUDA分配峰值，后者不代表Mitsuba/驱动的完整GPU峰值。

## 数据与边界

原始结果目录：`outputs/paper_interfaces_20261014/spark_v6`。清单包含 379 个文件、6169875366 bytes；清单SHA256为 `fda3440db83a255cf512f3c95a0403b6b6421f39a1ba425826380d953d88d35c`。两端逐文件核对回执单独保存，不能仅凭目录复制成功宣称校验通过。

字段语义见 [接口说明](PAPER_INTERFACES.md)、[逐项文献追踪](PAPER_FIELD_TRACEABILITY.md)、[机器可读目录](paper_interface_catalog.json)。默认配置仍不启用新增接口；专用入口为 `configs/rt_sys/paper_interfaces.json`。

CQI 是参考 SINR 量化，失步/恢复阈值是可配置工程假设，掉线率是无线连接会话口径。能量仅含已建模 RF 数据/导频；整机指标不可用。保留 A1/E2 语义仿真、理想 CSI、等效干扰接收及 none/Chase HARQ 边界。RET 零倾角针对受约束模拟阵列模型回归，不要求等同旧全数字 RZF。

早期 spark_v1/v2/v3/v4/v5 运行已主动停止并保留，不计为完整验收；本报告只统计 spark_v6 的冻结源码结果。spark_v2 的固定功率测试使用了旧调度配置PRB数，导致正常向量按长度拒绝；最终时间线使用实际NR网格并检查正常动作确实生效。随后补上了功率向量与休眠/上限组合的原子预算检查、定向切换驻留保护，以及不改变统计结果的线性E2状态计数。

## 两端数据校验与配置指纹

Mac与Spark已对矩阵379个文件、6,169,875,366bytes以及切换专项59个文件、18,865,573bytes逐文件核验SHA256，两端清单哈希一致。回执保存在 `outputs/paper_interfaces_20261014/*_verification_{mac,spark}_v6.json`。

18个闭环用例的生效配置SHA256、变化时隙和RET模拟权重目录保存在 `outputs/paper_interfaces_20261014/local_v6/runtime_config_fingerprints.json`；动态数字RZF权重仍保留在原逐资源trace中。源码归档、本地默认/固定动作数值重放、180种功率转换核对、原目录1113文件未改动检查及控制执行图亦保存在该目录。

[Street实际执行图](../outputs/paper_interfaces_20261014/local_v6/street_control_trace.png)；[Munich实际执行图](../outputs/paper_interfaces_20261014/local_v6/munich_control_trace.png)。

源码/配置/权重与重放支持材料另有20个文件、26,504,305bytes通过两端严格清单校验。三组共458个文件、6,215,245,244bytes；完整汇总为 `outputs/paper_interfaces_20261014/verification_v6.json`。所有仿真任务已结束。
