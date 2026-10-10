# 完整接入测试

本记录对应两个地图、3 小区、12/24 UE、3 个种子、每次 1000 时隙。结果以实际矩阵审计文件为准，不作为算法泛化结论。

## 执行与审计

```bash
source scripts/spark/activate.sh
python experiments/run_rt_sys_matrix.py --workers 3 --skip-completed
python experiments/audit_rt_sys_matrix.py
python experiments/validate_rt_sys_controls.py --device cuda
python -m pytest -q tests --ignore=tests/test_xapp_experiment_utils.py --disable-warnings
```

矩阵支持每个场景单独日志、失败汇总和续跑。默认 workers=1；本次采用三个进程并行。进程耗时受到并行竞争和缓存状态影响，不能当成单独占用设备时的性能基准。

逐时隙审计检查完整 GPU 执行、12/24 个 UE 记录、实际预编码列归一化、发射能量与资源掩码一致、功率预算、TB 与队列预算、ACK/NACK 交付记账、控制观测延迟、未支持动作的明确理由。控制探针另外在真实 RT/SYS GPU 上验证邻区功率、资源分离、静默、波束和休眠，并通过原控制延迟链路核对功率、波束、休眠和调度策略。闭环探针直接检查每个 UE 的队列扣除及 HARQ 进程字节与汇总一致性。

控制探针使用脚本签发确定动作，以隔离因果；不同变体共享传播信道缓存、业务到达和随机种子。它不替代真实 xApp 的算法性能比较。已有专项的已知信道张量继续验证正交/相关用户及切换服务链路。

## 本轮发现与修复

首轮 street / 12 UE 在第 143 时隙拒绝了 target MCS 大于 cap 的重传 grant。旧 MAC 的目标与上限会在重传以及 slice-aware 提升后冲突；新 RT/SYS 适配要求显式一致。已在专用 RtSysScheduler 中将目标限制到原策略 cap，并将最低目标限定到 SYS 表 1 的校准入口 MCS 3，然后重新计算 grant/TB。旧调度器保持原状。新增 3 个回归用例覆盖重传、低时延目标及最低校准入口。

控制对照还发现：同一进程反复加载地图时，Mitsuba 为匿名对象分配的 `no-name-N` 名称递增，导致相同场景的缓存键不同。现按场景加载顺序绑定材料，继续哈希完整 XML/mesh 资产和材料属性。新增用例确认运行时名称变化不影响缓存键，材料变化仍使缓存失效。早期矩阵进程使用旧键完成的有效物理结果保留其原键；后续输出使用稳定键。该修复不改变 RT 信道计算、轨迹或物理参数。

## 回归范围

原有缺失模块 tests/test_xapp_experiment_utils.py 显式排除。此前复现的四项原有失败单独记录：场景注册表数量断言、Windows 配置路径、两个 xApp 选择断言。不得把它们记为本轮新增失败，也不得把完整测试套件记为全部通过。

## 内存解释

矩阵每秒采样 cudaMemGetInfo 与 Linux MemAvailable。CUDA 计数覆盖设备总体可用内存，能够观察 Torch 和 RT/Dr.Jit 之外的设备内存变化，但也包含并行任务、其他进程及 GB10 共享内存效应，不能用于精确归因单个项目。另保留每个进程峰值 RSS 和 Torch 分配峰值；Torch 计数不包含全部 RT 分配。
