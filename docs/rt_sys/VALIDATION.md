# 本地接入验收记录

状态：**本地 CPU 接入完成；Spark GPU 验收未执行。** 数据摘要见同目录 `local_validation.json`，详细结果保留在副本 `outputs/rt_sys/`。

原目录的 1113 个清单文件逐一校验，SHA256 均未改变。副本首个 Git 提交保留复制时快照；输出、虚拟环境、LLVM 与缓存均不进入源码提交。

## 已执行检查

| 检查 | 结果 |
|---|---|
| 实际环境 | macOS ARM64、Python 3.12.14、PyTorch 2.14.1、Sionna/Sionna RT 2.2.0 |
| 实际 RT | `llvm_ad_mono_polarized`，真实传播路径/CFR 调用 |
| 实际 SYS | CPU PHYAbstraction/EESM/TB 调用；未使用 TensorFlow |
| 数值、缓存、闭环、真实 Torch/SYS 与 HARQ 专项 | 20 项通过 |
| 回归（排除原有缺失模块的测试） | 187 项通过、2 个 subtest 通过、4 项原有失败 |
| 两地图部署几何 | 各 3 BS；12/24 UE 起点和完整预设路段通过 mesh intersection 检查 |
| 两地图闭环 | 各 2 小区、4 UE、40 时隙，完成原有 rApp/xApp/调度/反馈链路 |
| 扩展规模冒烟 | munich、3 小区、24 UE、4 时隙，完成 |
| 实际 RT + Torch + SYS MU 接口 | 两地图各 2 小区/4 UE，强制重叠 grant，每小区 2 个用户流；各 28 个符号资源组实际求解 |
| RT 缓存 vs 重算 | `rtol=1e-4, atol=1e-10`；原始 CPU 检查的相对 L2 误差约 4e-8 |
| 固定种子完整闭环重放 | street 40 时隙，160 行逐 UE CSV 完全一致 |
| 1 ms / 10 ms 对照 | street 2 小区/4 UE/40 时隙，均交付 42885 字节；只证明此短样例，未证明长期近似误差可忽略 |
| Spark SSH / CUDA / 1000-slot 矩阵 | 未执行 |

专项覆盖：邻小区静默/移开重叠资源后干扰降低；邻小区功率和自身功率响应；物理波束子空间改变信号；平均功率守恒；未调度、无路径和深度休眠零交付；grant/队列/TB 预算；正交 MU 用户复用收益、相关用户性能下降；多层接收；切换后选择新服务链路；跨频及 guard-band grant 拒绝；unsupported 动作原因；缓存/seed 重放；NACK 仅登记实际 TB 字节，后续 ACK 清除对应 HARQ 并扣除队列。

小型闭环中，原 MAC 未自行选择 MU 组。因此新增 `experiments/validate_rt_sys_mu.py` 单独以重叠 grant 验证真实 RT/SYS 接口，未把 SU 运行标为 MU 性能验证。正交/相关信道的因果比较使用已知信道张量；地图上强制 MU 的吞吐量不作为复用收益结论。

## 原有回归问题

在原始目录使用相同 Python 环境、禁用 bytecode/cache 写入，以下 4 项失败全部复现：

- `test_scenario_matrix_registry.py`：测试期待 7 个场景，现有注册表展开为 9 个。
- `test_sensitivity_xapps.py`：硬编码 Windows 配置路径，当前 Mac 上无法加载。
- `test_xapp_selection_rapp.py` 的两个选择测试：现有依赖修复逻辑增加 latency scheduler，与旧断言不符。

另外，`test_xapp_experiment_utils.py` 引用原目录本来就不存在的 `experiments.xapp_experiment_utils`，直接运行全套测试会在 collection 阶段失败。本次回归显式排除此文件，没有把这些原有问题修改成通过。

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv-local/bin/python -m pytest -q tests \
  --ignore=tests/test_xapp_experiment_utils.py --disable-warnings
.venv-local/bin/python scripts/verify_source_snapshot.py
```

## 尚待 Spark 完成

ARM64 Linux、GB10/驱动、PyTorch CUDA 与 RT GPU 的真实调用，以及两地图 12 次、各 1000 时隙的接入矩阵；记录整体 GPU 内存与运行性能，并生成准确的 Spark 依赖锁。所有本地运行均为 `spark_verified: false`。

新模式默认 SYS BLER/EESM 的校准范围、表 3 缺失、理想 CSI、导频开销模型和无 HARQ 软合并均已在主说明注明。完整 PHY 校准、自定义地图/动态障碍物、扩展控制动作及正式跨场景泛化协议尚未纳入本阶段结论。
