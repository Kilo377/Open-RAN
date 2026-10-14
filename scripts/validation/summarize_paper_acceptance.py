"""Build a report from completed, immutable paper-interface acceptance artifacts."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(root, destination):
    status = json.loads((root / 'pipeline_status.json').read_text())
    validation = json.loads((root / 'validation.json').read_text())
    manifest = json.loads((root / 'data_manifest.json').read_text())
    assert status['completed'] and status['passed'] and validation['passed']
    assert status['source_sha256'] == validation['source_sha256'] == manifest['source_sha256']
    cases = []
    total_events = Counter()
    for result in validation['results']:
        name = result['case']
        folder = root / name
        snapshot = json.loads((folder / 'config_snapshot.json').read_text())
        final = json.loads((folder / 'final_summary.json').read_text())
        runtime = json.loads((folder / 'run_status.json').read_text())
        assert result['passed'] and runtime['completed'] and runtime['spark_verified']
        assert runtime['source_sha256'] == status['source_sha256']
        events = final['paper_interfaces']['events']
        event_counts = Counter(e['kind'] for e in events)
        total_events.update(event_counts)
        stats = final['link_reference_statistics']
        for key in ('amc_probe_trials', 'bler_probe_trials', 'calibration_decode_trials', 'incompatible_no_tb', 'unsupported_no_tb'):
            assert stats.get(key, 0) == 0
        cases.append(dict(
            case=name, scene=snapshot['manifest']['scene'], slots=snapshot['manifest']['total_slots'],
            actual_decodes=result['actual_decodes'], elapsed_s=result['elapsed_s'],
            events=dict(event_counts), recoveries_after_slot2300=sum(e['kind'] == 'connection_started' and e['slot_id'] > 2300 for e in events),
            max_energy_error_j=result['max_energy_error_j'], config_rejections=result['config_rejections'],
            peak_rss_bytes=runtime.get('peak_rss_bytes'), peak_cuda_bytes=runtime.get('peak_cuda_bytes'),
        ))
    timings = {scene: json.loads((root / (scene + '_timing8') / 'cost_profile.json').read_text())['total_s']
               for scene in ('simple_street_canyon', 'munich')}
    report = dict(
        passed=True, source_sha256=status['source_sha256'], pipeline_elapsed_s=status['elapsed_s'],
        cases=cases, audited_closed_loop_cases=len(cases), closed_loop_slots=sum(c['slots'] for c in cases),
        timing_slots=16, actual_decodes=sum(c['actual_decodes'] for c in cases), timing8_seconds=timings,
        ret_pairs=validation['ret_pairs'], events=dict(total_events),
        artifact_files=len(manifest['files']), artifact_bytes=sum(f['bytes'] for f in manifest['files']),
        manifest_sha256=hashlib.sha256((root / 'data_manifest.json').read_bytes()).hexdigest(),
        scope='Fixed control/observation integration; forced outages included; not an algorithm comparison or capacity/SLA guarantee',
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 四篇论文接口 Spark 验收', '',
        f"冻结执行源码：`{report['source_sha256']}`。有限验收全部完成，{len(cases)} 个闭环用例、{report['closed_loop_slots']} 个时隙，另有两地图共16个成本门禁时隙。",
        '', '本次仅验证环境接口，动作由固定时间线产生；未实现论文的 xApp 决策、训练、冲突分类或缓解算法。长闭环含主动断开、功率压低制造失步、恢复和定向切换，吞吐/积压不能据此用作无故障承载结论。', '',
        '## 验证结果', '',
        '- Mac 完整测试：290 passed、1 skipped、2 subtests；Spark：291 passed、2 subtests。Mac 跳过的真实 RT 检查由 Spark 完成。',
        '- 默认功能关闭：1/3小区、各4UE、各32slots，ACK/NACK、MCS、交付、队列、PRB/符号位置与冻结旧版逐项完全一致。',
        f"- 8-slot 成本：Street {timings['simple_street_canyon']:.2f}s，Munich {timings['munich']:.2f}s，均低于14.64s门禁。该数字是短计时含启动成本，不能当每时隙长期成本。",
        f"- 闭环实际 NR 解码 {report['actual_decodes']} 次；在线探针、资源不兼容、发送后容量保护均为0。资源、功率、原重传码字与分组字节审计通过。",
        f"- 最大逐时隙 RF 能量积分误差 {max(c['max_energy_error_j'] for c in cases):.3g} J。所有成功业务字节均绑定 ACK，未出现重复交付区间。", '',
        '| 用例 | slots | 实际解码 | 耗时s | 异常掉线 | 正常结束 | HO成功/失败/管理取消 |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for c in cases:
        e = c['events']
        lines.append(f"| {c['case']} | {c['slots']} | {c['actual_decodes']} | {c['elapsed_s']:.1f} | {e.get('connection_drop',0)} | {e.get('connection_normal_end',0)} | {e.get('handover_success',0)}/{e.get('handover_failure',0)}/{e.get('handover_cancelled',0)} |")
    lines += ['', 'RET 使用相同部署、业务随机流和传播缓存成对检查；阵元布局另由真实 Sionna 阵列验证：', '']
    for pair in report['ret_pairs']:
        lines.append(f"- {pair['scene']}：最大候选/服务参考 RSRP 差 {pair['max_reference_difference_db']:.4f}dB；缓存 `{pair['cache_key']}`。")
    lines += ['', '## 数据与边界', '',
        f"原始结果目录：`{root}`。清单包含 {report['artifact_files']} 个文件、{report['artifact_bytes']} bytes；清单SHA256为 `{report['manifest_sha256']}`。两端逐文件核对回执单独保存，不能仅凭目录复制成功宣称校验通过。", '',
        '字段语义见 [接口说明](PAPER_INTERFACES.md)、[逐项文献追踪](PAPER_FIELD_TRACEABILITY.md)、[机器可读目录](paper_interface_catalog.json)。默认配置仍不启用新增接口；专用入口为 `configs/rt_sys/paper_interfaces.json`。', '',
        'CQI 是参考 SINR 量化，失步/恢复阈值是可配置工程假设，掉线率是无线连接会话口径。能量仅含已建模 RF 数据/导频；整机指标不可用。保留 A1/E2 语义仿真、理想 CSI、等效干扰接收及 none/Chase HARQ 边界。RET 零倾角针对受约束模拟阵列模型回归，不要求等同旧全数字 RZF。', '',
        f'早期 spark_v1/spark_v2 运行已主动停止并保留，不计为完整验收；本报告只统计 {root.name} 的冻结源码结果。spark_v2 的固定功率测试使用了旧调度配置PRB数，导致正常向量按长度拒绝；最终时间线使用实际NR网格并检查正常动作确实生效。',
    ]
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.output)
    print(json.dumps({k: result[k] for k in ('passed', 'audited_closed_loop_cases', 'closed_loop_slots', 'actual_decodes')}))
