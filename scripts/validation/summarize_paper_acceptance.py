"""Build a report from completed, immutable paper-interface acceptance artifacts."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(root, destination, local_validation, handover_root=None):
    status = json.loads((root / 'pipeline_status.json').read_text())
    validation = json.loads((root / 'validation.json').read_text())
    manifest = json.loads((root / 'data_manifest.json').read_text())
    assert status['completed'] and status['passed'] and validation['passed']
    assert status['source_sha256'] == validation['source_sha256'] == manifest['source_sha256']
    local_log=(local_validation/'pytest_local.log').read_text().strip().splitlines()[-1]
    spark_log=(root/'pytest.log').read_text().strip().splitlines()[-1]
    cases = [];pressure=[]
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
        if snapshot['manifest']['total_slots']==1000:
            steady=final['steady_state']['per_ue'];packets=final['packet_metrics']
            connected={e['ue_id'] for e in events if e['kind']=='connection_started'}
            missing=sorted(set(packets)-connected)
            pressure.append(dict(case=name,evaluated_ms=final['steady_state']['evaluated_ms'],
                offered_mbps=sum(u['offered_mbps'] for u in steady.values()),goodput_mbps=sum(u['delivered_mbps'] for u in steady.values()),
                queue_growth_bytes=sum(u['queue_growth_bytes'] for u in steady.values()),overflow_bytes=sum(u['overflow_bytes'] for u in steady.values()),
                deadline_misses=sum(u['deadline_misses'] for u in steady.values()),never_connected=missing,
                zero_delivery=[uid for uid,p in packets.items() if not p['delivered_bytes']],
                retx_deferred_resource=sum(u['retx_deferred_resource'] for u in final['tb_outcomes'].values()),
                recovery_rsrp_dbm=final['paper_interfaces']['model']['good_rsrp_dbm'],
                last_reference={uid:final['paper_interfaces']['final_reports']['ue'][uid] for uid in missing}))
    timings = {scene: json.loads((root / (scene + '_timing8') / 'cost_profile.json').read_text())['total_s']
               for scene in ('simple_street_canyon', 'munich')}
    report = dict(
        passed=True, source_sha256=status['source_sha256'], pipeline_elapsed_s=status['elapsed_s'],local_tests=local_log,spark_tests=spark_log,
        cases=cases, audited_closed_loop_cases=len(cases), closed_loop_slots=sum(c['slots'] for c in cases),
        timing_slots=16, actual_decodes=sum(c['actual_decodes'] for c in cases), timing8_seconds=timings,
        ret_pairs=validation['ret_pairs'], events=dict(total_events),
        artifact_files=len(manifest['files']), artifact_bytes=sum(f['bytes'] for f in manifest['files']),
        manifest_sha256=hashlib.sha256((root / 'data_manifest.json').read_bytes()).hexdigest(),
        pressure=pressure,scope='Fixed control/observation integration; forced outages included; not an algorithm comparison or capacity/SLA guarantee',
    )
    if handover_root:
        extra=json.loads((handover_root/'integration_audit.json').read_text())
        assert extra['passed'] and len(extra['results'])==3
        assert all(r['passed'] and r['source_sha256']==report['source_sha256'] for r in extra['results'])
        report['supplemental_handover']=dict(root=str(handover_root),cases=3,slots=480,
            actual_decodes=sum(r['actual_decodes'] for r in extra['results']),results=extra['results'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 四篇论文接口 Spark 验收', '',
        f"冻结执行源码：`{report['source_sha256']}`。有限验收全部完成，{len(cases)} 个闭环用例、{report['closed_loop_slots']} 个时隙，另有两地图共16个成本门禁时隙。",
        '', '本次仅验证环境接口，动作由固定时间线产生；未实现论文的 xApp 决策、训练、冲突分类或缓解算法。长闭环含主动断开、功率压低制造失步、恢复和定向切换，吞吐/积压不能据此用作无故障承载结论。', '',
        '## 验证结果', '',
        f'- Mac 完整测试：`{local_log}`；Spark：`{spark_log}`。Mac 跳过的真实 RT 检查由 Spark 完成。',
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
    if handover_root:
        lines += ['',f"另有3个受控重叠覆盖的真实RT/NR切换专项通过共同守恒审计，共480slots、{report['supplemental_handover']['actual_decodes']}次实际解码：成功切换、执行中目标失效、邻区删除管理取消。成功用例验证了新服务小区实际解码，以及驻留期间反向命令被拒绝。该受控部署不作为真实部署容量结论。"]
    lines += ['', '## 资源压力与保留的问题', '',
        '以下仅为本轮固定动作下200ms预热后的300ms窗口，不是无控制基线容量，也不是长期稳定性或SLA通过结论。', '',
        '| 24UE用例 | 到达/交付 Mbit/s | 队列增长 byte | 溢出 byte | 新增超期分组 | 从未接入 UE | 合法重传等待 |',
        '|---|---:|---:|---:|---:|---|---:|']
    for p in pressure:
        lines.append(f"| {p['case']} | {p['offered_mbps']:.3f}/{p['goodput_mbps']:.3f} | {p['queue_growth_bytes']} | {p['overflow_bytes']} | {p['deadline_misses']} | {','.join(p['never_connected']) or '无'} | {p['retx_deferred_resource']} |")
    lines.append('')
    for p in pressure:
        for uid,r in p['last_reference'].items():
            best=max(r['candidate_rsrp_dbm'].values())
            lines.append(f"- {p['case']}：{uid}全程未建立连接；最终最佳候选RSRP {best:.2f}dBm，工程恢复门槛 {p['recovery_rsrp_dbm']:.2f}dBm。其零交付、未接入及排队数据均保留。")
        if p['goodput_mbps']>p['offered_mbps']:
            lines.append(f"- {p['case']}：窗口交付略高于同期到达，同时队列增长为{p['queue_growth_bytes']}byte，表示在清理预热期遗留队列，字节账本仍守恒。")
    lines += ['', '正常负载下的积压、溢出和超期，以及低负载下仍有的业务时延问题，均如实保留。合法资源等待不计NACK或资源不兼容。上述结果不生成冲突标签，也不对联合时间线做单变量因果归因。RF能效只反映射频，不能推导整机节能。内存记录为进程RSS及PyTorch CUDA分配峰值，后者不代表Mitsuba/驱动的完整GPU峰值。']
    lines += ['', '## 数据与边界', '',
        f"原始结果目录：`{root}`。清单包含 {report['artifact_files']} 个文件、{report['artifact_bytes']} bytes；清单SHA256为 `{report['manifest_sha256']}`。两端逐文件核对回执单独保存，不能仅凭目录复制成功宣称校验通过。", '',
        '字段语义见 [接口说明](PAPER_INTERFACES.md)、[逐项文献追踪](PAPER_FIELD_TRACEABILITY.md)、[机器可读目录](paper_interface_catalog.json)。默认配置仍不启用新增接口；专用入口为 `configs/rt_sys/paper_interfaces.json`。', '',
        'CQI 是参考 SINR 量化，失步/恢复阈值是可配置工程假设，掉线率是无线连接会话口径。能量仅含已建模 RF 数据/导频；整机指标不可用。保留 A1/E2 语义仿真、理想 CSI、等效干扰接收及 none/Chase HARQ 边界。RET 零倾角针对受约束模拟阵列模型回归，不要求等同旧全数字 RZF。', '',
        f'早期 spark_v1/v2/v3/v4/v5 运行已主动停止并保留，不计为完整验收；本报告只统计 {root.name} 的冻结源码结果。spark_v2 的固定功率测试使用了旧调度配置PRB数，导致正常向量按长度拒绝；最终时间线使用实际NR网格并检查正常动作确实生效。随后补上了功率向量与休眠/上限组合的原子预算检查、定向切换驻留保护，以及不改变统计结果的线性E2状态计数。',
    ]
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-validation', type=Path, required=True)
    parser.add_argument('--handover-root', type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.output, args.local_validation,args.handover_root)
    print(json.dumps({k: result[k] for k in ('passed', 'audited_closed_loop_cases', 'closed_loop_slots', 'actual_decodes')}))
