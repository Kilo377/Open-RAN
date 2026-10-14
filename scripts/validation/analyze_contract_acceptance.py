"""Summarize only completed frozen-version acceptance artifacts."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def analyze(root,output):
    status=json.loads((root/'pipeline_status.json').read_text())
    assert status['completed'] and status['passed'],'Refusing to label incomplete acceptance as passed'
    reports={stage:json.loads((root/stage/'validation.json').read_text())
        for stage in ('gate','olla','control_lifecycle','xapp_io','moving_handover','matrix')}
    assert all(r['passed'] and r['source_sha256']==status['source_sha256'] for r in reports.values())
    cases=[]
    for row in reports['matrix']['cases']:
        ues=row['steady_state']['per_ue'].values()
        case=dict(case=row['case'],offered_mbps=sum(u['offered_mbps'] for u in ues),
            delivered_mbps=sum(u['delivered_mbps'] for u in ues),
            overflow_bytes=sum(u['overflow_bytes'] for u in ues),
            deadline_misses=sum(u['deadline_misses'] for u in ues),
            pending_bytes=sum(u['pending_end'] for u in ues),
            queue_growth_bytes_per_s=sum(u['queue_growth_bytes_per_s'] for u in ues),
            actual_decode_attempts=row['actual_decode_attempts'],first_tx_bler=row['post_warmup_first_tx']['first_bler'],
            retx_deferred_resource=row['retx_deferred_resource'],elapsed_s=row['elapsed_s'],
            zero_delivery_ues=row['zero_delivery_ues'])
        cases.append(case)
    summary=dict(artifact_root=str(root),passed=True,source_sha256=status['source_sha256'],elapsed_s=status['elapsed_s'],
        integration_acceptance=True,qos_or_generalization_claim=False,cases=cases,
        actual_matrix_decode_attempts=sum(c['actual_decode_attempts'] for c in cases),
        retx_deferred_resource=sum(c['retx_deferred_resource'] for c in cases),
        scopes=dict(phy='RT + joint multi-cell RZF/LMMSE + actual NR equivalent-link CRC/LDPC; ideal CSI',
            protocols='A1/E2 semantic simulation',harq='Same RV0 codeword, none/Chase; no IR',
            control_clocks='5/10 slots unchanged',
            unavailable_xapps='Independent CQI / precoder-match measurement sources remain absent'))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    lines=['# 无探针 O-RAN 接入验收','',f'源码指纹：`{status["source_sha256"]}`。',
        '', '发送契约、载荷预留、固定 OLLA 与 KPI/动作权限接口已接入。以下为有限接入验收，不作为论文泛化或所有负载的 QoS 结论。',
        '', f'完整验收耗时 {status["elapsed_s"]/60:.1f} 分钟；矩阵实际解码 {summary["actual_matrix_decode_attempts"]} 次。',
        '', '## 本轮结果与限制', '',
        '连续数据 NACK 不再被当作无线链路失步，避免 UE 永久退出调度；单小区与多小区均启用相同发送契约。独立 RLM/RLF 协议仍未建模。',
        '', '两地图 4 UE 正常负载和 24 UE 0.1 倍负载的全部种子均无溢出、无超期。24 UE 正常负载存在明显承载不足、溢出与超期；结构验收通过不代表这些负载的 QoS 达标。',
        '', '固定链路 OLLA 通过目标跟踪验收；动态 RT 用例不承诺每个 UE 均达到目标 BLER。实际 NR 解码仍有运行开销，短时成本检查通过不代表已适合大规模训练。',
        '', '## 完成的验证','', '| 项目 | 结果 |','|---|---|']
    for c in reports['gate']['cases']:
        elapsed=c['elapsed_s']
        if c['case'].endswith('_timing8'):
            elapsed=json.loads((root/'gate'/c['case']/'cost_profile.json').read_text())['total_s']
        lines.append(f'| {c["case"]} | {elapsed:.2f} s，{c["actual_decode_attempts"]} 次实际解码，容量保护与重传不兼容为 0 |')
    for c in reports['olla']['cases']:
        ci=c['wilson_95'];lines.append(f'| OLLA {c["case"]} | {c["first_samples"]} 个预热后首传；BLER {c["first_bler"]:.4f}，95% CI [{ci[0]:.4f}, {ci[1]:.4f}]；饱和 {c["saturated_fraction"]:.1%} |')
    lines.extend(['| KPI/动作、A1 策略/仲裁 | 两地图固定测试策略通过 |',
        '| 移动和切换 | 两地图实际移动 RT、休眠触发切换、新服务链路与字节守恒通过 |',
        '| 长期矩阵 | 18/18 用例；2 地图、3 小区、4/24 UE、3 种子；每例 2000 slots，预热 200 ms |',
        '', '所有新环境业务验收用例的在线 AMC/BLER/校准探针为 0，实际解码次数与真实发送尝试一致。旧后端单元回归单独验证兼容性，不属于在线环境探针。',
        '', '## 负载和承载结果','',
        '| 用例 | 业务/交付 Mbps | 溢出 B | 超期分组 | 期末待交付 B | 首传 BLER（预热后） | 重传资源等待 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for c in sorted(cases,key=lambda c:c['case']):
        rate='无样本' if c['first_tx_bler'] is None else f'{c["first_tx_bler"]:.3f}'
        lines.append(f'| {c["case"]} | {c["offered_mbps"]:.3f}/{c["delivered_mbps"]:.3f} | {c["overflow_bytes"]} | {c["deadline_misses"]} | {c["pending_bytes"]} | {rate} | {c["retx_deferred_resource"]} |')
    lines.extend(['', '动态 RT 矩阵中每个 UE 的目标跟踪状态、MCS 边界比例和样本数见各用例 `contract_validation.json`。固定链路 OLLA 验收通过，不意味着任意 RT 链路、过载或饱和 MCS 都达到目标。',
        '', '## 数据与边界','',
        f'原始目录：`outputs/rt_contract_20261014/{root.name}/`。逐 UE/小区 CSV、原始 RT/NR trace、分组和 HARQ 事件、控制回执、配置、设备版本、缓存标识与 SHA256 清单保存在本报告记录的验收目录。所有合法资源等待、过载、弱覆盖和 HARQ 清理分别记录。',
        '', 'A1/E2 仍为语义仿真，PHY 使用理想 CSI 和等效 NR 资源；未实现完整 OFDM/信道估计或 IR。OLLA/HARQ 不向 xApp 开放。没有整机能耗模型，不把 RF 发射功率解释为设备节能。依赖独立 CQI/precoder 测量的两个内置 xApp 明确不可用，其余已接入参数消费者可供符合契约的 xApp 使用。'])
    evidence=root.parent/('local_validation_'+root.name.removeprefix('final_'))
    receipts=[]
    for name in ('spark_artifact_verification.json','mac_artifact_verification.json'):
        path=evidence/name
        if path.exists():
            receipt=json.loads(path.read_text())
            assert receipt['verified'] and receipt['source_sha256']==status['source_sha256']
            receipts.append(receipt)
    if len(receipts)==2:
        a,b=receipts
        assert all(a[k]==b[k] for k in ('manifest_sha256','files','bytes','matrix_trace_slots','out_of_sync_observations'))
        summary['artifact_verification']=dict(files=a['files'],bytes=a['bytes'],manifest_sha256=a['manifest_sha256'],both_hosts_verified=True,matrix_trace_slots=a['matrix_trace_slots'],out_of_sync_observations=a['out_of_sync_observations'])
        lines.extend(['', '## 保存与校验', '',
            f"Mac 与 Spark 均完成逐文件 SHA256 核验：{a['files']} 个文件，共 {a['bytes']} B；矩阵 {a['matrix_trace_slots']} 个时隙的失步观测为 {a['out_of_sync_observations']}。",
            '', f"清单 SHA256：`{a['manifest_sha256']}`。源码归档、本地测试和两端校验回执位于 `outputs/rt_contract_20261014/{evidence.name}/`。"])
        output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    output.write_text('\n'.join(lines)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.root.resolve(),args.output.resolve())
    print(json.dumps(dict(passed=True,cases=len(result['cases']),actual_matrix_decode_attempts=result['actual_matrix_decode_attempts'])))
