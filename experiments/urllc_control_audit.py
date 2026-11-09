"""Read-only URLLC control-receipt classification for a moving UE network.

Only a recorded, atomically rejected stale-membership transaction can be
classified as a handover no-op. Raw receipts remain intact; none is rewritten,
filtered from ACK checks, converted into an applied action or training label.
"""
from collections import Counter
import math

AUDIT_VERSION = 'membership-noop-v1'
IMPLEMENTATION_ID = 'latency_urllc_rf'
USERS = frozenset(('ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'))


def is_urllc_receipt(row):
    provenance = row.get('provenance', [])
    provenance = provenance if isinstance(provenance, list) else [provenance]
    return any(isinstance(p, dict) and p.get('implementation_id') == IMPLEMENTATION_ID
        for p in provenance)


def classify_stale_rejections(receipts, decisions):
    """Return evidence for guarded HO no-ops; fail on every other rejection.

    Membership is proved by the receipt's actual before-state. The exact request
    must occur in a recorded prior decision; before==after proves it had no
    effect. Applied ACKs and later owner confirmations are checked separately.
    """
    classified = []
    for row in receipts:
        if not is_urllc_receipt(row) or row.get('status') == 'applied':
            continue
        if (row.get('status') != 'rejected'
                or row.get('reason') != 'ue_scheduler_weight_target_outside_cell_scope'):
            raise ValueError('Unclassified URLLC control rejection')
        cid = row.get('target_cell_id'); before = row.get('before'); after = row.get('after')
        fields = row.get('fields', [])
        if (row.get('apply_scope') != 'cell' or not isinstance(cid, str)
                or row.get('target_cell_ids') != [cid] or not isinstance(before, dict)
                or before != after or cid not in before or not isinstance(before[cid], dict)
                or not isinstance(fields, list) or len(fields) != 2
                or not all(isinstance(field, dict) for field in fields)
                or {r.get('field') for r in fields} != {'scheduler_mode', 'ue_scheduler_weights'}):
            raise ValueError('Rejected scheduler transaction is not an atomic membership no-op')
        requested = {r['field']: r.get('value') for r in fields}
        weights = requested['ue_scheduler_weights']; mode = requested['scheduler_mode']
        if (mode not in ('round_robin', 'latency_aware', 'qos_deadline')
                or not isinstance(weights, dict) or not set(weights) <= USERS):
            raise ValueError('Rejected transaction has an invalid scheduler scope/mode')
        upper = 4. if mode == 'qos_deadline' else 2.
        if any(type(w) not in (int, float) or not math.isfinite(w) or not .5 <= w <= upper
                for w in weights.values()):
            raise ValueError('Rejected transaction has an invalid scheduler weight')
        actual = before[cid].get('ue_scheduler_weights')
        if not isinstance(actual, dict) or not set(actual) <= USERS:
            raise ValueError('Rejected transaction lacks actual target-cell membership')
        absent = sorted(set(weights)-set(actual)); issued = row.get('issued_at_slot')
        matching = [d for d in decisions if type(d.get('slot_id')) is int and type(issued) is int
            and d['slot_id'] <= issued and d.get('snapshot', {}).get(cid) == requested]
        if not absent or not matching:
            raise ValueError('Rejected transaction is not a recorded stale membership decision')
        classified.append(dict(slot_id=row.get('slot_id'), issued_at_slot=issued, target_cell_id=cid,
            reason=row['reason'], stale_ue_ids=absent, actual_member_ue_ids=sorted(actual),
            matched_decision_slot=max(d['slot_id'] for d in matching), atomic_noop=True))
    return classified


def rejection_audit_metadata(classified):
    """Expose all classified rejected receipts without modifying raw records."""
    return dict(rejection_audit_version=AUDIT_VERSION,
        rejected_receipts_total=len(classified), accepted_stale_scope_noop_rejections=len(classified),
        rejection_reasons=dict(Counter(row['reason'] for row in classified)),
        rejection_classifications=classified, raw_control_receipts_preserved=True,
        all_original_confirmation_ack_and_training_window_checks_retained=True)
