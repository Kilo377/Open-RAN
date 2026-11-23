# Runtime snapshot 20261123

This snapshot preserves the current `validated_common_v1` network profiles and
the independently measured parameter-to-KPI relationships used by the SMO.

- `network/`: the four original traffic, UE-service, BS-service and base profiles.
- `calibration/`: 239 conditional first-order estimates over 95 P–K pairs,
  experimental design, audited measurements and baseline-origin equivalence.
  The calibration covers 81 fixed-parameter runs and three independent seeds.
- `provenance/`: original calibration certificate and runtime recipe, unchanged.
- `snapshot.json`: relative artifact paths, SHA-256 checksums and original sources.

All copied experiment files retain their original bytes. The certificates
describe the archived experiment and its original physical source; paths inside
them may refer to the original workstation or workers. The main UI used 90%
relation-level screening while retaining the original estimate intervals and
conditional contexts. Loading the matrix in the standalone graph displays the
supplied evidence; it does not recreate the certified multi-scenario UI.

Models are versioned separately under `artifacts/xapps/chongyu/20261123/`.
Raw run recordings remain in the external ORAN archives and are not removed or
uploaded by this snapshot. No intent-combination deployment outcomes are used
to fit the P–K matrix.

See [the dual-AGX workflow](../../../docs/rt_sys/DUAL_AGX_WORKFLOW.md).
