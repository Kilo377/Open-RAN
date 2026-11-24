# Competitor-priority action gains

Two additional conditional contexts on existing P–K edges. The graph topology, candidate controls and observed Evaluation are unchanged.

The six matched fixed-control pairs use four independent training seeds. Both arms retain latency-aware scheduling and MEC priority2; only priorities of the same competing population change from1 to0.5. Raw prefix, traffic/service configuration, protected simulator code, task identities, frozen fixed-policy checkpoints and normal control receipts were audited. Three contexts with another competitor population are excluded.

Coefficients use equal seed weights and90% exact seed-block bootstrap intervals. Units are ms/weight and W/weight. The ordinary scalar gains retain precedence. Subtract derivatives of disjoint contained groups from the measured joint derivative, then project the residual onto the uncovered population. A common step of the full group therefore reproduces its measured derivative without double counting. This is a first-order reference approximation, not an identified per-UE causal derivative or a KPI forecast; incompatible overlapping reference groups remain unavailable.

No heldout, learned-policy deployment, or16-combination evaluation results are fitting inputs. The measurements and recomputable certificate remain separate from the original immutable calibration.

Rebuild from retained raw study data:

```sh
python scripts/smo/fit-competitor-priority.py --study /path/to/X12-study --manifest /path/to/base-scenario-manifest.json --output /path/to/new-supplement
```
