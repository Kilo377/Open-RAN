# Chongyu release 20261117

X-01 controls all nine existing DRX/offload actions (`deploy_drx=true`). The 221-row frozen RF checkpoint is byte-identical to 20261109; no retraining was performed. DRX profiles are cycle/on-duration/inactivity 40/4/8, 20/4/1 and 80/4/1 ms. URLLC DRX protection remains unchanged.

Two matched evaluation seeds each ran 3 seconds of Base, native-DRX X-01, and DRX-enabled X-01. Relative to Base, enabled X-01 reduced target-group energy by 14.70% and all-six-UE energy by 5.21%; network DL goodput fell 1.47% and mean completed-DL-packet latency rose 10.46%. Relative to the same X-01 with native DRX, target energy fell 1.85% and all-UE energy fell 0.67%; DL goodput fell 1.16%, mean DL packet latency rose 12.30%, and mature-cohort MEC workload task latency rose 4.86%. These are policy-level effects with coupled offload adaptation, not a scalar DRX derivative. Both seeds actually executed a non-native DRX profile.

`evidence/x01.json` contains per-seed measurements, scope, source/model fingerprints and the retained raw-data archive checksum. The former locked-policy evidence remains `evidence/x01-legacy-locked.json`. Two short paired seeds do not establish a population confidence interval or long-run generalization.

The other five forests/MLP/QGBDT checkpoints and their evidence are inherited unchanged. Models, measured training rows, raw recordings and historical releases are preserved.
