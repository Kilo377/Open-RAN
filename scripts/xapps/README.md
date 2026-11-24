# MEC latency training

Prepare retained real run metrics:

```sh
python scripts/xapps/prepare-mec-study.py --study /path/to/study
```

Fit the paired, seed-balanced contextual policy (NumPy / scikit-learn required):

```sh
python scripts/xapps/train-mec-latency.py --study /path/to/study --version my-model --balanced-seeds --followup-labels --latency-only --empirical-action-filter --minimum-predicted-gain 0
```

`--dataset-file` refits a pinned exported training dataset without adding later runs. `--action-only` fits the noncontextual control. Exported forests require no scikit-learn during inference. Models and optimizer checkpoints for online PPO are recorded by the normal hot-xApp runner when an explicitly separate training policy enables learning. Published releases reject training-mode policies.

The current study uses the unchanged validated 6UE fast/system environment and100ms MEC task deadlines. Development and final heldout seeds are excluded from fitting. The validation report distinguishes actual terminal followup from the unchanged online-window Evaluation.
