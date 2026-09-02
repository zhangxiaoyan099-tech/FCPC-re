# New FCPC beta ablation

## Purpose

Use CIFAR-10 development seed 42 to select the initial New-FCPC regularization
coefficient `beta`.  Every other setting is inherited from
`configs/cifar10_full_comparison_base.yaml`:

- all 45,000 post-validation training samples are assigned exactly once;
- 10 clients, 6 clients per round, 200 rounds and 1 local epoch;
- ResNet-18, batch size 128 and CUDA device 0;
- weighted-JS pair complementarity, optimal matching and pair-center reference;
- cosine learning-rate schedule and cosine beta decay to zero.

The default grid is:

```text
0.005, 0.01, 0.02, 0.05, 0.1, 0.2
```

The subsequent refinement on development seed 42 tested `0.001`, `0.0025`
and `0.0075`.  Validation accuracy selected `beta = 0.001` under the existing
cosine decay to zero.  This value is frozen for formal seeds 43, 44 and 45;
seed 42 is not included in the final multi-seed estimate.

Only validation accuracy may be used to select beta.  Do not inspect or use
test accuracy while making this choice.

## Smoke test

```bash
cd ~/FCPC-re/TII_R1/fcpc/fcpc

python scripts/run_new_fcpc_beta_ablation.py \
  --betas 0.01 \
  --seed 42 \
  --rounds 2 \
  --force
```

## Full sequential run

Use only one training process on the single RTX 3090:

```bash
mkdir -p outputs/cifar10_beta_ablation

nohup python -u scripts/run_new_fcpc_beta_ablation.py \
  --betas 0.005,0.01,0.02,0.05,0.1,0.2 \
  --seed 42 \
  > outputs/cifar10_beta_ablation/beta_seed42_runner.log 2>&1 &

echo $!
```

Do not add `--force` to the formal command.  A completed beta run is skipped
automatically; an incomplete run is restarted from round 1 because the current
trainer does not resume local/server optimizer state from a partial checkpoint.

## Monitoring

```bash
tail -f outputs/cifar10_beta_ablation/beta_seed42_runner.log
```

Press `Ctrl+C` to leave `tail`; training continues.

```bash
watch -n 2 nvidia-smi
```

```bash
wc -l outputs/cifar10_beta_ablation/logs/*.csv
```

A complete 200-round CSV has 201 lines, including the header.

## Selection rule

The runner prints the validation-only table after all runs:

```text
VALIDATION-ONLY SUMMARY
beta    best_round    best_val_acc
...
SELECTED_BY_VALIDATION beta=...
```

If two beta values have exactly the same best validation accuracy, the script
chooses the smaller beta.  After beta is frozen, run fresh formal seeds and
report mean and standard deviation of the validation-selected checkpoint test
accuracy.  Seed 42 is a development run and should not be reused as an
independent final test seed after tuning.
