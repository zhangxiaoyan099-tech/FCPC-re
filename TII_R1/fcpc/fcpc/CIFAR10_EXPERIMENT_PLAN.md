# CIFAR-10 FedAvg/FCPC execution plan

This plan separates pipeline validation, shared-parameter selection, and the
200-round FedAvg/FCPC comparison. The 200-round runs are exploratory until the
same protocol has been repeated with multiple seeds.

## Key concepts

### Beta

FCPC optimizes

\[
L_{total}=L_{task}+\beta R_{FCPC}.
\]

`beta` controls the strength of the paired-client penalty. `beta=0` removes
the FCPC contribution. A value that is too large can make the paired-model
constraint dominate classification. The CSV therefore records the raw and
weighted terms separately:

- `train_task_loss`;
- `train_fcpc_raw_loss`;
- `train_fcpc_weighted_loss` (`beta * raw`);
- `train_total_loss`.

`beta` is not the Dirichlet `alpha` and is not `lambda_jsdn`. `alpha` controls
label skew, while `lambda_jsdn` controls how label and quantity differences
are combined for pairing.

### Dual skew

`dual_skew` applies label skew followed by quantity skew. Labels are allocated
class-by-class with a Dirichlet distribution. Smaller `alpha` means stronger
label imbalance. Quantity skew then retains different proportions of samples
at different clients. The current `alpha=0.1` setting is strongly non-IID.

## Code-level preparation

The experiment branch provides:

1. optional CIFAR augmentation and normalization;
2. a CIFAR-style ResNet-18 stem for 32x32 inputs;
3. SGD momentum, weight decay and Nesterov options;
4. constant, cosine and step round-level learning-rate schedules;
5. a deterministic stratified validation holdout;
6. separate task, baseline-adapter and FCPC loss logging;
7. final and best-validation checkpoints.

The validation view never uses random augmentation. Configuration selection
uses validation accuracy; the test split is not used every round.

## Generate configurations

Run from the project directory:

```bash
python -m scripts.generate_cifar10_experiments \
  --data-root ./data \
  --selected-lr 0.03 \
  --selected-beta 0.001

mkdir -p outputs/cifar10_plan
```

This creates configurations under `configs/generated_cifar10/`.

## Stage 1: tests and GPU smoke run

```bash
python -m unittest discover -s tests -v
python -m src.main --config configs/smoke_synthetic.yaml
```

The synthetic CSV must contain finite `train_task_loss` and
`train_total_loss`. FCPC runs should have a finite FCPC loss; FedAvg runs must
report zero weighted FCPC loss.

## Stage 2: IID learnability check

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.main \
  --config configs/generated_cifar10/cifar10_iid_fedavg_r10.yaml \
  2>&1 | tee outputs/cifar10_plan/iid-r10.txt
```

Stop and debug the shared training pipeline if accuracy remains near random.
This run is diagnostic and is not the main FCPC comparison.

## Stage 3: 40-round FedAvg learning-rate screen

Run the three `fedavg_lr*_r40.yaml` configurations. Select the learning rate
using the mean validation accuracy over the final five rounds, not a single
test-set spike. Regenerate configs with the selected value:

```bash
python -m scripts.summarize_cifar10_runs \
  --pattern "*fedavg_lr*_r40.csv"
```

```bash
python -m scripts.generate_cifar10_experiments --selected-lr SELECTED_LR
```

## Stage 4: 200-round FedAvg reference

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.main \
  --config configs/generated_cifar10/cifar10_dual_a0p1_fedavg_selected_r200.yaml \
  2>&1 | tee outputs/cifar10_plan/fedavg-selected-r200.txt
```

Archive the YAML, CSV, final checkpoint, best-validation checkpoint, terminal
log, Git commit, and GPU/runtime record together.

## Stage 5: 30-round beta screen

Run the five `fcpc_beta*_r30.yaml` configurations. Inspect both validation
accuracy and

\[
\frac{L_{FCPC,weighted}}{L_{task}}.
\]

A persistent ratio much greater than one is evidence that the paired penalty
dominates classification. Choose `beta` by validation behavior, regenerate
the final configuration, and do not modify shared FedAvg parameters:

```bash
python -m scripts.summarize_cifar10_runs \
  --pattern "*fcpc_beta*_r30.csv"
```

```bash
python -m scripts.generate_cifar10_experiments \
  --selected-lr SELECTED_LR \
  --selected-beta SELECTED_BETA
```

## Stage 6: 200-round FCPC run

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.main \
  --config configs/generated_cifar10/cifar10_dual_a0p1_fcpc_selected_r200.yaml \
  2>&1 | tee outputs/cifar10_plan/fcpc-selected-r200.txt
```

FedAvg and FCPC start independently from the same seed and initialization.
FCPC must not continue from the trained FedAvg checkpoint.

## Stage 7: confirmation

If the seed-42 FCPC run is promising, repeat the frozen protocol with at least
three seeds and report mean plus standard deviation. Keep selection rules and
the checkpoint used for testing fixed before inspecting final test results.
