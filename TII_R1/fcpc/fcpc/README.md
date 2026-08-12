# FCPC Reproducibility Workspace

This directory is the reconstruction and revision workspace for the FCPC
resubmission. It is not yet a reproduction package for every number in the
rejected manuscript: MOON, FedDyn, FBLG, and FedCFA remain explicit extension
stubs and must not be reported as reproduced from this code.

## Environment

Install the dependencies in `requirements.txt`, then run commands from this
directory so that the `src` package is importable.

```powershell
python -m unittest discover -s tests -v
python -m src.main --config configs\smoke_synthetic.yaml
```

The synthetic smoke run exercises dataset creation, non-IID partitioning,
LDP metadata perturbation, round-aware pairing, differentiable FCPC
regularization, aggregation, evaluation, checkpointing, resource profiling,
and communication-byte accounting.

## Pairing scalability benchmark

```powershell
python -m scripts.benchmark_pairing `
  --clients 10 50 100 500 1000 `
  --classes 10 `
  --optimal-max-clients 100 `
  --output outputs\benchmarks\pairing.csv
```

Available strategies are:

- `fair_greedy_dissimilar` (revised FCPC default)
- `greedy_dissimilar`
- `optimal`
- `random`
- `greedy_similar`

For odd participating sets, the fair strategy rotates the unpaired
regularization role. The unmatched client still trains and is aggregated.

## Logged system metrics

Each training CSV contains accuracy/loss together with:

- pairing and round wall-clock time;
- process CPU mean/peak utilization;
- peak RSS;
- GPU mean/peak utilization and peak allocated memory when available;
- server download/upload bytes;
- partner-model upload bytes;
- per-round and cumulative model traffic.

The byte accounting treats each accepted pair as two directed full-model
partner uploads. Therefore, with an even full-participation set, FCPC sends
50% more total model payload per round than FedAvg; FCPC does not add a
communication round.

## UCI HAR sensor experiment

The official UCI HAR archive should be extracted to:

```text
data/uci_har/UCI HAR Dataset/
```

Then validate or run:

```powershell
python -m src.main --config configs\uci_har_natural_fcpc.yaml --dry-run
python -m src.main --config configs\uci_har_natural_fcpc.yaml
```

The loader uses the nine raw inertial-signal channels (128 readings per
window), and the natural subject identifiers define FL clients. The official
download is available from:

```text
https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip
```

A partial archive in `data/downloads/uci_har.zip` can be resumed with
`curl.exe -L -C - <URL> -o data\downloads\uci_har.zip`.

## Known work remaining before submission

1. Implement or integrate exact official baseline code under a common
   experiment protocol.
2. Add multi-seed sweep orchestration and statistical summaries.
3. Add feature-skew and distributed concept-drift schedules.
4. Run the full pairing ablation and resource/communication study.
5. Freeze verified results before updating result tables and claims.
