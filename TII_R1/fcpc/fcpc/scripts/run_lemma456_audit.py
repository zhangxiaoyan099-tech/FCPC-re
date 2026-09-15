"""Frozen-checkpoint empirical audit of FCPC-grad Lemmas 4--6.

The audit compares gradient-center FCPC (xi=1) with its matched historical-
center counterfactual (xi=0).  Both replays start from the same checkpoint and
consume the same per-client mini-batch trace.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import cos, pi
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.run_at_m_audit import (
    build_loader,
    checkpoint_signature,
    choose_pairing,
    clone_state,
    assert_finite_state,
    compute_client_gradients,
    create_neutral_checkpoints,
    evaluate,
    load_neutral_checkpoint,
    matrices_for_metadata,
    mixed_seed,
    model_factory,
    partner_weight,
    prepare_data,
    prepare_metadata,
    select_device,
)
from src.algorithms import build_algorithm
from src.experiments.lemma456 import compute_gain_metrics, compute_proxy_chain_metrics
from src.fcpc.regularizer import (
    blend_state_centers,
    clip_state_center_to_global,
    pair_update_proxy_center,
    state_l2_norm,
    weighted_state_center,
)
from src.federated.aggregation import fedavg_aggregate
from src.federated.client import Client
from src.utils.config import load_config
from src.utils.seed import set_seed


PAIR_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "client_i",
    "client_j",
    "pair_mass",
    "theta",
    "d_pair_norm",
    "g_pair_norm",
    "delta_pair",
    "epsilon_pair",
    "epsilon_over_gamma",
    "lemma4_condition_rhs",
    "lemma4_condition_slack",
    "lemma4_sufficient_holds",
    "descent_margin",
    "descent_lower_bound",
    "lower_bound_slack",
    "proxy_is_favorable",
    "center_clip_scale",
    "lambda_i",
    "lambda_j",
    "lambda_bar",
    "weighted_proxy_margin",
]

MAIN_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "pair_list",
    "smoothness_L",
    "q_candidate",
    "global_gradient_norm",
    "global_gradient_norm_sq",
    "history_gamma",
    "pair_count",
    "lemma4_sufficient_fraction",
    "proxy_favorable_fraction",
    "all_pairs_lemma4_sufficient",
    "all_pairs_proxy_favorable",
    "P_t_norm",
    "lemma6_projection",
    "lemma6_weighted_margin_sum",
    "lemma6_identity_gap",
    "lemma6_lower_bound",
    "lemma6_lower_bound_slack",
    "P_t_descent_cosine",
    "first_order_q_ratio",
    "Q_internal",
    "Q_proxy_vs_mix0",
    "Q_counterfactual",
    "Q_observed_loss_gain",
    "proxy_Q_over_grad_sq",
    "Q_over_grad_sq",
    "Z_t_norm",
    "trajectory_error_norm",
    "trajectory_retention_ratio",
    "epsilon_trajectory_bound",
    "proxy_q_condition_slack",
    "proxy_q_condition_holds",
    "inequality_rhs",
    "inequality_slack",
    "inequality_holds",
    "mix0_objective_loss",
    "grad_objective_loss",
    "mix0_val_loss",
    "grad_val_loss",
    "mix0_val_acc",
    "grad_val_acc",
    "beta",
    "gradient_mix",
    "step_scale",
    "local_steps",
    "learning_rate",
    "gradient_model_mode",
    "gradient_batch_size",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse compatible neutral checkpoints containing broadcast histories",
    )
    return parser.parse_args()


def _effective_betas(
    data: Mapping[str, Any], pairing, beta: float, weighting: str
) -> dict[int, float]:
    values = {client_id: 0.0 for client_id in range(int(data["num_clients"]))}
    for client_i, client_j in pairing.pairs:
        values[client_i] = float(beta) * partner_weight(
            data["sample_counts"][client_i],
            data["sample_counts"][client_j],
            weighting,
        )
        values[client_j] = float(beta) * partner_weight(
            data["sample_counts"][client_j],
            data["sample_counts"][client_i],
            weighting,
        )
    return values


def _scheduled_beta(replay: Mapping[str, Any], checkpoint_round: int) -> float:
    """Use the coefficient of the next replayed round on the training schedule."""
    base = float(replay.get("beta", 0.2))
    minimum = float(replay.get("min_beta", 0.0))
    schedule = str(replay.get("beta_schedule", "constant")).lower()
    if schedule in {"constant", "none"}:
        return base
    if schedule in {"cosine", "cosine_decay"}:
        total_rounds = int(replay.get("beta_schedule_rounds", 200))
        progress = int(checkpoint_round) / max(total_rounds - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        return minimum + 0.5 * (base - minimum) * (1.0 + cos(pi * progress))
    raise ValueError(f"unsupported replay beta schedule: {schedule}")


def _replay(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    pairing,
    *,
    gradient_mix: float,
    batch_seed: int,
    device: str,
    center_base: str = "history",
    pair_gate_values: Mapping[tuple[int, int], float] | None = None,
) -> tuple[
    dict[int, Mapping[str, object]],
    Mapping[str, object],
    dict[int, float],
    dict[tuple[int, int], float],
]:
    replay = config["audit"]["replay"]
    global_state = checkpoint["model_state"]
    previous_states = checkpoint["client_previous_states"]
    previous_global_states = checkpoint.get("client_previous_global_states")
    if previous_global_states is None:
        raise ValueError(
            "checkpoint lacks client_previous_global_states; regenerate it without "
            "--reuse-checkpoints so d_i = endpoint_i - broadcast_i is identifiable"
        )
    probe_model = model_factory(config, data)
    parameter_names = {name for name, _ in probe_model.named_parameters()}
    global_norm = state_l2_norm(global_state, parameter_names=parameter_names)
    relative_limit = replay.get("center_max_relative_distance")
    absolute_limit = replay.get("center_max_distance")
    if absolute_limit not in (None, ""):
        center_limit = float(absolute_limit)
    elif relative_limit not in (None, ""):
        center_limit = float(relative_limit) * global_norm
    else:
        center_limit = None

    beta = _scheduled_beta(replay, int(checkpoint["round"]))
    weighting = str(replay.get("partner_weighting", "uniform"))
    step_scale = float(replay.get("grad_center_step_scale", 0.5))
    effective_betas = _effective_betas(data, pairing, beta, weighting)
    references: dict[int, Mapping[str, object]] = {}
    clip_scales: dict[tuple[int, int], float] = {}
    center_base = str(center_base).lower()
    if center_base not in {"history", "global"}:
        raise ValueError("center_base must be 'history' or 'global'")
    gate_values = pair_gate_values or {}
    for client_i, client_j in pairing.pairs:
        history_center = weighted_state_center(
            previous_states[client_i],
            previous_states[client_j],
            data["sample_counts"][client_i],
            data["sample_counts"][client_j],
            fallback_state=global_state,
        )
        gradient_center = pair_update_proxy_center(
            previous_states[client_i],
            previous_states[client_j],
            previous_global_states[client_i],
            previous_global_states[client_j],
            data["sample_counts"][client_i],
            data["sample_counts"][client_j],
            global_state,
            step_scale=step_scale,
        )
        base_center = history_center if center_base == "history" else global_state
        key = (min(client_i, client_j), max(client_i, client_j))
        gate = float(gate_values.get(key, 1.0))
        if not 0.0 <= gate <= 1.0:
            raise ValueError("pair gate values must be in [0, 1]")
        center = blend_state_centers(
            base_center,
            gradient_center,
            gradient_mix=float(gradient_mix) * gate,
        )
        center, _, clip_scale = clip_state_center_to_global(
            center,
            global_state,
            max_distance=center_limit,
            parameter_names=parameter_names,
        )
        references[client_i] = center
        references[client_j] = center
        clip_scales[key] = float(clip_scale)

    optimizer = replay.get("optimizer", {})
    if str(optimizer.get("name", "sgd")).lower() != "sgd":
        raise ValueError("Lemma 4--6 replay requires plain SGD")
    lr = float(optimizer.get("lr", 0.03))
    momentum = float(optimizer.get("momentum", 0.0))
    weight_decay = float(optimizer.get("weight_decay", 0.0))
    if momentum != 0.0 or weight_decay != 0.0:
        raise ValueError("Lemma 4--6 replay requires momentum=weight_decay=0")
    local_steps = int(replay.get("local_steps", 2))
    if local_steps <= 0:
        raise ValueError("audit.replay.local_steps must be positive")
    batch_size = int(replay.get("batch_size", 64))
    if int(replay.get("num_workers", 0)) != 0:
        raise ValueError("matched counterfactual replay requires num_workers=0")
    mean_count = float(np.mean(list(data["sample_counts"].values())))
    algorithm = build_algorithm("fedavg")
    gradient_model_mode = str(
        config.get("audit", {}).get("gradient_model_mode", "eval")
    ).lower()
    local_states: dict[int, Mapping[str, object]] = {}
    for client_id in range(int(data["num_clients"])):
        loader = build_loader(
            data["dataset"],
            data["client_indices"][client_id],
            batch_size=batch_size,
            shuffle=True,
            seed=mixed_seed(batch_seed, client_id),
            num_workers=0,
            pin_memory=bool(config.get("federated", {}).get("pin_memory", False)),
        )
        client = Client(
            client_id=client_id,
            train_loader=loader,
            sample_count=int(data["sample_counts"][client_id]),
            label_histogram=data["histograms"][client_id],
        )
        client.previous_state = clone_state(previous_states[client_id])
        client.previous_global_state = clone_state(previous_global_states[client_id])
        set_seed(mixed_seed(batch_seed + 1_000_000, client_id))
        state, metrics = client.local_train(
            model_factory(config, data),
            algorithm,
            global_state,
            paired_previous_state=references.get(client_id),
            use_fcpc=client_id in pairing.pair_map,
            beta=effective_betas[client_id],
            lr=lr,
            optimizer_name="sgd",
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            local_epochs=1,
            device=device,
            max_batches=local_steps,
            mean_sample_count=mean_count,
            fcpc_update_rule="proximal",
            freeze_batchnorm_stats=gradient_model_mode == "eval",
            return_metrics=True,
        )
        if int(metrics["processed_batches"]) != local_steps:
            raise ValueError(
                f"client {client_id} produced {metrics['processed_batches']} batches; "
                f"the audit requires exactly {local_steps} equal steps"
            )
        local_states[client_id] = state
    aggregate = fedavg_aggregate(
        [local_states[index] for index in range(int(data["num_clients"]))],
        [data["sample_counts"][index] for index in range(int(data["num_clients"]))],
        weighted=True,
    )
    return local_states, aggregate, effective_betas, clip_scales


def _validate_history_config(config: Mapping[str, Any]) -> float:
    warmup = config["audit"]["warmup"]
    optimizer = warmup.get("optimizer", {})
    if str(optimizer.get("name", "sgd")).lower() != "sgd":
        raise ValueError("controlled history requires SGD")
    if float(optimizer.get("momentum", 0.0)) != 0.0:
        raise ValueError("controlled history requires momentum=0")
    if float(optimizer.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("controlled history requires weight_decay=0")
    steps = warmup.get("max_batches_per_client")
    if steps in (None, 0):
        raise ValueError("set warmup.max_batches_per_client to a fixed positive step count")
    if int(warmup.get("local_epochs", 1)) != 1:
        raise ValueError("controlled history currently requires local_epochs=1")
    return float(optimizer.get("lr", 0.03)) * int(steps)


def _evaluate_gradient_probe_objective(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    state: Mapping[str, object],
    device: str,
) -> float:
    """Evaluate the same weighted empirical objective used by the gradient probe."""
    import torch
    from torch import nn

    audit = config.get("audit", {})
    batch_size = int(audit.get("gradient_batch_size", 128))
    max_batches = audit.get("gradient_max_batches")
    max_batches = None if max_batches in (None, 0) else int(max_batches)
    counts = {key: float(value) for key, value in data["sample_counts"].items()}
    total_count = sum(counts.values())
    model = model_factory(config, data)
    model.load_state_dict(state)
    model.to(device)
    model_mode = str(audit.get("gradient_model_mode", "eval")).lower()
    if model_mode == "train":
        model.train()
    elif model_mode == "eval":
        model.eval()
    else:
        raise ValueError("audit.gradient_model_mode must be 'eval' or 'train'")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    objective = 0.0
    with torch.no_grad():
        for client_id in range(int(data["num_clients"])):
            indices = list(data["client_indices"][client_id])
            if max_batches is not None:
                indices = indices[: batch_size * max_batches]
            loader = build_loader(
                data["dataset"],
                indices,
                batch_size=batch_size,
                shuffle=False,
                seed=0,
                # Repeated counterfactual evaluation would otherwise create a
                # fresh worker pool per client and dominate the audit runtime.
                num_workers=0,
                pin_memory=bool(config.get("federated", {}).get("pin_memory", False)),
            )
            local_sum = 0.0
            local_count = 0
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=device.startswith("cuda"))
                targets = targets.to(device, non_blocking=device.startswith("cuda"))
                local_sum += float(criterion(model(inputs), targets).item())
                local_count += int(targets.numel())
            if local_count <= 0:
                raise ValueError(f"client {client_id} has no objective-probe examples")
            objective += (counts[client_id] / total_count) * (local_sum / local_count)
    return float(objective)


def _write_summary(rows: list[Mapping[str, Any]], path: Path) -> None:
    keys = ("checkpoint_round", "panel", "pairing_strategy", "smoothness_L")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    fields = [
        *keys,
        "n",
        "lemma4_sufficient_fraction_mean",
        "proxy_favorable_fraction_mean",
        "lemma6_projection_mean",
        "P_t_descent_cosine_mean",
        "Q_counterfactual_mean",
        "Q_counterfactual_std",
        "Q_observed_loss_gain_mean",
        "epsilon_trajectory_mean",
        "q_max_with_epsilon",
        "proxy_q_ratio_mean",
        "q_no_epsilon",
        "proxy_q_condition_hold_fraction",
        "inequality_hold_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(grouped.items()):
            array = lambda name: np.asarray(
                [float(row[name]) for row in values], dtype=np.float64
            )
            q_values = array("Q_counterfactual")
            epsilon_values = array("epsilon_trajectory_bound")
            grad_sq = float(values[0]["global_gradient_norm_sq"])
            writer.writerow(
                {
                    **dict(zip(keys, key)),
                    "n": len(values),
                    "lemma4_sufficient_fraction_mean": array(
                        "lemma4_sufficient_fraction"
                    ).mean(),
                    "proxy_favorable_fraction_mean": array(
                        "proxy_favorable_fraction"
                    ).mean(),
                    "lemma6_projection_mean": array("lemma6_projection").mean(),
                    "P_t_descent_cosine_mean": array("P_t_descent_cosine").mean(),
                    "Q_counterfactual_mean": q_values.mean(),
                    "Q_counterfactual_std": q_values.std(ddof=0),
                    "Q_observed_loss_gain_mean": array(
                        "Q_observed_loss_gain"
                    ).mean(),
                    "epsilon_trajectory_mean": epsilon_values.mean(),
                    "q_max_with_epsilon": (q_values.mean() + epsilon_values.mean())
                    / (grad_sq + 1e-12),
                    "proxy_q_ratio_mean": array("proxy_Q_over_grad_sq").mean(),
                    "q_no_epsilon": q_values.mean() / (grad_sq + 1e-12),
                    "proxy_q_condition_hold_fraction": array(
                        "proxy_q_condition_holds"
                    ).mean(),
                    "inequality_hold_fraction": array("inequality_holds").mean(),
                }
            )


def run(config: Mapping[str, Any], *, reuse_checkpoints: bool) -> dict[str, str]:
    seed = int(config.get("seed", 42))
    audit = config.get("audit", {})
    output_dir = Path(audit.get("output_dir", "outputs/lemma456_audit"))
    checkpoint_dir = Path(
        audit.get("checkpoint_dir", output_dir / "neutral_checkpoints")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    history_gamma = _validate_history_config(config)
    data = prepare_data(config)
    device = select_device(str(config.get("federated", {}).get("device", "auto")))
    print(f"runtime_device: {device}; history_gamma={history_gamma:.6e}", flush=True)
    if not reuse_checkpoints:
        create_neutral_checkpoints(config, data, checkpoint_dir)
    signature = checkpoint_signature(config)
    metadata = prepare_metadata(
        data,
        epsilon=float(audit.get("epsilon", 1.0)),
        ldp_seed=int(audit.get("ldp_seed", seed + 20_000)),
    )
    panels = [str(value).lower() for value in audit.get("panels", ["raw"])]
    strategies = [str(value).lower() for value in audit.get("strategies", ["optimal"])]
    pairing_seeds = [int(value) for value in audit.get("random_pairing_seeds", [0])]
    batch_seeds = [int(value) for value in audit.get("batch_seeds", [100, 101, 102])]
    checkpoints = [int(value) for value in audit.get("checkpoints", [5, 20, 50])]
    l_values = [float(value) for value in audit.get("smoothness_L_values", [0.0])]
    q_candidate = float(audit.get("q_candidate", 0.0))
    replay = audit["replay"]
    gradient_mix = float(replay.get("grad_center_mix", 1.0))
    if gradient_mix != 1.0:
        raise ValueError("the primary Lemma 4--6 audit requires grad_center_mix=1")
    step_scale = float(replay.get("grad_center_step_scale", 0.5))
    local_steps = int(replay.get("local_steps", 2))
    learning_rate = float(replay.get("optimizer", {}).get("lr", 0.03))
    parameter_names = [name for name, _ in model_factory(config, data).named_parameters()]

    metrics_path = output_dir / "lemma456_metrics.csv"
    pairs_path = output_dir / "lemma456_pairs.csv"
    summary_path = output_dir / "lemma456_summary.csv"
    all_rows: list[dict[str, Any]] = []
    with (
        metrics_path.open("w", newline="", encoding="utf-8") as metrics_file,
        pairs_path.open("w", newline="", encoding="utf-8") as pairs_file,
    ):
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=MAIN_FIELDS)
        pair_writer = csv.DictWriter(pairs_file, fieldnames=PAIR_FIELDS)
        metrics_writer.writeheader()
        pair_writer.writeheader()
        for checkpoint_round in checkpoints:
            checkpoint = load_neutral_checkpoint(
                checkpoint_dir / f"fedavg_round_{checkpoint_round:03d}.pt", signature
            )
            if "client_previous_global_states" not in checkpoint:
                raise ValueError(
                    "neutral checkpoints predate the Lemma 4--6 audit; rerun without "
                    "--reuse-checkpoints"
                )
            assert_finite_state(
                checkpoint["model_state"],
                where=f"loaded neutral checkpoint round {checkpoint_round}",
            )
            gradients = compute_client_gradients(
                config, data, checkpoint["model_state"], device
            )
            for panel in panels:
                matrices = matrices_for_metadata(
                    metadata[panel], lambda_jsdn=float(audit.get("lambda_jsdn", 0.3))
                )
                jobs = []
                for strategy in strategies:
                    seeds = pairing_seeds if strategy == "random" else [seed]
                    jobs.extend((strategy, pairing_seed) for pairing_seed in seeds)
                for strategy, pairing_seed in jobs:
                    pairing, _, _ = choose_pairing(
                        strategy, matrices, seed=pairing_seed
                    )
                    for batch_seed in batch_seeds:
                        mix0_states, mix0_aggregate, _, _ = _replay(
                            config,
                            data,
                            checkpoint,
                            pairing,
                            gradient_mix=0.0,
                            batch_seed=batch_seed,
                            device=device,
                        )
                        grad_states, grad_aggregate, effective_betas, clip_scales = _replay(
                            config,
                            data,
                            checkpoint,
                            pairing,
                            gradient_mix=1.0,
                            batch_seed=batch_seed,
                            device=device,
                        )
                        chain, pair_rows, proxy_component = compute_proxy_chain_metrics(
                            global_state=checkpoint["model_state"],
                            previous_states=checkpoint["client_previous_states"],
                            previous_global_states=checkpoint[
                                "client_previous_global_states"
                            ],
                            client_gradients=gradients,
                            sample_counts=data["sample_counts"],
                            pairing=pairing,
                            parameter_names=parameter_names,
                            history_gamma=history_gamma,
                            learning_rate=learning_rate,
                            local_steps=local_steps,
                            effective_betas=effective_betas,
                            gradient_mix=gradient_mix,
                            step_scale=step_scale,
                            pair_clip_scales=clip_scales,
                        )
                        mix0_validation = evaluate(config, data, mix0_aggregate, device)
                        grad_validation = evaluate(config, data, grad_aggregate, device)
                        mix0_objective = _evaluate_gradient_probe_objective(
                            config, data, mix0_aggregate, device
                        )
                        grad_objective = _evaluate_gradient_probe_objective(
                            config, data, grad_aggregate, device
                        )
                        common = {
                            "model_seed": seed,
                            "checkpoint_round": checkpoint_round,
                            "panel": panel,
                            "pairing_strategy": strategy,
                            "pairing_seed": pairing_seed if strategy == "random" else "",
                            "batch_seed": batch_seed,
                            "pair_list": json.dumps(pairing.pairs, separators=(",", ":")),
                            **chain,
                            "mix0_val_loss": mix0_validation["loss"],
                            "grad_val_loss": grad_validation["loss"],
                            "mix0_objective_loss": mix0_objective,
                            "grad_objective_loss": grad_objective,
                            "mix0_val_acc": mix0_validation["acc"],
                            "grad_val_acc": grad_validation["acc"],
                            "beta": _scheduled_beta(replay, checkpoint_round),
                            "gradient_mix": gradient_mix,
                            "step_scale": step_scale,
                            "local_steps": local_steps,
                            "learning_rate": learning_rate,
                            "gradient_model_mode": str(
                                audit.get("gradient_model_mode", "eval")
                            ).lower(),
                            "gradient_batch_size": int(
                                audit.get("gradient_batch_size", 128)
                            ),
                        }
                        for smoothness_l in l_values:
                            gain = compute_gain_metrics(
                                global_state=checkpoint["model_state"],
                                grad_states=grad_states,
                                baseline_states=mix0_states,
                                client_gradients=gradients,
                                sample_counts=data["sample_counts"],
                                parameter_names=parameter_names,
                                proxy_component=proxy_component,
                                smoothness_l=smoothness_l,
                                observed_loss_grad=grad_objective,
                                observed_loss_baseline=mix0_objective,
                                q_candidate=q_candidate,
                            )
                            row = {**common, **gain}
                            metrics_writer.writerow(
                                {field: row.get(field, "") for field in MAIN_FIELDS}
                            )
                            all_rows.append(row)
                        metrics_file.flush()
                        for pair_row in pair_rows:
                            enriched = {**common, **pair_row}
                            pair_writer.writerow(
                                {field: enriched.get(field, "") for field in PAIR_FIELDS}
                            )
                        pairs_file.flush()
                        print(
                            f"lemma456: t={checkpoint_round}, panel={panel}, "
                            f"strategy={strategy}, batch_seed={batch_seed}, "
                            f"L4={chain['lemma4_sufficient_fraction']:.1%}, "
                            f"favorable={chain['proxy_favorable_fraction']:.1%}, "
                            f"-<g,P>={chain['lemma6_projection']:+.3e}, "
                            f"observed_Q={mix0_objective - grad_objective:+.3e}",
                            flush=True,
                        )
    _write_summary(all_rows, summary_path)
    return {
        "metrics_path": str(metrics_path),
        "pairs_path": str(pairs_path),
        "summary_path": str(summary_path),
        "checkpoint_dir": str(checkpoint_dir),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run(config, reuse_checkpoints=args.reuse_checkpoints)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
