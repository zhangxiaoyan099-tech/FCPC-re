"""Clean counterfactual audit for an oracle-gated FCPC-grad proxy.

``proxy_off`` and every candidate use the same proximal update.  The only
difference is the displacement of a pair centre from the current global model:

    proxy_off: c_p = w_t
    ungated:   c_p = w_t + s d_p
    oracle:    c_p = w_t + s 1{-<g_t,d_p> > 0} d_p

The oracle sees the empirical global gradient and is therefore diagnostic,
not deployable.  Its purpose is to separate wrong proxy directions from
proximal-transfer and trajectory-response failures.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.run_at_m_audit import (
    assert_finite_state,
    checkpoint_signature,
    choose_pairing,
    compute_client_gradients,
    create_neutral_checkpoints,
    evaluate,
    load_neutral_checkpoint,
    matrices_for_metadata,
    model_factory,
    prepare_data,
    prepare_metadata,
    select_device,
)
from scripts.run_lemma456_audit import (
    _evaluate_gradient_probe_objective,
    _replay,
    _scheduled_beta,
    _validate_history_config,
)
from src.experiments.lemma456 import (
    compute_gain_metrics,
    compute_pair_oracle_gates,
    compute_proxy_chain_metrics,
)
from src.utils.config import load_config


MAIN_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "method",
    "pair_list",
    "smoothness_L",
    "global_gradient_norm",
    "global_gradient_norm_sq",
    "pair_count",
    "proxy_favorable_fraction",
    "proxy_gate_fraction",
    "accepted_pair_mass",
    "P_t_norm",
    "lemma6_projection",
    "P_t_descent_cosine",
    "first_order_q_ratio",
    "Q_proxy_vs_baseline",
    "Q_counterfactual",
    "Q_observed_loss_gain",
    "proxy_Q_over_grad_sq",
    "Q_over_grad_sq",
    "proxy_minus_epsilon_over_grad_sq",
    "Z_t_norm",
    "trajectory_error_norm",
    "trajectory_retention_ratio",
    "epsilon_trajectory_bound",
    "proxy_q_condition_holds",
    "inequality_holds",
    "proxy_off_objective_loss",
    "method_objective_loss",
    "proxy_off_val_loss",
    "method_val_loss",
    "proxy_off_val_acc",
    "method_val_acc",
    "beta",
    "step_scale",
    "local_steps",
    "learning_rate",
    "oracle_min_margin",
    "oracle_min_cosine",
]

PAIR_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "method",
    "client_i",
    "client_j",
    "pair_mass",
    "theta",
    "d_pair_norm",
    "oracle_margin",
    "oracle_cosine",
    "oracle_gate",
    "proxy_gate",
    "proxy_is_favorable",
    "descent_margin",
    "lambda_i",
    "lambda_j",
    "lambda_bar",
    "weighted_proxy_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse compatible neutral checkpoints",
    )
    return parser.parse_args()


def _write_summary(rows: list[Mapping[str, Any]], path: Path) -> None:
    keys = (
        "checkpoint_round",
        "method",
        "step_scale",
        "panel",
        "pairing_strategy",
        "smoothness_L",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    fields = [
        *keys,
        "n",
        "gate_fraction",
        "favorable_fraction",
        "lemma6_projection_mean",
        "descent_cosine_mean",
        "Q_proxy_mean",
        "Q_counterfactual_mean",
        "Q_counterfactual_std",
        "Q_observed_mean",
        "epsilon_trajectory_mean",
        "proxy_q_ratio_mean",
        "counterfactual_q_ratio_mean",
        "counterfactual_q_ratio_lcb95",
        "proxy_minus_epsilon_ratio_mean",
        "proxy_q_positive_fraction",
        "counterfactual_q_positive_fraction",
        "observed_gain_positive_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(grouped.items()):
            array = lambda name: np.asarray(
                [float(row[name]) for row in values], dtype=np.float64
            )
            q_proxy = array("Q_proxy_vs_baseline")
            q_actual = array("Q_counterfactual")
            observed = array("Q_observed_loss_gain")
            q_actual_ratio = array("Q_over_grad_sq")
            writer.writerow(
                {
                    **dict(zip(keys, key)),
                    "n": len(values),
                    "gate_fraction": array("proxy_gate_fraction").mean(),
                    "favorable_fraction": array("proxy_favorable_fraction").mean(),
                    "lemma6_projection_mean": array("lemma6_projection").mean(),
                    "descent_cosine_mean": array("P_t_descent_cosine").mean(),
                    "Q_proxy_mean": q_proxy.mean(),
                    "Q_counterfactual_mean": q_actual.mean(),
                    "Q_counterfactual_std": q_actual.std(ddof=0),
                    "Q_observed_mean": observed.mean(),
                    "epsilon_trajectory_mean": array(
                        "epsilon_trajectory_bound"
                    ).mean(),
                    "proxy_q_ratio_mean": array("proxy_Q_over_grad_sq").mean(),
                    "counterfactual_q_ratio_mean": q_actual_ratio.mean(),
                    "counterfactual_q_ratio_lcb95": q_actual_ratio.mean()
                    - 1.96
                    * q_actual_ratio.std(ddof=1 if len(values) > 1 else 0)
                    / max(len(values), 1) ** 0.5,
                    "proxy_minus_epsilon_ratio_mean": array(
                        "proxy_minus_epsilon_over_grad_sq"
                    ).mean(),
                    "proxy_q_positive_fraction": float((q_proxy > 0.0).mean()),
                    "counterfactual_q_positive_fraction": float(
                        (q_actual > 0.0).mean()
                    ),
                    "observed_gain_positive_fraction": float(
                        (observed > 0.0).mean()
                    ),
                }
            )


def run(config: Mapping[str, Any], *, reuse_checkpoints: bool) -> dict[str, str]:
    seed = int(config.get("seed", 42))
    audit = config.get("audit", {})
    output_dir = Path(audit.get("output_dir", "outputs/fcpc_grad_oracle_audit"))
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
    checkpoints = [int(value) for value in audit.get("checkpoints", [1, 10, 25, 50])]
    l_values = [float(value) for value in audit.get("smoothness_L_values", [0.0])]
    oracle = audit.get("oracle", {})
    min_margin = float(oracle.get("min_margin", 0.0))
    min_cosine = float(oracle.get("min_cosine", 0.0))
    replay = audit["replay"]
    configured_step_scale = float(replay.get("grad_center_step_scale", 0.5))
    step_scales = [
        float(value)
        for value in oracle.get("step_scales", [configured_step_scale])
    ]
    if not step_scales or any(value < 0.0 for value in step_scales):
        raise ValueError("audit.oracle.step_scales must contain non-negative values")
    local_steps = int(replay.get("local_steps", 2))
    learning_rate = float(replay.get("optimizer", {}).get("lr", 0.03))
    parameter_names = [name for name, _ in model_factory(config, data).named_parameters()]

    metrics_path = output_dir / "oracle_metrics.csv"
    pairs_path = output_dir / "oracle_pairs.csv"
    summary_path = output_dir / "oracle_summary.csv"
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
                raise ValueError("regenerate neutral checkpoints with broadcast histories")
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
                    oracle_gates, oracle_rows = compute_pair_oracle_gates(
                        previous_states=checkpoint["client_previous_states"],
                        previous_global_states=checkpoint[
                            "client_previous_global_states"
                        ],
                        client_gradients=gradients,
                        sample_counts=data["sample_counts"],
                        pairing=pairing,
                        parameter_names=parameter_names,
                        min_margin=min_margin,
                        min_cosine=min_cosine,
                    )
                    all_gates = {key: 1.0 for key in oracle_gates}
                    oracle_by_pair = {
                        (min(int(row["client_i"]), int(row["client_j"])),
                         max(int(row["client_i"]), int(row["client_j"]))): row
                        for row in oracle_rows
                    }

                    for batch_seed in batch_seeds:
                        off_states, off_aggregate, _, _ = _replay(
                            config,
                            data,
                            checkpoint,
                            pairing,
                            gradient_mix=0.0,
                            center_base="global",
                            pair_gate_values={key: 0.0 for key in oracle_gates},
                            batch_seed=batch_seed,
                            device=device,
                        )
                        off_validation = evaluate(config, data, off_aggregate, device)
                        off_objective = _evaluate_gradient_probe_objective(
                            config, data, off_aggregate, device
                        )

                        for step_scale in step_scales:
                            for method, gates in (
                                ("ungated", all_gates),
                                ("oracle", oracle_gates),
                            ):
                                states, aggregate, effective_betas, clip_scales = (
                                    _replay(
                                        config,
                                        data,
                                        checkpoint,
                                        pairing,
                                        gradient_mix=1.0,
                                        center_base="global",
                                        pair_gate_values=gates,
                                        batch_seed=batch_seed,
                                        device=device,
                                        step_scale_override=step_scale,
                                    )
                                )
                                chain, pair_rows, proxy_component = (
                                    compute_proxy_chain_metrics(
                                        global_state=checkpoint["model_state"],
                                        previous_states=checkpoint[
                                            "client_previous_states"
                                        ],
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
                                        gradient_mix=1.0,
                                        step_scale=step_scale,
                                        pair_clip_scales=clip_scales,
                                        pair_gate_values=gates,
                                    )
                                )
                                validation = evaluate(config, data, aggregate, device)
                                objective = _evaluate_gradient_probe_objective(
                                    config, data, aggregate, device
                                )
                                common = {
                                    "model_seed": seed,
                                    "checkpoint_round": checkpoint_round,
                                    "panel": panel,
                                    "pairing_strategy": strategy,
                                    "pairing_seed": (
                                        pairing_seed if strategy == "random" else ""
                                    ),
                                    "batch_seed": batch_seed,
                                    "method": method,
                                    "pair_list": json.dumps(
                                        pairing.pairs, separators=(",", ":")
                                    ),
                                    **chain,
                                    "proxy_off_objective_loss": off_objective,
                                    "method_objective_loss": objective,
                                    "proxy_off_val_loss": off_validation["loss"],
                                    "method_val_loss": validation["loss"],
                                    "proxy_off_val_acc": off_validation["acc"],
                                    "method_val_acc": validation["acc"],
                                    "beta": _scheduled_beta(
                                        replay, checkpoint_round
                                    ),
                                    "step_scale": step_scale,
                                    "local_steps": local_steps,
                                    "learning_rate": learning_rate,
                                    "oracle_min_margin": min_margin,
                                    "oracle_min_cosine": min_cosine,
                                }
                                for smoothness_l in l_values:
                                    gain = compute_gain_metrics(
                                        global_state=checkpoint["model_state"],
                                        grad_states=states,
                                        baseline_states=off_states,
                                        client_gradients=gradients,
                                        sample_counts=data["sample_counts"],
                                        parameter_names=parameter_names,
                                        proxy_component=proxy_component,
                                        smoothness_l=smoothness_l,
                                        observed_loss_grad=objective,
                                        observed_loss_baseline=off_objective,
                                        q_candidate=float(
                                            audit.get("q_candidate", 0.0)
                                        ),
                                    )
                                    row = {**common, **gain}
                                    metrics_writer.writerow(
                                        {field: row.get(field, "") for field in MAIN_FIELDS}
                                    )
                                    all_rows.append(row)
                                metrics_file.flush()

                                for pair_row in pair_rows:
                                    key = (
                                        min(int(pair_row["client_i"]), int(pair_row["client_j"])),
                                        max(int(pair_row["client_i"]), int(pair_row["client_j"])),
                                    )
                                    enriched = {
                                        **common,
                                        **oracle_by_pair[key],
                                        **pair_row,
                                    }
                                    pair_writer.writerow(
                                        {field: enriched.get(field, "") for field in PAIR_FIELDS}
                                    )
                                pairs_file.flush()
                                print(
                                    f"oracle_audit: t={checkpoint_round}, method={method}, "
                                    f"step_scale={step_scale:g}, batch_seed={batch_seed}, "
                                    f"gate={chain['proxy_gate_fraction']:.1%}, "
                                    f"-<g,P>={chain['lemma6_projection']:+.3e}, "
                                    f"observed_Q={off_objective - objective:+.3e}",
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
    outputs = run(load_config(args.config), reuse_checkpoints=args.reuse_checkpoints)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
