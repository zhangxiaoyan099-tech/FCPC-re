"""Empirical diagnostics for the conditional Lemma 4--6 proof chain.

The functions in this module deliberately operate on frozen model states and
flattened empirical gradients.  They do not claim to prove smoothness or an
expectation theorem for a neural network; instead they expose every measurable
term that appears in the sufficient conditions.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src.experiments.at_m import flatten_state_delta, squared_l2
from src.fcpc.pairing import PairingResult


def _weights(sample_counts: Mapping[int, float]) -> tuple[list[int], dict[int, float]]:
    client_ids = sorted(int(client_id) for client_id in sample_counts)
    counts = {
        client_id: max(float(sample_counts[client_id]), 0.0)
        for client_id in client_ids
    }
    total = sum(counts.values())
    if total <= 0.0:
        raise ValueError("sample counts must have positive total mass")
    return client_ids, {client_id: counts[client_id] / total for client_id in client_ids}


def aggregate_delta(
    states: Mapping[int, Mapping[str, object]],
    global_state: Mapping[str, object],
    sample_counts: Mapping[int, float],
    parameter_names: Sequence[str],
):
    """Return the sample-weighted server update vector."""
    import torch

    client_ids, weights = _weights(sample_counts)
    first = flatten_state_delta(states[client_ids[0]], global_state, parameter_names)
    result = torch.zeros_like(first)
    for client_id in client_ids:
        result.add_(
            flatten_state_delta(states[client_id], global_state, parameter_names),
            alpha=weights[client_id],
        )
    return result


def aggregate_gradient(
    client_gradients: Mapping[int, object],
    sample_counts: Mapping[int, float],
):
    """Return the sample-weighted empirical global gradient."""
    import torch

    client_ids, weights = _weights(sample_counts)
    first = client_gradients[client_ids[0]].detach().cpu().float().reshape(-1)
    result = torch.zeros_like(first)
    for client_id in client_ids:
        gradient = client_gradients[client_id].detach().cpu().float().reshape(-1)
        if gradient.shape != first.shape:
            raise ValueError("all client gradients must have the same shape")
        result.add_(gradient, alpha=weights[client_id])
    return result


def proximal_transfer_coefficient(
    *, beta: float, learning_rate: float, local_steps: int
) -> float:
    """Return ``1-(1-rho)^H`` for the exact quadratic proximal map."""
    beta = float(beta)
    learning_rate = float(learning_rate)
    local_steps = int(local_steps)
    if beta < 0.0 or learning_rate < 0.0 or local_steps < 0:
        raise ValueError("beta, learning_rate, and local_steps must be non-negative")
    contraction = 1.0 / (1.0 + 2.0 * learning_rate * beta)
    return float(1.0 - contraction**local_steps)


def compute_pair_oracle_gates(
    *,
    previous_states: Mapping[int, Mapping[str, object]],
    previous_global_states: Mapping[int, Mapping[str, object]],
    client_gradients: Mapping[int, object],
    sample_counts: Mapping[int, float],
    pairing: PairingResult,
    parameter_names: Sequence[str],
    min_margin: float = 0.0,
    min_cosine: float = 0.0,
    eps: float = 1e-12,
) -> tuple[dict[tuple[int, int], float], list[dict[str, float | bool | int]]]:
    """Return an unavailable-in-practice oracle gate for each client pair.

    A pair update proxy ``d_p`` is admitted only when its projection on the
    *true empirical* global descent direction is sufficiently positive:

    ``-<g_t, d_p> > min_margin`` and ``cos(d_p, -g_t) >= min_cosine``.

    The helper is deliberately an audit oracle, not a deployable FCPC rule.
    It establishes whether rejecting harmful historical directions would be
    enough to repair the Lemma 4--6 mechanism before a history-only gate is
    designed.
    """
    client_ids, weights = _weights(sample_counts)
    if set(client_gradients) != set(client_ids):
        raise ValueError("client_gradients must cover every client")
    global_gradient = aggregate_gradient(client_gradients, sample_counts)
    global_norm = float(global_gradient.norm().item())
    gates: dict[tuple[int, int], float] = {}
    rows: list[dict[str, float | bool | int]] = []
    for client_i, client_j in pairing.pairs:
        pair_mass = weights[client_i] + weights[client_j]
        theta = weights[client_i] / pair_mass
        update_i = flatten_state_delta(
            previous_states[client_i], previous_global_states[client_i], parameter_names
        )
        update_j = flatten_state_delta(
            previous_states[client_j], previous_global_states[client_j], parameter_names
        )
        d_pair = theta * update_i + (1.0 - theta) * update_j
        d_norm = float(d_pair.norm().item())
        margin = float(-global_gradient.dot(d_pair).item())
        cosine = (
            margin / (global_norm * d_norm)
            if global_norm * d_norm > eps
            else float("nan")
        )
        accepted = bool(
            d_norm > eps
            and margin > float(min_margin)
            and cosine >= float(min_cosine)
        )
        key = (min(client_i, client_j), max(client_i, client_j))
        gates[key] = 1.0 if accepted else 0.0
        rows.append(
            {
                "client_i": client_i,
                "client_j": client_j,
                "pair_mass": pair_mass,
                "theta": theta,
                "d_pair_norm": d_norm,
                "oracle_margin": margin,
                "oracle_cosine": cosine,
                "oracle_gate": accepted,
            }
        )
    return gates, rows


def compute_proxy_chain_metrics(
    *,
    global_state: Mapping[str, object],
    previous_states: Mapping[int, Mapping[str, object]],
    previous_global_states: Mapping[int, Mapping[str, object]],
    client_gradients: Mapping[int, object],
    sample_counts: Mapping[int, float],
    pairing: PairingResult,
    parameter_names: Sequence[str],
    history_gamma: float,
    learning_rate: float,
    local_steps: int,
    effective_betas: Mapping[int, float],
    gradient_mix: float,
    step_scale: float,
    pair_clip_scales: Mapping[tuple[int, int], float] | None = None,
    pair_gate_values: Mapping[tuple[int, int], float] | None = None,
    eps: float = 1e-12,
) -> tuple[dict[str, float | bool], list[dict[str, float | bool | int]], object]:
    """Measure the Lemma 4 pair conditions and Lemma 5--6 proxy component.

    ``history_gamma`` is fixed by the controlled historical round.  The
    residual ``epsilon_p = ||d_p + history_gamma * g_p||`` therefore includes
    stochastic-gradient noise, local drift, and one-round staleness instead of
    being fitted after observing ``d_p``.
    """
    import torch

    if history_gamma <= 0.0:
        raise ValueError("history_gamma must be positive")
    client_ids, weights = _weights(sample_counts)
    if set(client_gradients) != set(client_ids):
        raise ValueError("client_gradients must cover every client")
    global_gradient = aggregate_gradient(client_gradients, sample_counts)
    gradient_norm = float(global_gradient.norm().item())
    gradient_norm_sq = squared_l2(global_gradient)
    proxy_component = torch.zeros_like(global_gradient)
    rows: list[dict[str, float | bool | int]] = []
    weighted_margin = 0.0
    weighted_lower_bound = 0.0
    sufficient_count = 0
    parallel_sufficient_count = 0
    favorable_count = 0
    clip_scales = pair_clip_scales or {}
    gate_values = pair_gate_values or {}
    accepted_pair_mass = 0.0
    server_coefficient = 0.0
    server_gradient_error = torch.zeros_like(global_gradient)
    server_history_residual = torch.zeros_like(global_gradient)
    gradient_heterogeneity = 0.0
    history_residual_energy = 0.0
    gradient_bound_factor = 0.0
    history_bound_factor = 0.0

    for client_i, client_j in pairing.pairs:
        pair_mass = weights[client_i] + weights[client_j]
        theta = weights[client_i] / pair_mass
        update_i = flatten_state_delta(
            previous_states[client_i], previous_global_states[client_i], parameter_names
        )
        update_j = flatten_state_delta(
            previous_states[client_j], previous_global_states[client_j], parameter_names
        )
        d_pair = theta * update_i + (1.0 - theta) * update_j
        g_pair = (
            theta * client_gradients[client_i].detach().cpu().float().reshape(-1)
            + (1.0 - theta)
            * client_gradients[client_j].detach().cpu().float().reshape(-1)
        )
        gradient_error = g_pair - global_gradient
        history_residual = d_pair + float(history_gamma) * g_pair
        delta = float(gradient_error.norm().item())
        epsilon = float(history_residual.norm().item())
        condition_rhs = delta + epsilon / float(history_gamma)
        sufficient = bool(gradient_norm > condition_rhs)
        margin = float(-global_gradient.dot(d_pair).item())
        gradient_error_inner = float(global_gradient.dot(gradient_error).item())
        history_residual_inner = float(
            global_gradient.dot(history_residual).item()
        )
        if gradient_norm > eps:
            delta_parallel = max(0.0, -gradient_error_inner / gradient_norm)
            epsilon_parallel = max(0.0, history_residual_inner / gradient_norm)
            exact_directional_slack = margin / (
                float(history_gamma) * gradient_norm
            )
        else:
            delta_parallel = 0.0
            epsilon_parallel = 0.0
            exact_directional_slack = 0.0
        parallel_condition_rhs = (
            delta_parallel + epsilon_parallel / float(history_gamma)
        )
        parallel_sufficient = bool(gradient_norm > parallel_condition_rhs)
        parallel_lower_bound = float(
            history_gamma
            * gradient_norm
            * (gradient_norm - parallel_condition_rhs)
        )
        lower_bound = float(
            history_gamma * gradient_norm * (gradient_norm - condition_rhs)
        )
        favorable = bool(margin > 0.0)
        sufficient_count += int(sufficient)
        parallel_sufficient_count += int(parallel_sufficient)
        favorable_count += int(favorable)

        key = (min(client_i, client_j), max(client_i, client_j))
        clip_scale = float(clip_scales.get(key, 1.0))
        pair_gate = float(gate_values.get(key, 1.0))
        if not 0.0 <= pair_gate <= 1.0:
            raise ValueError("pair gate values must be in [0, 1]")
        accepted_pair_mass += pair_mass * pair_gate
        lambda_i = (
            float(gradient_mix)
            * float(step_scale)
            * clip_scale
            * pair_gate
            * proximal_transfer_coefficient(
                beta=float(effective_betas.get(client_i, 0.0)),
                learning_rate=learning_rate,
                local_steps=local_steps,
            )
        )
        lambda_j = (
            float(gradient_mix)
            * float(step_scale)
            * clip_scale
            * pair_gate
            * proximal_transfer_coefficient(
                beta=float(effective_betas.get(client_j, 0.0)),
                learning_rate=learning_rate,
                local_steps=local_steps,
            )
        )
        lambda_bar = theta * lambda_i + (1.0 - theta) * lambda_j
        proxy_component.add_(d_pair, alpha=pair_mass * lambda_bar)
        weighted_gamma_lambda = (
            pair_mass * lambda_bar * float(history_gamma)
        )
        server_coefficient += weighted_gamma_lambda
        server_gradient_error.add_(gradient_error, alpha=weighted_gamma_lambda)
        server_history_residual.add_(
            history_residual, alpha=pair_mass * lambda_bar
        )
        gradient_heterogeneity += pair_mass * squared_l2(gradient_error)
        history_residual_energy += pair_mass * squared_l2(history_residual)
        gradient_bound_factor += (
            pair_mass * (lambda_bar * float(history_gamma)) ** 2
        )
        history_bound_factor += pair_mass * lambda_bar**2
        weighted_margin += pair_mass * lambda_bar * margin
        weighted_lower_bound += pair_mass * lambda_bar * lower_bound
        rows.append(
            {
                "client_i": client_i,
                "client_j": client_j,
                "pair_mass": pair_mass,
                "theta": theta,
                "d_pair_norm": float(d_pair.norm().item()),
                "g_pair_norm": float(g_pair.norm().item()),
                "delta_pair": delta,
                "epsilon_pair": epsilon,
                "epsilon_over_gamma": epsilon / float(history_gamma),
                "gradient_error_inner": gradient_error_inner,
                "history_residual_inner": history_residual_inner,
                "delta_parallel": delta_parallel,
                "epsilon_parallel": epsilon_parallel,
                "epsilon_parallel_over_gamma": epsilon_parallel
                / float(history_gamma),
                "lemma4_parallel_condition_rhs": parallel_condition_rhs,
                "lemma4_parallel_condition_slack": gradient_norm
                - parallel_condition_rhs,
                "lemma4_parallel_sufficient_holds": parallel_sufficient,
                "lemma4_parallel_lower_bound": parallel_lower_bound,
                "lemma4_exact_directional_slack": exact_directional_slack,
                "lemma4_condition_rhs": condition_rhs,
                "lemma4_condition_slack": gradient_norm - condition_rhs,
                "lemma4_sufficient_holds": sufficient,
                "descent_margin": margin,
                "descent_lower_bound": lower_bound,
                "lower_bound_slack": margin - lower_bound,
                "proxy_is_favorable": favorable,
                "proxy_gate": pair_gate,
                "center_clip_scale": clip_scale,
                "lambda_i": lambda_i,
                "lambda_j": lambda_j,
                "lambda_bar": lambda_bar,
                "weighted_proxy_margin": pair_mass * lambda_bar * margin,
            }
        )

    lemma6_lhs = float(-global_gradient.dot(proxy_component).item())
    identity_gap = lemma6_lhs - weighted_margin
    proxy_norm = float(proxy_component.norm().item())
    cosine = (
        lemma6_lhs / (gradient_norm * proxy_norm)
        if gradient_norm * proxy_norm > eps
        else float("nan")
    )
    pair_count = len(pairing.pairs)
    reconstructed_proxy = (
        -server_coefficient * global_gradient
        - server_gradient_error
        + server_history_residual
    )
    server_decomposition_gap = float(
        (proxy_component - reconstructed_proxy).norm().item()
    )
    if gradient_norm > eps:
        server_delta_parallel = max(
            0.0,
            -float(global_gradient.dot(server_gradient_error).item())
            / gradient_norm,
        )
        server_epsilon_parallel = max(
            0.0,
            float(global_gradient.dot(server_history_residual).item())
            / gradient_norm,
        )
    else:
        server_delta_parallel = 0.0
        server_epsilon_parallel = 0.0
    server_parallel_slack = (
        server_coefficient * gradient_norm
        - server_delta_parallel
        - server_epsilon_parallel
    )
    server_norm_slack = (
        server_coefficient * gradient_norm
        - float(server_gradient_error.norm().item())
        - float(server_history_residual.norm().item())
    )
    gradient_error_bound = float(
        max(gradient_bound_factor * gradient_heterogeneity, 0.0) ** 0.5
    )
    history_residual_bound = float(
        max(history_bound_factor * history_residual_energy, 0.0) ** 0.5
    )
    server_second_moment_slack = (
        server_coefficient * gradient_norm
        - gradient_error_bound
        - history_residual_bound
    )
    metrics: dict[str, float | bool] = {
        "global_gradient_norm": gradient_norm,
        "global_gradient_norm_sq": gradient_norm_sq,
        "history_gamma": float(history_gamma),
        "pair_count": float(pair_count),
        "lemma4_sufficient_fraction": (
            sufficient_count / pair_count if pair_count else float("nan")
        ),
        "lemma4_parallel_sufficient_fraction": (
            parallel_sufficient_count / pair_count
            if pair_count
            else float("nan")
        ),
        "proxy_favorable_fraction": (
            favorable_count / pair_count if pair_count else float("nan")
        ),
        "proxy_gate_fraction": (
            sum(float(value) for value in gate_values.values()) / pair_count
            if pair_count and gate_values
            else (1.0 if pair_count else float("nan"))
        ),
        "accepted_pair_mass": accepted_pair_mass if gate_values else 1.0,
        "all_pairs_lemma4_sufficient": bool(pair_count and sufficient_count == pair_count),
        "all_pairs_proxy_favorable": bool(pair_count and favorable_count == pair_count),
        "P_t_norm": proxy_norm,
        "lemma6_projection": lemma6_lhs,
        "lemma6_weighted_margin_sum": weighted_margin,
        "lemma6_identity_gap": identity_gap,
        "lemma6_lower_bound": weighted_lower_bound,
        "lemma6_lower_bound_slack": lemma6_lhs - weighted_lower_bound,
        "P_t_descent_cosine": cosine,
        "first_order_q_ratio": lemma6_lhs / (gradient_norm_sq + eps),
        "server_coefficient_C_t": server_coefficient,
        "server_gradient_error_norm": float(server_gradient_error.norm().item()),
        "server_history_residual_norm": float(
            server_history_residual.norm().item()
        ),
        "server_delta_parallel": server_delta_parallel,
        "server_epsilon_parallel": server_epsilon_parallel,
        "server_parallel_condition_slack": server_parallel_slack,
        "server_parallel_condition_holds": bool(server_parallel_slack > 0.0),
        "server_norm_condition_slack": server_norm_slack,
        "server_norm_condition_holds": bool(server_norm_slack > 0.0),
        "gradient_heterogeneity_H_t": gradient_heterogeneity,
        "history_residual_energy_A_t": history_residual_energy,
        "server_gradient_error_bound": gradient_error_bound,
        "server_history_residual_bound": history_residual_bound,
        "server_second_moment_condition_slack": server_second_moment_slack,
        "server_second_moment_condition_holds": bool(
            server_second_moment_slack > 0.0
        ),
        "server_proxy_decomposition_gap": server_decomposition_gap,
    }
    return metrics, rows, proxy_component


def compute_gain_metrics(
    *,
    global_state: Mapping[str, object],
    grad_states: Mapping[int, Mapping[str, object]],
    baseline_states: Mapping[int, Mapping[str, object]],
    client_gradients: Mapping[int, object],
    sample_counts: Mapping[int, float],
    parameter_names: Sequence[str],
    proxy_component: object,
    smoothness_l: float,
    observed_loss_grad: float | None = None,
    observed_loss_baseline: float | None = None,
    q_candidate: float = 0.0,
    eps: float = 1e-12,
) -> dict[str, float | bool]:
    """Evaluate internal and matched-counterfactual one-round gain terms."""
    smoothness_l = float(smoothness_l)
    if smoothness_l < 0.0:
        raise ValueError("smoothness_l must be non-negative")
    global_gradient = aggregate_gradient(client_gradients, sample_counts)
    gradient_norm_sq = squared_l2(global_gradient)
    delta_grad = aggregate_delta(
        grad_states, global_state, sample_counts, parameter_names
    )
    delta_baseline = aggregate_delta(
        baseline_states, global_state, sample_counts, parameter_names
    )
    z_value = delta_grad - delta_baseline
    p_value = proxy_component.detach().cpu().float().reshape(-1)
    if z_value.shape != p_value.shape:
        raise ValueError("proxy and counterfactual update vectors have different shapes")
    trajectory_error = z_value - p_value
    b_internal = delta_grad - p_value

    def q_value(base, correction) -> float:
        return float(
            -global_gradient.dot(correction).item()
            - smoothness_l * base.dot(correction).item()
            - 0.5 * smoothness_l * squared_l2(correction)
        )

    q_internal = q_value(b_internal, p_value)
    q_proxy_vs_baseline = q_value(delta_baseline, p_value)
    q_counterfactual = q_value(delta_baseline, z_value)
    trajectory_norm = float(trajectory_error.norm().item())
    epsilon_trajectory = float(
        (
            float(global_gradient.norm().item())
            + smoothness_l * float(delta_baseline.norm().item())
            + smoothness_l * float(p_value.norm().item())
        )
        * trajectory_norm
        + 0.5 * smoothness_l * trajectory_norm**2
    )
    rhs = float(q_candidate) * gradient_norm_sq - epsilon_trajectory
    proxy_q_rhs = float(q_candidate) * gradient_norm_sq
    observed_gain = float("nan")
    if observed_loss_grad is not None and observed_loss_baseline is not None:
        observed_gain = float(observed_loss_baseline) - float(observed_loss_grad)
    z_norm = float(z_value.norm().item())
    p_norm = float(p_value.norm().item())
    return {
        "smoothness_L": smoothness_l,
        "q_candidate": float(q_candidate),
        "Q_internal": q_internal,
        "Q_proxy_vs_baseline": q_proxy_vs_baseline,
        "Q_proxy_vs_mix0": q_proxy_vs_baseline,
        "Q_counterfactual": q_counterfactual,
        "Q_observed_loss_gain": observed_gain,
        "proxy_Q_over_grad_sq": q_proxy_vs_baseline / (gradient_norm_sq + eps),
        "Q_over_grad_sq": q_counterfactual / (gradient_norm_sq + eps),
        "proxy_minus_epsilon_over_grad_sq": (
            q_proxy_vs_baseline - epsilon_trajectory
        )
        / (gradient_norm_sq + eps),
        "P_t_norm": p_norm,
        "Z_t_norm": z_norm,
        "trajectory_error_norm": trajectory_norm,
        "trajectory_retention_ratio": p_norm / (z_norm + eps),
        "epsilon_trajectory_bound": epsilon_trajectory,
        "proxy_q_condition_slack": q_proxy_vs_baseline - proxy_q_rhs,
        "proxy_q_condition_holds": bool(q_proxy_vs_baseline + eps >= proxy_q_rhs),
        "inequality_rhs": rhs,
        "inequality_slack": q_counterfactual - rhs,
        "inequality_holds": bool(q_counterfactual + eps >= rhs),
    }
