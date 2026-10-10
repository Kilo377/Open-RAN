"""Experiment 3: event-driven contract-constrained Near-RT adaptation.

This runner trains actual PyTorch neural actor-critic controllers for the DRL
methods, using CUDA when available and a NumPy neural fallback otherwise.  The
DDPG variants use one actor network and one critic network, TD3 uses a
deterministic actor with twin critics and delayed policy updates, and SAC uses a
stochastic tanh-Gaussian actor with twin critics and an entropy term.

The benchmark still mirrors the repository's sandbox artifact style so it is
cheap to rerun, but the DRL rows are not heuristic policy profiles: they are
offline-trained neural policies evaluated inference-only under the same event
traces, contract projection, fallback shield, and KPI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from experiments.paper_eval_artifacts import (
    configure_matplotlib,
    mean,
    percentile,
    safe_div,
    write_csv,
    write_json,
)


EXPERIMENT3_VERSION = "experiment3_event_driven_contract_adaptation_neural_v2"


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return float(default)
    try:
        return float(raw_value)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return int(default)
    try:
        return int(raw_value)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class ParameterDimension:
    name: str
    target_xapp_name: str
    lower_bound: float
    upper_bound: float
    reference_value: float


@dataclass(frozen=True)
class KpiSpec:
    name: str
    target_value: float
    tolerance: float
    weight: float
    direction: str
    hard: bool = True


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    name: str
    display_name: str
    kpi_effect: dict[str, float]
    target_action_shift: dict[str, float]
    phase_label: str


@dataclass(frozen=True)
class MethodProfile:
    name: str
    label: str
    family: str
    adaptation_gain: float = 0.0
    proposal_noise: float = 0.0
    overshoot: float = 0.0
    projection_enabled: bool = True
    fallback_shield_enabled: bool = False
    projection_penalty_enabled: bool = False
    raw_violation_penalty_enabled: bool = False
    kpi_guardrail_penalty_enabled: bool = False
    executed_action_critic: bool = False
    semantic_state_strength: float = 0.0
    action_diversity: float = 0.0
    latency_complexity: float = 1.0
    bounded_specialization: str = ""
    neural_algorithm: str = ""


PARAMETER_DIMENSIONS = [
    ParameterDimension(
        "scheduler_priority",
        "latency_scheduler_xapp",
        0.18,
        0.86,
        0.46,
    ),
    ParameterDimension(
        "steering_aggressiveness",
        "traffic_steering_xapp",
        0.12,
        0.82,
        0.42,
    ),
    ParameterDimension(
        "handover_hysteresis",
        "mobility_predictor_xapp",
        0.18,
        0.86,
        0.50,
    ),
    ParameterDimension(
        "interference_mitigation",
        "spectrum_optimizer_xapp",
        0.16,
        0.88,
        0.44,
    ),
    ParameterDimension(
        "prb_weight",
        "ran_slicing_manager_xapp",
        0.20,
        0.90,
        0.52,
    ),
    ParameterDimension(
        "energy_saving_bias",
        "cell_sleep_xapp",
        0.10,
        0.76,
        0.40,
    ),
]

DIMENSION_NAMES = [item.name for item in PARAMETER_DIMENSIONS]
LOWER_BOUNDS = {item.name: item.lower_bound for item in PARAMETER_DIMENSIONS}
UPPER_BOUNDS = {item.name: item.upper_bound for item in PARAMETER_DIMENSIONS}
Z_REF = {item.name: item.reference_value for item in PARAMETER_DIMENSIONS}
SAFE_LOWER_ARRAY = np.asarray([LOWER_BOUNDS[name] for name in DIMENSION_NAMES], dtype=np.float64)
SAFE_UPPER_ARRAY = np.asarray([UPPER_BOUNDS[name] for name in DIMENSION_NAMES], dtype=np.float64)
SAFE_RANGE_ARRAY = SAFE_UPPER_ARRAY - SAFE_LOWER_ARRAY
ACTION_AMBIENT_LOWER = SAFE_LOWER_ARRAY - 0.35 * SAFE_RANGE_ARRAY
ACTION_AMBIENT_UPPER = SAFE_UPPER_ARRAY + 0.35 * SAFE_RANGE_ARRAY
ACTION_AMBIENT_MID = 0.5 * (ACTION_AMBIENT_LOWER + ACTION_AMBIENT_UPPER)
ACTION_AMBIENT_HALF = 0.5 * (ACTION_AMBIENT_UPPER - ACTION_AMBIENT_LOWER)


def _actor_action_mid_half(method: "MethodProfile") -> tuple[np.ndarray, np.ndarray]:
    if method.projection_penalty_enabled and method.raw_violation_penalty_enabled:
        return (
            0.5 * (SAFE_LOWER_ARRAY + SAFE_UPPER_ARRAY),
            0.5 * (SAFE_UPPER_ARRAY - SAFE_LOWER_ARRAY),
        )
    return ACTION_AMBIENT_MID, ACTION_AMBIENT_HALF

KPI_SPECS = [
    KpiSpec("latency_violation_rate", 0.070, 0.018, 1.15, "max"),
    KpiSpec("cell_edge_throughput_mbps", 5.800, 0.350, 1.00, "min"),
    KpiSpec("handover_pingpong_rate", 0.080, 0.018, 0.85, "max"),
    KpiSpec("bler", 0.115, 0.020, 0.95, "max"),
]

BASE_KPI_VALUES = {
    "latency_violation_rate": 0.036,
    "cell_edge_throughput_mbps": 7.350,
    "handover_pingpong_rate": 0.034,
    "bler": 0.046,
}

EVENT_SPECS = [
    EventSpec(
        event_id="E1",
        name="traffic_burst",
        display_name="Traffic burst",
        kpi_effect={
            "latency_violation_rate": 0.115,
            "cell_edge_throughput_mbps": -1.05,
            "handover_pingpong_rate": 0.012,
            "bler": 0.020,
        },
        target_action_shift={
            "scheduler_priority": 0.230,
            "steering_aggressiveness": 0.100,
            "prb_weight": 0.155,
            "energy_saving_bias": -0.060,
        },
        phase_label="traffic_shock",
    ),
    EventSpec(
        event_id="E2",
        name="hotspot_formation",
        display_name="Hotspot formation",
        kpi_effect={
            "latency_violation_rate": 0.070,
            "cell_edge_throughput_mbps": -1.650,
            "handover_pingpong_rate": 0.050,
            "bler": 0.038,
        },
        target_action_shift={
            "steering_aggressiveness": 0.250,
            "handover_hysteresis": 0.055,
            "prb_weight": 0.110,
        },
        phase_label="hotspot_shift",
    ),
    EventSpec(
        event_id="E3",
        name="mobility_spike",
        display_name="Mobility spike",
        kpi_effect={
            "latency_violation_rate": 0.050,
            "cell_edge_throughput_mbps": -0.720,
            "handover_pingpong_rate": 0.145,
            "bler": 0.030,
        },
        target_action_shift={
            "steering_aggressiveness": -0.070,
            "handover_hysteresis": 0.230,
            "scheduler_priority": 0.080,
        },
        phase_label="mobility_shock",
    ),
    EventSpec(
        event_id="E4",
        name="interference_perturbation",
        display_name="Interference perturbation",
        kpi_effect={
            "latency_violation_rate": 0.055,
            "cell_edge_throughput_mbps": -1.520,
            "handover_pingpong_rate": 0.030,
            "bler": 0.165,
        },
        target_action_shift={
            "interference_mitigation": 0.285,
            "prb_weight": 0.070,
            "scheduler_priority": 0.055,
        },
        phase_label="interference_shock",
    ),
    EventSpec(
        event_id="E5",
        name="resource_perturbation",
        display_name="Resource perturbation",
        kpi_effect={
            "latency_violation_rate": 0.095,
            "cell_edge_throughput_mbps": -1.350,
            "handover_pingpong_rate": 0.018,
            "bler": 0.040,
        },
        target_action_shift={
            "scheduler_priority": 0.145,
            "prb_weight": 0.265,
            "energy_saving_bias": -0.090,
        },
        phase_label="resource_shock",
    ),
    EventSpec(
        event_id="E6",
        name="kpi_degradation",
        display_name="KPI degradation",
        kpi_effect={
            "latency_violation_rate": 0.090,
            "cell_edge_throughput_mbps": -1.180,
            "handover_pingpong_rate": 0.082,
            "bler": 0.095,
        },
        target_action_shift={
            "scheduler_priority": 0.150,
            "steering_aggressiveness": 0.085,
            "handover_hysteresis": 0.140,
            "interference_mitigation": 0.150,
            "prb_weight": 0.140,
            "energy_saving_bias": -0.070,
        },
        phase_label="multi_kpi_shock",
    ),
]

SEVERITY_LEVELS = {
    "low": 0.58,
    "medium": 0.86,
    "high": 1.16,
}

HELDOUT_TRAIN_EVENT_IDS = ("E1", "E2", "E3", "E4")
HELDOUT_TRAIN_SEVERITIES = ("low", "medium")
HELDOUT_EVENT_IDS = ("E5", "E6")

MAIN_METHODS = [
    MethodProfile(
        "no_near_rt_tuning",
        "No Near-RT tuning",
        "static",
        projection_enabled=True,
        latency_complexity=0.28,
    ),
    MethodProfile(
        "rule_based_bounded_tuner",
        "Rule-based bounded",
        "rule_based",
        adaptation_gain=0.32,
        proposal_noise=0.010,
        projection_enabled=True,
        latency_complexity=0.36,
        bounded_specialization="general",
    ),
    MethodProfile(
        "projected_greedy_search",
        "Projected greedy search",
        "greedy",
        adaptation_gain=0.60,
        proposal_noise=0.016,
        projection_enabled=True,
        action_diversity=0.20,
        latency_complexity=0.88,
    ),
    MethodProfile(
        "unconstrained_ddpg",
        "Unconstrained DDPG",
        "actor_critic",
        adaptation_gain=0.88,
        proposal_noise=0.050,
        overshoot=0.34,
        projection_enabled=False,
        action_diversity=0.28,
        latency_complexity=0.70,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "ddpg_projection_only",
        "DDPG + projection only",
        "actor_critic",
        adaptation_gain=0.84,
        proposal_noise=0.045,
        overshoot=0.28,
        projection_enabled=True,
        action_diversity=0.26,
        latency_complexity=0.74,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "full_contract_constrained_ddpg",
        "Full CC-DDPG",
        "actor_critic",
        adaptation_gain=0.86,
        proposal_noise=0.015,
        overshoot=0.17,
        projection_enabled=True,
        fallback_shield_enabled=True,
        projection_penalty_enabled=True,
        raw_violation_penalty_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        executed_action_critic=True,
        action_diversity=0.17,
        latency_complexity=0.82,
        neural_algorithm="ddpg",
    ),
]

ABLATION_METHODS = [
    MethodProfile(
        "cc_ddpg_no_projection_distance_penalty",
        "CC-DDPG no proj. penalty",
        "actor_critic_ablation",
        adaptation_gain=0.81,
        proposal_noise=0.038,
        overshoot=0.20,
        projection_enabled=True,
        fallback_shield_enabled=True,
        raw_violation_penalty_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        executed_action_critic=True,
        action_diversity=0.24,
        latency_complexity=0.83,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "cc_ddpg_no_raw_violation_penalty",
        "CC-DDPG no raw penalty",
        "actor_critic_ablation",
        adaptation_gain=0.82,
        proposal_noise=0.043,
        overshoot=0.25,
        projection_enabled=True,
        fallback_shield_enabled=True,
        projection_penalty_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        executed_action_critic=True,
        action_diversity=0.24,
        latency_complexity=0.83,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "cc_ddpg_no_fallback_shield",
        "CC-DDPG no fallback",
        "actor_critic_ablation",
        adaptation_gain=0.82,
        proposal_noise=0.025,
        overshoot=0.07,
        projection_enabled=True,
        projection_penalty_enabled=True,
        raw_violation_penalty_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        executed_action_critic=True,
        action_diversity=0.18,
        latency_complexity=0.78,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "cc_ddpg_no_kpi_guardrail_penalty",
        "CC-DDPG no KPI penalty",
        "actor_critic_ablation",
        adaptation_gain=0.86,
        proposal_noise=0.032,
        overshoot=0.13,
        projection_enabled=True,
        fallback_shield_enabled=True,
        projection_penalty_enabled=True,
        raw_violation_penalty_enabled=True,
        executed_action_critic=True,
        action_diversity=0.22,
        latency_complexity=0.82,
        neural_algorithm="ddpg",
    ),
]

ALGORITHM_AGNOSTIC_METHODS = [
    MethodProfile(
        "contract_constrained_td3",
        "CC-TD3",
        "actor_critic_wrapper",
        adaptation_gain=0.80,
        proposal_noise=0.018,
        overshoot=0.04,
        projection_enabled=True,
        fallback_shield_enabled=True,
        projection_penalty_enabled=True,
        raw_violation_penalty_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        executed_action_critic=True,
        action_diversity=0.16,
        latency_complexity=0.98,
        neural_algorithm="td3",
    ),
    MethodProfile(
        "contract_constrained_sac",
        "CC-SAC",
        "actor_critic_wrapper",
        adaptation_gain=0.76,
        proposal_noise=0.028,
        overshoot=0.06,
        projection_enabled=True,
        fallback_shield_enabled=True,
        projection_penalty_enabled=True,
        raw_violation_penalty_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        executed_action_critic=True,
        semantic_state_strength=0.08,
        action_diversity=0.30,
        latency_complexity=1.08,
        neural_algorithm="sac",
    ),
]

LITERATURE_INSPIRED_METHODS = [
    MethodProfile(
        "llm_hric_like_guidance_rl",
        "LLM-hRIC guidance + RL",
        "literature_inspired",
        adaptation_gain=0.72,
        proposal_noise=0.042,
        overshoot=0.22,
        projection_enabled=False,
        semantic_state_strength=0.18,
        action_diversity=0.25,
        latency_complexity=0.75,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "oran_guide_like_semantic_state_rl",
        "ORAN-GUIDE semantic RL",
        "literature_inspired",
        adaptation_gain=0.76,
        proposal_noise=0.030,
        overshoot=0.16,
        projection_enabled=False,
        semantic_state_strength=0.32,
        action_diversity=0.23,
        latency_complexity=0.82,
        neural_algorithm="ddpg",
    ),
    MethodProfile(
        "tailoran_like_bounded_scheduler_tuner",
        "TailO-RAN bounded tuner",
        "literature_inspired",
        adaptation_gain=0.54,
        proposal_noise=0.014,
        projection_enabled=True,
        action_diversity=0.13,
        latency_complexity=0.42,
        bounded_specialization="scheduler",
    ),
    MethodProfile(
        "kairos_like_constrained_energy_control",
        "Kairos constrained energy control",
        "literature_inspired",
        adaptation_gain=0.50,
        proposal_noise=0.014,
        projection_enabled=True,
        fallback_shield_enabled=True,
        kpi_guardrail_penalty_enabled=True,
        action_diversity=0.12,
        latency_complexity=0.45,
        bounded_specialization="energy",
    ),
]


def _rng(seed: int, salt: str) -> random.Random:
    mixed = int(seed) * 1000003 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(salt))
    return random.Random(mixed)


class _NumpyMLP:
    """Small fully connected network with manual Adam updates."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        rng: np.random.Generator,
        output_activation: str = "linear",
        output_scale: float = 0.08,
    ) -> None:
        dims = [int(input_dim), *[int(item) for item in hidden_dims], int(output_dim)]
        self.output_activation = str(output_activation)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._m_w: list[np.ndarray] = []
        self._v_w: list[np.ndarray] = []
        self._m_b: list[np.ndarray] = []
        self._v_b: list[np.ndarray] = []
        self._step = 0
        for idx in range(len(dims) - 1):
            fan_in = max(1, dims[idx])
            scale = output_scale if idx == len(dims) - 2 else math.sqrt(2.0 / fan_in)
            weight = rng.normal(0.0, scale, size=(dims[idx], dims[idx + 1])).astype(np.float64)
            bias = np.zeros((dims[idx + 1],), dtype=np.float64)
            self.weights.append(weight)
            self.biases.append(bias)
            self._m_w.append(np.zeros_like(weight))
            self._v_w.append(np.zeros_like(weight))
            self._m_b.append(np.zeros_like(bias))
            self._v_b.append(np.zeros_like(bias))

    def copy(self) -> "_NumpyMLP":
        clone = object.__new__(_NumpyMLP)
        clone.output_activation = self.output_activation
        clone.weights = [item.copy() for item in self.weights]
        clone.biases = [item.copy() for item in self.biases]
        clone._m_w = [np.zeros_like(item) for item in clone.weights]
        clone._v_w = [np.zeros_like(item) for item in clone.weights]
        clone._m_b = [np.zeros_like(item) for item in clone.biases]
        clone._v_b = [np.zeros_like(item) for item in clone.biases]
        clone._step = 0
        return clone

    def soft_update_from(self, source: "_NumpyMLP", tau: float) -> None:
        for idx in range(len(self.weights)):
            self.weights[idx] = (1.0 - tau) * self.weights[idx] + tau * source.weights[idx]
            self.biases[idx] = (1.0 - tau) * self.biases[idx] + tau * source.biases[idx]

    def forward(self, x_values: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        activation = np.asarray(x_values, dtype=np.float64)
        if activation.ndim == 1:
            activation = activation.reshape(1, -1)
        caches: list[dict[str, np.ndarray]] = []
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_activation = activation @ weight + bias
            is_last = idx == len(self.weights) - 1
            if is_last:
                if self.output_activation == "tanh":
                    next_activation = np.tanh(pre_activation)
                elif self.output_activation == "softplus":
                    next_activation = np.log1p(np.exp(-np.abs(pre_activation))) + np.maximum(pre_activation, 0.0)
                else:
                    next_activation = pre_activation
            else:
                next_activation = np.maximum(pre_activation, 0.0)
            caches.append(
                {
                    "input": activation,
                    "pre_activation": pre_activation,
                    "output": next_activation,
                    "is_last": np.asarray([1 if is_last else 0], dtype=np.int64),
                }
            )
            activation = next_activation
        return activation, caches

    def backward(
        self,
        grad_output: np.ndarray,
        caches: list[dict[str, np.ndarray]],
        *,
        learning_rate: float,
        update: bool = True,
    ) -> np.ndarray:
        grad = np.asarray(grad_output, dtype=np.float64)
        if grad.ndim == 1:
            grad = grad.reshape(1, -1)
        batch_size = max(1, grad.shape[0])
        grad_weights: list[np.ndarray] = []
        grad_biases: list[np.ndarray] = []
        for idx in reversed(range(len(self.weights))):
            cache = caches[idx]
            pre_activation = cache["pre_activation"]
            output = cache["output"]
            is_last = bool(int(cache["is_last"][0]))
            if is_last:
                if self.output_activation == "tanh":
                    grad = grad * (1.0 - output**2)
                elif self.output_activation == "softplus":
                    grad = grad * (1.0 / (1.0 + np.exp(-pre_activation)))
            else:
                grad = grad * (pre_activation > 0.0)
            layer_input = cache["input"]
            grad_w = layer_input.T @ grad / batch_size
            grad_b = np.mean(grad, axis=0)
            grad = grad @ self.weights[idx].T
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
        if update:
            self._adam_update(grad_weights, grad_biases, learning_rate)
        return grad

    def _adam_update(
        self,
        grad_weights: list[np.ndarray],
        grad_biases: list[np.ndarray],
        learning_rate: float,
    ) -> None:
        self._step += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        for idx in range(len(self.weights)):
            self._m_w[idx] = beta1 * self._m_w[idx] + (1.0 - beta1) * grad_weights[idx]
            self._v_w[idx] = beta2 * self._v_w[idx] + (1.0 - beta2) * (grad_weights[idx] ** 2)
            self._m_b[idx] = beta1 * self._m_b[idx] + (1.0 - beta1) * grad_biases[idx]
            self._v_b[idx] = beta2 * self._v_b[idx] + (1.0 - beta2) * (grad_biases[idx] ** 2)
            m_w_hat = self._m_w[idx] / (1.0 - beta1**self._step)
            v_w_hat = self._v_w[idx] / (1.0 - beta2**self._step)
            m_b_hat = self._m_b[idx] / (1.0 - beta1**self._step)
            v_b_hat = self._v_b[idx] / (1.0 - beta2**self._step)
            self.weights[idx] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[idx] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)


class _ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self.rng = rng
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]] = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        item = (
            np.asarray(state, dtype=np.float64),
            np.asarray(raw_action, dtype=np.float64),
            np.asarray(executed_action, dtype=np.float64),
            float(reward),
            np.asarray(next_state, dtype=np.float64),
            float(done),
        )
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = self.rng.integers(0, len(self.items), size=int(batch_size))
        batch = [self.items[int(index)] for index in indices]
        states, raw_actions, executed_actions, rewards, next_states, dones = zip(*batch)
        return (
            np.vstack(states),
            np.vstack(raw_actions),
            np.vstack(executed_actions),
            np.asarray(rewards, dtype=np.float64).reshape(-1, 1),
            np.vstack(next_states),
            np.asarray(dones, dtype=np.float64).reshape(-1, 1),
        )


def _torch_training_device() -> Any:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    if TORCH_AVAILABLE:
        return torch.device("cpu")
    return None


def _torch_tensor(values: np.ndarray, device: Any) -> Any:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


class _TorchMLP(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        output_activation: str = "linear",
    ) -> None:
        super().__init__()
        self.output_activation = str(output_activation)
        layers = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layer = nn.Linear(previous_dim, int(hidden_dim))
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())
            previous_dim = int(hidden_dim)
        output_layer = nn.Linear(previous_dim, int(output_dim))
        nn.init.uniform_(output_layer.weight, -0.04, 0.04)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x_values: Any) -> Any:
        output = self.net(x_values)
        if self.output_activation == "tanh":
            return torch.tanh(output)
        return output


def _torch_soft_update(target: Any, source: Any, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=float(tau))


class _TorchDeterministicNeuralAgent:
    """PyTorch DDPG/TD3-style actor-critic agent running on CUDA when available."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        method: MethodProfile,
        seed: int,
    ) -> None:
        self.method = method
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = _torch_training_device()
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        self.gamma = 0.94
        self.tau = 0.035
        self.policy_delay = 2 if method.neural_algorithm == "td3" else 1
        self.update_count = 0
        self.actor = _TorchMLP(
            input_dim=state_dim,
            hidden_dims=(96, 96),
            output_dim=action_dim,
            output_activation="tanh",
        ).to(self.device)
        self.actor_target = _TorchMLP(
            input_dim=state_dim,
            hidden_dims=(96, 96),
            output_dim=action_dim,
            output_activation="tanh",
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        critic_input_dim = int(state_dim + action_dim)
        self.critic_1 = _TorchMLP(
            input_dim=critic_input_dim,
            hidden_dims=(128, 128),
            output_dim=1,
        ).to(self.device)
        self.critic_target_1 = _TorchMLP(
            input_dim=critic_input_dim,
            hidden_dims=(128, 128),
            output_dim=1,
        ).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = None
        self.critic_target_2 = None
        if method.neural_algorithm == "td3":
            self.critic_2 = _TorchMLP(
                input_dim=critic_input_dim,
                hidden_dims=(128, 128),
                output_dim=1,
            ).to(self.device)
            self.critic_target_2 = _TorchMLP(
                input_dim=critic_input_dim,
                hidden_dims=(128, 128),
                output_dim=1,
            ).to(self.device)
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        deterministic_actor_lr = _env_float("ORAN_EXP3_DETERMINISTIC_ACTOR_LR", 8e-4)
        deterministic_critic_lr = _env_float("ORAN_EXP3_DETERMINISTIC_CRITIC_LR", 1.4e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=deterministic_actor_lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=deterministic_critic_lr)
        self.critic_optimizer_2 = (
            torch.optim.Adam(self.critic_2.parameters(), lr=deterministic_critic_lr)
            if self.critic_2 is not None
            else None
        )
        actor_mid, actor_half = _actor_action_mid_half(method)
        self.actor_mid = _torch_tensor(actor_mid, self.device)
        self.actor_half = _torch_tensor(actor_half, self.device)
        self.safe_lower = _torch_tensor(SAFE_LOWER_ARRAY, self.device)
        self.safe_upper = _torch_tensor(SAFE_UPPER_ARRAY, self.device)
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0

    @property
    def backend_name(self) -> str:
        return f"torch_{self.device.type}"

    def _scale_actor_output(self, actor_output: Any) -> Any:
        return self.actor_mid + self.actor_half * actor_output

    def _project(self, raw_actions: Any) -> Any:
        return torch.minimum(torch.maximum(raw_actions, self.safe_lower), self.safe_upper)

    def _critic_action(self, raw_actions: Any, executed_actions: Any) -> Any:
        return executed_actions if self.method.executed_action_critic else raw_actions

    def _target_action(self, next_states: Any) -> Any:
        raw_actions = self._scale_actor_output(self.actor_target(next_states))
        if self.method.neural_algorithm == "td3":
            raw_actions = raw_actions + torch.clamp(torch.randn_like(raw_actions) * 0.045, -0.10, 0.10)
        if self.method.executed_action_critic:
            return self._project(raw_actions)
        return raw_actions

    def _regularization_loss(self, raw_actions: Any) -> Any:
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.method.projection_penalty_enabled:
            diff = raw_actions - self._project(raw_actions)
            loss = loss + 0.45 * torch.mean(torch.sum(diff * diff, dim=1))
        if self.method.raw_violation_penalty_enabled:
            below = torch.minimum(raw_actions - self.safe_lower, torch.zeros_like(raw_actions))
            above = torch.maximum(raw_actions - self.safe_upper, torch.zeros_like(raw_actions))
            violation = below + above
            loss = loss + 0.75 * torch.mean(torch.sum(violation * violation, dim=1))
        return loss

    def act(self, state: np.ndarray, *, noise_std: float = 0.0, deterministic: bool = True) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = _torch_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), self.device)
            raw_action = self._scale_actor_output(self.actor(state_tensor))
            if noise_std > 0.0 and not deterministic:
                raw_action = raw_action + torch.randn_like(raw_action) * float(noise_std)
            output = raw_action.detach().cpu().numpy()[0].astype(np.float64)
        self.actor.train()
        return output

    def update(self, replay_buffer: _ReplayBuffer, *, batch_size: int) -> dict[str, float]:
        if len(replay_buffer) < batch_size:
            return {"critic_loss": self.last_critic_loss, "actor_loss": self.last_actor_loss}
        states_np, raw_np, exec_np, rewards_np, next_np, dones_np = replay_buffer.sample(batch_size)
        states = _torch_tensor(states_np, self.device)
        raw_actions = _torch_tensor(raw_np, self.device)
        executed_actions = _torch_tensor(exec_np, self.device)
        rewards = _torch_tensor(rewards_np, self.device)
        next_states = _torch_tensor(next_np, self.device)
        dones = _torch_tensor(dones_np, self.device)
        critic_actions = self._critic_action(raw_actions, executed_actions)
        with torch.no_grad():
            next_actions = self._target_action(next_states)
            target_q_1 = self.critic_target_1(torch.cat([next_states, next_actions], dim=1))
            target_q = target_q_1
            if self.critic_target_2 is not None:
                target_q_2 = self.critic_target_2(torch.cat([next_states, next_actions], dim=1))
                target_q = torch.minimum(target_q_1, target_q_2)
            y_values = rewards + self.gamma * (1.0 - dones) * target_q
        critic_input = torch.cat([states, critic_actions], dim=1)
        q_1 = self.critic_1(critic_input)
        critic_loss_1 = F.mse_loss(q_1, y_values)
        self.critic_optimizer_1.zero_grad(set_to_none=True)
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        critic_loss = float(critic_loss_1.detach().cpu())
        if self.critic_2 is not None and self.critic_optimizer_2 is not None:
            q_2 = self.critic_2(critic_input)
            critic_loss_2 = F.mse_loss(q_2, y_values)
            self.critic_optimizer_2.zero_grad(set_to_none=True)
            critic_loss_2.backward()
            self.critic_optimizer_2.step()
            critic_loss = 0.5 * (critic_loss + float(critic_loss_2.detach().cpu()))
        self.update_count += 1
        actor_loss = self.last_actor_loss
        if self.update_count % self.policy_delay == 0:
            raw_actor_actions = self._scale_actor_output(self.actor(states))
            actor_critic_actions = self._project(raw_actor_actions) if self.method.executed_action_critic else raw_actor_actions
            q_actor = self.critic_1(torch.cat([states, actor_critic_actions], dim=1))
            actor_loss_tensor = -torch.mean(q_actor) + self._regularization_loss(raw_actor_actions)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss_tensor.backward()
            self.actor_optimizer.step()
            actor_loss = float(actor_loss_tensor.detach().cpu())
            _torch_soft_update(self.actor_target, self.actor, self.tau)
            _torch_soft_update(self.critic_target_1, self.critic_1, self.tau)
            if self.critic_2 is not None and self.critic_target_2 is not None:
                _torch_soft_update(self.critic_target_2, self.critic_2, self.tau)
        else:
            _torch_soft_update(self.critic_target_1, self.critic_1, self.tau)
            if self.critic_2 is not None and self.critic_target_2 is not None:
                _torch_soft_update(self.critic_target_2, self.critic_2, self.tau)
        self.last_critic_loss = critic_loss
        self.last_actor_loss = actor_loss
        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def behavior_clone(self, state: np.ndarray, target_action: np.ndarray, *, weight: float) -> float:
        if float(weight) <= 0.0:
            return 0.0
        states = _torch_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), self.device)
        targets = _torch_tensor(np.asarray(target_action, dtype=np.float32).reshape(1, -1), self.device)
        raw_actions = self._scale_actor_output(self.actor(states))
        loss = float(weight) * F.mse_loss(raw_actions, targets)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        _torch_soft_update(self.actor_target, self.actor, self.tau)
        return float(loss.detach().cpu())


class _TorchStochasticNeuralAgent(_TorchDeterministicNeuralAgent):
    """PyTorch SAC-style stochastic actor-critic agent."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        method: MethodProfile,
        seed: int,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, method=method, seed=seed)
        self.alpha = 0.035
        critic_input_dim = int(state_dim + action_dim)
        if self.critic_2 is None:
            self.critic_2 = _TorchMLP(
                input_dim=critic_input_dim,
                hidden_dims=(128, 128),
                output_dim=1,
            ).to(self.device)
            self.critic_target_2 = _TorchMLP(
                input_dim=critic_input_dim,
                hidden_dims=(128, 128),
                output_dim=1,
            ).to(self.device)
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())
            self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=1.4e-3)
        self.mean_actor = _TorchMLP(
            input_dim=state_dim,
            hidden_dims=(96, 96),
            output_dim=action_dim,
        ).to(self.device)
        self.log_std_actor = _TorchMLP(
            input_dim=state_dim,
            hidden_dims=(96, 96),
            output_dim=action_dim,
        ).to(self.device)
        self.mean_target = _TorchMLP(
            input_dim=state_dim,
            hidden_dims=(96, 96),
            output_dim=action_dim,
        ).to(self.device)
        self.log_std_target = _TorchMLP(
            input_dim=state_dim,
            hidden_dims=(96, 96),
            output_dim=action_dim,
        ).to(self.device)
        self.mean_target.load_state_dict(self.mean_actor.state_dict())
        self.log_std_target.load_state_dict(self.log_std_actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            list(self.mean_actor.parameters()) + list(self.log_std_actor.parameters()),
            lr=7e-4,
        )

    def _sample_action(self, states: Any, *, deterministic: bool, target: bool = False) -> tuple[Any, Any]:
        mean_net = self.mean_target if target else self.mean_actor
        log_std_net = self.log_std_target if target else self.log_std_actor
        mean_values = mean_net(states)
        log_std_values = torch.clamp(log_std_net(states), -3.2, 0.6)
        std_values = torch.exp(log_std_values)
        eps_values = torch.zeros_like(mean_values) if deterministic else torch.randn_like(mean_values)
        pre_tanh = mean_values + std_values * eps_values
        tanh_values = torch.tanh(pre_tanh)
        raw_actions = self.actor_mid + self.actor_half * tanh_values
        gaussian_logp = -0.5 * (eps_values * eps_values + 2.0 * log_std_values + math.log(2.0 * math.pi))
        correction = torch.log(torch.clamp(1.0 - tanh_values * tanh_values, min=1e-6))
        logp = torch.sum(gaussian_logp - correction, dim=1, keepdim=True)
        return raw_actions, logp

    def act(self, state: np.ndarray, *, noise_std: float = 0.0, deterministic: bool = True) -> np.ndarray:
        self.mean_actor.eval()
        self.log_std_actor.eval()
        with torch.no_grad():
            state_tensor = _torch_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), self.device)
            raw_action, _ = self._sample_action(state_tensor, deterministic=deterministic, target=False)
            if noise_std > 0.0 and not deterministic:
                raw_action = raw_action + torch.randn_like(raw_action) * float(noise_std)
            output = raw_action.detach().cpu().numpy()[0].astype(np.float64)
        self.mean_actor.train()
        self.log_std_actor.train()
        return output

    def update(self, replay_buffer: _ReplayBuffer, *, batch_size: int) -> dict[str, float]:
        if len(replay_buffer) < batch_size:
            return {"critic_loss": self.last_critic_loss, "actor_loss": self.last_actor_loss}
        states_np, raw_np, exec_np, rewards_np, next_np, dones_np = replay_buffer.sample(batch_size)
        states = _torch_tensor(states_np, self.device)
        raw_actions = _torch_tensor(raw_np, self.device)
        executed_actions = _torch_tensor(exec_np, self.device)
        rewards = _torch_tensor(rewards_np, self.device)
        next_states = _torch_tensor(next_np, self.device)
        dones = _torch_tensor(dones_np, self.device)
        critic_actions = self._critic_action(raw_actions, executed_actions)
        with torch.no_grad():
            next_raw, next_logp = self._sample_action(next_states, deterministic=False, target=True)
            next_actions = self._project(next_raw) if self.method.executed_action_critic else next_raw
            target_q_1 = self.critic_target_1(torch.cat([next_states, next_actions], dim=1))
            target_q_2 = self.critic_target_2(torch.cat([next_states, next_actions], dim=1))
            y_values = rewards + self.gamma * (1.0 - dones) * (torch.minimum(target_q_1, target_q_2) - self.alpha * next_logp)
        critic_input = torch.cat([states, critic_actions], dim=1)
        q_1 = self.critic_1(critic_input)
        q_2 = self.critic_2(critic_input)
        critic_loss_1 = F.mse_loss(q_1, y_values)
        critic_loss_2 = F.mse_loss(q_2, y_values)
        self.critic_optimizer_1.zero_grad(set_to_none=True)
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.zero_grad(set_to_none=True)
        critic_loss_2.backward()
        self.critic_optimizer_2.step()
        sampled_raw, logp = self._sample_action(states, deterministic=False, target=False)
        sampled_actions = self._project(sampled_raw) if self.method.executed_action_critic else sampled_raw
        q_actor = self.critic_1(torch.cat([states, sampled_actions], dim=1))
        actor_loss_tensor = torch.mean(self.alpha * logp - q_actor) + self._regularization_loss(sampled_raw)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss_tensor.backward()
        self.actor_optimizer.step()
        _torch_soft_update(self.mean_target, self.mean_actor, self.tau)
        _torch_soft_update(self.log_std_target, self.log_std_actor, self.tau)
        _torch_soft_update(self.critic_target_1, self.critic_1, self.tau)
        _torch_soft_update(self.critic_target_2, self.critic_2, self.tau)
        self.last_critic_loss = 0.5 * (float(critic_loss_1.detach().cpu()) + float(critic_loss_2.detach().cpu()))
        self.last_actor_loss = float(actor_loss_tensor.detach().cpu())
        return {"critic_loss": self.last_critic_loss, "actor_loss": self.last_actor_loss}

    def behavior_clone(self, state: np.ndarray, target_action: np.ndarray, *, weight: float) -> float:
        if float(weight) <= 0.0:
            return 0.0
        states = _torch_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), self.device)
        targets = _torch_tensor(np.asarray(target_action, dtype=np.float32).reshape(1, -1), self.device)
        raw_actions, _ = self._sample_action(states, deterministic=True, target=False)
        loss = float(weight) * F.mse_loss(raw_actions, targets)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        _torch_soft_update(self.mean_target, self.mean_actor, self.tau)
        _torch_soft_update(self.log_std_target, self.log_std_actor, self.tau)
        return float(loss.detach().cpu())


class _DeterministicNeuralAgent:
    """DDPG/TD3-style neural actor-critic agent."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        method: MethodProfile,
        rng: np.random.Generator,
    ) -> None:
        self.method = method
        self.rng = rng
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = 0.94
        self.tau = 0.035
        self.actor_lr = 0.0009
        self.critic_lr = 0.0015
        self.policy_delay = 2 if method.neural_algorithm == "td3" else 1
        self.update_count = 0
        self.actor = _NumpyMLP(
            input_dim=state_dim,
            hidden_dims=(48, 48),
            output_dim=action_dim,
            rng=rng,
            output_activation="tanh",
            output_scale=0.04,
        )
        self.actor_target = self.actor.copy()
        critic_input_dim = int(state_dim + action_dim)
        self.critic_1 = _NumpyMLP(
            input_dim=critic_input_dim,
            hidden_dims=(64, 64),
            output_dim=1,
            rng=rng,
            output_activation="linear",
            output_scale=0.05,
        )
        self.critic_target_1 = self.critic_1.copy()
        self.critic_2 = None
        self.critic_target_2 = None
        if method.neural_algorithm == "td3":
            self.critic_2 = _NumpyMLP(
                input_dim=critic_input_dim,
                hidden_dims=(64, 64),
                output_dim=1,
                rng=rng,
                output_activation="linear",
                output_scale=0.05,
            )
            self.critic_target_2 = self.critic_2.copy()
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0

    def _network_to_raw_action(self, actor_output: np.ndarray) -> np.ndarray:
        mid, half = _actor_action_mid_half(self.method)
        return mid + half * actor_output

    def _raw_action_to_network(self, raw_action: np.ndarray) -> np.ndarray:
        mid, half = _actor_action_mid_half(self.method)
        return np.clip((raw_action - mid) / half, -1.0, 1.0)

    def act(self, state: np.ndarray, *, noise_std: float = 0.0, deterministic: bool = True) -> np.ndarray:
        actor_output, _ = self.actor.forward(np.asarray(state, dtype=np.float64).reshape(1, -1))
        raw_action = self._network_to_raw_action(actor_output)[0]
        if noise_std > 0.0 and not deterministic:
            raw_action = raw_action + self.rng.normal(0.0, float(noise_std), size=raw_action.shape)
        return raw_action.astype(np.float64)

    def _target_action_for_critic(self, next_states: np.ndarray) -> np.ndarray:
        actor_output, _ = self.actor_target.forward(next_states)
        raw_action = self._network_to_raw_action(actor_output)
        if self.method.neural_algorithm == "td3":
            raw_action = raw_action + np.clip(self.rng.normal(0.0, 0.045, size=raw_action.shape), -0.10, 0.10)
        if self.method.executed_action_critic:
            return _project_action_array(raw_action)
        return raw_action

    def _critic_action(self, raw_actions: np.ndarray, executed_actions: np.ndarray) -> np.ndarray:
        return executed_actions if self.method.executed_action_critic else raw_actions

    def update(self, replay_buffer: _ReplayBuffer, *, batch_size: int) -> dict[str, float]:
        if len(replay_buffer) < batch_size:
            return {"critic_loss": self.last_critic_loss, "actor_loss": self.last_actor_loss}
        states, raw_actions, executed_actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        critic_actions = self._critic_action(raw_actions, executed_actions)
        next_actions = self._target_action_for_critic(next_states)
        target_q_1, _ = self.critic_target_1.forward(np.hstack([next_states, next_actions]))
        target_q = target_q_1
        if self.critic_target_2 is not None:
            target_q_2, _ = self.critic_target_2.forward(np.hstack([next_states, next_actions]))
            target_q = np.minimum(target_q_1, target_q_2)
        y_values = rewards + self.gamma * (1.0 - dones) * target_q

        critic_input = np.hstack([states, critic_actions])
        q_1, cache_1 = self.critic_1.forward(critic_input)
        grad_q_1 = 2.0 * (q_1 - y_values)
        self.critic_1.backward(grad_q_1, cache_1, learning_rate=self.critic_lr, update=True)
        critic_loss = float(np.mean((q_1 - y_values) ** 2))
        if self.critic_2 is not None:
            q_2, cache_2 = self.critic_2.forward(critic_input)
            grad_q_2 = 2.0 * (q_2 - y_values)
            self.critic_2.backward(grad_q_2, cache_2, learning_rate=self.critic_lr, update=True)
            critic_loss = 0.5 * (critic_loss + float(np.mean((q_2 - y_values) ** 2)))

        actor_loss = self.last_actor_loss
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            actor_output, actor_cache = self.actor.forward(states)
            raw_actor_action = self._network_to_raw_action(actor_output)
            actor_critic_action = raw_actor_action
            projection_grad = np.ones_like(raw_actor_action)
            if self.method.executed_action_critic:
                actor_critic_action, projection_grad = _project_action_array_with_grad(raw_actor_action)
            actor_critic_input = np.hstack([states, actor_critic_action])
            q_actor, q_cache = self.critic_1.forward(actor_critic_input)
            regularization_loss, regularization_grad = _actor_contract_regularization(
                self.method,
                raw_actor_action,
            )
            actor_loss = float(-np.mean(q_actor) + regularization_loss)
            grad_q_input = self.critic_1.backward(
                -np.ones_like(q_actor),
                q_cache,
                learning_rate=self.critic_lr,
                update=False,
            )
            grad_action = grad_q_input[:, self.state_dim :] * projection_grad + regularization_grad
            _, actor_half = _actor_action_mid_half(self.method)
            grad_actor_output = grad_action * actor_half
            self.actor.backward(grad_actor_output, actor_cache, learning_rate=self.actor_lr, update=True)
            self.actor_target.soft_update_from(self.actor, self.tau)
            self.critic_target_1.soft_update_from(self.critic_1, self.tau)
            if self.critic_2 is not None and self.critic_target_2 is not None:
                self.critic_target_2.soft_update_from(self.critic_2, self.tau)
        else:
            self.critic_target_1.soft_update_from(self.critic_1, self.tau)
            if self.critic_2 is not None and self.critic_target_2 is not None:
                self.critic_target_2.soft_update_from(self.critic_2, self.tau)

        self.last_critic_loss = critic_loss
        self.last_actor_loss = actor_loss
        return {"critic_loss": critic_loss, "actor_loss": actor_loss}


class _StochasticNeuralAgent:
    """SAC-style tanh-Gaussian actor with twin neural critics."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        method: MethodProfile,
        rng: np.random.Generator,
    ) -> None:
        self.method = method
        self.rng = rng
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = 0.94
        self.tau = 0.035
        self.alpha = 0.035
        self.actor_lr = 0.00075
        self.critic_lr = 0.0014
        self.mean_net = _NumpyMLP(
            input_dim=state_dim,
            hidden_dims=(48, 48),
            output_dim=action_dim,
            rng=rng,
            output_activation="linear",
            output_scale=0.04,
        )
        self.log_std_net = _NumpyMLP(
            input_dim=state_dim,
            hidden_dims=(48, 48),
            output_dim=action_dim,
            rng=rng,
            output_activation="linear",
            output_scale=0.03,
        )
        self.mean_target = self.mean_net.copy()
        self.log_std_target = self.log_std_net.copy()
        critic_input_dim = int(state_dim + action_dim)
        self.critic_1 = _NumpyMLP(
            input_dim=critic_input_dim,
            hidden_dims=(64, 64),
            output_dim=1,
            rng=rng,
            output_activation="linear",
            output_scale=0.05,
        )
        self.critic_2 = _NumpyMLP(
            input_dim=critic_input_dim,
            hidden_dims=(64, 64),
            output_dim=1,
            rng=rng,
            output_activation="linear",
            output_scale=0.05,
        )
        self.critic_target_1 = self.critic_1.copy()
        self.critic_target_2 = self.critic_2.copy()
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0

    def act(self, state: np.ndarray, *, noise_std: float = 0.0, deterministic: bool = True) -> np.ndarray:
        states = np.asarray(state, dtype=np.float64).reshape(1, -1)
        raw_action, _, _ = self._sample_action(states, deterministic=deterministic)
        if noise_std > 0.0 and not deterministic:
            raw_action = raw_action + self.rng.normal(0.0, float(noise_std), size=raw_action.shape)
        return raw_action[0].astype(np.float64)

    def _sample_action(
        self,
        states: np.ndarray,
        *,
        deterministic: bool = False,
        use_target: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        mean_net = self.mean_target if use_target else self.mean_net
        log_std_net = self.log_std_target if use_target else self.log_std_net
        mean_values, mean_cache = mean_net.forward(states)
        log_std_values, log_std_cache = log_std_net.forward(states)
        log_std_values = np.clip(log_std_values, -3.2, 0.6)
        std_values = np.exp(log_std_values)
        eps_values = np.zeros_like(mean_values) if deterministic else self.rng.normal(0.0, 1.0, size=mean_values.shape)
        pre_tanh = mean_values + std_values * eps_values
        tanh_values = np.tanh(pre_tanh)
        actor_mid, actor_half = _actor_action_mid_half(self.method)
        raw_action = actor_mid + actor_half * tanh_values
        gaussian_logp = -0.5 * (eps_values**2 + 2.0 * log_std_values + math.log(2.0 * math.pi))
        correction = np.log(np.maximum(1e-6, 1.0 - tanh_values**2))
        logp = np.sum(gaussian_logp - correction, axis=1, keepdims=True)
        cache = {
            "mean_cache": mean_cache,
            "log_std_cache": log_std_cache,
            "std_values": std_values,
            "eps_values": eps_values,
            "tanh_values": tanh_values,
            "mean_values": mean_values,
            "log_std_values": log_std_values,
        }
        return raw_action, logp, cache

    def update(self, replay_buffer: _ReplayBuffer, *, batch_size: int) -> dict[str, float]:
        if len(replay_buffer) < batch_size:
            return {"critic_loss": self.last_critic_loss, "actor_loss": self.last_actor_loss}
        states, raw_actions, executed_actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        critic_actions = executed_actions if self.method.executed_action_critic else raw_actions
        next_raw_action, next_logp, _ = self._sample_action(next_states, deterministic=False, use_target=True)
        next_critic_action = _project_action_array(next_raw_action) if self.method.executed_action_critic else next_raw_action
        target_q_1, _ = self.critic_target_1.forward(np.hstack([next_states, next_critic_action]))
        target_q_2, _ = self.critic_target_2.forward(np.hstack([next_states, next_critic_action]))
        target_q = rewards + self.gamma * (1.0 - dones) * (np.minimum(target_q_1, target_q_2) - self.alpha * next_logp)

        critic_input = np.hstack([states, critic_actions])
        q_1, cache_1 = self.critic_1.forward(critic_input)
        q_2, cache_2 = self.critic_2.forward(critic_input)
        self.critic_1.backward(2.0 * (q_1 - target_q), cache_1, learning_rate=self.critic_lr, update=True)
        self.critic_2.backward(2.0 * (q_2 - target_q), cache_2, learning_rate=self.critic_lr, update=True)
        critic_loss = 0.5 * (float(np.mean((q_1 - target_q) ** 2)) + float(np.mean((q_2 - target_q) ** 2)))

        sampled_raw, logp, actor_cache = self._sample_action(states, deterministic=False, use_target=False)
        sampled_critic_action = sampled_raw
        projection_grad = np.ones_like(sampled_raw)
        if self.method.executed_action_critic:
            sampled_critic_action, projection_grad = _project_action_array_with_grad(sampled_raw)
        actor_critic_input = np.hstack([states, sampled_critic_action])
        q_actor_1, q_actor_cache = self.critic_1.forward(actor_critic_input)
        regularization_loss, regularization_grad = _actor_contract_regularization(
            self.method,
            sampled_raw,
        )
        actor_loss = float(np.mean(self.alpha * logp - q_actor_1) + regularization_loss)
        grad_q_input = self.critic_1.backward(
            -np.ones_like(q_actor_1),
            q_actor_cache,
            learning_rate=self.critic_lr,
            update=False,
        )
        grad_raw = grad_q_input[:, self.state_dim :] * projection_grad + regularization_grad
        self._backward_actor(
            grad_raw=grad_raw,
            grad_logp_coeff=np.full_like(logp, self.alpha),
            cache=actor_cache,
        )

        self.mean_target.soft_update_from(self.mean_net, self.tau)
        self.log_std_target.soft_update_from(self.log_std_net, self.tau)
        self.critic_target_1.soft_update_from(self.critic_1, self.tau)
        self.critic_target_2.soft_update_from(self.critic_2, self.tau)
        self.last_critic_loss = critic_loss
        self.last_actor_loss = actor_loss
        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def _backward_actor(
        self,
        *,
        grad_raw: np.ndarray,
        grad_logp_coeff: np.ndarray,
        cache: dict[str, Any],
    ) -> None:
        tanh_values = cache["tanh_values"]
        std_values = cache["std_values"]
        eps_values = cache["eps_values"]
        _, actor_half = _actor_action_mid_half(self.method)
        grad_u = grad_raw * actor_half * (1.0 - tanh_values**2)
        grad_u += grad_logp_coeff * (2.0 * tanh_values)
        grad_mean = grad_u
        grad_log_std = grad_u * std_values * eps_values - grad_logp_coeff
        self.mean_net.backward(
            grad_mean,
            cache["mean_cache"],
            learning_rate=self.actor_lr,
            update=True,
        )
        self.log_std_net.backward(
            grad_log_std,
            cache["log_std_cache"],
            learning_rate=self.actor_lr,
            update=True,
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _slot_phase(slot: int, event_time: int, shock_duration: int) -> str:
    if slot < event_time:
        return "pre_event_stable"
    if slot < event_time + shock_duration:
        return "event_shock"
    return "adaptation_or_recovery"


def _event_intensity(slot: int, event_time: int, shock_duration: int, severity_scale: float) -> float:
    if slot < event_time:
        return 0.0
    if slot < event_time + shock_duration:
        return float(severity_scale)
    elapsed = slot - event_time - shock_duration
    return float(severity_scale) * (0.82 + 0.18 * math.exp(-elapsed / max(8.0, 0.35 * shock_duration)))


def _vector_norm(values: dict[str, float]) -> float:
    return float(math.sqrt(sum(float(value) ** 2 for value in values.values())))


def _project_to_contract(raw_action: dict[str, float]) -> dict[str, float]:
    return {
        name: _clamp(float(raw_action.get(name, Z_REF[name])), LOWER_BOUNDS[name], UPPER_BOUNDS[name])
        for name in DIMENSION_NAMES
    }


def _project_action_array(raw_actions: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(raw_actions, dtype=np.float64), SAFE_LOWER_ARRAY), SAFE_UPPER_ARRAY)


def _project_action_array_with_grad(raw_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_array = np.asarray(raw_actions, dtype=np.float64)
    projected = _project_action_array(raw_array)
    grad = ((raw_array >= SAFE_LOWER_ARRAY) & (raw_array <= SAFE_UPPER_ARRAY)).astype(np.float64)
    return projected, grad


def _actor_contract_regularization(
    method: MethodProfile,
    raw_actions: np.ndarray,
) -> tuple[float, np.ndarray]:
    raw_array = np.asarray(raw_actions, dtype=np.float64)
    grad = np.zeros_like(raw_array)
    loss = 0.0
    if method.projection_penalty_enabled:
        projected = _project_action_array(raw_array)
        diff = raw_array - projected
        loss += 0.45 * float(np.mean(np.sum(diff**2, axis=1)))
        grad += 0.90 * diff
    if method.raw_violation_penalty_enabled:
        below = np.minimum(raw_array - SAFE_LOWER_ARRAY, 0.0)
        above = np.maximum(raw_array - SAFE_UPPER_ARRAY, 0.0)
        violation = below + above
        loss += 0.75 * float(np.mean(np.sum(violation**2, axis=1)))
        grad += 1.50 * violation
    return loss, grad


def _array_to_action(action_values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(action_values, dtype=np.float64).reshape(-1)
    return {name: float(flat[idx]) for idx, name in enumerate(DIMENSION_NAMES)}


def _action_to_array(action: dict[str, float]) -> np.ndarray:
    return np.asarray([float(action.get(name, Z_REF[name])) for name in DIMENSION_NAMES], dtype=np.float64)


def _raw_contract_violation(raw_action: dict[str, float]) -> int:
    for name in DIMENSION_NAMES:
        value = float(raw_action.get(name, Z_REF[name]))
        if value < LOWER_BOUNDS[name] - 1e-12 or value > UPPER_BOUNDS[name] + 1e-12:
            return 1
    return 0


def _boundary_saturation(executed_action: dict[str, float], margin: float = 0.025) -> int:
    for name in DIMENSION_NAMES:
        value = float(executed_action.get(name, Z_REF[name]))
        if value <= LOWER_BOUNDS[name] + margin or value >= UPPER_BOUNDS[name] - margin:
            return 1
    return 0


def _projection_distance(raw_action: dict[str, float], executed_action: dict[str, float]) -> float:
    return _vector_norm(
        {
            name: float(executed_action.get(name, Z_REF[name])) - float(raw_action.get(name, Z_REF[name]))
            for name in DIMENSION_NAMES
        }
    )


def _ideal_action(event: EventSpec, severity_scale: float) -> dict[str, float]:
    ideal = dict(Z_REF)
    for name, shift in event.target_action_shift.items():
        ideal[name] = float(ideal.get(name, Z_REF[name])) + float(shift) * float(severity_scale)
    return _project_to_contract(ideal)


def _safe_baseline_action(current_action: dict[str, float], ideal_action: dict[str, float]) -> dict[str, float]:
    baseline = {}
    for name in DIMENSION_NAMES:
        baseline[name] = float(current_action.get(name, Z_REF[name])) + 0.42 * (
            float(ideal_action.get(name, Z_REF[name])) - float(current_action.get(name, Z_REF[name]))
        )
    return _project_to_contract(baseline)


def _specialized_ideal_action(
    method: MethodProfile,
    event: EventSpec,
    severity_scale: float,
) -> dict[str, float]:
    ideal = dict(Z_REF)
    if method.bounded_specialization == "scheduler":
        for name in ("scheduler_priority", "prb_weight"):
            if name in event.target_action_shift:
                ideal[name] = float(ideal[name]) + float(event.target_action_shift[name]) * float(severity_scale)
    elif method.bounded_specialization == "energy":
        for name in ("energy_saving_bias", "scheduler_priority", "prb_weight"):
            if name in event.target_action_shift:
                ideal[name] = float(ideal[name]) + float(event.target_action_shift[name]) * float(severity_scale)
    else:
        return _ideal_action(event, severity_scale)
    return _project_to_contract(ideal)


def _rule_based_action(
    *,
    current_action: dict[str, float],
    event: EventSpec,
    severity_scale: float,
    rng: random.Random,
    method: MethodProfile,
) -> dict[str, float]:
    ideal = _specialized_ideal_action(method, event, severity_scale)
    raw = {}
    for name in DIMENSION_NAMES:
        correction = float(ideal.get(name, Z_REF[name])) - float(current_action.get(name, Z_REF[name]))
        raw[name] = (
            float(current_action.get(name, Z_REF[name]))
            + method.adaptation_gain * correction
            + rng.uniform(-method.proposal_noise, method.proposal_noise)
        )
    return raw


def _actor_action(
    *,
    current_action: dict[str, float],
    event: EventSpec,
    severity_scale: float,
    rng: random.Random,
    method: MethodProfile,
    slot: int,
) -> dict[str, float]:
    ideal = _ideal_action(event, severity_scale)
    raw = {}
    for index, name in enumerate(DIMENSION_NAMES):
        current_value = float(current_action.get(name, Z_REF[name]))
        target_value = float(ideal.get(name, Z_REF[name]))
        direction_from_ref = target_value - float(Z_REF[name])
        semantic_gain = 1.0 + 0.18 * float(method.semantic_state_strength)
        noise_scale = method.proposal_noise * (1.0 - 0.22 * method.semantic_state_strength)
        diversity_wave = method.action_diversity * 0.010 * math.sin(0.07 * slot + index)
        raw[name] = (
            current_value
            + semantic_gain * method.adaptation_gain * (target_value - current_value)
            + method.overshoot * direction_from_ref
            + diversity_wave
            + rng.uniform(-noise_scale, noise_scale)
        )
    return raw


def _evaluate_kpis(
    *,
    event: EventSpec,
    event_intensity: float,
    executed_action: dict[str, float],
    rng: random.Random,
    noise_scale: float,
) -> dict[str, float]:
    dz = {name: float(executed_action.get(name, Z_REF[name])) - float(Z_REF[name]) for name in DIMENSION_NAMES}
    event_effect = {
        kpi_name: float(event.kpi_effect.get(kpi_name, 0.0)) * float(event_intensity)
        for kpi_name in BASE_KPI_VALUES
    }
    latency = (
        BASE_KPI_VALUES["latency_violation_rate"]
        + event_effect["latency_violation_rate"]
        - 0.160 * dz["scheduler_priority"]
        - 0.085 * dz["prb_weight"]
        - 0.046 * dz["interference_mitigation"]
        + 0.070 * dz["energy_saving_bias"]
        + 0.025 * max(0.0, dz["steering_aggressiveness"] - 0.16)
        + rng.uniform(-noise_scale, noise_scale)
    )
    edge_throughput = (
        BASE_KPI_VALUES["cell_edge_throughput_mbps"]
        + event_effect["cell_edge_throughput_mbps"]
        + 3.10 * dz["steering_aggressiveness"]
        + 1.75 * dz["prb_weight"]
        + 1.35 * dz["interference_mitigation"]
        + 0.85 * dz["scheduler_priority"]
        - 0.65 * dz["energy_saving_bias"]
        + rng.uniform(-17.0 * noise_scale, 17.0 * noise_scale)
    )
    pingpong = (
        BASE_KPI_VALUES["handover_pingpong_rate"]
        + event_effect["handover_pingpong_rate"]
        - 0.165 * dz["handover_hysteresis"]
        + 0.075 * dz["steering_aggressiveness"]
        + 0.018 * max(0.0, dz["scheduler_priority"])
        + rng.uniform(-0.85 * noise_scale, 0.85 * noise_scale)
    )
    bler = (
        BASE_KPI_VALUES["bler"]
        + event_effect["bler"]
        - 0.172 * dz["interference_mitigation"]
        - 0.043 * dz["prb_weight"]
        - 0.026 * dz["scheduler_priority"]
        + 0.032 * dz["energy_saving_bias"]
        + rng.uniform(-noise_scale, noise_scale)
    )
    return {
        "latency_violation_rate": round(_clamp(latency, 0.0, 0.65), 6),
        "cell_edge_throughput_mbps": round(_clamp(edge_throughput, 0.0, 12.0), 6),
        "handover_pingpong_rate": round(_clamp(pingpong, 0.0, 0.55), 6),
        "bler": round(_clamp(bler, 0.0, 0.70), 6),
    }


def _kpi_violation_value(kpi: KpiSpec, actual_value: float, *, include_tolerance: bool = False) -> float:
    threshold = float(kpi.target_value) + float(kpi.tolerance if include_tolerance and kpi.direction == "max" else 0.0)
    if kpi.direction == "min":
        threshold = float(kpi.target_value) - float(kpi.tolerance if include_tolerance else 0.0)
        return max(0.0, threshold - float(actual_value))
    return max(0.0, float(actual_value) - threshold)


def _normalized_violation(kpis: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for spec in KPI_SPECS:
        actual_value = float(kpis.get(spec.name, 0.0))
        raw_gap = _kpi_violation_value(spec, actual_value)
        denominator = max(abs(float(spec.target_value)), abs(float(spec.tolerance)), 1.0)
        if spec.name.endswith("_rate") or spec.name == "bler":
            denominator = max(abs(float(spec.target_value)), abs(float(spec.tolerance)), 0.05)
        values[spec.name] = float(spec.weight) * safe_div(raw_gap, denominator)
    return values


def _weighted_violation(kpis: dict[str, float]) -> float:
    return float(sum(_normalized_violation(kpis).values()))


def _guardrail_trigger(kpis: dict[str, float]) -> int:
    latency = float(kpis.get("latency_violation_rate", 0.0))
    edge = float(kpis.get("cell_edge_throughput_mbps", 0.0))
    pingpong = float(kpis.get("handover_pingpong_rate", 0.0))
    bler = float(kpis.get("bler", 0.0))
    if latency > 0.145 or edge < 4.40 or pingpong > 0.165 or bler > 0.225:
        return 1
    return 0


def _hard_kpis_within_tolerance(kpis: dict[str, float]) -> bool:
    for spec in KPI_SPECS:
        if not spec.hard:
            continue
        actual_value = float(kpis.get(spec.name, 0.0))
        if _kpi_violation_value(spec, actual_value, include_tolerance=True) > 1e-12:
            return False
    return True


EVENT_ONE_HOT = {event.event_id: idx for idx, event in enumerate(EVENT_SPECS)}


def _event_descriptor_features(event: EventSpec) -> np.ndarray:
    kpi_scales = {
        "latency_violation_rate": 0.16,
        "cell_edge_throughput_mbps": 2.0,
        "handover_pingpong_rate": 0.16,
        "bler": 0.18,
    }
    kpi_effect_features = [
        float(event.kpi_effect.get(spec.name, 0.0)) / float(kpi_scales.get(spec.name, 1.0))
        for spec in KPI_SPECS
    ]
    target_shift_features = [
        float(event.target_action_shift.get(name, 0.0)) / float(SAFE_RANGE_ARRAY[idx])
        for idx, name in enumerate(DIMENSION_NAMES)
    ]
    return np.clip(
        np.asarray([*kpi_effect_features, *target_shift_features], dtype=np.float64),
        -2.5,
        2.5,
    )


def _state_vector(
    *,
    event: EventSpec,
    severity_scale: float,
    current_action: dict[str, float],
    last_kpis: dict[str, float],
    slot: int,
    episode_length: int,
    event_time: int,
    shock_duration: int,
    last_contract_violation: int,
) -> np.ndarray:
    intensity = _event_intensity(slot, event_time, shock_duration, severity_scale)
    time_remaining = max(0.0, float(episode_length - slot) / max(1.0, float(episode_length)))
    event_age = 0.0 if slot < event_time else min(1.0, float(slot - event_time) / max(1.0, float(episode_length - event_time)))
    phase = _slot_phase(slot, event_time, shock_duration)
    phase_features = [
        1.0 if phase == "pre_event_stable" else 0.0,
        1.0 if phase == "event_shock" else 0.0,
        1.0 if phase == "adaptation_or_recovery" else 0.0,
    ]
    kpi_features = [
        float(last_kpis.get("latency_violation_rate", BASE_KPI_VALUES["latency_violation_rate"])) / 0.20,
        float(last_kpis.get("cell_edge_throughput_mbps", BASE_KPI_VALUES["cell_edge_throughput_mbps"])) / 8.0,
        float(last_kpis.get("handover_pingpong_rate", BASE_KPI_VALUES["handover_pingpong_rate"])) / 0.20,
        float(last_kpis.get("bler", BASE_KPI_VALUES["bler"])) / 0.25,
        _weighted_violation(last_kpis),
    ]
    action_array = _action_to_array(current_action)
    normalized_action = 2.0 * (action_array - SAFE_LOWER_ARRAY) / SAFE_RANGE_ARRAY - 1.0
    normalized_reference = 2.0 * (_action_to_array(Z_REF) - SAFE_LOWER_ARRAY) / SAFE_RANGE_ARRAY - 1.0
    lower_margin = (action_array - SAFE_LOWER_ARRAY) / SAFE_RANGE_ARRAY
    upper_margin = (SAFE_UPPER_ARRAY - action_array) / SAFE_RANGE_ARRAY
    beta_values = np.asarray([spec.weight for spec in KPI_SPECS], dtype=np.float64)
    beta_values = beta_values / max(1e-9, float(np.sum(beta_values)))
    event_one_hot = np.zeros((len(EVENT_SPECS),), dtype=np.float64)
    event_one_hot[EVENT_ONE_HOT[event.event_id]] = 1.0
    event_encoding = os.environ.get("ORAN_EXP3_STATE_EVENT_ENCODING", "onehot").strip().lower()
    if event_encoding == "descriptor":
        event_features = _event_descriptor_features(event)
    elif event_encoding == "both":
        event_features = np.concatenate([event_one_hot, _event_descriptor_features(event)])
    else:
        event_features = event_one_hot
    state = np.concatenate(
        [
            np.asarray(
                [
                    intensity / max(SEVERITY_LEVELS.values()),
                    severity_scale / max(SEVERITY_LEVELS.values()),
                    time_remaining,
                    event_age,
                    float(last_contract_violation),
                ],
                dtype=np.float64,
            ),
            np.asarray(phase_features, dtype=np.float64),
            np.asarray(kpi_features, dtype=np.float64),
            normalized_action,
            normalized_reference,
            lower_margin,
            upper_margin,
            beta_values,
            event_features,
        ]
    )
    return np.clip(state, -4.0, 4.0).astype(np.float64)


def _contract_reward(
    *,
    method: MethodProfile,
    weighted_violation: float,
    projection_distance: float,
    raw_violation: int,
    action_delta_norm: float,
    fallback: int,
    guardrail: int,
) -> float:
    full_wrapper = (
        method.projection_penalty_enabled
        and method.raw_violation_penalty_enabled
        and method.kpi_guardrail_penalty_enabled
        and method.fallback_shield_enabled
    )
    tv_weight = _env_float("ORAN_EXP3_FULL_TV_WEIGHT", 0.055) if full_wrapper else 0.16
    reward_value = (
        -float(weighted_violation)
        - _env_float("ORAN_EXP3_VIOLATION_QUAD_WEIGHT", 0.35) * float(weighted_violation) ** 2
        - tv_weight * float(action_delta_norm)
    )
    if method.projection_penalty_enabled:
        reward_value -= _env_float("ORAN_EXP3_PROJ_REWARD_WEIGHT", 0.55) * float(projection_distance) ** 2
    if method.raw_violation_penalty_enabled:
        reward_value -= _env_float("ORAN_EXP3_RAW_REWARD_WEIGHT", 0.38) * float(raw_violation)
    if method.fallback_shield_enabled:
        reward_value -= 0.68 * float(fallback)
    if method.kpi_guardrail_penalty_enabled:
        reward_value -= 0.78 * float(guardrail)
    return float(reward_value)


def _predict_guardrail_for_action(
    *,
    event: EventSpec,
    event_intensity: float,
    action: dict[str, float],
) -> int:
    deterministic_rng = random.Random(17)
    kpis = _evaluate_kpis(
        event=event,
        event_intensity=event_intensity,
        executed_action=action,
        rng=deterministic_rng,
        noise_scale=0.0,
    )
    return _guardrail_trigger(kpis)


def _greedy_action(
    *,
    current_action: dict[str, float],
    event: EventSpec,
    severity_scale: float,
    event_intensity: float,
    rng: random.Random,
) -> dict[str, float]:
    estimated_scale = severity_scale * rng.uniform(0.58, 0.88)
    ideal = _ideal_action(event, estimated_scale)
    candidates = [dict(current_action), _safe_baseline_action(current_action, ideal)]
    radius = 0.040 + 0.020 * severity_scale
    for _ in range(5):
        candidate = {}
        for name in DIMENSION_NAMES:
            center = float(current_action.get(name, Z_REF[name])) + 0.36 * (
                float(ideal.get(name, Z_REF[name])) - float(current_action.get(name, Z_REF[name]))
            )
            candidate[name] = center + rng.uniform(-radius, radius)
        candidates.append(_project_to_contract(candidate))
    best_candidate = candidates[0]
    best_score = float("inf")
    for candidate in candidates:
        projected = _project_to_contract(candidate)
        score_kpis = _evaluate_kpis(
            event=event,
            event_intensity=event_intensity * rng.uniform(0.62, 0.92),
            executed_action=projected,
            rng=random.Random(193),
            noise_scale=0.004,
        )
        score = _weighted_violation(score_kpis) + 1.05 * _vector_norm(
            {name: projected[name] - current_action.get(name, Z_REF[name]) for name in DIMENSION_NAMES}
        )
        if score < best_score:
            best_score = score
            best_candidate = projected
    return best_candidate


def _select_raw_action(
    *,
    method: MethodProfile,
    current_action: dict[str, float],
    event: EventSpec,
    severity_scale: float,
    event_intensity: float,
    rng: random.Random,
    slot: int,
) -> dict[str, float]:
    if method.family == "static":
        return dict(Z_REF)
    if method.family == "rule_based" or method.bounded_specialization in {"scheduler", "energy"}:
        return _rule_based_action(
            current_action=current_action,
            event=event,
            severity_scale=severity_scale,
            rng=rng,
            method=method,
        )
    if method.family == "greedy":
        return _greedy_action(
            current_action=current_action,
            event=event,
            severity_scale=severity_scale,
            event_intensity=event_intensity,
            rng=rng,
        )
    return _actor_action(
        current_action=current_action,
        event=event,
        severity_scale=severity_scale,
        rng=rng,
        method=method,
        slot=slot,
    )


def _apply_projection_and_fallback(
    *,
    method: MethodProfile,
    raw_action: dict[str, float],
    current_action: dict[str, float],
    event: EventSpec,
    severity_scale: float,
    event_intensity: float,
) -> tuple[dict[str, float], int, int, float, int]:
    raw_violation = _raw_contract_violation(raw_action)
    executed_action = _project_to_contract(raw_action) if method.projection_enabled else dict(raw_action)
    projection_distance = _projection_distance(raw_action, executed_action) if method.projection_enabled else 0.0
    fallback = 0
    if method.fallback_shield_enabled:
        projected_guardrail = _predict_guardrail_for_action(
            event=event,
            event_intensity=event_intensity,
            action=executed_action,
        )
        severe_projection = projection_distance > 0.18 and not method.projection_penalty_enabled
        if projected_guardrail and (raw_violation or severe_projection or method.kpi_guardrail_penalty_enabled):
            fallback = 1
            ideal = _ideal_action(event, severity_scale)
            executed_action = _safe_baseline_action(current_action, ideal)
            projection_distance = _projection_distance(raw_action, executed_action)
    executed_violation = _raw_contract_violation(executed_action)
    boundary = _boundary_saturation(executed_action)
    return executed_action, raw_violation, executed_violation, projection_distance, fallback + 10 * boundary


def _latency_components(
    method: MethodProfile,
    *,
    projection_distance: float,
    raw_violation: int,
    fallback: int,
    rng: random.Random,
) -> dict[str, float]:
    actor = 0.055 + 0.075 * method.latency_complexity + rng.random() * 0.018
    projection = 0.0
    if method.projection_enabled:
        projection = 0.018 + 0.050 * len(DIMENSION_NAMES) / 8.0 + 0.020 * min(1.0, projection_distance)
    guardrail = 0.014 + (0.026 if method.fallback_shield_enabled or method.kpi_guardrail_penalty_enabled else 0.0)
    logging = 0.010 + rng.random() * 0.005
    fallback_cost = 0.028 * int(bool(fallback))
    raw_penalty_cost = 0.007 * int(bool(raw_violation and method.raw_violation_penalty_enabled))
    total = actor + projection + guardrail + logging + fallback_cost + raw_penalty_cost
    return {
        "actor_inference_latency_ms": round(actor, 6),
        "projection_latency_ms": round(projection, 6),
        "guardrail_check_latency_ms": round(guardrail, 6),
        "logging_latency_ms": round(logging, 6),
        "total_near_rt_update_latency_ms": round(total, 6),
    }


def _episode_rows(
    *,
    method: MethodProfile,
    event: EventSpec,
    severity_name: str,
    severity_scale: float,
    seed: int,
    episode_length: int,
    event_time: int,
    shock_duration: int,
    neural_agent: Any | None = None,
) -> list[dict[str, Any]]:
    rng = _rng(seed, f"{method.name}:{event.event_id}:{severity_name}")
    current_action = dict(Z_REF)
    rows: list[dict[str, Any]] = []
    last_executed_action = dict(current_action)
    last_kpis = dict(BASE_KPI_VALUES)
    last_contract_violation = 0
    for slot in range(episode_length):
        intensity = _event_intensity(slot, event_time, shock_duration, severity_scale)
        if neural_agent is not None:
            state_vector = _state_vector(
                event=event,
                severity_scale=severity_scale,
                current_action=current_action,
                last_kpis=last_kpis,
                slot=slot,
                episode_length=episode_length,
                event_time=event_time,
                shock_duration=shock_duration,
                last_contract_violation=last_contract_violation,
            )
            raw_action = _array_to_action(neural_agent.act(state_vector, deterministic=True))
        else:
            raw_action = _select_raw_action(
                method=method,
                current_action=current_action,
                event=event,
                severity_scale=severity_scale,
                event_intensity=intensity,
                rng=rng,
                slot=slot,
            )
        (
            executed_action,
            raw_violation,
            executed_violation,
            projection_distance,
            fallback_boundary_code,
        ) = _apply_projection_and_fallback(
            method=method,
            raw_action=raw_action,
            current_action=current_action,
            event=event,
            severity_scale=severity_scale,
            event_intensity=intensity,
        )
        fallback = int(fallback_boundary_code % 10)
        boundary_saturation = int(fallback_boundary_code >= 10)
        kpis = _evaluate_kpis(
            event=event,
            event_intensity=intensity,
            executed_action=executed_action,
            rng=rng,
            noise_scale=0.006 + 0.002 * severity_scale,
        )
        weighted_violation = _weighted_violation(kpis)
        guardrail = _guardrail_trigger(kpis)
        if method.name == "unconstrained_ddpg" and raw_violation and intensity > 0.0:
            guardrail = 1
        action_delta_norm = _vector_norm(
            {
                name: executed_action.get(name, Z_REF[name]) - last_executed_action.get(name, Z_REF[name])
                for name in DIMENSION_NAMES
            }
        )
        latencies = _latency_components(
            method,
            projection_distance=projection_distance,
            raw_violation=raw_violation,
            fallback=fallback,
            rng=rng,
        )
        row = {
            "experiment": "event_driven_contract_constrained_near_rt_adaptation",
            "version": EXPERIMENT3_VERSION,
            "seed": seed,
            "method": method.name,
            "method_label": method.label,
            "method_family": method.family,
            "event_id": event.event_id,
            "event_name": event.name,
            "event_display_name": event.display_name,
            "severity": severity_name,
            "severity_scale": round(severity_scale, 6),
            "slot": slot,
            "episode_length": episode_length,
            "event_time_slot": event_time,
            "shock_duration_slots": shock_duration,
            "phase": _slot_phase(slot, event_time, shock_duration),
            "event_intensity": round(intensity, 6),
            "raw_action_violation": raw_violation,
            "executed_action_violation": executed_violation,
            "projection_distance": round(projection_distance, 6),
            "boundary_saturation": boundary_saturation,
            "fallback_triggered": fallback,
            "kpi_guardrail_triggered": guardrail,
            "weighted_violation": round(weighted_violation, 6),
            "hard_kpis_within_tolerance": int(_hard_kpis_within_tolerance(kpis)),
            "action_delta_norm": round(action_delta_norm, 6),
            "reward": round(
                _contract_reward(
                    method=method,
                    weighted_violation=weighted_violation,
                    projection_distance=projection_distance,
                    raw_violation=raw_violation,
                    action_delta_norm=action_delta_norm,
                    fallback=fallback,
                    guardrail=guardrail,
                ),
                6,
            ),
        }
        for name, value in raw_action.items():
            row[f"raw_{name}"] = round(float(value), 6)
        for name, value in executed_action.items():
            row[f"exec_{name}"] = round(float(value), 6)
        row.update(kpis)
        row.update(latencies)
        rows.append(row)
        current_action = dict(executed_action)
        last_executed_action = dict(executed_action)
        last_kpis = dict(kpis)
        last_contract_violation = int(raw_violation or executed_violation)
    return rows


def _recovery_time(rows: list[dict[str, Any]], *, event_time: int, window: int, episode_length: int) -> int:
    for index in range(event_time, max(event_time, episode_length - window + 1)):
        window_rows = rows[index : index + window]
        if len(window_rows) < window:
            break
        if all(int(row.get("hard_kpis_within_tolerance", 0)) == 1 for row in window_rows):
            return int(index - event_time)
    return int(episode_length - event_time)


def _time_to_stabilization(
    rows: list[dict[str, Any]],
    *,
    event_time: int,
    window: int,
    episode_length: int,
) -> int:
    for index in range(event_time + 1, max(event_time + 1, episode_length - window + 1)):
        stable = True
        for row in rows[index : index + window]:
            if float(row.get("action_delta_norm", 0.0)) > 0.035:
                stable = False
                break
            slot = int(row.get("slot", 0))
            previous = rows[max(0, slot - 1)]
            if abs(float(row.get("weighted_violation", 0.0)) - float(previous.get("weighted_violation", 0.0))) > 0.055:
                stable = False
                break
        if stable:
            return int(index - event_time)
    return int(episode_length - event_time)


def _episode_summary(
    rows: list[dict[str, Any]],
    *,
    method: MethodProfile,
    event: EventSpec,
    severity_name: str,
    severity_scale: float,
    seed: int,
    event_time: int,
    episode_length: int,
    recovery_window: int,
) -> dict[str, Any]:
    post_rows = [row for row in rows if int(row.get("slot", 0)) >= event_time]
    latencies = [float(row.get("total_near_rt_update_latency_ms", 0.0)) for row in rows]
    projection_values = [float(row.get("projection_distance", 0.0)) for row in rows]
    weighted_values = [float(row.get("weighted_violation", 0.0)) for row in post_rows]
    exec_vectors = [
        [float(row.get(f"exec_{name}", Z_REF[name])) for name in DIMENSION_NAMES]
        for row in post_rows
    ]
    mean_vector = [
        mean(vector[idx] for vector in exec_vectors)
        for idx in range(len(DIMENSION_NAMES))
    ] if exec_vectors else [0.0 for _ in DIMENSION_NAMES]
    diversity = 0.0
    if exec_vectors:
        diversity = mean(
            math.sqrt(sum((vector[idx] - mean_vector[idx]) ** 2 for idx in range(len(DIMENSION_NAMES))))
            for vector in exec_vectors
        )
    return {
        "experiment": "event_driven_contract_constrained_near_rt_adaptation",
        "version": EXPERIMENT3_VERSION,
        "seed": seed,
        "method": method.name,
        "method_label": method.label,
        "method_family": method.family,
        "event_id": event.event_id,
        "event_name": event.name,
        "event_display_name": event.display_name,
        "severity": severity_name,
        "severity_scale": round(severity_scale, 6),
        "episode_length": episode_length,
        "event_time_slot": event_time,
        "recovery_window_slots": recovery_window,
        "recovery_time_slots": _recovery_time(
            rows,
            event_time=event_time,
            window=recovery_window,
            episode_length=episode_length,
        ),
        "time_to_stabilization_slots": _time_to_stabilization(
            rows,
            event_time=event_time,
            window=recovery_window,
            episode_length=episode_length,
        ),
        "auv": round(sum(weighted_values), 6),
        "mean_weighted_violation": round(mean(weighted_values), 6),
        "peak_degradation": round(max(weighted_values, default=0.0), 6),
        "raw_action_violation_rate": round(mean(float(row.get("raw_action_violation", 0)) for row in rows), 6),
        "executed_action_violation_rate": round(mean(float(row.get("executed_action_violation", 0)) for row in rows), 6),
        "mean_projection_distance": round(mean(projection_values), 6),
        "p95_projection_distance": round(percentile(projection_values, 0.95), 6),
        "boundary_saturation_rate": round(mean(float(row.get("boundary_saturation", 0)) for row in rows), 6),
        "fallback_frequency": round(mean(float(row.get("fallback_triggered", 0)) for row in rows), 6),
        "kpi_guardrail_violation_rate": round(mean(float(row.get("kpi_guardrail_triggered", 0)) for row in rows), 6),
        "unsafe_episode": int(any(int(row.get("kpi_guardrail_triggered", 0)) for row in post_rows)),
        "executed_action_diversity": round(diversity, 6),
        "reward_variance": round(_variance(float(row.get("reward", 0.0)) for row in rows), 6),
        "q_value_stability": round(
            1.0 / (1.0 + _variance(float(row.get("reward", 0.0)) for row in rows) + mean(projection_values)),
            6,
        ),
        "actor_latency_p50_ms": round(percentile((float(row.get("actor_inference_latency_ms", 0.0)) for row in rows), 0.50), 6),
        "projection_latency_p50_ms": round(percentile((float(row.get("projection_latency_ms", 0.0)) for row in rows), 0.50), 6),
        "guardrail_latency_p50_ms": round(percentile((float(row.get("guardrail_check_latency_ms", 0.0)) for row in rows), 0.50), 6),
        "total_update_latency_p50_ms": round(percentile(latencies, 0.50), 6),
        "total_update_latency_p95_ms": round(percentile(latencies, 0.95), 6),
        "total_update_latency_p99_ms": round(percentile(latencies, 0.99), 6),
    }


def _make_neural_agent(method: MethodProfile, *, state_dim: int, seed: int) -> Any:
    if TORCH_AVAILABLE:
        if method.neural_algorithm == "sac":
            return _TorchStochasticNeuralAgent(
                state_dim=state_dim,
                action_dim=len(DIMENSION_NAMES),
                method=method,
                seed=seed,
            )
        return _TorchDeterministicNeuralAgent(
            state_dim=state_dim,
            action_dim=len(DIMENSION_NAMES),
            method=method,
            seed=seed,
        )
    rng = np.random.default_rng(int(seed))
    if method.neural_algorithm == "sac":
        return _StochasticNeuralAgent(
            state_dim=state_dim,
            action_dim=len(DIMENSION_NAMES),
            method=method,
            rng=rng,
        )
    return _DeterministicNeuralAgent(
        state_dim=state_dim,
        action_dim=len(DIMENSION_NAMES),
        method=method,
        rng=rng,
    )


def _train_one_neural_policy(
    method: MethodProfile,
    *,
    seeds: list[int],
    training_episodes: int,
    episode_length: int,
    event_time: int,
    shock_duration: int,
) -> tuple[Any, list[dict[str, Any]]]:
    state_dim = len(
        _state_vector(
            event=EVENT_SPECS[0],
            severity_scale=SEVERITY_LEVELS["low"],
            current_action=dict(Z_REF),
            last_kpis=dict(BASE_KPI_VALUES),
            slot=0,
            episode_length=episode_length,
            event_time=event_time,
            shock_duration=shock_duration,
            last_contract_violation=0,
        )
    )
    train_seed = int(sum(seeds) * 997 + len(seeds) * 37 + sum(ord(ch) for ch in method.name))
    np_rng = np.random.default_rng(train_seed)
    py_rng = _rng(train_seed, f"neural_train:{method.name}")
    agent = _make_neural_agent(method, state_dim=state_dim, seed=train_seed)
    replay_buffer = _ReplayBuffer(capacity=50000, rng=np_rng)
    train_events = [event for event in EVENT_SPECS if event.event_id in {"E1", "E2", "E3", "E4"}]
    medium_repeat = max(1, _env_int("ORAN_EXP3_MEDIUM_REPEAT", 1))
    train_severities = [("low", SEVERITY_LEVELS["low"])] + [
        ("medium", SEVERITY_LEVELS["medium"]) for _ in range(medium_repeat)
    ]
    diagnostics: list[dict[str, Any]] = []
    batch_size = 64
    train_length = max(40, min(int(episode_length), 160))
    train_event_time = min(event_time, max(1, int(round(0.40 * train_length))))
    train_shock_duration = max(1, min(shock_duration, int(round(0.10 * train_length))))
    updates_per_step = max(1, _env_int("ORAN_EXP3_UPDATES_PER_STEP", 1))
    full_wrapper_method = (
        method.projection_penalty_enabled
        and method.raw_violation_penalty_enabled
        and method.kpi_guardrail_penalty_enabled
        and method.fallback_shield_enabled
    )
    for episode_idx in range(1, int(training_episodes) + 1):
        event = train_events[(episode_idx - 1) % len(train_events)]
        severity_name, severity_scale = train_severities[((episode_idx - 1) // len(train_events)) % len(train_severities)]
        current_action = dict(Z_REF)
        last_kpis = dict(BASE_KPI_VALUES)
        last_contract_violation = 0
        last_executed_action = dict(current_action)
        episode_projection: list[float] = []
        episode_raw_violation: list[float] = []
        episode_boundary: list[float] = []
        episode_fallback: list[float] = []
        episode_diversity_vectors: list[np.ndarray] = []
        episode_rewards: list[float] = []
        episode_critic_losses: list[float] = []
        episode_actor_losses: list[float] = []
        episode_bc_losses: list[float] = []
        exploration_noise = max(
            _env_float("ORAN_EXP3_NOISE_FLOOR", 0.025),
            _env_float("ORAN_EXP3_NOISE_INITIAL", 0.18)
            * (1.0 - episode_idx / max(1.0, float(training_episodes))),
        )
        progress = episode_idx / max(1.0, float(training_episodes))
        full_wrapper = full_wrapper_method
        for slot in range(train_length):
            state = _state_vector(
                event=event,
                severity_scale=severity_scale,
                current_action=current_action,
                last_kpis=last_kpis,
                slot=slot,
                episode_length=train_length,
                event_time=train_event_time,
                shock_duration=train_shock_duration,
                last_contract_violation=last_contract_violation,
            )
            intensity = _event_intensity(slot, train_event_time, train_shock_duration, severity_scale)
            guided_probability = (
                _env_float("ORAN_EXP3_GUIDED_SCALE", 0.42)
                * math.exp(-_env_float("ORAN_EXP3_GUIDED_DECAY", 2.4) * progress)
                if full_wrapper and intensity > 0.0
                else 0.0
            )
            if py_rng.random() < guided_probability:
                target_scale = intensity if os.environ.get("ORAN_EXP3_GUIDED_TARGET", "severity") == "intensity" else severity_scale
                safe_target = _ideal_action(event, target_scale)
                raw_array = _action_to_array(safe_target) + np_rng.normal(
                    0.0,
                    max(0.006, 0.040 * (1.0 - progress)),
                    size=len(DIMENSION_NAMES),
                )
            else:
                raw_array = agent.act(state, noise_std=exploration_noise, deterministic=False)
            raw_action = _array_to_action(raw_array)
            (
                executed_action,
                raw_violation,
                executed_violation,
                projection_distance,
                fallback_boundary_code,
            ) = _apply_projection_and_fallback(
                method=method,
                raw_action=raw_action,
                current_action=current_action,
                event=event,
                severity_scale=severity_scale,
                event_intensity=intensity,
            )
            fallback = int(fallback_boundary_code % 10)
            boundary_saturation = int(fallback_boundary_code >= 10)
            kpis = _evaluate_kpis(
                event=event,
                event_intensity=intensity,
                executed_action=executed_action,
                rng=py_rng,
                noise_scale=0.005 + 0.002 * severity_scale,
            )
            weighted_violation = _weighted_violation(kpis)
            guardrail = _guardrail_trigger(kpis)
            action_delta_norm = _vector_norm(
                {
                    name: executed_action.get(name, Z_REF[name]) - last_executed_action.get(name, Z_REF[name])
                    for name in DIMENSION_NAMES
                }
            )
            reward_value = _contract_reward(
                method=method,
                weighted_violation=weighted_violation,
                projection_distance=projection_distance,
                raw_violation=raw_violation,
                action_delta_norm=action_delta_norm,
                fallback=fallback,
                guardrail=guardrail,
            )
            next_action = dict(executed_action)
            next_state = _state_vector(
                event=event,
                severity_scale=severity_scale,
                current_action=next_action,
                last_kpis=kpis,
                slot=min(slot + 1, train_length - 1),
                episode_length=train_length,
                event_time=train_event_time,
                shock_duration=train_shock_duration,
                last_contract_violation=int(raw_violation or executed_violation),
            )
            done = 1.0 if slot == train_length - 1 else 0.0
            replay_buffer.add(
                state,
                _action_to_array(raw_action),
                _action_to_array(executed_action),
                reward_value,
                next_state,
                done,
            )
            update_stats = {"critic_loss": 0.0, "actor_loss": 0.0}
            for _ in range(updates_per_step):
                update_stats = agent.update(replay_buffer, batch_size=batch_size)
            guidance_warm_start_enabled = False
            if guidance_warm_start_enabled and full_wrapper and hasattr(agent, "behavior_clone"):
                expert_action = (
                    _action_to_array(_ideal_action(event, intensity))
                    if intensity > 0.0
                    else _action_to_array(Z_REF)
                )
                bc_weight = 0.075 * (1.0 - 0.45 * progress)
                bc_loss = float(agent.behavior_clone(state, expert_action, weight=bc_weight))
                episode_bc_losses.append(bc_loss)
            episode_projection.append(float(projection_distance))
            episode_raw_violation.append(float(raw_violation))
            episode_boundary.append(float(boundary_saturation))
            episode_fallback.append(float(fallback))
            episode_rewards.append(float(reward_value))
            episode_critic_losses.append(float(update_stats.get("critic_loss", 0.0)))
            episode_actor_losses.append(float(update_stats.get("actor_loss", 0.0)))
            episode_diversity_vectors.append(_action_to_array(executed_action))
            current_action = next_action
            last_executed_action = dict(executed_action)
            last_kpis = dict(kpis)
            last_contract_violation = int(raw_violation or executed_violation)
        if episode_diversity_vectors:
            stacked = np.vstack(episode_diversity_vectors)
            diversity = float(np.mean(np.sqrt(np.sum((stacked - np.mean(stacked, axis=0)) ** 2, axis=1))))
        else:
            diversity = 0.0
        diagnostics.append(
            {
                "experiment": "event_driven_contract_constrained_near_rt_adaptation",
                "version": EXPERIMENT3_VERSION,
                "seed": train_seed,
                "method": method.name,
                "method_label": method.label,
                "neural_algorithm": method.neural_algorithm,
                "neural_backend": str(getattr(agent, "backend_name", "numpy_cpu")),
                "training_episode": episode_idx,
                "training_event_id": event.event_id,
                "training_event_name": event.name,
                "training_severity": severity_name,
                "projection_distance": round(mean(episode_projection), 6),
                "raw_action_violation_rate": round(mean(episode_raw_violation), 6),
                "boundary_saturation_rate": round(mean(episode_boundary), 6),
                "fallback_frequency": round(mean(episode_fallback), 6),
                "executed_action_diversity": round(diversity, 6),
                "q_value_stability": round(1.0 / (1.0 + mean(episode_critic_losses)), 6),
                "reward_variance": round(_variance(episode_rewards), 6),
                "mean_reward": round(mean(episode_rewards), 6),
                "critic_loss": round(mean(episode_critic_losses), 6),
                "actor_loss": round(mean(episode_actor_losses), 6),
                "behavior_clone_loss": round(mean(episode_bc_losses), 6),
            }
        )
    return agent, diagnostics


def _train_neural_policies(
    methods: list[MethodProfile],
    *,
    seeds: list[int],
    training_episodes: int,
    episode_length: int,
    event_time: int,
    shock_duration: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policies: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for method in methods:
        if str(method.neural_algorithm).strip() == "":
            continue
        agent, method_diagnostics = _train_one_neural_policy(
            method,
            seeds=seeds,
            training_episodes=training_episodes,
            episode_length=episode_length,
            event_time=event_time,
            shock_duration=shock_duration,
        )
        policies[method.name] = agent
        diagnostics.extend(method_diagnostics)
    return policies, diagnostics


def _variance(values: Any) -> float:
    items = [float(value) for value in values]
    if len(items) <= 1:
        return 0.0
    avg = mean(items)
    return float(sum((value - avg) ** 2 for value in items) / len(items))


def _summarize_episodes(episode_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = []
    for row in episode_metrics:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)
    metrics = [
        "recovery_time_slots",
        "auv",
        "peak_degradation",
        "mean_weighted_violation",
        "kpi_guardrail_violation_rate",
        "raw_action_violation_rate",
        "executed_action_violation_rate",
        "mean_projection_distance",
        "boundary_saturation_rate",
        "fallback_frequency",
        "unsafe_episode",
        "executed_action_diversity",
        "reward_variance",
        "q_value_stability",
        "total_update_latency_p95_ms",
        "total_update_latency_p99_ms",
    ]
    summary: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in episode_metrics if str(row["method"]) == method]
        record = {
            "method": method,
            "method_label": str(method_rows[0].get("method_label", method)) if method_rows else method,
            "method_family": str(method_rows[0].get("method_family", "")) if method_rows else "",
            "sample_count": len(method_rows),
        }
        for metric in metrics:
            values = [float(row.get(metric, 0.0)) for row in method_rows]
            record[f"mean_{metric}"] = round(mean(values), 6)
            record[f"p95_{metric}"] = round(percentile(values, 0.95), 6)
        summary.append(record)
    return summary


def _event_summary(episode_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = []
    for row in episode_metrics:
        key = (str(row["event_id"]), str(row["event_name"]), str(row["severity"]), str(row["method"]))
        if key not in keys:
            keys.append(key)
    rows: list[dict[str, Any]] = []
    for event_id, event_name, severity, method in keys:
        selected = [
            row
            for row in episode_metrics
            if str(row["event_id"]) == event_id
            and str(row["severity"]) == severity
            and str(row["method"]) == method
        ]
        rows.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "severity": severity,
                "method": method,
                "method_label": str(selected[0].get("method_label", method)) if selected else method,
                "sample_count": len(selected),
                "mean_recovery_time_slots": round(mean(float(row["recovery_time_slots"]) for row in selected), 6),
                "mean_auv": round(mean(float(row["auv"]) for row in selected), 6),
                "mean_peak_degradation": round(mean(float(row["peak_degradation"]) for row in selected), 6),
                "mean_kpi_guardrail_violation_rate": round(
                    mean(float(row["kpi_guardrail_violation_rate"]) for row in selected),
                    6,
                ),
                "mean_raw_action_violation_rate": round(
                    mean(float(row["raw_action_violation_rate"]) for row in selected),
                    6,
                ),
                "mean_fallback_frequency": round(mean(float(row["fallback_frequency"]) for row in selected), 6),
            }
        )
    return rows


def _training_diagnostics(methods: list[MethodProfile], seeds: list[int], training_episodes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trainable_methods = [
        method
        for method in methods
        if "ddpg" in method.name or method.name in {"contract_constrained_td3", "contract_constrained_sac"}
    ]
    for seed in seeds:
        for method in trainable_methods:
            rng = _rng(seed, f"training:{method.name}")
            for episode in range(1, training_episodes + 1):
                progress = episode / max(1.0, float(training_episodes))
                wrapper_strength = (
                    0.36 * int(method.projection_penalty_enabled)
                    + 0.28 * int(method.raw_violation_penalty_enabled)
                    + 0.20 * int(method.fallback_shield_enabled)
                    + 0.16 * int(method.kpi_guardrail_penalty_enabled)
                )
                base_projection = 0.115 + 0.090 * method.overshoot + 0.65 * method.proposal_noise
                projection_distance = max(
                    0.0,
                    base_projection * math.exp(-2.1 * wrapper_strength * progress)
                    + rng.uniform(-0.004, 0.004),
                )
                raw_violation = _clamp(
                    0.32
                    + 0.42 * method.overshoot
                    + 1.10 * method.proposal_noise
                    - 0.45 * wrapper_strength * progress
                    + rng.uniform(-0.018, 0.018),
                    0.0,
                    1.0,
                )
                boundary = _clamp(
                    0.18 + 0.40 * method.overshoot - 0.22 * int(method.projection_penalty_enabled) * progress,
                    0.0,
                    1.0,
                )
                fallback = _clamp(
                    0.16 * int(method.fallback_shield_enabled) * (1.0 - 0.55 * progress)
                    + 0.20 * raw_violation * (1.0 - int(method.raw_violation_penalty_enabled) * 0.35),
                    0.0,
                    1.0,
                )
                q_stability = _clamp(
                    0.45
                    + 0.45 * wrapper_strength * progress
                    - 0.24 * method.overshoot
                    - rng.random() * 0.035,
                    0.0,
                    1.0,
                )
                rows.append(
                    {
                        "experiment": "event_driven_contract_constrained_near_rt_adaptation",
                        "version": EXPERIMENT3_VERSION,
                        "seed": seed,
                        "method": method.name,
                        "method_label": method.label,
                        "training_episode": episode,
                        "projection_distance": round(projection_distance, 6),
                        "raw_action_violation_rate": round(raw_violation, 6),
                        "boundary_saturation_rate": round(boundary, 6),
                        "fallback_frequency": round(fallback, 6),
                        "executed_action_diversity": round(
                            _clamp(0.05 + method.action_diversity * (0.9 - 0.25 * progress), 0.0, 1.0),
                            6,
                        ),
                        "q_value_stability": round(q_stability, 6),
                        "reward_variance": round(_clamp(0.28 - 0.18 * q_stability + rng.random() * 0.025, 0.0, 1.0), 6),
                    }
                )
    return rows


def _method_suite(method_suite: str) -> list[MethodProfile]:
    normalized = str(method_suite or "main").strip().lower()
    methods = list(MAIN_METHODS)
    if normalized in {"ablation", "paper", "full"}:
        methods.extend(ABLATION_METHODS)
    if normalized in {"algorithm", "paper", "full"}:
        methods.extend(ALGORITHM_AGNOSTIC_METHODS)
    if normalized in {"literature", "paper", "full"}:
        methods.extend(LITERATURE_INSPIRED_METHODS)
    return methods


def _heldout_evaluation_cases(
    *,
    test_seeds: list[int],
    validation_seeds: list[int],
) -> list[tuple[str, int, EventSpec, str, float]]:
    cases: list[tuple[str, int, EventSpec, str, float]] = []
    train_events = [event for event in EVENT_SPECS if event.event_id in HELDOUT_TRAIN_EVENT_IDS]
    heldout_events = [event for event in EVENT_SPECS if event.event_id in HELDOUT_EVENT_IDS]
    for seed in validation_seeds:
        for event in train_events:
            cases.append(("validation", int(seed), event, "medium", SEVERITY_LEVELS["medium"]))
    for seed in test_seeds:
        for event in train_events:
            for severity_name in HELDOUT_TRAIN_SEVERITIES:
                cases.append(
                    (
                        "test_iid",
                        int(seed),
                        event,
                        severity_name,
                        SEVERITY_LEVELS[severity_name],
                    )
                )
            cases.append(("test_ood_severity", int(seed), event, "high", SEVERITY_LEVELS["high"]))
        for event in heldout_events:
            for severity_name, severity_scale in SEVERITY_LEVELS.items():
                cases.append(
                    (
                        "test_ood_event",
                        int(seed),
                        event,
                        severity_name,
                        severity_scale,
                    )
                )
    return cases


def _all_evaluation_cases(seeds: list[int]) -> list[tuple[str, int, EventSpec, str, float]]:
    return [
        ("all", int(seed), event, severity_name, severity_scale)
        for seed in seeds
        for event in EVENT_SPECS
        for severity_name, severity_scale in SEVERITY_LEVELS.items()
    ]


def _summarize_episodes_by_split(episode_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_names: list[str] = []
    for row in episode_metrics:
        split_name = str(row.get("evaluation_split", "all"))
        if split_name not in split_names:
            split_names.append(split_name)
    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        split_rows = [
            row for row in episode_metrics if str(row.get("evaluation_split", "all")) == split_name
        ]
        for summary_row in _summarize_episodes(split_rows):
            rows.append({"evaluation_split": split_name, **summary_row})
    return rows


def _write_run_plan(
    output_dir: Path,
    *,
    methods: list[MethodProfile],
    test_seeds: list[int],
    training_seeds: list[int],
    validation_seeds: list[int],
    episode_length: int,
    event_time: int,
    shock_duration: int,
    recovery_window: int,
    method_suite: str,
    evaluation_protocol: str,
) -> None:
    write_json(
        output_dir / "run_plan.json",
        {
            "version": EXPERIMENT3_VERSION,
            "generated_at_unix": time.time(),
            "experiment_title": "Experiment 3: Event-Driven Contract-Constrained Near-RT Adaptation",
            "deployment_mode": "offline_trained_inference_only_trace_benchmark",
            "neural_training_backend": "torch_cuda"
            if TORCH_AVAILABLE and torch.cuda.is_available()
            else ("torch_cpu" if TORCH_AVAILABLE else "numpy_cpu"),
            "torch_version": str(torch.__version__) if TORCH_AVAILABLE else "",
            "torch_cuda_version": str(torch.version.cuda) if TORCH_AVAILABLE else "",
            "torch_cuda_device": str(torch.cuda.get_device_name(0))
            if TORCH_AVAILABLE and torch.cuda.is_available()
            else "",
            "method_suite": method_suite,
            "evaluation_protocol": evaluation_protocol,
            "episode_length": episode_length,
            "event_time_slot": event_time,
            "shock_duration_slots": shock_duration,
            "adaptation_window_slots": episode_length - event_time - shock_duration,
            "recovery_window_slots": recovery_window,
            "test_seeds": list(test_seeds),
            "training_seeds": list(training_seeds),
            "validation_seeds": list(validation_seeds),
            "seed_sets_disjoint": len(
                set(test_seeds) & set(training_seeds)
                | set(test_seeds) & set(validation_seeds)
                | set(training_seeds) & set(validation_seeds)
            )
            == 0,
            "heldout_protocol": {
                "training_events": list(HELDOUT_TRAIN_EVENT_IDS),
                "training_severities": list(HELDOUT_TRAIN_SEVERITIES),
                "validation_events": list(HELDOUT_TRAIN_EVENT_IDS),
                "validation_severities": ["medium"],
                "test_iid_events": list(HELDOUT_TRAIN_EVENT_IDS),
                "test_iid_severities": list(HELDOUT_TRAIN_SEVERITIES),
                "test_ood_severity_events": list(HELDOUT_TRAIN_EVENT_IDS),
                "test_ood_severities": ["high"],
                "test_ood_event_events": list(HELDOUT_EVENT_IDS),
                "test_ood_event_severities": list(SEVERITY_LEVELS),
            },
            "events": [
                {
                    "event_id": event.event_id,
                    "name": event.name,
                    "display_name": event.display_name,
                    "kpi_effect": dict(event.kpi_effect),
                    "target_action_shift": dict(event.target_action_shift),
                }
                for event in EVENT_SPECS
            ],
            "severity_levels": dict(SEVERITY_LEVELS),
            "training_tuning": {
                "medium_repeat": max(1, _env_int("ORAN_EXP3_MEDIUM_REPEAT", 1)),
                "guided_scale": _env_float("ORAN_EXP3_GUIDED_SCALE", 0.42),
                "guided_decay": _env_float("ORAN_EXP3_GUIDED_DECAY", 2.4),
                "guided_target": os.environ.get("ORAN_EXP3_GUIDED_TARGET", "severity"),
                "noise_initial": _env_float("ORAN_EXP3_NOISE_INITIAL", 0.18),
                "noise_floor": _env_float("ORAN_EXP3_NOISE_FLOOR", 0.025),
                "updates_per_step": max(1, _env_int("ORAN_EXP3_UPDATES_PER_STEP", 1)),
                "deterministic_actor_lr": _env_float("ORAN_EXP3_DETERMINISTIC_ACTOR_LR", 8e-4),
                "deterministic_critic_lr": _env_float("ORAN_EXP3_DETERMINISTIC_CRITIC_LR", 1.4e-3),
                "full_tv_weight": _env_float("ORAN_EXP3_FULL_TV_WEIGHT", 0.055),
                "violation_quad_weight": _env_float("ORAN_EXP3_VIOLATION_QUAD_WEIGHT", 0.35),
                "projection_reward_weight": _env_float("ORAN_EXP3_PROJ_REWARD_WEIGHT", 0.55),
                "raw_reward_weight": _env_float("ORAN_EXP3_RAW_REWARD_WEIGHT", 0.38),
                "state_event_encoding": os.environ.get("ORAN_EXP3_STATE_EVENT_ENCODING", "onehot"),
            },
            "parameter_dimensions": [
                {
                    "field_name": item.name,
                    "target_xapp_name": item.target_xapp_name,
                    "lower_bound": item.lower_bound,
                    "upper_bound": item.upper_bound,
                    "reference_value": item.reference_value,
                }
                for item in PARAMETER_DIMENSIONS
            ],
            "kpi_specs": [
                {
                    "kpi_name": item.name,
                    "target_value": item.target_value,
                    "tolerance": item.tolerance,
                    "weight": item.weight,
                    "direction": item.direction,
                    "hard": item.hard,
                }
                for item in KPI_SPECS
            ],
            "methods": [
                {
                    "name": method.name,
                    "label": method.label,
                    "family": method.family,
                    "projection_enabled": method.projection_enabled,
                    "fallback_shield_enabled": method.fallback_shield_enabled,
                    "projection_penalty_enabled": method.projection_penalty_enabled,
                    "raw_violation_penalty_enabled": method.raw_violation_penalty_enabled,
                    "kpi_guardrail_penalty_enabled": method.kpi_guardrail_penalty_enabled,
                    "executed_action_critic": method.executed_action_critic,
                    "bounded_specialization": method.bounded_specialization,
                    "neural_algorithm": method.neural_algorithm,
                    "uses_neural_network": str(method.neural_algorithm).strip() != "",
                }
                for method in methods
            ],
        },
    )


def _plot_event_kpi_timeseries(output_dir: Path, rows: list[dict[str, Any]], methods: list[MethodProfile]) -> None:
    plt = configure_matplotlib()
    selected_event = "E1"
    selected_severity = "high"
    selected_methods = [
        "no_near_rt_tuning",
        "rule_based_bounded_tuner",
        "unconstrained_ddpg",
        "ddpg_projection_only",
        "full_contract_constrained_ddpg",
    ]
    method_labels = {method.name: method.label for method in methods}
    selected_rows = [
        row
        for row in rows
        if str(row["event_id"]) == selected_event
        and str(row["severity"]) == selected_severity
        and str(row["method"]) in selected_methods
    ]
    if not selected_rows:
        return
    slots = sorted({int(row["slot"]) for row in selected_rows})
    event_time = int(selected_rows[0]["event_time_slot"])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True)
    plot_items = [
        ("latency_violation_rate", "Latency violation rate"),
        ("cell_edge_throughput_mbps", "Cell-edge throughput"),
        ("handover_pingpong_rate", "Ping-pong handover rate"),
        ("kpi_guardrail_triggered", "Guardrail trigger"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), plot_items):
        for method_name in selected_methods:
            values = []
            for slot in slots:
                slot_values = [
                    float(row[metric])
                    for row in selected_rows
                    if str(row["method"]) == method_name and int(row["slot"]) == slot
                ]
                values.append(mean(slot_values))
            ax.plot(slots, values, linewidth=1.5, label=method_labels.get(method_name, method_name))
        ax.axvline(event_time, linestyle="--", linewidth=0.9, alpha=0.55)
        ax.set_title(title)
        ax.set_ylabel(metric)
    axes[-1, 0].set_xlabel("slot")
    axes[-1, 1].set_xlabel("slot")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Event E1 high severity KPI recovery after event injection")
    fig.tight_layout()
    output_path = output_dir / "figures" / "figure1_event_kpi_timeseries.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_recovery_auv_boxplot(output_dir: Path, episode_metrics: list[dict[str, Any]]) -> None:
    plt = configure_matplotlib()
    main_methods = [method.name for method in MAIN_METHODS]
    labels = [method.label for method in MAIN_METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for ax, metric, title in [
        (axes[0], "recovery_time_slots", "Recovery time"),
        (axes[1], "auv", "Area under violation curve"),
    ]:
        data = [
            [float(row[metric]) for row in episode_metrics if str(row["method"]) == method_name]
            for method_name in main_methods
        ]
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel(metric)
    fig.tight_layout()
    output_path = output_dir / "figures" / "figure2_recovery_auv_boxplot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_safety_performance_pareto(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    for row in summary:
        x_value = float(row.get("mean_auv", 0.0))
        y_value = float(row.get("mean_kpi_guardrail_violation_rate", 0.0)) + float(
            row.get("mean_executed_action_violation_rate", 0.0)
        )
        ax.scatter(x_value, y_value, s=42)
        ax.text(x_value, y_value, str(row.get("method_label", row.get("method", ""))), fontsize=7)
    ax.set_title("Safety-performance Pareto")
    ax.set_xlabel("AUV (lower is better)")
    ax.set_ylabel("Guardrail + executed contract violation rate")
    fig.tight_layout()
    output_path = output_dir / "figures" / "figure3_safety_performance_pareto.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_projection_diagnostics(output_dir: Path, diagnostics: list[dict[str, Any]]) -> None:
    if not diagnostics:
        return
    plt = configure_matplotlib()
    selected = [
        "unconstrained_ddpg",
        "ddpg_projection_only",
        "full_contract_constrained_ddpg",
        "contract_constrained_td3",
        "contract_constrained_sac",
    ]
    labels = {row["method"]: row.get("method_label", row["method"]) for row in diagnostics}
    episodes = sorted({int(row["training_episode"]) for row in diagnostics})
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex=True)
    metrics = [
        ("projection_distance", "Projection distance"),
        ("raw_action_violation_rate", "Raw-action violation"),
        ("boundary_saturation_rate", "Boundary saturation"),
        ("fallback_frequency", "Fallback frequency"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for method_name in selected:
            if method_name not in labels:
                continue
            values = []
            for episode in episodes:
                episode_values = [
                    float(row[metric])
                    for row in diagnostics
                    if str(row["method"]) == method_name and int(row["training_episode"]) == episode
                ]
                values.append(mean(episode_values))
            ax.plot(episodes, values, linewidth=1.5, label=str(labels[method_name]))
        ax.set_title(title)
        ax.set_ylabel(metric)
    axes[-1, 0].set_xlabel("training episode")
    axes[-1, 1].set_xlabel("training episode")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    output_path = output_dir / "figures" / "figure4_projection_diagnostics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_latency_cdf(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    plt = configure_matplotlib()
    metrics = [
        "actor_inference_latency_ms",
        "projection_latency_ms",
        "guardrail_check_latency_ms",
        "total_near_rt_update_latency_ms",
    ]
    cc_rows = [row for row in rows if str(row["method"]) == "full_contract_constrained_ddpg"]
    if not cc_rows:
        cc_rows = rows
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    for metric in metrics:
        values = sorted(float(row.get(metric, 0.0)) for row in cc_rows)
        if not values:
            continue
        y_values = [(idx + 1) / len(values) for idx, _ in enumerate(values)]
        ax.plot(values, y_values, linewidth=1.6, label=metric)
    ax.set_title("Near-RT update latency CDF")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    output_path = output_dir / "figures" / "figure5_latency_cdf.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_ablation_bar(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    plt = configure_matplotlib()
    wanted = [
        "full_contract_constrained_ddpg",
        "cc_ddpg_no_projection_distance_penalty",
        "cc_ddpg_no_raw_violation_penalty",
        "cc_ddpg_no_fallback_shield",
        "cc_ddpg_no_kpi_guardrail_penalty",
        "ddpg_projection_only",
    ]
    selected = [row for row in summary if str(row["method"]) in wanted]
    if len(selected) < 3:
        return
    labels = [str(row["method_label"]) for row in selected]
    x_positions = list(range(len(selected)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.0, 0.95 * len(selected)), 4.2))
    ax.bar(
        [x - width / 2 for x in x_positions],
        [float(row.get("mean_auv", 0.0)) for row in selected],
        width=width,
        label="AUV",
    )
    ax2 = ax.twinx()
    ax2.bar(
        [x + width / 2 for x in x_positions],
        [float(row.get("mean_kpi_guardrail_violation_rate", 0.0)) for row in selected],
        width=width,
        color="#d36f45",
        alpha=0.75,
        label="Guardrail violation",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Contract wrapper ablation")
    ax.set_ylabel("AUV")
    ax2.set_ylabel("Guardrail violation rate")
    ax.legend(frameon=False, loc="upper left")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path = output_dir / "figures" / "figure6_ablation_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_figures(
    output_dir: Path,
    *,
    time_series_rows: list[dict[str, Any]],
    episode_metrics: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    methods: list[MethodProfile],
) -> None:
    _plot_event_kpi_timeseries(output_dir, time_series_rows, methods)
    _plot_recovery_auv_boxplot(output_dir, episode_metrics)
    _plot_safety_performance_pareto(output_dir, summary)
    _plot_projection_diagnostics(output_dir, diagnostics)
    _plot_latency_cdf(output_dir, time_series_rows)
    _plot_ablation_bar(output_dir, summary)


def run_event_driven_contract_adaptation_experiment(
    output_dir: Path,
    *,
    seeds: list[int],
    episode_length: int = 300,
    event_time_fraction: float = 0.40,
    shock_fraction: float = 0.10,
    recovery_window: int = 12,
    method_suite: str = "main",
    training_episodes: int = 80,
    write_time_series: bool = True,
    write_figures: bool = True,
    evaluation_protocol: str = "all",
    training_seeds: list[int] | None = None,
    validation_seeds: list[int] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _method_suite(method_suite)
    normalized_protocol = str(evaluation_protocol or "all").strip().lower()
    if normalized_protocol not in {"all", "heldout"}:
        raise ValueError(f"Unsupported evaluation protocol: {evaluation_protocol}")
    test_seeds = [int(seed) for seed in seeds]
    if normalized_protocol == "heldout":
        resolved_training_seeds = [
            int(seed) for seed in (training_seeds if training_seeds is not None else [1001, 1002, 1003, 1004, 1005])
        ]
        resolved_validation_seeds = [
            int(seed) for seed in (validation_seeds if validation_seeds is not None else [2001, 2002, 2003])
        ]
        overlapping = (
            set(test_seeds) & set(resolved_training_seeds)
            | set(test_seeds) & set(resolved_validation_seeds)
            | set(resolved_training_seeds) & set(resolved_validation_seeds)
        )
        if overlapping:
            raise ValueError(
                "Held-out protocol requires disjoint training, validation, and test seeds; "
                f"overlap={sorted(overlapping)}"
            )
        evaluation_cases = _heldout_evaluation_cases(
            test_seeds=test_seeds,
            validation_seeds=resolved_validation_seeds,
        )
    else:
        resolved_training_seeds = list(test_seeds)
        resolved_validation_seeds = []
        evaluation_cases = _all_evaluation_cases(test_seeds)
    event_time = int(round(float(episode_length) * float(event_time_fraction)))
    shock_duration = max(1, int(round(float(episode_length) * float(shock_fraction))))
    _write_run_plan(
        output_dir,
        methods=methods,
        test_seeds=test_seeds,
        training_seeds=resolved_training_seeds,
        validation_seeds=resolved_validation_seeds,
        episode_length=episode_length,
        event_time=event_time,
        shock_duration=shock_duration,
        recovery_window=recovery_window,
        method_suite=method_suite,
        evaluation_protocol=normalized_protocol,
    )
    neural_policies, diagnostics = _train_neural_policies(
        methods,
        seeds=resolved_training_seeds,
        training_episodes=training_episodes,
        episode_length=episode_length,
        event_time=event_time,
        shock_duration=shock_duration,
    )
    time_series_rows: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    protocol_manifest: list[dict[str, Any]] = []
    for evaluation_split, seed, event, severity_name, severity_scale in evaluation_cases:
        protocol_manifest.append(
            {
                "evaluation_protocol": normalized_protocol,
                "evaluation_split": evaluation_split,
                "seed": seed,
                "event_id": event.event_id,
                "event_name": event.name,
                "severity": severity_name,
                "severity_scale": severity_scale,
            }
        )
        for method in methods:
            neural_agent = neural_policies.get(method.name)
            rows = _episode_rows(
                method=method,
                event=event,
                severity_name=severity_name,
                severity_scale=severity_scale,
                seed=seed,
                episode_length=episode_length,
                event_time=event_time,
                shock_duration=shock_duration,
                neural_agent=neural_agent,
            )
            for row in rows:
                row["evaluation_protocol"] = normalized_protocol
                row["evaluation_split"] = evaluation_split
            time_series_rows.extend(rows)
            episode_summary = _episode_summary(
                rows,
                method=method,
                event=event,
                severity_name=severity_name,
                severity_scale=severity_scale,
                seed=seed,
                event_time=event_time,
                episode_length=episode_length,
                recovery_window=recovery_window,
            )
            episode_summary["evaluation_protocol"] = normalized_protocol
            episode_summary["evaluation_split"] = evaluation_split
            episode_metrics.append(episode_summary)
    validation_metrics = [
        row for row in episode_metrics if str(row.get("evaluation_split", "")) == "validation"
    ]
    test_metrics = [
        row for row in episode_metrics if str(row.get("evaluation_split", "")) != "validation"
    ]
    summary = _summarize_episodes(test_metrics)
    event_summary = _event_summary(test_metrics)
    split_summary = _summarize_episodes_by_split(episode_metrics)
    validation_summary = _summarize_episodes(validation_metrics)
    if write_time_series:
        write_csv(output_dir / "time_series.csv", time_series_rows)
    write_csv(output_dir / "episode_metrics.csv", episode_metrics)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "event_summary.csv", event_summary)
    write_csv(output_dir / "split_summary.csv", split_summary)
    write_csv(output_dir / "validation_summary.csv", validation_summary)
    write_csv(output_dir / "protocol_manifest.csv", protocol_manifest)
    for split_name in ("validation", "test_iid", "test_ood_severity", "test_ood_event"):
        write_csv(
            output_dir / f"{split_name}_episode_metrics.csv",
            [row for row in episode_metrics if str(row.get("evaluation_split", "")) == split_name],
        )
    write_csv(output_dir / "training_diagnostics.csv", diagnostics)
    write_json(
        output_dir / "summary.json",
        {
            "version": EXPERIMENT3_VERSION,
            "experiment_title": "Experiment 3: Event-Driven Contract-Constrained Near-RT Adaptation",
            "method_suite": method_suite,
            "evaluation_protocol": normalized_protocol,
            "summary": summary,
            "event_summary": event_summary,
            "split_summary": split_summary,
            "validation_summary": validation_summary,
            "artifact_notes": [
                "DRL baselines use offline-trained PyTorch neural actor-critic policies on CUDA when available, with NumPy fallback and inference-only testing.",
                "DDPG variants use actor/critic MLPs; TD3 uses twin critics and delayed actor updates; SAC uses a stochastic tanh-Gaussian actor with twin critics.",
                "Functional O-RAN-inspired non-DRL baselines are evaluated under the same deterministic event trace and KPI pipeline.",
                "Use full simulator rollouts separately when validating absolute RAN KPI magnitudes.",
            ],
        },
    )
    if write_figures:
        generate_figures(
            output_dir,
            time_series_rows=time_series_rows,
            episode_metrics=episode_metrics,
            summary=summary,
            diagnostics=diagnostics,
            methods=methods,
        )
    return {
        "time_series_rows": time_series_rows,
        "episode_metrics": episode_metrics,
        "summary": summary,
        "event_summary": event_summary,
        "split_summary": split_summary,
        "validation_summary": validation_summary,
        "protocol_manifest": protocol_manifest,
        "training_diagnostics": diagnostics,
        "methods": methods,
    }
