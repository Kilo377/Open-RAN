"""
Single-user MIMO helper logic shared by simulator state derivation and PHY.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SingleUserMimoEvaluation:
    pre_eq_sinr_db: float
    post_eq_sinr_db: float
    channel_rank_limit: int
    precoder_match_score: float
    rx_combining_gain_db: float
    inter_layer_interference_db: float
    spatial_mode: str


class SingleUserMimoModel:
    """
    Lightweight single-user MIMO abstraction used before full waveform MIMO.
    """

    def build_precoder_matrix(
        self,
        *,
        num_tx_ant: int,
        layer_count: int,
        precoder_id: int,
    ) -> list[list[float]]:
        normalized_num_tx_ant = max(1, int(num_tx_ant))
        normalized_layer_count = max(1, int(layer_count))
        amplitude = 1.0 / math.sqrt(float(normalized_num_tx_ant))
        basis = [1.0, 0.5, -1.0, -0.5]

        matrix: list[list[float]] = []
        for ant_index in range(normalized_num_tx_ant):
            row: list[float] = []
            for layer_index in range(normalized_layer_count):
                basis_index = (int(precoder_id) + ant_index * (layer_index + 1) + layer_index) % len(basis)
                row.append(round(amplitude * basis[basis_index], 4))
            matrix.append(row)
        return matrix

    def build_combiner_vector(
        self,
        *,
        num_rx_ant: int,
        combiner_id: int,
    ) -> list[float]:
        normalized_num_rx_ant = max(1, int(num_rx_ant))
        amplitude = 1.0 / math.sqrt(float(normalized_num_rx_ant))
        basis = [1.0, 0.5, -1.0, -0.5]
        return [
            round(amplitude * basis[(int(combiner_id) + rx_index) % len(basis)], 4)
            for rx_index in range(normalized_num_rx_ant)
        ]

    def evaluate(
        self,
        *,
        pre_eq_sinr_db: float,
        num_tx_ant: int,
        num_rx_ant: int,
        layer_count: int,
        rank_indicator: int,
        precoder_id: int,
        combiner_id: int,
        reported_ri: int | None,
        reported_pmi: int | None,
        delay_spread_ns: float,
        doppler_hz: float,
        channel_model_type: str,
        k_factor_db: float = 0.0,
        angle_spread_deg: float = 35.0,
        spatial_correlation_level: float = 0.5,
    ) -> SingleUserMimoEvaluation:
        normalized_pre_eq_sinr_db = float(pre_eq_sinr_db)
        normalized_num_tx_ant = max(1, int(num_tx_ant))
        normalized_num_rx_ant = max(1, int(num_rx_ant))
        normalized_layer_count = max(1, int(layer_count))
        normalized_rank_indicator = max(1, int(rank_indicator))
        max_spatial_rank = max(1, min(normalized_num_tx_ant, normalized_num_rx_ant))
        reported_rank_limit = max(
            1,
            int(reported_ri if reported_ri is not None else normalized_rank_indicator),
        )

        if normalized_pre_eq_sinr_db < 8.0:
            quality_rank_limit = 1
        elif normalized_pre_eq_sinr_db < 18.0:
            quality_rank_limit = min(2, max_spatial_rank)
        else:
            quality_rank_limit = max_spatial_rank

        channel_rank_limit = max(
            1,
            min(max_spatial_rank, reported_rank_limit, quality_rank_limit),
        )

        reference_pmi = max(
            0,
            int(reported_pmi if reported_pmi is not None else combiner_id),
        )
        pmi_span = max(1, max_spatial_rank)
        pmi_delta = abs(int(precoder_id) - reference_pmi)
        precoder_match_score = max(
            0.0,
            min(1.0, 1.0 - float(pmi_delta) / float(pmi_span)),
        )

        rx_combining_gain_db = 10.0 * math.log10(float(max(1, min(normalized_num_rx_ant, channel_rank_limit))))
        desired_layer_gain_db = 0.8 * math.log2(float(max(1, normalized_layer_count)))
        mismatch_penalty_db = 3.0 * (1.0 - precoder_match_score)
        rank_penalty_db = 4.5 * max(0, normalized_layer_count - channel_rank_limit)
        inter_layer_interference_db = (
            float(max(0, normalized_layer_count - 1))
            * (0.75 + 1.75 * (1.0 - precoder_match_score))
        )
        delay_penalty_db = min(1.5, max(0.0, float(delay_spread_ns)) / 2000.0)
        doppler_penalty_db = min(1.0, max(0.0, float(doppler_hz)) / 300.0)
        k_factor_bonus_db = min(1.0, max(0.0, float(k_factor_db)) / 12.0)
        angle_spread_bonus_db = min(0.6, max(0.0, float(angle_spread_deg)) / 90.0)
        spatial_corr_penalty_db = min(
            1.2,
            max(0.0, float(spatial_correlation_level)) * max(0, normalized_layer_count - 1),
        )
        channel_model_type_normalized = str(channel_model_type).strip().lower()
        channel_model_bonus_db = 0.5 if ("cdl" in channel_model_type_normalized or channel_model_type_normalized.endswith("_d")) else 0.0

        post_eq_sinr_db = (
            normalized_pre_eq_sinr_db
            + 0.6 * rx_combining_gain_db
            + desired_layer_gain_db
            + channel_model_bonus_db
            + k_factor_bonus_db
            + angle_spread_bonus_db
            - mismatch_penalty_db
            - rank_penalty_db
            - inter_layer_interference_db
            - spatial_corr_penalty_db
            - delay_penalty_db
            - doppler_penalty_db
        )

        if normalized_layer_count == 1 and max_spatial_rank > 1:
            post_eq_sinr_db += 0.4 * math.log2(float(max_spatial_rank))
        elif normalized_layer_count > 1 and channel_rank_limit >= normalized_layer_count and normalized_pre_eq_sinr_db >= 10.0:
            post_eq_sinr_db += 1.0 * precoder_match_score

        spatial_mode = "su_mimo" if normalized_layer_count > 1 else "su_siso"
        return SingleUserMimoEvaluation(
            pre_eq_sinr_db=normalized_pre_eq_sinr_db,
            post_eq_sinr_db=float(post_eq_sinr_db),
            channel_rank_limit=int(channel_rank_limit),
            precoder_match_score=float(precoder_match_score),
            rx_combining_gain_db=float(rx_combining_gain_db),
            inter_layer_interference_db=float(inter_layer_interference_db),
            spatial_mode=spatial_mode,
        )
