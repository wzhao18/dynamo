// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode step-cost worker selection for data-parallel decode engines. It is the default policy
//! for a decode pool with no explicitly selected policy (see `DEFAULT_DECODE_POLICY_TYPE`).
//!
//! Every candidate is one `(worker, dp_rank)` of a lockstep DP decode engine. The ranks step
//! together, so the slowest rank sets every decode step; the policy therefore places each new
//! request on the rank whose predicted per-step cost after admission is lowest:
//!
//! `cost(r) = per_request * (active_requests(r) + 1)
//!          + per_mtok * (active_tokens(r) + isl) / 1e6`
//!
//! where `active_tokens(r) = active_blocks(r) * block_size` is the router's booked KV footprint of
//! the rank. Ties resolve to the rank with fewer active requests.
//!
//! The default selector balances the KV footprint alone. On Kimi K3.1 DCP4xDP4 decode (VR200,
//! AgentX c576) that keeps the summed context within 3% across ranks but leaves the request count
//! 1.17x (p50) / 1.30x (p90) max/mean, and the served step period grows with the busiest rank's
//! request count (+0.095 ms per request, regression over 9.8K lockstep steps). The per-request
//! term prices that. Defaults: 0.095 ms per request, 0.26 ms per million context tokens (decode
//! indexer logits scale with summed context per rank).
//!
//! Placement only: no request's computation changes.

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "decode-step-cost";

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// ms per running request on a rank per decode step.
    per_request: f64,
    /// ms per million context tokens on a rank per decode step.
    per_mtok: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            per_request: 0.095,
            per_mtok: 0.26,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        for (name, value) in [("per_request", self.per_request), ("per_mtok", self.per_mtok)] {
            if !value.is_finite() || value < 0.0 {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be a finite number >= 0"
                )));
            }
        }
        if self.per_request == 0.0 && self.per_mtok == 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "per_request and per_mtok cannot both be 0",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Rank {
    active_requests: usize,
    active_blocks: f64,
}

fn row_costs(parameters: &Parameters, isl: usize, block_size: u32, ranks: &[Rank]) -> Vec<f64> {
    let bs = block_size.max(1) as f64;
    ranks
        .iter()
        .map(|rank| {
            let tokens = rank.active_blocks * bs + isl as f64;
            parameters.per_request * (rank.active_requests + 1) as f64
                + parameters.per_mtok * tokens / 1e6
        })
        .collect()
}

fn select_row(costs: &[f64], ranks: &[Rank]) -> Option<usize> {
    (0..costs.len()).min_by(|&a, &b| {
        costs[a]
            .total_cmp(&costs[b])
            .then(ranks[a].active_requests.cmp(&ranks[b].active_requests))
    })
}

struct DecodeStepCostPicker {
    parameters: Parameters,
}

impl WorkerPicker for DecodeStepCostPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if load.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        let ranks: Vec<Rank> = load
            .iter()
            .map(|load| Rank {
                active_requests: load.active_requests(),
                active_blocks: load.decode_cost_blocks(),
            })
            .collect();
        let costs = row_costs(
            &self.parameters,
            context.isl_tokens(),
            context.block_size(),
            &ranks,
        );
        select_row(&costs, &ranks)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(DecodeStepCostPicker { parameters }),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: u32 = 16384;

    fn r(active_requests: usize, active_blocks: f64) -> Rank {
        Rank {
            active_requests,
            active_blocks,
        }
    }

    #[test]
    fn equal_footprint_prefers_fewer_requests() {
        // Same booked KV, one rank carries 49 requests and another 39: the count term decides.
        let p = Parameters::default();
        let ranks = [r(49, 450.0), r(39, 450.0), r(44, 450.0)];
        let costs = row_costs(&p, 120_000, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn footprint_still_counts() {
        // Equal counts: the smaller footprint wins.
        let p = Parameters::default();
        let ranks = [r(40, 500.0), r(40, 400.0)];
        let costs = row_costs(&p, 120_000, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn count_and_footprint_trade_off_at_the_fitted_rate() {
        // 1 request = per_request / per_mtok million tokens = 0.365M tokens ~ 22.3 blocks.
        let p = Parameters::default();
        let blocks_per_request = p.per_request / p.per_mtok * 1e6 / BS as f64;
        let ranks = [r(41, 400.0), r(40, 400.0 + blocks_per_request + 1.0)];
        let costs = row_costs(&p, 0, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(0));
        let ranks = [r(41, 400.0), r(40, 400.0 + blocks_per_request - 1.0)];
        let costs = row_costs(&p, 0, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn ties_go_to_fewer_requests() {
        let p = Parameters {
            per_request: 0.0,
            per_mtok: 1.0,
        };
        let ranks = [r(5, 10.0), r(3, 10.0)];
        let costs = row_costs(&p, 0, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn validation_rejects_bad_parameters() {
        assert!(Parameters { per_request: -1.0, per_mtok: 0.26 }.validate().is_err());
        assert!(Parameters { per_request: f64::NAN, per_mtok: 0.26 }.validate().is_err());
        assert!(Parameters { per_request: 0.0, per_mtok: 0.0 }.validate().is_err());
        assert!(Parameters::default().validate().is_ok());
    }

    #[test]
    fn empty_candidates_select_nothing() {
        let p = Parameters::default();
        assert_eq!(select_row(&row_costs(&p, 0, BS, &[]), &[]), None);
    }
}
