// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill step-cost worker selection for data-parallel prefill engines behind a shared KV store.
//!
//! Every candidate is one `(worker, dp_rank)`. The policy estimates, in milliseconds, when this
//! prompt's prefill would finish on each rank and picks the earliest:
//!
//! - `backlog(r)`: prefill work already routed to the rank, from the router's booked uncached
//!   tokens and request count, `uncached * (active_prefill_tokens + backlog_context *
//!   active_prefill_tokens) + per_request * active_requests`.
//! - `request(r)`: this prompt's own compute on the rank. The cached prefix is the longer of the
//!   rank's device-KV overlap and the shared-store prefix (every rank can load the store), so the
//!   uncached tokens `q` are the same on every rank unless a rank holds more on the GPU; cost is
//!   `uncached * q + uncached_times_context * q * (cached + q / 2) + per_request`.
//! - `load(r)`: time before the prompt can start when the rank must load the store prefix beyond
//!   its device overlap, `load_fixed + load_per_token * load_tokens` (0 when the device holds the
//!   whole cached prefix). Loading overlaps the rank's backlog, so the start time is
//!   `max(load, backlog)`.
//! - DP lockstep: ranks of a DP engine step together, so the busiest rank sets every step. Raising
//!   a rank above the current busiest rank's backlog costs `balance_weight` times the excess.
//!
//! `cost(r) = max(load(r), backlog(r)) + request(r) + balance_weight * max(0, backlog(r) +
//! request(r) - max_backlog)`. Ties resolve to the rank with fewer active requests.
//!
//! With `store_retention_secs > 0` the policy also enables the router's global-prefix store model
//! (blocks routed earlier to any worker are resident in the store for that long), which supplies
//! the shared-store prefix and books store-resident tokens as cached.
//!
//! Coefficients default to the Kimi K3.1 DEP16 prefill fit on VR200 (step cost per rank: 0.0253
//! ms/token, 1.03e-7 ms per token x context token, 18.7 ms per chunk; Mooncake load about 0.14 s +
//! 19 KB/token at ~5 GB/s plus two lockstep steps of admission pipeline).

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerCacheInput, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "prefill-step-cost";

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// ms per uncached token.
    uncached: f64,
    /// ms per (uncached token x context token).
    uncached_times_context: f64,
    /// ms per request (per chunk) on a rank.
    per_request: f64,
    /// Mean context (tokens) assumed for booked backlog tokens.
    backlog_context: f64,
    /// ms before a store load completes and the prompt is schedulable (lookup, transfer start,
    /// admission steps).
    load_fixed: f64,
    /// ms per token loaded from the store.
    load_per_token: f64,
    /// Weight of the DP-lockstep excess over the busiest rank's backlog.
    balance_weight: f64,
    /// Store prefix retention in seconds; > 0 enables the router's global-prefix store model.
    store_retention_secs: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            uncached: 0.0253,
            uncached_times_context: 1.03e-7,
            per_request: 18.7,
            backlog_context: 150_000.0,
            load_fixed: 1_850.0,
            load_per_token: 0.0038,
            balance_weight: 1.0,
            store_retention_secs: 0,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        let nonneg = [
            ("uncached", self.uncached),
            ("uncached_times_context", self.uncached_times_context),
            ("per_request", self.per_request),
            ("backlog_context", self.backlog_context),
            ("load_fixed", self.load_fixed),
            ("load_per_token", self.load_per_token),
            ("balance_weight", self.balance_weight),
        ];
        for (name, value) in nonneg {
            if !value.is_finite() || value < 0.0 {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be a finite number >= 0"
                )));
            }
        }
        Ok(())
    }

    fn backlog_ms(&self, rank: &Rank) -> f64 {
        let tokens = rank.active_prefill_tokens as f64;
        self.uncached * tokens
            + self.uncached_times_context * tokens * self.backlog_context
            + self.per_request * rank.active_requests as f64
    }
}

/// The per-rank signals the policy uses.
#[derive(Debug, Clone, Copy)]
struct Rank {
    device_overlap_blocks: f64,
    active_prefill_tokens: usize,
    active_requests: usize,
}

impl Rank {
    fn from_inputs(cache: &WorkerCacheInput, load: &WorkerLoadInput) -> Self {
        Self {
            device_overlap_blocks: cache.device_overlap_blocks(),
            active_prefill_tokens: load.active_prefill_tokens(),
            active_requests: load.active_requests(),
        }
    }
}

/// Per-row costs; split out so tests can check the arithmetic without a router.
fn row_costs(
    parameters: &Parameters,
    isl: usize,
    block_size: u32,
    shared_prefix_tokens: usize,
    ranks: &[Rank],
) -> Vec<f64> {
    let backlog: Vec<f64> = ranks.iter().map(|r| parameters.backlog_ms(r)).collect();
    let max_backlog = backlog.iter().copied().fold(0.0, f64::max);
    let isl_f = isl as f64;
    ranks
        .iter()
        .zip(&backlog)
        .map(|(rank, &backlog)| {
            let device = (rank.device_overlap_blocks * block_size as f64).min(isl_f);
            let store = shared_prefix_tokens as f64;
            let cached = device.max(store);
            let q = isl_f - cached;
            let request = parameters.uncached * q
                + parameters.uncached_times_context * q * (cached + q / 2.0)
                + parameters.per_request;
            let load_tokens = (store - device).max(0.0);
            let load = if load_tokens > 0.0 {
                parameters.load_fixed + parameters.load_per_token * load_tokens
            } else {
                0.0
            };
            let excess = (backlog + request - max_backlog).max(0.0);
            load.max(backlog) + request + parameters.balance_weight * excess
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

struct PrefillStepCostPicker {
    parameters: Parameters,
}

impl WorkerPicker for PrefillStepCostPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if cache.is_empty() || cache.len() != load.len() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        let ranks: Vec<Rank> = cache
            .iter()
            .zip(load)
            .map(|(cache, load)| Rank::from_inputs(cache, load))
            .collect();
        let costs = row_costs(
            &self.parameters,
            context.isl_tokens(),
            context.block_size(),
            context.shared_prefix_tokens(),
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
    dynamo_kv_router::scheduling::global_prefix::enable(parameters.store_retention_secs);

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(PrefillStepCostPicker { parameters }),
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

    fn rank(device_overlap_blocks: f64, active_prefill_tokens: usize, active_requests: usize) -> Rank {
        Rank {
            device_overlap_blocks,
            active_prefill_tokens,
            active_requests,
        }
    }

    #[test]
    fn idle_ranks_prefer_the_device_hit() {
        let p = Parameters::default();
        // 140K prompt, 131K in the store; rank 1 holds 8 blocks (131072 tokens) on the GPU.
        let ranks = [rank(0.0, 0, 0), rank(8.0, 0, 0)];
        let costs = row_costs(&p, 140_000, BS, 131_072, &ranks);
        assert!(costs[1] < costs[0], "{costs:?}");
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn busy_hit_rank_loses_to_idle_rank_when_backlog_exceeds_load_time() {
        let p = Parameters::default();
        // The hit rank has a cold 100K prompt queued (~2.5 s + context), more than a store load.
        let ranks = [rank(0.0, 0, 0), rank(8.0, 100_000, 1)];
        let costs = row_costs(&p, 140_000, BS, 131_072, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(0), "{costs:?}");
    }

    #[test]
    fn hit_rank_with_short_backlog_keeps_the_prompt() {
        let p = Parameters::default();
        // One continuation (9K fresh) queued on the hit rank: still sooner than a store load.
        let ranks = [rank(0.0, 0, 0), rank(8.0, 9_000, 1)];
        let costs = row_costs(&p, 140_000, BS, 131_072, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1), "{costs:?}");
    }

    #[test]
    fn cold_prompt_goes_to_least_backlog() {
        let p = Parameters::default();
        let ranks = [rank(0.0, 20_000, 2), rank(0.0, 5_000, 3), rank(0.0, 60_000, 1)];
        let costs = row_costs(&p, 90_000, BS, 0, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1), "{costs:?}");
    }

    #[test]
    fn store_prefix_counts_as_cached_on_every_rank() {
        let p = Parameters::default();
        let ranks = [rank(0.0, 0, 0), rank(0.0, 0, 0)];
        let costs = row_costs(&p, 140_000, BS, 131_072, &ranks);
        // Fresh 8928 tokens after a store load of 131072 tokens, not a 140K cold prefill.
        let fresh = 140_000.0 - 131_072.0;
        let request = p.uncached * fresh
            + p.uncached_times_context * fresh * (131_072.0 + fresh / 2.0)
            + p.per_request;
        // Idle engine: the request itself raises the lockstep maximum by its own cost.
        let expected = p.load_fixed + p.load_per_token * 131_072.0 + request + p.balance_weight * request;
        assert!((costs[0] - expected).abs() < 1e-6, "{costs:?} vs {expected}");
        let cold = p.uncached * 140_000.0
            + p.uncached_times_context * 140_000.0 * 70_000.0
            + p.per_request;
        assert!(costs[0] < cold, "{costs:?} vs cold {cold}");
    }

    #[test]
    fn rejects_negative_coefficients() {
        let p = Parameters {
            uncached: -1.0,
            ..Parameters::default()
        };
        assert!(p.validate().is_err());
    }
}
