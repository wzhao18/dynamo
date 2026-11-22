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
//!          + per_mtok * projected_tokens(r) / 1e6`
//!
//! `projected_tokens(r) = decode_cost_blocks(r) * block_size` is the router's booked KV footprint
//! of the rank plus this request's prompt (the router's projected decode load after admission).
//! Ties resolve to the rank with fewer active requests.
//!
//! `active_requests(r)` is the router's own booking count: a request is booked on its rank in the
//! same scheduler-actor step that selects it, and released when its decode stream finishes, is
//! cancelled, or fails (including a failed KV transfer), with the request-lease reaper as the
//! backstop. It therefore counts requests that are routed but not yet decoding (KV still in
//! transfer), and back-to-back selections in a burst see each other: four equal ranks take eight
//! burst requests 2/2/2/2.
//!
//! The default selector balances the KV footprint alone and charges active requests nothing
//! (`decode_active_request_weight` 0). On Kimi K3.1 DCP4xDP4 decode (VR200, AgentX c576/c608) the
//! busiest rank then carries ~7 requests above the mean (1.17x p50 max/mean), what count-blind
//! placement gives, and every rank launches at the busiest rank's size. Defaults: 0.094 ms per
//! request (served lockstep step slope below 64 requests per rank, c608 minutes 10-60; 0.095 at
//! c576) and 0.26 ms per million context tokens (decode indexer logits scale with summed context
//! per rank). With these, one request outweighs ~0.36M tokens of footprint, so the count decides.
//!
//! Placement only: no request's computation changes.

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
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
            per_request: 0.094,
            per_mtok: 0.26,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        for (name, value) in [
            ("per_request", self.per_request),
            ("per_mtok", self.per_mtok),
        ] {
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
    projected_blocks: f64,
}

/// `projected_blocks` already includes this request's prompt blocks, so the request's own
/// context is not added a second time.
fn row_costs(parameters: &Parameters, block_size: u32, ranks: &[Rank]) -> Vec<f64> {
    let bs = block_size.max(1) as f64;
    ranks
        .iter()
        .map(|rank| {
            let tokens = rank.projected_blocks * bs;
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
                projected_blocks: load.decode_cost_blocks(),
            })
            .collect();
        let costs = row_costs(&self.parameters, context.block_size(), &ranks);
        let row = select_row(&costs, &ranks)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        log_pick(input.candidates(), &ranks, &costs, row);
        Ok(row)
    }
}

/// One INFO line per decode placement: the chosen rank and every rank's router in-flight count
/// before admission, ordered by (worker_id, dp_rank), so a served run can measure the spread.
fn log_pick(candidates: &[ScoredWorkerCandidate], ranks: &[Rank], costs: &[f64], row: usize) {
    let mut order: Vec<usize> = (0..ranks.len()).collect();
    if candidates.len() == ranks.len() {
        order.sort_by_key(|&i| {
            let worker = candidates[i].worker();
            (worker.worker_id, worker.dp_rank)
        });
    }
    let active: Vec<usize> = order.iter().map(|&i| ranks[i].active_requests).collect();
    let (worker_id, dp_rank) = candidates
        .get(row)
        .map(|candidate| {
            let worker = candidate.worker();
            (worker.worker_id, worker.dp_rank)
        })
        .unwrap_or((0, 0));
    tracing::info!(
        worker_id,
        dp_rank,
        cost_ms = costs[row],
        active_requests = ?active,
        "decode-step-cost pick"
    );
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    tracing::info!(
        per_request_ms = parameters.per_request,
        per_mtok_ms = parameters.per_mtok,
        "decode-step-cost worker selection active for decode"
    );
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

    fn r(active_requests: usize, projected_blocks: f64) -> Rank {
        Rank {
            active_requests,
            projected_blocks,
        }
    }

    #[test]
    fn default_coefficients_match_the_served_fit() {
        let p = Parameters::default();
        assert_eq!(p.per_request, 0.094);
        assert_eq!(p.per_mtok, 0.26);
    }

    #[test]
    fn equal_footprint_prefers_fewer_requests() {
        // Same booked KV, one rank carries 49 requests and another 39: the count term decides.
        let p = Parameters::default();
        let ranks = [r(49, 450.0), r(39, 450.0), r(44, 450.0)];
        let costs = row_costs(&p, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn footprint_still_counts() {
        // Equal counts: the smaller footprint wins.
        let p = Parameters::default();
        let ranks = [r(40, 500.0), r(40, 400.0)];
        let costs = row_costs(&p, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn count_and_footprint_trade_off_at_the_fitted_rate() {
        // 1 request = per_request / per_mtok million tokens = 0.36M tokens ~ 22 blocks.
        let p = Parameters::default();
        let blocks_per_request = p.per_request / p.per_mtok * 1e6 / BS as f64;
        let ranks = [r(41, 400.0), r(40, 400.0 + blocks_per_request + 1.0)];
        let costs = row_costs(&p, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(0));
        let ranks = [r(41, 400.0), r(40, 400.0 + blocks_per_request - 1.0)];
        let costs = row_costs(&p, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn served_c608_imbalance_goes_to_the_emptiest_rank() {
        // c608 per-rank counts with a ~7-request spread and per-rank context within 3%
        // (149K-token requests, 16384-token blocks): the count decides, not the footprint.
        let p = Parameters::default();
        let ranks = [r(52, 480.0), r(45, 470.0), r(43, 475.0), r(41, 485.0)];
        let costs = row_costs(&p, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(3));
    }

    #[test]
    fn ties_go_to_fewer_requests() {
        let p = Parameters {
            per_request: 0.0,
            per_mtok: 1.0,
        };
        let ranks = [r(5, 10.0), r(3, 10.0)];
        let costs = row_costs(&p, BS, &ranks);
        assert_eq!(select_row(&costs, &ranks), Some(1));
    }

    #[test]
    fn validation_rejects_bad_parameters() {
        assert!(
            Parameters {
                per_request: -1.0,
                per_mtok: 0.26
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                per_request: f64::NAN,
                per_mtok: 0.26
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                per_request: 0.0,
                per_mtok: 0.0
            }
            .validate()
            .is_err()
        );
        assert!(Parameters::default().validate().is_ok());
    }

    #[test]
    fn empty_candidates_select_nothing() {
        let p = Parameters::default();
        assert_eq!(select_row(&row_costs(&p, BS, &[]), &[]), None);
    }
}

/// The policy inside the real scheduler actor: router-side in-flight accounting (booked at
/// selection, released at finish / abort / failure / lease expiry) is what it reads.
#[cfg(test)]
mod scheduler_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::overlap::OverlapSignals;
    use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
    use dynamo_kv_router::scheduling::{AdmissionAttempt, ScheduleMode, ScheduleRequest};
    use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;
    use dynamo_kv_router::test_utils::SimpleWorkerConfig;
    use dynamo_kv_router::{
        ActiveSequencesMultiWorker, KvRouterConfig, LocalScheduler, NoopSequencePublisher,
        RouterQueuePolicy, RoutingPartitionRef, WorkerSelectionPolicy, WorkerType,
    };
    use tokio_util::sync::CancellationToken;

    const WORKER: u64 = 7;
    const RANKS: u32 = 4;
    /// Prompt blocks per request (a ~149K-token AgentX request at 16384-token blocks).
    const BLOCKS: u64 = 9;

    type Scheduler =
        LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, WorkerSelectionPolicy>;

    struct Harness {
        scheduler: Arc<Scheduler>,
        slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        cancel: CancellationToken,
        _cfg_tx: tokio::sync::watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            self.cancel.cancel();
        }
    }

    /// One DP decode engine with `RANKS` ranks; the decode policy comes from the registry with
    /// no decode policy selected, i.e. the default the frontend resolves.
    fn harness() -> Harness {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        super::register(&mut registry).unwrap();
        let config = KvRouterConfig::default();
        let factory = registry
            .resolve_for_worker_type(&config, WorkerType::Decode)
            .unwrap()
            .expect("decode resolves to decode-step-cost by default");
        let policy = factory(
            &config,
            WorkerType::Decode,
            RoutingPartitionRef::new("model", "default"),
        );

        let workers = HashMap::from([(
            WORKER,
            SimpleWorkerConfig {
                data_parallel_start_rank: 0,
                data_parallel_size: RANKS,
                ..Default::default()
            },
        )]);
        let dp_range = HashMap::from([(WORKER, (0, RANKS))]);
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            BS as usize,
            dp_range,
            false,
            0,
            "decode",
        ));
        let (cfg_tx, cfg_rx) = tokio::sync::watch::channel(workers);
        let cancel = CancellationToken::new();
        let scheduler = Arc::new(LocalScheduler::new(
            Arc::clone(&slots),
            cfg_rx,
            None,
            BS,
            policy,
            RouterQueuePolicy::Fcfs,
            None,
            Duration::from_secs(60),
            false,
            cancel.clone(),
            "decode",
            false,
        ));
        Harness {
            scheduler,
            slots,
            cancel,
            _cfg_tx: cfg_tx,
        }
    }

    const BS: u32 = 16384;

    fn request(id: &str, n: u64) -> ScheduleRequest {
        ScheduleRequest {
            mode: ScheduleMode::Tracked {
                request_id: id.to_string(),
            },
            // Unique, non-shared blocks (decode routing assumes no KV reuse).
            token_seq: Some((0..BLOCKS).map(|b| (n << 16) | b).collect()),
            block_hashes: None,
            isl_tokens: (BLOCKS * BS as u64) as usize,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
        }
    }

    fn counts(h: &Harness) -> Vec<usize> {
        let by_rank = h.slots.active_request_counts();
        (0..RANKS)
            .map(|dp_rank| {
                by_rank
                    .get(&WorkerWithDpRank::new(WORKER, dp_rank))
                    .copied()
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Eight requests arrive in one burst, before any of them starts decoding: each selection
    /// sees the bookings of the ones before it, so four equal ranks end at 2/2/2/2 (a count-blind
    /// or engine-reported count would put all eight on the momentarily emptiest rank).
    #[tokio::test]
    async fn burst_of_eight_spreads_two_per_rank() {
        let h = harness();
        let mut tasks = Vec::new();
        for n in 0..8u64 {
            let scheduler = Arc::clone(&h.scheduler);
            tasks.push(tokio::spawn(async move {
                scheduler
                    .schedule_request(request(&format!("burst-{n}"), n))
                    .await
                    .unwrap()
                    .best_worker
            }));
        }
        let mut chosen = vec![0usize; RANKS as usize];
        for task in tasks {
            let worker = task.await.unwrap();
            assert_eq!(worker.worker_id, WORKER);
            chosen[worker.dp_rank as usize] += 1;
        }
        assert_eq!(chosen, vec![2, 2, 2, 2]);
        assert_eq!(counts(&h), vec![2, 2, 2, 2]);
    }

    /// Sequential back-to-back selections spread the same way.
    #[tokio::test]
    async fn back_to_back_selections_spread_two_per_rank() {
        let h = harness();
        for n in 0..8u64 {
            h.scheduler
                .schedule_request(request(&format!("seq-{n}"), n))
                .await
                .unwrap();
        }
        assert_eq!(counts(&h), vec![2, 2, 2, 2]);
    }

    /// After uneven finishes, the next request goes to the rank with the fewest in flight.
    #[tokio::test]
    async fn next_request_fills_the_rank_that_finished() {
        let h = harness();
        let mut by_rank: HashMap<u32, Vec<String>> = HashMap::new();
        for n in 0..8u64 {
            let id = format!("r-{n}");
            let worker = h
                .scheduler
                .schedule_request(request(&id, n))
                .await
                .unwrap()
                .best_worker;
            by_rank.entry(worker.dp_rank).or_default().push(id);
        }
        // Rank 2 finishes both of its requests.
        for id in by_rank.remove(&2).unwrap() {
            h.scheduler.free(&id).await.unwrap();
        }
        assert_eq!(counts(&h), vec![2, 2, 0, 2]);
        let worker = h
            .scheduler
            .schedule_request(request("r-8", 8))
            .await
            .unwrap()
            .best_worker;
        assert_eq!(worker.dp_rank, 2);
    }

    /// Every way a routed request leaves releases its booking, so the in-flight count the policy
    /// reads drains to 0: finish (stream end), abort / error / KV-transfer failure (the frontend's
    /// request guard releases its booking through the scheduler's booking cleanup on all three),
    /// and the request-lease expiry backstop.
    #[tokio::test]
    async fn in_flight_count_drains_to_zero_on_finish_abort_failure_and_expiry() {
        let h = harness();
        let mut bookings = Vec::new();
        for n in 0..8u64 {
            let id = format!("d-{n}");
            let admitted = h
                .scheduler
                .schedule_request_admitted(request(&id, n))
                .await
                .unwrap();
            let AdmissionAttempt::Tracked(attempt_id) = admitted.attempt else {
                panic!("tracked request must carry an attempt id");
            };
            bookings.push(SchedulerBookingDescriptor {
                request_id: id,
                worker: admitted.response.best_worker,
                attempt_id,
            });
        }
        assert_eq!(counts(&h), vec![2, 2, 2, 2]);
        let cleanup = h.scheduler.booking_cleanup();

        // Finish: decode stream ended.
        h.scheduler.free(&bookings[0].request_id).await.unwrap();
        h.scheduler.free(&bookings[1].request_id).await.unwrap();
        // Abort before or during decode (client cancel).
        cleanup.enqueue_and_wait(bookings[2].clone()).await.unwrap();
        cleanup.enqueue_and_wait(bookings[3].clone()).await.unwrap();
        // Error frame / failed KV transfer: the same guarded release, issued before the error
        // is yielded downstream.
        cleanup.enqueue_and_wait(bookings[4].clone()).await.unwrap();
        cleanup.enqueue_and_wait(bookings[5].clone()).await.unwrap();
        // Lease expiry backstop.
        cleanup.enqueue_expired(bookings[6].clone());
        cleanup.enqueue_expired(bookings[7].clone());
        // A second release of an already released booking is a no-op, never a negative count.
        cleanup.enqueue_and_wait(bookings[2].clone()).await.unwrap();
        h.scheduler.free(&bookings[0].request_id).await.ok();

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if counts(&h).iter().all(|&count| count == 0) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("in-flight count must drain to 0 on every rank");
        assert_eq!(counts(&h), vec![0, 0, 0, 0]);

        // A stale release of a booking whose request id was re-admitted must not free the new
        // attempt.
        let readmitted = h
            .scheduler
            .schedule_request_admitted(request(&bookings[4].request_id, 4))
            .await
            .unwrap();
        cleanup.enqueue_and_wait(bookings[4].clone()).await.unwrap();
        let total: usize = counts(&h).iter().sum();
        assert_eq!(total, 1);
        let AdmissionAttempt::Tracked(attempt_id) = readmitted.attempt else {
            panic!("tracked request must carry an attempt id");
        };
        cleanup
            .enqueue_and_wait(SchedulerBookingDescriptor {
                request_id: bookings[4].request_id.clone(),
                worker: readmitted.response.best_worker,
                attempt_id,
            })
            .await
            .unwrap();
        assert_eq!(counts(&h), vec![0, 0, 0, 0]);
    }
}
