// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Least-active decode worker selection with bounded-load session affinity. It is the default
//! policy for a disaggregated decode pool with no explicitly selected policy (see
//! `DEFAULT_DECODE_POLICY_TYPE`).
//!
//! Every candidate is one `(worker, dp_rank)` of a lockstep DP decode engine. The ranks step
//! together and every rank pads to the busiest one, so the decode step is set by the rank with
//! the most running requests. The policy places each request on the rank with the fewest
//! router-tracked active requests; ties rotate. KV overlap is ignored: a disaggregated decode
//! worker receives the prompt's KV by transfer, and decode workers publish no KV events.
//!
//! Session affinity, bounded by load: a decode worker keeps a session's KV in its local prefix
//! cache, so a turn placed on its session's previous rank transfers only the new suffix. The turn
//! stays on that rank while the rank's active count is at most `mean + affinity_slack` (mean over
//! the candidate ranks, before this request); otherwise it goes to the least-active rank and the
//! session moves there.
//!
//! `active_requests(r)` is the router's own booking count (`WorkerInputs::LOAD`): a request is
//! booked on its rank in the same scheduler-actor step that selects it and released when its
//! decode stream finishes, is cancelled, or fails. Back-to-back selections therefore see each
//! other at once; a load snapshot that lags the router's own assignments would herd every arrival
//! in the lag window onto one rank.
//!
//! Measured basis (Kimi K3.1 DCP4xDP4 decode, AgentX c440, 70K decode requests, replay of the
//! served arrivals and residencies with only the DP rank changed): the served KV-cost selector
//! left the busiest rank 5.85 running requests above the mean (random placement: 5.79). Pure
//! least-active placement gives 1.06; `affinity_slack` 1 gives 1.50 while keeping 84% of
//! continuation turns on their session's rank (25% today), which raises the decode-local prefix
//! hit and cuts the transferred prompt tokens from ~62% to ~34% of the prompt.
//!
//! Placement only: no request's computation changes.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::WorkerWithDpRank;
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
pub const POLICY_TYPE: &str = "least-active-decode";

/// Prune idle sessions once per this many tracked placements.
const PRUNE_EVERY: u64 = 4096;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// A session's turn stays on its previous rank while that rank's active count is at most
    /// the candidate mean plus this many requests.
    affinity_slack: f64,
    /// Forget a session's rank after this many seconds without a turn.
    session_ttl_secs: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            affinity_slack: 1.0,
            session_ttl_secs: 3600,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.affinity_slack.is_finite() || self.affinity_slack < 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "affinity_slack must be a finite number >= 0",
            ));
        }
        Ok(())
    }
}

/// Why a rank was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reason {
    /// The session's previous rank, within the load bound.
    Affinity,
    /// The least-active rank (no session, a new session, or a session moved off an
    /// over-loaded or unavailable rank).
    LeastActive,
}

impl Reason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Affinity => "affinity",
            Self::LeastActive => "least-active",
        }
    }
}

struct SessionRank {
    rank: WorkerWithDpRank,
    last_seen: Instant,
}

/// Placement state, independent of the router types so the replay tests drive it directly.
struct Placer {
    parameters: Parameters,
    rotation: usize,
    tracked_picks: u64,
    sessions: HashMap<String, SessionRank>,
}

/// One session-aware placement request.
struct Session<'a> {
    id: &'a str,
    is_final: bool,
}

impl Placer {
    fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            rotation: 0,
            tracked_picks: 0,
            sessions: HashMap::new(),
        }
    }

    /// Return the chosen row. `ranks[i]` and `active[i]` describe row `i`; row order is
    /// arbitrary, so ties rotate in `(worker_id, dp_rank)` order. State changes only when
    /// `tracked` (the router books the chosen rank).
    fn place(
        &mut self,
        now: Instant,
        session: Option<Session<'_>>,
        tracked: bool,
        ranks: &[WorkerWithDpRank],
        active: &[usize],
    ) -> Option<(usize, Reason)> {
        if ranks.is_empty() || ranks.len() != active.len() {
            return None;
        }
        let ttl = Duration::from_secs(self.parameters.session_ttl_secs);
        let mean = active.iter().sum::<usize>() as f64 / active.len() as f64;
        let affine = session
            .as_ref()
            .and_then(|session| self.sessions.get(session.id))
            .filter(|entry| now.saturating_duration_since(entry.last_seen) <= ttl)
            .and_then(|entry| ranks.iter().position(|rank| *rank == entry.rank))
            .filter(|&row| active[row] as f64 <= mean + self.parameters.affinity_slack);

        let (row, reason) = match affine {
            Some(row) => (row, Reason::Affinity),
            None => {
                let least = *active.iter().min()?;
                let mut ties: Vec<usize> = (0..ranks.len())
                    .filter(|&row| active[row] == least)
                    .collect();
                ties.sort_by_key(|&row| (ranks[row].worker_id, ranks[row].dp_rank));
                (ties[self.rotation % ties.len()], Reason::LeastActive)
            }
        };

        if tracked {
            if reason == Reason::LeastActive {
                self.rotation = self.rotation.wrapping_add(1);
            }
            if let Some(session) = session {
                if session.is_final {
                    self.sessions.remove(session.id);
                } else {
                    self.sessions.insert(
                        session.id.to_owned(),
                        SessionRank {
                            rank: ranks[row],
                            last_seen: now,
                        },
                    );
                }
            }
            self.tracked_picks += 1;
            if self.tracked_picks.is_multiple_of(PRUNE_EVERY) {
                self.sessions
                    .retain(|_, entry| now.saturating_duration_since(entry.last_seen) <= ttl);
            }
        }
        Some((row, reason))
    }
}

struct LeastActiveDecodePicker {
    placer: Placer,
}

impl WorkerPicker for LeastActiveDecodePicker {
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
        let ranks: Vec<WorkerWithDpRank> = input
            .candidates()
            .iter()
            .map(|candidate| candidate.worker())
            .collect();
        let active: Vec<usize> = load.iter().map(|load| load.active_requests()).collect();
        let session = context.session_context().map(|session| Session {
            id: session.session_id(),
            is_final: session.session_final() == Some(true),
        });
        let (row, reason) = self
            .placer
            .place(
                Instant::now(),
                session,
                context.is_tracked(),
                &ranks,
                &active,
            )
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        log_pick(context, &ranks, &active, row, reason);
        Ok(row)
    }
}

/// One INFO line per decode placement (joinable to the selector's "Selected worker" line by
/// request_id): the chosen rank, the reason, and the router in-flight count of every candidate
/// rank ordered by (worker_id, dp_rank), before this request.
fn log_pick(
    context: &WorkerSelectionContext<'_>,
    ranks: &[WorkerWithDpRank],
    active: &[usize],
    row: usize,
    reason: Reason,
) {
    let mut order: Vec<usize> = (0..ranks.len()).collect();
    order.sort_by_key(|&i| (ranks[i].worker_id, ranks[i].dp_rank));
    let active_sorted: Vec<usize> = order.iter().map(|&i| active[i]).collect();
    tracing::info!(
        request_id = context.request_id(),
        session = context.session_context().is_some(),
        worker_id = ranks[row].worker_id,
        dp_rank = ranks[row].dp_rank,
        reason = reason.as_str(),
        active_requests = ?active_sorted,
        "least-active-decode pick"
    );
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    tracing::info!(
        affinity_slack = parameters.affinity_slack,
        session_ttl_secs = parameters.session_ttl_secs,
        "least-active-decode worker selection active for decode"
    );
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(LeastActiveDecodePicker {
                    placer: Placer::new(parameters),
                }),
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

    fn ranks(n: u32) -> Vec<WorkerWithDpRank> {
        (0..n).map(|r| WorkerWithDpRank::new(7, r)).collect()
    }

    fn s(id: &str) -> Option<Session<'_>> {
        Some(Session {
            id,
            is_final: false,
        })
    }

    #[test]
    fn picks_the_least_active_rank() {
        let mut p = Placer::new(Parameters::default());
        let r = ranks(4);
        let now = Instant::now();
        assert_eq!(
            p.place(now, None, true, &r, &[5, 3, 4, 6]),
            Some((1, Reason::LeastActive))
        );
    }

    #[test]
    fn ties_rotate_in_rank_order_regardless_of_row_order() {
        let mut p = Placer::new(Parameters::default());
        // Rows in reverse rank order.
        let r: Vec<_> = ranks(4).into_iter().rev().collect();
        let now = Instant::now();
        let picked: Vec<u32> = (0..8)
            .map(|_| r[p.place(now, None, true, &r, &[2, 2, 2, 2]).unwrap().0].dp_rank)
            .collect();
        assert_eq!(picked, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn query_only_selection_changes_no_state() {
        let mut p = Placer::new(Parameters::default());
        let r = ranks(4);
        let now = Instant::now();
        p.place(now, s("a"), false, &r, &[0, 0, 0, 0]);
        assert!(p.sessions.is_empty());
        assert_eq!(p.rotation, 0);
    }

    #[test]
    fn session_stays_within_the_load_bound_and_moves_beyond_it() {
        let mut p = Placer::new(Parameters::default());
        let r = ranks(4);
        let now = Instant::now();
        // First turn: least active is rank 2.
        assert_eq!(
            p.place(now, s("a"), true, &r, &[3, 3, 1, 3]),
            Some((2, Reason::LeastActive))
        );
        // Rank 2 is at mean + 1 (mean 4, count 5): stay, although rank 0 is emptier.
        assert_eq!(
            p.place(now, s("a"), true, &r, &[2, 4, 5, 5]),
            Some((2, Reason::Affinity))
        );
        // Rank 2 is above mean + 1 (mean 4.25, count 6): move to the least-active rank.
        assert_eq!(
            p.place(now, s("a"), true, &r, &[2, 4, 6, 5]),
            Some((0, Reason::LeastActive))
        );
        // The session follows its new rank.
        assert_eq!(
            p.place(now, s("a"), true, &r, &[3, 3, 3, 3]),
            Some((0, Reason::Affinity))
        );
    }

    #[test]
    fn unavailable_expired_or_final_sessions_fall_back_to_least_active() {
        let mut p = Placer::new(Parameters {
            affinity_slack: 1.0,
            session_ttl_secs: 10,
        });
        let r = ranks(4);
        let t0 = Instant::now();
        assert_eq!(p.place(t0, s("a"), true, &r, &[1, 0, 1, 1]).unwrap().0, 1);
        // The session's rank is not a candidate.
        let others = [r[0], r[2], r[3]];
        assert_eq!(
            p.place(t0, s("a"), false, &others, &[2, 1, 2]),
            Some((1, Reason::LeastActive))
        );
        // Expired.
        assert_eq!(
            p.place(
                t0 + Duration::from_secs(11),
                s("a"),
                false,
                &r,
                &[1, 1, 0, 1]
            ),
            Some((2, Reason::LeastActive))
        );
        // A final turn still uses affinity and then forgets the session.
        let fin = Some(Session {
            id: "a",
            is_final: true,
        });
        assert_eq!(
            p.place(t0, fin, true, &r, &[1, 1, 0, 1]),
            Some((1, Reason::Affinity))
        );
        assert!(p.sessions.is_empty());
    }

    #[test]
    fn idle_sessions_are_pruned() {
        let mut p = Placer::new(Parameters {
            affinity_slack: 1.0,
            session_ttl_secs: 1,
        });
        let r = ranks(4);
        let t0 = Instant::now();
        p.place(t0, s("old"), true, &r, &[0, 0, 0, 0]);
        let later = t0 + Duration::from_secs(5);
        for _ in 1..PRUNE_EVERY {
            p.place(later, None, true, &r, &[0, 0, 0, 0]);
        }
        assert!(p.sessions.is_empty());
    }

    #[test]
    fn validation() {
        assert!(Parameters::default().validate().is_ok());
        for slack in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                Parameters {
                    affinity_slack: slack,
                    session_ttl_secs: 3600
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn empty_candidates_select_nothing() {
        let mut p = Placer::new(Parameters::default());
        assert_eq!(p.place(Instant::now(), None, true, &[], &[]), None);
    }

    /// Trace replay: arrivals with multi-turn sessions and heavy-tailed decode residencies on
    /// 4 lockstep DP ranks at ~31 running requests per rank (the served c440 regime). Counts
    /// are booked at placement and released at finish, as the router does. Imbalance is the
    /// time-weighted mean of (max - mean) running requests per rank.
    mod replay {
        use super::*;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        const RANKS: usize = 4;

        /// Small deterministic generator (SplitMix64).
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
            fn unit(&mut self) -> f64 {
                ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            }
            fn exp(&mut self, mean: f64) -> f64 {
                -mean * self.unit().ln()
            }
        }

        struct Turn {
            t: f64,
            session: usize,
            dur: f64,
        }

        /// Sessions arrive as a Poisson stream; each runs several turns separated by think
        /// time. Residency is a mixture: most turns are short, a tail is long.
        fn trace(seed: u64) -> Vec<Turn> {
            let mut rng = Rng(seed);
            let mut turns = Vec::new();
            let mut t = 0.0;
            let mut session = 0;
            while t < 1500.0 {
                t += rng.exp(1.0 / 3.6);
                let n = 1 + (rng.exp(6.0) as usize).min(40);
                let mut ts = t;
                for _ in 0..n {
                    let dur = if rng.unit() < 0.8 {
                        rng.exp(2.0)
                    } else {
                        rng.exp(18.0)
                    };
                    turns.push(Turn {
                        t: ts,
                        session,
                        dur,
                    });
                    ts += dur + rng.exp(8.0);
                }
                session += 1;
            }
            turns.sort_by(|a, b| a.t.total_cmp(&b.t));
            turns
        }

        enum Mode {
            Random,
            Policy(f64),
        }

        struct Result {
            imbalance: f64,
            mean_running: f64,
            continuation_same_rank: f64,
        }

        fn run(turns: &[Turn], mode: Mode) -> Result {
            let ranks: Vec<WorkerWithDpRank> = (0..RANKS as u32)
                .map(|r| WorkerWithDpRank::new(1, r))
                .collect();
            let mut placer = Placer::new(Parameters {
                affinity_slack: match mode {
                    Mode::Policy(slack) => slack,
                    Mode::Random => 0.0,
                },
                session_ttl_secs: 3600,
            });
            let mut rng = Rng(99);
            let base = Instant::now();
            let mut active = [0usize; RANKS];
            // Min-heap of (finish time in us, rank).
            let mut finishes: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
            let mut last_rank: HashMap<usize, usize> = HashMap::new();
            let (mut continuations, mut same) = (0usize, 0usize);
            let (w0, w1) = (300.0, 1500.0);
            let (mut acc, mut acc_running, mut t_last) = (0.0, 0.0, 0.0f64);
            let mut advance = |t: f64, active: &[usize; RANKS], acc: &mut f64, acc_r: &mut f64| {
                let (a, b) = (t_last.max(w0), t.min(w1));
                if b > a {
                    let max = *active.iter().max().unwrap() as f64;
                    let mean = active.iter().sum::<usize>() as f64 / RANKS as f64;
                    *acc += (max - mean) * (b - a);
                    *acc_r += mean * (b - a);
                }
                t_last = t;
            };
            for turn in turns {
                while let Some(&Reverse((tf, rank))) = finishes.peek() {
                    if tf as f64 / 1e6 > turn.t {
                        break;
                    }
                    finishes.pop();
                    advance(tf as f64 / 1e6, &active, &mut acc, &mut acc_running);
                    active[rank] -= 1;
                }
                advance(turn.t, &active, &mut acc, &mut acc_running);
                let id = turn.session.to_string();
                let rank = match mode {
                    Mode::Random => (rng.next() % RANKS as u64) as usize,
                    Mode::Policy(_) => {
                        let now = base + Duration::from_secs_f64(turn.t);
                        placer.place(now, s(&id), true, &ranks, &active).unwrap().0
                    }
                };
                if let Some(&prev) = last_rank.get(&turn.session) {
                    continuations += 1;
                    same += usize::from(prev == rank);
                }
                last_rank.insert(turn.session, rank);
                active[rank] += 1;
                finishes.push(Reverse((((turn.t + turn.dur) * 1e6) as u64, rank)));
            }
            Result {
                imbalance: acc / (w1 - w0),
                mean_running: acc_running / (w1 - w0),
                continuation_same_rank: same as f64 / continuations.max(1) as f64,
            }
        }

        #[test]
        fn least_active_with_bounded_affinity_balances_the_served_regime() {
            let turns = trace(20_261_123);
            let random = run(&turns, Mode::Random);
            let pure = run(&turns, Mode::Policy(0.0));
            let affine = run(&turns, Mode::Policy(1.0));
            let loose = run(&turns, Mode::Policy(2.0));
            eprintln!(
                "mean running/rank {:.1}; Bmax-Bmean random {:.2}, slack0 {:.2}, slack1 {:.2} \
                 (same-rank {:.2}), slack2 {:.2} (same-rank {:.2}); random same-rank {:.2}",
                random.mean_running,
                random.imbalance,
                pure.imbalance,
                affine.imbalance,
                affine.continuation_same_rank,
                loose.imbalance,
                loose.continuation_same_rank,
                random.continuation_same_rank,
            );
            // The regime: ~31 running per rank and random placement leaving several requests.
            assert!((25.0..40.0).contains(&random.mean_running));
            assert!(random.imbalance > 4.0);
            // Least-active (own assignments counted at once) removes most of it.
            assert!(pure.imbalance < 1.3);
            assert!(affine.imbalance <= 2.0);
            assert!(affine.imbalance < random.imbalance / 2.5);
            // Affinity keeps most continuations on their session's rank (random: ~1/4).
            assert!(random.continuation_same_rank < 0.35);
            assert!(affine.continuation_same_rank > 0.75);
            assert!(loose.continuation_same_rank >= affine.continuation_same_rank);
        }
    }
}
