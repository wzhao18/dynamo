// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-side model of a global, write-through prefix store (vLLM `MooncakeStoreConnector`).
//!
//! With a cluster-wide KV store every prefill worker saves the blocks it computes and any worker
//! can load them, so a prefix hit does not depend on which worker served the earlier turn. The
//! router cannot query the store, but it sees every prompt it routes: a block chain routed (to any
//! worker) within the store's retention is resident for every worker. This registry keeps that
//! set of chained block hashes and reports, for a new prompt, the leading resident blocks and the
//! tokens they imply. The prefill router offers them to worker selection as shared-cache hits and
//! the scheduler credits them as cached when it books the request, so the booked uncached tokens
//! of active prefills follow the store instead of the device-only overlap.
//!
//! The registry is enabled by a worker-selection policy that models the store (the built-in
//! `prefill-step-cost` policy with `store_retention_secs > 0`), so it is off unless such a policy
//! is configured for the prefill role.
//!
//! Based on Juntian Liu's global-prefix store model (Inferact dynamo-k31-vr 6d04b25141).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use dashmap::DashMap;

use crate::protocols::LocalBlockHash;

const MAX_ENTRIES: usize = 4_000_000;

pub struct GlobalPrefixRegistry {
    ttl_secs: u64,
    start: Instant,
    /// Chained block hash -> (expiry in seconds since `start`, longest routed prompt in tokens
    /// that contains this block).
    entries: DashMap<u64, (u64, u32)>,
    inserts_since_prune: AtomicU64,
    queries: AtomicU64,
    query_tokens: AtomicU64,
    hit_tokens: AtomicU64,
}

/// Chain per-block content hashes into prefix hashes, so block `i` identifies the whole prefix
/// `0..=i` (equal local hashes at different depths or after different prefixes do not collide).
pub fn chain_hashes(block_hashes: &[LocalBlockHash]) -> Vec<u64> {
    let mut prev: u64 = 0x5bd1_e995_9e37_79b9;
    block_hashes
        .iter()
        .map(|hash| {
            let mut x = prev.rotate_left(23) ^ hash.0.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            // splitmix64 finalizer
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^= x >> 31;
            prev = x;
            x
        })
        .collect()
}

impl GlobalPrefixRegistry {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            ttl_secs,
            start: Instant::now(),
            entries: DashMap::new(),
            inserts_since_prune: AtomicU64::new(0),
            queries: AtomicU64::new(0),
            query_tokens: AtomicU64::new(0),
            hit_tokens: AtomicU64::new(0),
        }
    }

    fn now_secs(&self) -> u64 {
        self.start.elapsed().as_secs()
    }

    /// Leading blocks of `chain` resident in the store, and the resident tokens they imply for a
    /// prompt of `isl` tokens. When every full block of the prompt is resident, the prompt is
    /// assumed to extend an earlier routed prompt (agentic turns append), so the resident tokens
    /// run to that prompt's length, capped at `isl`; otherwise they stop at the last matched block.
    pub fn resident_prefix(&self, chain: &[u64], block_size: u32, isl: usize) -> (usize, usize) {
        let now = self.now_secs();
        let mut blocks = 0;
        let mut last_len = 0usize;
        for hash in chain {
            match self.entries.get(hash) {
                Some(entry) if entry.0 > now => {
                    blocks += 1;
                    last_len = entry.1 as usize;
                }
                _ => break,
            }
        }
        let block_tokens = blocks * block_size as usize;
        let tokens = if blocks > 0 && blocks == chain.len() {
            last_len.max(block_tokens).min(isl)
        } else {
            block_tokens.min(isl)
        };
        let queries = self.queries.fetch_add(1, Ordering::Relaxed) + 1;
        let total = self.query_tokens.fetch_add(isl as u64, Ordering::Relaxed) + isl as u64;
        let hit = self.hit_tokens.fetch_add(tokens as u64, Ordering::Relaxed) + tokens as u64;
        if queries % 2000 == 0 {
            tracing::info!(
                queries,
                resident_token_share = hit as f64 / total.max(1) as f64,
                entries = self.entries.len(),
                "global-prefix registry"
            );
        }
        (blocks, tokens)
    }

    /// Record that `chain` (a prompt of `isl` tokens) will be computed and saved by the worker
    /// that serves it.
    pub fn insert(&self, chain: &[u64], isl: usize) {
        let expiry = self.now_secs() + self.ttl_secs;
        let isl = u32::try_from(isl).unwrap_or(u32::MAX);
        for hash in chain {
            self.entries
                .entry(*hash)
                .and_modify(|entry| {
                    entry.0 = expiry;
                    entry.1 = entry.1.max(isl);
                })
                .or_insert((expiry, isl));
        }
        let n = self
            .inserts_since_prune
            .fetch_add(chain.len() as u64, Ordering::Relaxed);
        if n > (MAX_ENTRIES / 8) as u64 {
            self.inserts_since_prune.store(0, Ordering::Relaxed);
            let now = self.now_secs();
            self.entries.retain(|_, entry| entry.0 > now);
            if self.entries.len() > MAX_ENTRIES {
                // Over capacity with live entries: drop the oldest half by expiry.
                let mut expiries: Vec<u64> = self.entries.iter().map(|e| e.value().0).collect();
                expiries.sort_unstable();
                let cutoff = expiries[expiries.len() / 2];
                self.entries.retain(|_, entry| entry.0 > cutoff);
            }
        }
    }
}

static REGISTRY: OnceLock<GlobalPrefixRegistry> = OnceLock::new();

/// Enable the process-wide store model with the store's prefix retention. The first call wins;
/// later calls with a different retention are ignored with a warning.
pub fn enable(ttl_secs: u64) {
    if ttl_secs == 0 {
        return;
    }
    let registry = REGISTRY.get_or_init(|| {
        tracing::info!(ttl_secs, "router global-prefix (shared KV store) model enabled");
        GlobalPrefixRegistry::new(ttl_secs)
    });
    if registry.ttl_secs != ttl_secs {
        tracing::warn!(
            configured = registry.ttl_secs,
            requested = ttl_secs,
            "global-prefix store model already enabled with another retention; keeping the first"
        );
    }
}

/// The process-wide registry, or None when no policy enabled the store model.
pub fn registry() -> Option<&'static GlobalPrefixRegistry> {
    REGISTRY.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(ids: &[u64]) -> Vec<u64> {
        chain_hashes(&ids.iter().map(|&id| LocalBlockHash(id)).collect::<Vec<_>>())
    }

    #[test]
    fn chained_hashes_depend_on_prefix() {
        let a = chain(&[1, 2, 3]);
        let b = chain(&[9, 2, 3]);
        assert_eq!(a.len(), 3);
        assert_ne!(a[1], b[1]);
        assert_eq!(chain(&[1, 2]), a[..2].to_vec());
    }

    #[test]
    fn resident_prefix_counts_leading_blocks_only() {
        let registry = GlobalPrefixRegistry::new(60);
        registry.insert(&chain(&[1, 2, 3]), 3 * 16 + 5);
        // All full blocks resident: extends the earlier prompt, tokens run to its length.
        assert_eq!(registry.resident_prefix(&chain(&[1, 2, 3]), 16, 60), (3, 53));
        assert_eq!(registry.resident_prefix(&chain(&[1, 2, 3, 4]), 16, 70), (3, 48));
        assert_eq!(registry.resident_prefix(&chain(&[1, 9, 3]), 16, 60), (1, 16));
        assert_eq!(registry.resident_prefix(&chain(&[7, 1, 2]), 16, 60), (0, 0));
        // A shorter prompt than the routed one is capped at its own length.
        assert_eq!(registry.resident_prefix(&chain(&[1, 2, 3]), 16, 50), (3, 50));
    }

    #[test]
    fn expired_entries_are_not_resident() {
        let registry = GlobalPrefixRegistry::new(60);
        registry.insert(&chain(&[1, 2]), 32);
        for mut entry in registry.entries.iter_mut() {
            entry.value_mut().0 = 0;
        }
        assert_eq!(registry.resident_prefix(&chain(&[1, 2]), 16, 32), (0, 0));
    }
}
