// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_kv_router::selector::{WorkerSelectionInput, WorkerSelector};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{
    AdmissionFullPolicy, AdmissionOutcome, OccupancyReservation, PushRouter, RoutingOccupancyState,
};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::{
    entrypoint::{DecodeAdmissionConfig, DecodeAdmissionPolicy},
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
};

use super::builtin::BuiltinWorkerSelector;

pub(crate) struct HostedOccupancy {
    state: Arc<RoutingOccupancyState>,
    admission: Option<DecodeAdmissionConfig>,
}

fn admission_full(cap: u64, candidates: usize) -> anyhow::Error {
    anyhow::anyhow!(
        DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message(format!(
                "All {candidates} decode workers are at the admission cap ({cap} in flight)"
            ))
            .build()
    )
}

impl HostedOccupancy {
    pub(crate) fn new(
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<Self> {
        let state = router
            .routing_occupancy_state()
            .context("occupancy-aware router has no occupancy capability")?;
        Ok(Self {
            state,
            admission: None,
        })
    }

    pub(crate) fn set_admission(&mut self, admission: Option<DecodeAdmissionConfig>) {
        self.admission = admission;
    }

    pub(crate) fn admission(&self) -> Option<DecodeAdmissionConfig> {
        self.admission
    }

    #[cfg(test)]
    pub(crate) fn admission_queue_depth(&self) -> usize {
        self.state.admission_queue_depth()
    }

    /// Capped selection: only workers below `admission.cap` are eligible; when none
    /// is, wait in the router's FIFO (or reject). Pinned targets bypass the cap.
    pub(crate) async fn select_and_reserve_admitted(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        selector: &BuiltinWorkerSelector,
        admission: DecodeAdmissionConfig,
    ) -> Result<(HostedOccupancySelection, AdmissionOutcome)> {
        let policy = match admission.policy {
            DecodeAdmissionPolicy::Wait => AdmissionFullPolicy::Wait,
            DecodeAdmissionPolicy::Reject => AdmissionFullPolicy::Reject,
        };
        let candidate_count = std::sync::atomic::AtomicUsize::new(0);
        let (selection, outcome) = self
            .state
            .select_and_reserve_capped_with(
                admission.cap,
                policy,
                || {
                    let ids = router.selectable_worker_ids()?;
                    candidate_count.store(ids.len(), std::sync::atomic::Ordering::Relaxed);
                    Ok(ids)
                },
                |eligible, occupancy| {
                    selector
                        .select_worker(WorkerSelectionInput::<ModelRuntimeConfig>::hosted(
                            eligible,
                            Some(occupancy),
                        ))
                        .map(|selection| selection.worker.worker_id)
                        .map_err(anyhow::Error::from)
                },
                || {
                    admission_full(
                        admission.cap,
                        candidate_count.load(std::sync::atomic::Ordering::Relaxed),
                    )
                },
            )
            .await?;
        Ok((
            HostedOccupancySelection {
                worker_id: selection.worker_id(),
                candidate_count: selection.candidate_count(),
                occupancy: selection.load(),
                reservation: selection.into_reservation(),
            },
            outcome,
        ))
    }

    pub(crate) fn select_and_reserve(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        selector: &BuiltinWorkerSelector,
        pinned_worker: Option<u64>,
    ) -> Result<HostedOccupancySelection> {
        if let Some(worker_id) = pinned_worker {
            router.ensure_routable(worker_id)?;
            let reservation = self.state.reserve(worker_id);
            return Ok(HostedOccupancySelection {
                worker_id,
                candidate_count: 1,
                occupancy: reservation.load(),
                reservation,
            });
        }

        let candidates = router.selectable_worker_ids()?;
        let selection = self
            .state
            .select_and_reserve_with(&candidates, |occupancy| {
                selector
                    .select_worker(WorkerSelectionInput::<ModelRuntimeConfig>::hosted(
                        &candidates,
                        Some(occupancy),
                    ))
                    .map(|selection| selection.worker.worker_id)
            })?;
        let result = HostedOccupancySelection {
            worker_id: selection.worker_id(),
            candidate_count: selection.candidate_count(),
            occupancy: selection.load(),
            reservation: selection.into_reservation(),
        };
        Ok(result)
    }

    pub(crate) fn peek(
        &self,
        router: &PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        selector: &BuiltinWorkerSelector,
    ) -> Option<u64> {
        let candidates = router.selectable_worker_ids().ok()?;
        let occupancy = |worker_id| self.state.load(worker_id);
        selector
            .peek_worker(WorkerSelectionInput::hosted(&candidates, Some(&occupancy)))
            .ok()
    }
}

pub(crate) struct HostedOccupancySelection {
    pub(crate) worker_id: u64,
    pub(crate) candidate_count: usize,
    pub(crate) occupancy: u64,
    pub(crate) reservation: OccupancyReservation,
}
