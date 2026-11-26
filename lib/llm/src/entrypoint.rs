// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;
pub use input::{PreprocessedRouting, build_preprocessed_routing, http::HttpFrontend};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_kv_router::{PrefillLoadEstimator, config::KvRouterConfig};
use dynamo_runtime::{discovery::ModelCardInstanceId, pipeline::RouterMode};
use serde::{Deserialize, Serialize};

use crate::{
    backend::ExecutionContext, discovery::LoadThresholdConfig, engines::StreamingEngine,
    local_model::LocalModel, model_card::ModelDeploymentCard,
    session_affinity::SessionAffinityMode,
    types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
};

/// Callback type for chat engine factory (async)
pub type PrefillRoutedEngine = dynamo_runtime::pipeline::ServiceEngine<
    dynamo_runtime::pipeline::SingleIn<crate::protocols::common::preprocessor::PreprocessedRequest>,
    dynamo_runtime::pipeline::ManyOut<
        crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
    >,
>;

pub type ChatEngineFactoryCallback = Arc<
    dyn Fn(
            ModelCardInstanceId,
            ModelDeploymentCard,
            PrefillRoutedEngine,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<OpenAIChatCompletionsStreamingEngine>> + Send>,
        > + Send
        + Sync,
>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterConfig {
    pub router_mode: RouterMode,
    pub kv_router_config: KvRouterConfig,
    /// Load threshold configuration for overload detection
    pub load_threshold_config: LoadThresholdConfig,
    /// Deprecated compatibility field. Routing and readiness ignore this value.
    #[serde(default)]
    pub enforce_disagg: bool,
    #[serde(default)]
    pub session_affinity_ttl_secs: Option<u64>,
    #[serde(default)]
    pub session_affinity_mode: SessionAffinityMode,
    /// Optional per-worker in-flight cap for decode (and aggregated) worker selection.
    /// Unset keeps stock behaviour: the least-loaded worker is always taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_admission: Option<DecodeAdmissionConfig>,
}

/// What the router does when every decode worker is at the admission cap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecodeAdmissionPolicy {
    /// Hold the request in a router-local FIFO until a decode slot is released.
    #[default]
    Wait,
    /// Reject the request (ResourceExhausted).
    Reject,
}

impl std::str::FromStr for DecodeAdmissionPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "wait" => Ok(Self::Wait),
            "reject" => Ok(Self::Reject),
            other => Err(format!(
                "decode admission policy must be 'wait' or 'reject', got '{other}'"
            )),
        }
    }
}

/// Router-side decode admission: at most `cap` requests in flight per decode worker.
///
/// Requests beyond the cap are queued in strict FIFO order (or rejected), so the
/// queue wait shows up as time to first token instead of as co-scheduled decode
/// steps. Applies to the decode / aggregated hop of least-loaded and power-of-two
/// routing only; prefill selection and pinned (affinity / explicit) targets bypass it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecodeAdmissionConfig {
    pub cap: u64,
    #[serde(default)]
    pub policy: DecodeAdmissionPolicy,
    /// Upper bound on the queue wait; unset waits until admitted or cancelled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wait_secs: Option<f64>,
}

impl DecodeAdmissionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.cap == 0 {
            return Err("decode admission cap must be >= 1".to_string());
        }
        if let Some(max_wait) = self.max_wait_secs
            && !(max_wait.is_finite() && max_wait > 0.0)
        {
            return Err("decode admission max wait must be a finite value > 0".to_string());
        }
        Ok(())
    }
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
            session_affinity_ttl_secs: None,
            session_affinity_mode: SessionAffinityMode::Hard,
            decode_admission: None,
        }
    }

    pub fn with_load_threshold_config(mut self, config: LoadThresholdConfig) -> Self {
        self.load_threshold_config = config;
        self
    }

    #[deprecated(
        note = "enforce_disagg is ignored; topology and readiness come from registered worker types"
    )]
    pub fn with_enforce_disagg(mut self, enforce_disagg: bool) -> Self {
        self.enforce_disagg = enforce_disagg;
        self
    }

    pub fn with_session_affinity_ttl_secs(mut self, ttl_secs: u64) -> Self {
        self.session_affinity_ttl_secs = Some(ttl_secs);
        self
    }
}

#[derive(Clone)]
pub enum EngineConfig {
    /// Remote networked engines that we discover via etcd
    Dynamic {
        model: Box<LocalModel>,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    },

    /// A Text engine receives text, does it's own tokenization and prompt formatting.
    InProcessText {
        engine: Arc<dyn StreamingEngine>,
        model: Box<LocalModel>,
    },

    /// A Tokens engine receives tokens, expects to be wrapped with pre/post processors that handle tokenization.
    InProcessTokens {
        engine: ExecutionContext,
        model: Box<LocalModel>,
        is_prefill: bool,
        is_decode: bool,
    },
}

impl EngineConfig {
    pub fn local_model(&self) -> &LocalModel {
        use EngineConfig::*;
        match self {
            Dynamic { model, .. } => model,
            InProcessText { model, .. } => model,
            InProcessTokens { model, .. } => model,
        }
    }

    pub fn chat_engine_factory(&self) -> Option<&ChatEngineFactoryCallback> {
        match self {
            EngineConfig::Dynamic {
                chat_engine_factory,
                ..
            } => chat_engine_factory.as_ref(),
            _ => None,
        }
    }
}
