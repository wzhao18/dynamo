// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection policies Dynamo ships.
//!
//! Any build that enables the Python bindings' `custom-policy` feature links these, and that
//! feature is on by default, so a frontend deployment selects one through router-policy YAML
//! without rebuilding. A build with default features off, and the standalone EPP, link no catalog
//! at all and reject a configured policy type at startup.
//!
//! These do not go through the replaceable `dynamo-worker-selection-policy-catalog` alias, and they
//! register before it, so a custom image's catalog adds its policies alongside these rather than
//! displacing them.
//!
//! Each policy is one module here. To ship another, add a module and a line in [`register`], then
//! add a row to the policy table in the router configuration guide. Keep a policy in a single file
//! until it needs submodules, then promote it to a directory. If a policy ever needs a dependency
//! beyond `dynamo-kv-router`, put it behind its own default-on Cargo feature so a build can drop
//! it; every policy registered here is compiled into every artifact that links this crate.

mod decode_step_cost;
mod least_active_decode;
mod prefill_step_cost;
mod two_tier_cost_fn;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

/// Register every policy Dynamo ships.
///
/// `default` is reserved by the registry for Dynamo's built-in worker selector, so no policy here
/// can shadow it. A later catalog that reuses one of these type names fails registration rather
/// than overriding it.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    two_tier_cost_fn::register(registry)?;
    prefill_step_cost::register(registry)?;
    decode_step_cost::register(registry)?;
    least_active_decode::register(registry)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::services::selection::WorkerSelectionPolicyFactory;
    use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};

    use super::*;

    /// Resolve router-policy YAML exactly as the Python bindings do at startup, so these cover the
    /// real configuration path rather than the registrars in isolation.
    fn resolve(
        yaml: &str,
    ) -> (
        KvRouterConfig,
        Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError>,
    ) {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy_file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();
        let resolved = registry.resolve(&config);
        (config, resolved)
    }

    /// Catches a policy type name that drifts from its documentation, and proves the documented
    /// instance shape constructs for every stage it selects.
    #[test]
    fn resolves_documented_yaml() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  prefill: dynamo-two-tier-cost-fn
  decode: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");

        let partition = RoutingPartitionRef::new("model", "default");
        for worker_type in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            factory(&config, worker_type, partition);
        }
    }

    /// With only a prefill policy selected (the AgentX render), decode still resolves to a
    /// policy factory: the least-active decode default, not the KV-overlap/footprint selector.
    #[test]
    fn decode_defaults_to_least_active_when_unselected() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  prefill: ovh4-step-cost
  instances:
    - name: ovh4-step-cost
      type: prefill-step-cost
"#,
        );
        let factory = resolved.unwrap().expect("decode default makes a factory");
        factory(
            &config,
            WorkerType::Decode,
            RoutingPartitionRef::new("model", "default"),
        );
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();
        assert!(
            registry
                .resolve_for_worker_type(&KvRouterConfig::default(), WorkerType::Decode)
                .unwrap()
                .is_some()
        );
        assert!(
            registry
                .resolve_for_worker_type(&KvRouterConfig::default(), WorkerType::Prefill)
                .unwrap()
                .is_none()
        );
    }

    /// The policy YAML can select either decode policy explicitly, with parameters.
    #[test]
    fn resolves_explicit_decode_policy_yaml() {
        for yaml in [
            r#"
worker_selection:
  decode: lad
  instances:
    - name: lad
      type: least-active-decode
      parameters:
        affinity_slack: 2
        session_ttl_secs: 1800
"#,
            r#"
worker_selection:
  decode: dsc
  instances:
    - name: dsc
      type: decode-step-cost
"#,
        ] {
            let (config, resolved) = resolve(yaml);
            let factory = resolved.unwrap().expect("decode instance resolves");
            factory(
                &config,
                WorkerType::Decode,
                RoutingPartitionRef::new("model", "default"),
            );
        }
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  decode: lad
  instances:
    - name: lad
      type: least-active-decode
      parameters:
        affinity_slack: -1
"#,
        );
        assert!(resolved.is_err());
    }

    /// The OVHFIX-4 prefill routing YAML resolves for the prefill role.
    #[test]
    fn resolves_prefill_step_cost_yaml() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  prefill: ovh4-step-cost
  instances:
    - name: ovh4-step-cost
      type: prefill-step-cost
      parameters:
        store_retention_secs: 3600
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");
        factory(
            &config,
            WorkerType::Prefill,
            RoutingPartitionRef::new("model", "default"),
        );
        assert!(dynamo_kv_router::scheduling::global_prefix::registry().is_some());
    }

    /// An unknown parameter key is a mistake, most often a misremembered threshold name. It must
    /// fail startup rather than silently leaving the default in place.
    #[test]
    fn rejects_an_unknown_parameter_key() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_affinity_threshold: 0.8
"#,
        );

        let Err(error) = resolved else {
            panic!("an unknown parameter must fail resolution");
        };
        assert!(
            matches!(&error, WorkerSelectionPolicyRegistryError::Provider { policy_type, .. }
                if policy_type == two_tier_cost_fn::POLICY_TYPE),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("cache_affinity_threshold"),
            "the error should name the offending key: {error}"
        );
    }
}
