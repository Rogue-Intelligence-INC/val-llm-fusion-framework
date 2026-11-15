//! Valhalla Hub System - f64 GPU-Accelerated Performance Version
//!
//! Unified entry point integrating all systems
//! Uses f64 for high-performance GPU-accelerated processing
//!
//! Features:
//! - Fate-f64 signal coordination
//! - Quad systems processing
//! - Batch operations for GPU acceleration
//! - 70% resource utilization target
//! - **System storage**: full file-operation permissions

use serde::{Deserialize, Serialize};
use log::info;
use std::collections::HashMap;
use std::time::Instant;
use std::path::Path;
use uuid::Uuid;

pub mod cortex;
pub mod dl_front;
pub mod fair_score;
pub mod fair_suite;
pub mod grinder;
pub mod patch_export;
pub mod model_export;
pub mod pipeline;
pub mod task_decompose;
#[cfg(feature = "patch-lm")]
pub mod patch_lm;
pub mod hybrid;
#[cfg(feature = "patch-lm")]
pub mod fused_lm;
#[cfg(feature = "patch-lm")]
pub mod context_routing;
#[cfg(feature = "patch-lm")]
pub mod valhalla_base;
#[cfg(feature = "patch-lm")]
pub mod valhalla_base_protocol;
#[cfg(feature = "patch-lm")]
pub mod valhalla_base_client;
#[cfg(feature = "patch-lm")]
pub mod valhalla_base_bench;
#[cfg(feature = "patch-lm")]
pub mod frames_wiki;
#[cfg(feature = "patch-lm")]
pub mod valhalla_base_eval;
#[cfg(feature = "patch-lm")]
pub mod valhalla_base_http;
pub mod mcq_fate;
pub mod mcq_pairwise;
pub mod multihop_compose;
pub mod multihop_fate;
pub mod multimodal;
pub mod compose_fate;
pub mod model_fate;
pub mod fate_adaptive_algebra;
#[cfg(feature = "patch-lm")]
pub mod fate_io;
#[cfg(feature = "patch-lm")]
pub mod elastic_resources;
#[cfg(feature = "patch-lm")]
pub mod open_qa_interface;
#[cfg(feature = "patch-lm")]
pub mod multihop_bench;
pub use fate_adaptive_algebra::{
    fate_adaptive_context_boost, fate_adaptive_distractor_filter, fate_context_pollution,
    fate_land_tick, fate_landing_drift, fate_landing_fingerprints, fate_retrieve_context_lines,
    fate_signal_land_into_lines, fate_signal_landing_indices, fate_time_cite_bracket,
    line_query_score, resolve_evidence_line_indices, score_evidence_line_overlap,
    score_qa_algebra, score_qa_with_evidence, strip_all_citations, sync_algebra_meta,
    FateAlgebraMeta, FateAlgebraState,
};
#[cfg(feature = "patch-lm")]
pub use fate_io::{
    apply_unified_solution, ensure_unified_runtime, fate_egress, fate_egress_body,
    fate_ingress, fate_score_short_answer, unified_solution_config, FateIngressBundle,
};
#[cfg(feature = "patch-lm")]
pub use open_qa_interface::{
    run_open_qa, run_open_qa_scored, OpenQaFlags, OpenQaRequest, OpenQaResult,
};
#[cfg(feature = "patch-lm")]
pub use elastic_resources::{
    cuda_runtime_ready, elastic_parallel_workers, par_map_items, ElasticResourcePlan,
};
#[cfg(feature = "patch-lm")]
pub use fate_unified_qa::{run_fate_unified_qa, FateUnifiedQaOpts};
pub mod free_multimodal_fate;
pub mod multimodal_compose;
pub mod valhalla_adapter;
pub mod natural_channels;
pub mod native_qa;
pub mod signal_ingress;
pub mod fate_orchestrator;
pub mod fate_qa;
pub mod fate_parallel_longchain;
#[cfg(feature = "patch-lm")]
pub mod valhalla_address;
#[cfg(feature = "patch-lm")]
pub mod fate_pretrain_holdout;
#[cfg(feature = "patch-lm")]
pub mod fate_longchain_unified;
#[cfg(feature = "patch-lm")]
pub mod fate_unified_qa;
pub mod smoke_report;
pub mod monitor;
pub mod triad_dashboard;
pub mod triad_viz_types;
pub mod triad_viz;

pub use fate_orchestrator::{
    export_fragment, fate_warmup_enabled, fragment_mode, ingest_corpus_with_fragments,
    append_corpus_with_fragments_mode, production_fragment_mode,
    ingest_corpus_with_fragments_mode, ingest_corpus_with_fragments_mode_full,
    ingest_line_density_fragment,
    ingress_and_adapt, live_session_enabled, merge_fragments, merge_fragments_for_query,
    multi_fragment_enabled, pretrain_corpus_to_library, pulse_question_live,
    corpus_ingest_skip, dynamic_corpus_fragments_enabled, explicit_ingest_all_corpus,
    query_fragment_merge_enabled, query_fragment_top_k, run_fate_adapt, run_fate_convergence_cycles,
    run_fate_warmup, score_fragment_for_query, select_fragments_for_query,
    FateAdaptOutcome, FateCycleLimits, FateDensitySegmenter, FateOrchestratorConfig, FateWarmupOutcome,
    FragmentCloseMeta, FragmentMode,
    ManifoldFragmentLibrary, PretrainStats, QueryFragmentSelection, SnapshotFragment,
    FRAGMENT_LIBRARY_VERSION, MANIFOLD_PROTOCOL,
};

pub use native_qa::{run_native_qa, run_native_qa_batch, NativeQaBatchRequest, NativeQaRequest, NativeQaResponse};
pub use dl_front::{
    encode_modality, build_body_ports, hash_embed, project_port, BodyFrontPorts, DlFrontConfig,
    ModalityInput, DL_EMBED_DIM,
};
pub use signal_ingress::{
    build_model_patch, encode_payload, BodyTarget, EncodingMode, IngressBudget, IngressPayload,
    IngressStats, StemOrganCluster, TriadFeedReport, TriadSession, ValhallaModelPatch,
    PATCH_VECTOR_DIM,
};
pub use smoke_report::{SmokeReport, SMOKE_CORPUS};
pub use triad_viz_types::{
    StemOrganViz, TileNeuronViz, TileSynapseViz, TriadVizSnapshot, CycleVizPoint,
};

#[cfg(feature = "patch-lm")]
pub use valhalla_base::{
    context_fingerprint, item_to_corpus_line, run_valhalla_base, ValhallaBaseConfig,
    ValhallaBaseDecode, ValhallaBaseQuestion, ValhallaBaseRequest, ValhallaBaseResponse,
    ValhallaBaseRow, ValhallaBaseSessionState, ValhallaContextItem, ValhallaFateIngressRouting,
    ValhallaFateWeightMode, ValhallaFollowUpDecode, ValhallaMcqDecode, VALHALLA_BASE_PROTOCOL,
};
#[cfg(feature = "patch-lm")]
pub use valhalla_base_protocol::{
    example_names, example_request, execute_request, execute_request_json, parse_request_json,
    ensure_production_runtime, production_config, serialize_error, serialize_response,
    validate_request, validate_request_json, ValhallaBaseErrorBody, ValhallaBaseErrorKind,
};
#[cfg(feature = "patch-lm")]
pub use context_routing::{corpus_for_prompt, resolve_context_lines, targeted_context, ContextIngestMode};
#[cfg(feature = "patch-lm")]
pub use valhalla_base_client::{
    RunBatchOptions, RunBatchOutcome, RunOptions, ValhallaBaseCheckpoint, ValhallaBaseClient,
};
#[cfg(feature = "patch-lm")]
pub use valhalla_base_bench::{
    default_corpus_dirs, run_local_corpus_demo, run_multihop_fate_rag, run_multihop_fate_rag_ctrl,
    run_multihop_hybrid_rag, run_multihop_hybrid_rag_ctrl, run_multihop_rag, run_multihop_rag_ctrl,
    run_needle_session, run_longchain_fate_final_test, run_v1_extensions, write_report,
};
#[cfg(feature = "patch-lm")]
pub use valhalla_base_eval::{
    run_frames_slice, run_locomo_slice, run_long_chain_item, run_longbench_slice,
    run_mcq_batch_item, EvalControl, FramesEvalReport, LongChainEvalReport, LongbenchEvalReport,
};
#[cfg(feature = "patch-lm")]
pub use natural_channels::{natural_channels_enabled, pulse_question_natural, ChannelPick};
#[cfg(feature = "patch-lm")]
pub use fate_orchestrator::{
    apply_eval_corpus_cap, compact_manifold_snapshot, eval_chunk_size, eval_corpus_line_cap,
    export_session_after_ingest, fate_link_each_extend_enabled, fate_link_fragments_enabled, fragment_only_session_enabled,
    production_prelearn_count, stored_fragments, FateLinkOutcome,
};
#[cfg(feature = "patch-lm")]
pub use fate_parallel_longchain::{
    build_longchain_substrate, longchain_parallel_enabled, run_fate_parallel_longchain_item,
    run_fate_parallel_longchain_smoke, run_locomo_routing_ab, run_longchain_qa_parallel,
    spread_from_client, FateParallelLongchainReport, FateParallelLongchainRow, FateRoutingAbReport,
    FateRoutingAbRow,
};
#[cfg(feature = "patch-lm")]
pub use fate_longchain_unified::{
    run_fate_longchain_unified, AnchorDriftRow, CorpusBuildMode, FateLongchainUnifiedReport,
    UnifiedPhaseRow, UnifiedRouting, PROTOCOL as FATE_LONGCHAIN_UNIFIED_PROTOCOL,
};
#[cfg(feature = "patch-lm")]
pub use fate_pretrain_holdout::{
    adaptive_corpus_for_pretrain, adaptive_corpus_for_single, run_fate_pretrain_holdout,
    score_qa_pretrain, score_with_evidence, FatePretrainHoldoutReport, HoldoutItemResult,
    PRETRAIN_MODE,
};
pub use valhalla_base_client::PRETRAIN_PIPE_MODE;
#[cfg(feature = "patch-lm")]
pub use valhalla_base_http::{infer_sync, serve as serve_valhalla_base_http};

// System storage
pub use valhalla_storage::{SystemStorage, AllStorages, init_all_storages};

// Fate f64 (GPU-ready)
pub use fate_f64::{
    FateSystem, DeformationSignal, SystemType as FateSystemType,
    SystemPreferences, ResourceLimits, SignalBatch, FateEvaluation,
    TripleFateCombined, FateSlice, FateDataEntry, FateConvergenceState,
};

// Quad systems (f64)
pub use valhalla_metasystems_blackhole::{GravityField, GravityFieldConfig, EventHorizon, EventHorizonConfig};
pub use valhalla_metasystems_meshbrain::{MeshBrainSystem, MeshBrainConfig};
pub use valhalla_metasystems_multinova::{MultinovaProcessor, MultinovaConfig};
pub use valhalla_metasystems_baseforce::{BaseForceCoordinator, CoordinationResult};

// Space GPU high-dimensional computation
pub use valhalla_space_f64_gpu::{SpaceGPU, FractalDimensionCalculatorGPU, HyperVector, DimensionResult, SpaceStatus};

/// Hub configuration (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubConfig {
    pub max_requests_per_second: u64,
    pub routing_threshold: f64,
    pub load_balance_enabled: bool,
    pub resource_percent: u8,
    pub batch_size: usize,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            max_requests_per_second: 100000,
            routing_threshold: 0.5,
            load_balance_enabled: false, // Use Fate routing by default
            resource_percent: 70,
            batch_size: 1024,
        }
    }
}

/// Quad system type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QuadSystemType {
    BlackHole,
    MeshBrain,
    MultiNova,
    BaseForce,
}

impl From<QuadSystemType> for FateSystemType {
    fn from(q: QuadSystemType) -> Self {
        match q {
            QuadSystemType::BlackHole => FateSystemType::BlackHole,
            QuadSystemType::MeshBrain => FateSystemType::MeshBrain,
            QuadSystemType::MultiNova => FateSystemType::MultiNova,
            QuadSystemType::BaseForce => FateSystemType::BaseForce,
        }
    }
}

/// System status (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub is_initialized: bool,
    pub active_systems: Vec<String>,
    pub system_health: HashMap<String, f64>,
    pub last_update_tick: u64,
    pub fate_signal_tick: u64,
}

/// Hub statistics (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubStatistics {
    pub total_requests: u64,
    pub successful_routes: u64,
    pub failed_routes: u64,
    pub system_usage: HashMap<String, u64>,
    pub average_response_time_ms: f64,
    pub total_processing_time_ms: f64,
}

/// Hub system slice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubSlice {
    pub hub_id: String,
    pub statistics: HubStatistics,
    /// Embedded Fate slice
    pub fate_slice: FateSlice,
}

/// Valhalla Hub (f64 GPU-Accelerated)
/// How Fate feedback assigns quality to ingress signals.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FateWeightMode {
    /// quality_score = 1.0 always (high constraint — no differentiation).
    #[default]
    Fixed,
    /// quality/intensity from Fate evaluate_system (relaxed — learned weights).
    Evaluated,
}

/// Hub ingress routing when `fate_routing_free` is false.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum IngressRoutingMode {
    /// Round-robin across quad systems (symmetric — no Fate differentiation).
    #[default]
    RoundRobin,
    /// Route by corpus-line semantic keywords → asymmetric quad load.
    Semantic,
    /// Within each corpus-line ingest loop, cycle signals across all 4 quads.
    QuadCycle,
    /// 50% quad-cycle (even signals) + 50% semantic primary quad (odd signals) per line.
    HybridQuadSemantic,
}

pub struct ValhallaHub {
    pub hub_id: String,
    pub config: HubConfig,
    
    // Fate System (f64)
    pub fate_system: FateSystem,
    
    /// Triple-Fate storage system
    pub triple_fate: TripleFateCombined,
    
    // Quad Systems (f64)
    pub gravity_field: GravityField,
    pub event_horizon: EventHorizon,
    pub meshbrain: MeshBrainSystem,
    pub multinova: MultinovaProcessor,
    pub baseforce: BaseForceCoordinator,
    
    pub system_status: SystemStatus,
    pub statistics: HubStatistics,
    
    /// System storage - the Hub's own storage
    pub storage: Option<SystemStorage>,
    /// Storage of all subsystems
    pub all_storages: Option<AllStorages>,
    /// Space GPU high-dimensional computation
    pub space_gpu: SpaceGPU,
    /// Fate weight assignment mode for ingress feedback.
    pub fate_weight_mode: FateWeightMode,
    /// When false, Hub round-robins quad routing (Fate prefs locked).
    pub fate_routing_free: bool,
    /// Ingress routing while `fate_routing_free` is false.
    pub ingress_routing: IngressRoutingMode,
    /// Current corpus line for semantic ingress routing.
    ingress_line: Option<String>,
    /// Signal index within current corpus line (reset per line).
    ingress_line_signal_idx: usize,
}

impl ValhallaHub {
    pub fn new() -> Self {
        Self::with_config(HubConfig::default())
    }
    
    pub fn with_config(config: HubConfig) -> Self {
        info!("🌌 Initializing Valhalla Hub (f64 GPU-Accelerated)");
        
        // Initialize Fate with resource limits
        let fate_limits = ResourceLimits {
            max_memory_mb: 4096,
            max_cpu_percent: config.resource_percent,
            max_buffer_size: 100000,
            processing_interval_ms: 10,
            gpu_enabled: true,
            batch_size: config.batch_size,
        };
        let fate_system = FateSystem::new("hub_fate".to_string(), fate_limits);
        
        // Initialize Quad systems
        let gravity_config = GravityFieldConfig::default();
        let gravity_field = GravityField::new(gravity_config.clone());
        let horizon_config = EventHorizonConfig::default();
        let event_horizon = EventHorizon::new(horizon_config, gravity_field.clone());
        
        Self {
            hub_id: Uuid::new_v4().to_string(),
            config,
            fate_system,
            triple_fate: TripleFateCombined::new(),
            gravity_field,
            event_horizon,
            meshbrain: MeshBrainSystem::new(MeshBrainConfig::default()),
            multinova: MultinovaProcessor::new(MultinovaConfig::default()),
            baseforce: BaseForceCoordinator::new(),
            system_status: SystemStatus {
                is_initialized: true,
                active_systems: vec!["fate".into(), "blackhole".into(), "meshbrain".into(), "multinova".into(), "baseforce".into()],
                system_health: HashMap::new(),
                last_update_tick: 0,
                fate_signal_tick: 0,
            },
            statistics: HubStatistics {
                total_requests: 0,
                successful_routes: 0,
                failed_routes: 0,
                system_usage: HashMap::new(),
                average_response_time_ms: 0.0,
                total_processing_time_ms: 0.0,
            },
            storage: None,
            all_storages: None,
            space_gpu: SpaceGPU::new(),
            fate_weight_mode: FateWeightMode::Fixed,
            fate_routing_free: false,
            ingress_routing: IngressRoutingMode::RoundRobin,
            ingress_line: None,
            ingress_line_signal_idx: 0,
        }
    }
    
    /// Initialise with a storage path
    pub fn with_storage(base_path: &Path) -> std::io::Result<Self> {
        let mut instance = Self::new();
        instance.init_storage(base_path)?;
        Ok(instance)
    }
    
    /// Initialise all subsystem storage
    pub fn init_storage(&mut self, base_path: &Path) -> std::io::Result<()> {
        self.storage = Some(SystemStorage::new(base_path, "hub")?);
        self.all_storages = Some(init_all_storages(base_path)?);
        info!("📁 Storage initialized at {:?}", base_path);
        Ok(())
    }
    
    /// Get Hub storage
    pub fn storage(&self) -> Option<&SystemStorage> {
        self.storage.as_ref()
    }
    
    /// Get the named subsystem's storage
    pub fn system_storage(&self, system: QuadSystemType) -> Option<&SystemStorage> {
        self.all_storages.as_ref().map(|s| match system {
            QuadSystemType::BlackHole => &s.blackhole,
            QuadSystemType::MeshBrain => &s.meshbrain,
            QuadSystemType::MultiNova => &s.multinova,
            QuadSystemType::BaseForce => &s.baseforce,
        })
    }
    
    /// Get StemCell storage
    pub fn stemcell_storage(&self) -> Option<&SystemStorage> {
        self.all_storages.as_ref().map(|s| &s.stemcell)
    }
    
    /// Get Tile storage
    pub fn tile_storage(&self) -> Option<&SystemStorage> {
        self.all_storages.as_ref().map(|s| &s.tile)
    }
    
    /// Get Chamber storage
    pub fn chamber_storage(&self) -> Option<&SystemStorage> {
        self.all_storages.as_ref().map(|s| &s.chamber)
    }
    
    // ==================== Data-operation shortcuts ====================
    
    /// Store content into the Hub
    pub fn store(&self, name: &str, content: &str) -> std::io::Result<()> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .write(name, content)
    }
    
    /// Read Hub content
    pub fn load(&self, name: &str) -> std::io::Result<String> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .read(name)
    }
    
    /// Delete Hub content
    pub fn remove(&self, name: &str) -> std::io::Result<()> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .delete(name)
    }
    
    /// List all Hub content
    pub fn list_stored(&self) -> std::io::Result<Vec<String>> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .list_all()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Fate storage layer - delegated to the internal triple_fate
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Store data through Fate
    pub fn fate_store(&mut self, data: &[u8]) -> String {
        self.triple_fate.store_data(data)
    }
    
    /// Store a string through Fate
    pub fn fate_store_string(&mut self, content: &str) -> String {
        self.triple_fate.store_string(content)
    }
    
    /// System autonomously created data
    pub fn fate_create(&mut self, data: Vec<u8>) -> String {
        self.triple_fate.create_data(data)
    }
    
    /// Modify data stored by Fate
    pub fn fate_modify(&mut self, id: &str, new_data: Vec<u8>) -> bool {
        self.triple_fate.modify_data(id, new_data)
    }
    
    /// Read data stored by Fate
    pub fn fate_load(&self, id: &str) -> Option<&FateDataEntry> {
        self.triple_fate.load_data(id)
    }
    
    /// Delete data stored by Fate
    pub fn fate_remove(&mut self, id: &str) -> Option<FateDataEntry> {
        self.triple_fate.remove_data(id)
    }
    
    /// List all IDs stored by Fate
    pub fn fate_list(&self) -> Vec<&String> {
        self.triple_fate.list_stored()
    }
    
    /// Fate storage count
    pub fn fate_count(&self) -> usize {
        self.triple_fate.stored_count()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // System slice - freeze/thaw (bincode binary)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Save the system slice (.hub binary)
    pub fn save_slice(&self, path: &Path) -> std::io::Result<()> {
        let slice = HubSlice {
            hub_id: self.hub_id.clone(),
            statistics: self.statistics.clone(),
            fate_slice: self.triple_fate.freeze(),
        };
        let bytes = bincode::serialize(&slice)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }
    
    /// Restore from a slice
    pub fn load_slice(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let slice: HubSlice = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        let mut instance = Self::new();
        instance.hub_id = slice.hub_id;
        instance.statistics = slice.statistics;
        instance.triple_fate = TripleFateCombined::thaw(slice.fate_slice);
        Ok(instance)
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Space GPU high-dimensional computation
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Compute dimension (GPU accelerated)
    pub fn space_calculate_dimension(&mut self, data: &[f64]) -> DimensionResult {
        self.space_gpu.process_and_expand(data)
    }
    
    /// Batch-compute dimensions (GPU accelerated)
    pub fn space_calculate_batch(&mut self, data_batch: &[Vec<f64>]) -> Vec<DimensionResult> {
        self.space_gpu.process_batch(data_batch)
    }
    
    /// Discover unknown patterns
    pub fn space_on_discovery(&mut self, data: &[f64], pattern: &str) -> String {
        self.space_gpu.on_discovery(data, pattern)
    }
    
    /// Get the current dimension
    pub fn space_dimension(&self) -> f64 {
        self.space_gpu.get_dimension()
    }
    
    /// Get Space state
    pub fn space_status(&self) -> SpaceStatus {
        self.space_gpu.get_status()
    }
    
    /// Process request with Fate routing (f64)
    pub fn process(&mut self, request: HubRequest) -> HubResponse {
        let start = Instant::now();
        self.statistics.total_requests += 1;
        
        // Route based on request type
        let target = self.route_request(&request);
        let target_name = format!("{:?}", target);
        
        // Process based on target
        let result = match target {
            QuadSystemType::BlackHole => {
                let gravity = self.gravity_field.calculate_gravity(request.intensity);
                format!("BlackHole: gravity={:.6}", gravity)
            },
            QuadSystemType::MeshBrain => {
                let signal = self.meshbrain.generate_signal(request.intensity, [0.0, 0.0, 0.0]);
                format!("MeshBrain: intelligence={:.6}", signal.intelligence_level)
            },
            QuadSystemType::MultiNova => {
                let event = self.multinova.process_evolution(request.intensity);
                format!("MultiNova: is_nova={}, radiation={:.6}", event.is_nova, event.radiation)
            },
            QuadSystemType::BaseForce => {
                let coord = self.baseforce.coordinate(request.intensity, request.intensity, request.intensity);
                format!("BaseForce: activity={:.6}", coord.total_activity)
            },
        };
        
        // Feedback to Fate
        self.feedback_to_fate(&target, request.intensity, true);
        
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics.successful_routes += 1;
        self.statistics.total_processing_time_ms += elapsed;
        self.statistics.average_response_time_ms = 
            self.statistics.total_processing_time_ms / self.statistics.successful_routes as f64;
        *self.statistics.system_usage.entry(target_name).or_insert(0) += 1;
        
        self.system_status.fate_signal_tick = self.fate_system.get_logical_time();
        
        HubResponse {
            response_id: Uuid::new_v4().to_string(),
            success: true,
            message: result,
            data: None,
            processing_time_ms: elapsed,
        }
    }
    
    /// Restore Fate state from incubation snapshot (multi-round sessions).
    pub fn seed_fate_from_snapshot(
        &mut self,
        prefs: &[f64],
        routed: [u64; 4],
        logical_time: Option<u64>,
    ) {
        self.fate_system
            .seed_from_hub_snapshot(prefs, routed, logical_time);
    }

    pub fn set_fate_policy(
        &mut self,
        mode: FateWeightMode,
        routing_free: bool,
        ingress_routing: IngressRoutingMode,
    ) {
        self.fate_weight_mode = mode;
        self.fate_routing_free = routing_free;
        self.ingress_routing = ingress_routing;
    }

    pub fn set_ingress_line(&mut self, line: Option<String>) {
        let next = line.filter(|s| !s.trim().is_empty());
        if next.as_deref() != self.ingress_line.as_deref() {
            self.ingress_line_signal_idx = 0;
        }
        self.ingress_line = next;
    }

    fn quad_from_index(idx: usize) -> QuadSystemType {
        match idx % 4 {
            0 => QuadSystemType::BlackHole,
            1 => QuadSystemType::MeshBrain,
            2 => QuadSystemType::MultiNova,
            _ => QuadSystemType::BaseForce,
        }
    }

    /// Even signals → quad cycle; odd signals → semantic primary quad for this line.
    fn route_hybrid_quad_semantic(&mut self) -> QuadSystemType {
        let idx = self.ingress_line_signal_idx;
        self.ingress_line_signal_idx = idx.saturating_add(1);
        if idx % 2 == 0 {
            return Self::quad_from_index(idx / 2);
        }
        if let Some(ref line) = self.ingress_line {
            return Self::route_corpus_line_semantic(line);
        }
        Self::quad_from_index(idx)
    }

    /// Semantic quad routing from corpus line text (physics→BH, language→MB, process→MN, geo→BF).
    pub fn route_corpus_line_semantic(line: &str) -> QuadSystemType {
        let low = line.to_lowercase();
        let score = |keys: &[&str]| -> usize {
            keys.iter().filter(|k| low.contains(**k)).count()
        };
        let bh = score(&[
            "gravity", "force", "mass", "energy", "newton", "momentum", "physics", "weight",
            "friction", "work", "power", "velocity", "acceleration", "kinetic",
        ]);
        let mb = score(&[
            "english", "language", "word", "speak", "text", "formula", "symbol", "name",
            "chemical", "marsupial", "language",
        ]);
        let mn = score(&[
            "process", "boil", "freez", "temperature", "evapor", "condens", "reaction", "ph",
            "distill", "expand", "melt", "steam", "cloud", "vapor", "neutral",
        ]);
        let bf = score(&[
            "capital", "city", "country", "continent", "located", "australia", "france", "paris",
            "sydney", "region", "border", "coast", "flag", "currency", "oceania", "europe",
            "melbourne", "canberra", "outback", " reef",
        ]);
        let scores = [bh, mb, mn, bf];
        let max = *scores.iter().max().unwrap_or(&0);
        let idx = if max == 0 {
            line.bytes()
                .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize))
                % 4
        } else {
            scores
                .iter()
                .enumerate()
                .max_by_key(|(_, s)| *s)
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        match idx {
            0 => QuadSystemType::BlackHole,
            1 => QuadSystemType::MeshBrain,
            2 => QuadSystemType::MultiNova,
            _ => QuadSystemType::BaseForce,
        }
    }

    /// Feedback to Fate (f64)
    fn feedback_to_fate(&mut self, system: &QuadSystemType, intensity: f64, _success: bool) {
        self.feedback_to_fate_with_quality(system, intensity, None);
    }

    fn feedback_to_fate_with_quality(
        &mut self,
        system: &QuadSystemType,
        intensity: f64,
        qa_quality: Option<f64>,
    ) {
        let idx = match system {
            QuadSystemType::BlackHole => 0,
            QuadSystemType::MeshBrain => 1,
            QuadSystemType::MultiNova => 2,
            QuadSystemType::BaseForce => 3,
        };
        let (quality, eff_intensity) = match self.fate_weight_mode {
            FateWeightMode::Fixed => {
                let q = qa_quality.unwrap_or(1.0);
                (q, intensity)
            }
            FateWeightMode::Evaluated => {
                let ev = self.fate_system.evaluate_system(idx);
                let base_q = if ev.is_bootstrap {
                    1.0
                } else {
                    ev.quality.max(0.05)
                };
                let q = qa_quality.map(|qq| (base_q + qq) / 2.0).unwrap_or(base_q);
                let i = if ev.intensity > 0.0 {
                    intensity * (0.5 + ev.intensity * 0.5)
                } else {
                    intensity
                };
                (q, i)
            }
        };
        let mut signal = DeformationSignal::new(
            (*system).into(),
            eff_intensity,
            vec![0.0; 4],
        );
        signal.quality_score = quality.clamp(0.05, 1.0);
        self.fate_system.receive_signal(signal);
    }

    fn quad_index(system: &QuadSystemType) -> usize {
        match system {
            QuadSystemType::BlackHole => 0,
            QuadSystemType::MeshBrain => 1,
            QuadSystemType::MultiNova => 2,
            QuadSystemType::BaseForce => 3,
        }
    }

    /// QA outcome feedback — boosts semantic-primary quad on success, penalizes on failure.
    pub fn feedback_qa_outcome(&mut self, question: &str, on_topic: bool, routed: [u64; 4]) {
        let primary = Self::route_corpus_line_semantic(question);
        let primary_idx = Self::quad_index(&primary);
        let total: f64 = routed.iter().sum::<u64>().max(1) as f64;
        let route_share: [f64; 4] =
            std::array::from_fn(|i| routed[i] as f64 / total);

        let state = {
            let prefs: Vec<f64> = self
                .fate_system
                .calculate_system_preferences()
                .affinities
                .iter()
                .copied()
                .collect();
            crate::fate_adaptive_algebra::FateAlgebraState::from_hub_prefs(
                &prefs,
                &crate::fate_adaptive_algebra::FateAlgebraMeta::default(),
            )
        };

        let pass_i = state.qa_pass_intensity();
        let fail_i = state.qa_fail_intensity();

        for i in 0..4 {
            let system = Self::quad_from_index(i);
            let (quality, intensity) = if on_topic {
                if i == primary_idx {
                    (0.95, pass_i)
                } else {
                    (0.55 + 0.15 * route_share[i], fail_i * route_share[i].max(0.1))
                }
            } else if i == primary_idx {
                (fail_i, fail_i)
            } else {
                (0.35, fail_i * route_share[i].max(0.08))
            };
            self.feedback_to_fate_with_quality(&system, intensity, Some(quality));
        }
        self.mark_epoch_complete();
    }

    /// Line-level structural feedback into Fate (grinder Q↔line align, adaptive deltas).
    pub fn feedback_corpus_line_structural(
        &mut self,
        line: &str,
        align: f64,
        on_topic: bool,
    ) {
        let primary = Self::route_corpus_line_semantic(line);
        let state = {
            let prefs: Vec<f64> = self
                .fate_system
                .calculate_system_preferences()
                .affinities
                .iter()
                .copied()
                .collect();
            crate::fate_adaptive_algebra::FateAlgebraState::from_hub_prefs(
                &prefs,
                &crate::fate_adaptive_algebra::FateAlgebraMeta::default(),
            )
        };
        let intensity = if on_topic {
            state.affinity_pos_delta(align)
        } else {
            state.affinity_neg_delta(align)
        };
        let quality = if on_topic {
            0.5 + 0.45 * align.clamp(0.0, 1.0)
        } else {
            state.neg_coupling()
        };
        self.feedback_to_fate_with_quality(&primary, intensity, Some(quality));
    }
    
    /// Route request using Fate preferences (f64)
    fn route_request(&mut self, _request: &HubRequest) -> QuadSystemType {
        if !self.fate_routing_free {
            match self.ingress_routing {
                IngressRoutingMode::Semantic => {
                    if let Some(ref line) = self.ingress_line {
                        return Self::route_corpus_line_semantic(line);
                    }
                }
                IngressRoutingMode::QuadCycle => {
                    let idx = self.ingress_line_signal_idx;
                    self.ingress_line_signal_idx = idx.saturating_add(1);
                    return Self::quad_from_index(idx);
                }
                IngressRoutingMode::HybridQuadSemantic => {
                    return self.route_hybrid_quad_semantic();
                }
                IngressRoutingMode::RoundRobin => {}
            }
            let idx = (self.statistics.total_requests as usize) % 4;
            return Self::quad_from_index(idx);
        }
        // Periodically record snapshots to support convergence detection
        if self.statistics.total_requests % 1000 == 0 {
            self.fate_system.record_preference_snapshot();
        }
        
        // Check the Fate convergence mode - poll while still exploring
        let pattern = self.fate_system.detect_convergence_pattern();
        
        // Use preference routing only when Fate detects a "converging" or "fixed_point" mode
        // Otherwise keep polling and exploring
        let fate_ready = matches!(pattern, Some("converging") | Some("fixed_point"));
        
        if !fate_ready {
            // Fate has not formed a stable preference; round-robin allocation lets every subsystem receive signal
            let idx = (self.statistics.total_requests as usize) % 4;
            return match idx {
                0 => QuadSystemType::BlackHole,
                1 => QuadSystemType::MeshBrain,
                2 => QuadSystemType::MultiNova,
                _ => QuadSystemType::BaseForce,
            };
        }
        
        // Fate has formed a stable preference; Fate decides routing
        let prefs = self.fate_system.calculate_system_preferences();
        match prefs.preferred_system() {
            0 => QuadSystemType::BlackHole,
            1 => QuadSystemType::MeshBrain,
            2 => QuadSystemType::MultiNova,
            _ => QuadSystemType::BaseForce,
        }
    }
    
    /// Get Fate preferences
    pub fn get_fate_preferences(&self) -> SystemPreferences {
        self.fate_system.calculate_system_preferences()
    }
    
    /// Get system status
    pub fn get_status(&self) -> &SystemStatus {
        &self.system_status
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> &HubStatistics {
        &self.statistics
    }
    
    /// Get Fate system reference
    pub fn fate_system(&self) -> &FateSystem {
        &self.fate_system
    }
    
    /// Record the current Fate preference snapshot (for convergence detection)
    pub fn record_fate_snapshot(&mut self) {
        self.fate_system.record_preference_snapshot();
    }
    
    /// Mark one epoch (cycle) complete
    /// Only after this method is called can Fate judge convergence
    /// This ensures data must be fully processed before convergence can trigger
    pub fn mark_epoch_complete(&mut self) {
        self.fate_system.mark_epoch_complete();
    }
    
    /// Check whether data is still flowing in
    pub fn is_data_flowing(&self) -> bool {
        self.fate_system.is_data_flowing()
    }
    
    /// Fate autonomously judges convergence (algebraic marginal effect)
    /// Precondition: data fully processed (mark_epoch_complete called)
    pub fn has_fate_converged(&self) -> bool {
        self.fate_system.has_converged()
    }
    
    /// Get the Fate convergence mode
    pub fn get_fate_convergence_pattern(&self) -> Option<&'static str> {
        self.fate_system.detect_convergence_pattern()
    }
}

impl Default for ValhallaHub {
    fn default() -> Self {
        Self::new()
    }
}

/// Hub request (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubRequest {
    pub request_id: String,
    pub request_type: RequestType,
    pub intensity: f64,
    pub data: Option<serde_json::Value>,
    pub priority: Priority,
}

/// Request type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestType {
    Reasoning,
    Learning,
    Adaptation,
    DataProcessing,
    SystemCoordination,
}

/// Priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// Hub response (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubResponse {
    pub response_id: String,
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
    pub processing_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hub_with_fate() {
        let mut hub = ValhallaHub::new();
        
        // First request is bootstrap
        let req1 = HubRequest {
            request_id: "test_0".into(),
            request_type: RequestType::Reasoning,
            intensity: 0.5,
            data: None,
            priority: Priority::Normal,
        };
        let resp1 = hub.process(req1);
        assert!(resp1.message.contains("Bootstrap"));
        
        // Second request is real
        let req2 = HubRequest {
            request_id: "test_1".into(),
            request_type: RequestType::Reasoning,
            intensity: 0.5,
            data: None,
            priority: Priority::Normal,
        };
        let resp2 = hub.process(req2);
        assert!(resp2.success);
    }
    
    #[test]
    fn test_fate_integration() {
        let mut hub = ValhallaHub::new();
        
        for i in 0..10 {
            let request = HubRequest {
                request_id: format!("test_{}", i),
                request_type: RequestType::DataProcessing,
                intensity: 0.5,
                data: None,
                priority: Priority::Normal,
            };
            hub.process(request);
        }
        
        let prefs = hub.get_fate_preferences();
        assert_eq!(prefs.affinities.len(), 4);
    }
}
