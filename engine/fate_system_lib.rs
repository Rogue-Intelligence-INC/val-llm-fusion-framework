//! Fate System v3.0 - f64 GPU-Accelerated Version
//!
//! High-performance signal coordination using f64 arithmetic
//! Designed for GPU acceleration with batch operations
//!
//! Core capabilities:
//! 1. Signal coordination with f64 arithmetic (GPU-friendly)
//! 2. Batch vector operations (parallelizable)
//! 3. Internal space evolution
//! 4. Signal-based time (deformation signal count)

use serde::{Deserialize, Serialize};
use rayon::prelude::*;
#[cfg(feature = "native_cuda")]
use log::{info, warn};
use std::time::{SystemTime, UNIX_EPOCH};
use std::cell::Cell;

// Native CUDA acceleration (direct FFI, supports CUDA 13.1)
#[cfg(feature = "native_cuda")]
pub mod native_cuda;

// Triple-Fate: Multi-layer Fate architecture (from valhalla-rational)
pub mod triple_fate;
pub use triple_fate::{
    TripleFateCombined, FateLoop, FateInFate, BaseFate, MetaFate,
    TripleSignal, TripleFateState, TripleRule, TripleDiscovery,
    SignalSource, DiscoveryType, RuleAction, VoyagerSpaceF64,
    detect_pattern_f64, run_comparison_experiment, ExperimentResult,
    OrganDetector,
    // Fate 存储层
    FateSlice, FateDataEntry,
};

// ============================================================================
// System Types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SystemType {
    BlackHole,
    MeshBrain,
    MultiNova,
    BaseForce,
}

// ============================================================================
// Resource Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u64,
    pub max_cpu_percent: u8,
    pub max_buffer_size: usize,
    pub processing_interval_ms: u64,
    pub gpu_enabled: bool,
    pub batch_size: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096,
            max_cpu_percent: 70,
            max_buffer_size: 100000,
            processing_interval_ms: 10,
            gpu_enabled: true,
            batch_size: 1024,
        }
    }
}

// ============================================================================
// Deformation Signal (f64)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformationSignal {
    pub source_system: SystemType,
    pub intensity: f64,
    pub direction: Vec<f64>,
    pub signal_tick: u64,
    pub quality_score: f64,
}

impl DeformationSignal {
    pub fn new(source: SystemType, intensity: f64, direction: Vec<f64>) -> Self {
        Self {
            source_system: source,
            intensity,
            direction,
            signal_tick: 0,
            quality_score: 1.0, // No preset bias - neutral weight
        }
    }
    
    /// Compute weight = intensity × quality_score
    #[inline]
    pub fn weight(&self) -> f64 {
        self.intensity * self.quality_score
    }
}

// ============================================================================
// Batch Operations (GPU-friendly)
// ============================================================================

/// Batch of signals for GPU processing
#[derive(Debug, Clone)]
pub struct SignalBatch {
    pub intensities: Vec<f64>,
    pub quality_scores: Vec<f64>,
    pub directions: Vec<Vec<f64>>,
    pub system_indices: Vec<usize>,
}

impl SignalBatch {
    pub fn new() -> Self {
        Self {
            intensities: Vec::new(),
            quality_scores: Vec::new(),
            directions: Vec::new(),
            system_indices: Vec::new(),
        }
    }
    
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            intensities: Vec::with_capacity(cap),
            quality_scores: Vec::with_capacity(cap),
            directions: Vec::with_capacity(cap),
            system_indices: Vec::with_capacity(cap),
        }
    }
    
    pub fn push(&mut self, signal: &DeformationSignal) {
        self.intensities.push(signal.intensity);
        self.quality_scores.push(signal.quality_score);
        self.directions.push(signal.direction.clone());
        self.system_indices.push(match signal.source_system {
            SystemType::BlackHole => 0,
            SystemType::MeshBrain => 1,
            SystemType::MultiNova => 2,
            SystemType::BaseForce => 3,
        });
    }
    
    pub fn len(&self) -> usize {
        self.intensities.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.intensities.is_empty()
    }
    
    /// Compute weights in batch (parallelized)
    pub fn compute_weights_parallel(&self) -> Vec<f64> {
        self.intensities.par_iter()
            .zip(self.quality_scores.par_iter())
            .map(|(i, q)| i * q)
            .collect()
    }
    
    /// Compute weighted direction sum (parallelized)
    pub fn compute_weighted_direction_parallel(&self, dim: usize) -> Vec<f64> {
        let weights = self.compute_weights_parallel();
        let total_weight: f64 = weights.par_iter().sum();
        
        if total_weight.abs() < 1e-10 {
            return vec![0.0; dim];
        }
        
        let mut result = vec![0.0; dim];
        for (i, dir) in self.directions.iter().enumerate() {
            let w = weights[i];
            for (j, &d) in dir.iter().enumerate().take(dim) {
                result[j] += d * w;
            }
        }
        
        result.par_iter().map(|x| x / total_weight).collect()
    }
}

impl Default for SignalBatch {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// System Preferences (f64)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPreferences {
    pub affinities: Vec<f64>,
}

impl Default for SystemPreferences {
    fn default() -> Self {
        // 默认均匀分布，避免 NaN
        Self {
            affinities: vec![0.25, 0.25, 0.25, 0.25],
        }
    }
}

/// Persisted Fate convergence state — survives snapshot fork/restore.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateConvergenceState {
    pub preference_history: Vec<[f64; 4]>,
    pub system_weights: [f64; 4],
    pub system_counts: [u64; 4],
    pub signal_tick: u64,
    pub data_flowing: bool,
}

impl SystemPreferences {
    /// Get preferred system index
    pub fn preferred_system(&self) -> usize {
        self.affinities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Check if preferences have diverged (one system dominates)
    pub fn has_diverged(&self, threshold: f64) -> bool {
        let max = self.affinities.iter().cloned().fold(0.0_f64, f64::max);
        let sum: f64 = self.affinities.iter().sum();
        if sum.abs() < 1e-10 { return false; }
        (max / sum) > threshold
    }
}

// ============================================================================
// Fate Evaluation (f64)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateEvaluation {
    pub quality: f64,
    pub intensity: f64,
    pub is_bootstrap: bool,
}

// ============================================================================
// Resource Usage
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub memory_bytes: usize,
    pub buffer_usage: usize,
    pub buffer_capacity: usize,
    pub gpu_memory_mb: u64,
}

// ============================================================================
// Internal Space (f64)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalSpace {
    pub dimensions: usize,
    pub basis_vectors: Vec<Vec<f64>>,
    pub expansion_count: u64,
}

impl InternalSpace {
    pub fn new(dims: usize) -> Self {
        let mut basis = Vec::with_capacity(dims);
        for i in 0..dims {
            let mut vec = vec![0.0; dims];
            vec[i] = 1.0;
            basis.push(vec);
        }
        Self {
            dimensions: dims,
            basis_vectors: basis,
            expansion_count: 0,
        }
    }
}

impl Default for InternalSpace {
    fn default() -> Self {
        Self::new(4)
    }
}

// ============================================================================
// Fate System (f64 GPU-Accelerated)
// ============================================================================

pub struct FateSystem {
    pub entity_id: String,
    pub deformation_buffer: Vec<DeformationSignal>,
    pub resource_limits: ResourceLimits,
    pub is_active: bool,
    pub internal_space: InternalSpace,
    pub signal_tick: u64,

    // Native CUDA accelerator (optional). When enabled, we can offload
    // expensive vector ops (e.g. cosine similarity) inside Fate evaluation.
    #[cfg(feature = "native_cuda")]
    pub gpu_accelerator: Option<native_cuda::NativeCudaAccelerator>,

    // ====== GPU execution observability & resource control ======
    // These fields ensure the GPU path is "real": it is called only when allowed
    // by resource control, and we can see counters in Debug/log output.
    pub gpu_coherence_calls: Cell<u64>,
    pub gpu_coherence_skipped_cooldown: Cell<u64>,
    pub gpu_coherence_failures: Cell<u64>,
    pub last_gpu_coherence_wall_ms: Cell<u128>,
    /// Signal tick of the last actual GPU coherence computation.
    /// GPU coherence is Fate-triggered by new material only: it may
    /// run again once a full new batch of signals has arrived since
    /// this tick, never on repeated preference reads.
    pub last_gpu_pref_tick: Cell<u64>,
    /// Signal tick covered by the newest preference_history snapshot.
    /// Repeated epoch marks without new signals must not push
    /// duplicate snapshots (or re-run preference computation).
    last_snapshot_tick: u64,

    // System-specific accumulators (for fast preference calculation)
    system_weights: [f64; 4],
    system_counts: [u64; 4],
    
    // 偏好历史 (用于代数形式的收敛检测)
    preference_history: Vec<[f64; 4]>,
    
    // 数据流状态：只有数据处理完毕后才能判断收敛
    // 每次 receive_signal 时设为 true，表示有数据在流入
    // 调用 mark_epoch_complete 后设为 false，允许检查收敛
    data_flowing: bool,
}

impl std::fmt::Debug for FateSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("FateSystem");
        s.field("entity_id", &self.entity_id);
        s.field("deformation_buffer_len", &self.deformation_buffer.len());
        s.field("resource_limits", &self.resource_limits);
        s.field("is_active", &self.is_active);
        s.field("internal_space", &self.internal_space);
        s.field("signal_tick", &self.signal_tick);
        s.field("preference_history_len", &self.preference_history.len());
        s.field("data_flowing", &self.data_flowing);
        s.field("gpu_coherence_calls", &self.gpu_coherence_calls.get());
        s.field(
            "gpu_coherence_skipped_cooldown",
            &self.gpu_coherence_skipped_cooldown.get(),
        );
        s.field("gpu_coherence_failures", &self.gpu_coherence_failures.get());
        s.field("last_gpu_coherence_wall_ms", &self.last_gpu_coherence_wall_ms.get());
        #[cfg(feature = "native_cuda")]
        {
            let has_gpu = self.gpu_accelerator.is_some();
            s.field("gpu_accelerator", &has_gpu);
        }
        s.finish()
    }
}

impl Clone for FateSystem {
    fn clone(&self) -> Self {
        Self {
            entity_id: self.entity_id.clone(),
            deformation_buffer: self.deformation_buffer.clone(),
            resource_limits: self.resource_limits.clone(),
            is_active: self.is_active,
            internal_space: self.internal_space.clone(),
            signal_tick: self.signal_tick,
            gpu_coherence_calls: Cell::new(self.gpu_coherence_calls.get()),
            gpu_coherence_skipped_cooldown: Cell::new(self.gpu_coherence_skipped_cooldown.get()),
            gpu_coherence_failures: Cell::new(self.gpu_coherence_failures.get()),
            last_gpu_coherence_wall_ms: Cell::new(self.last_gpu_coherence_wall_ms.get()),
            last_gpu_pref_tick: Cell::new(self.last_gpu_pref_tick.get()),
            last_snapshot_tick: self.last_snapshot_tick,
            #[cfg(feature = "native_cuda")]
            gpu_accelerator: None, // GPU handles are not trivially clonable; clone without CUDA state.
            system_weights: self.system_weights,
            system_counts: self.system_counts,
            preference_history: self.preference_history.clone(),
            data_flowing: self.data_flowing,
        }
    }
}

impl FateSystem {
    fn now_wall_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    pub fn new(entity_id: String, resource_limits: ResourceLimits) -> Self {
        #[cfg(feature = "native_cuda")]
        let gpu_accelerator = if resource_limits.gpu_enabled {
            match native_cuda::NativeCudaAccelerator::new() {
                Ok(acc) => {
                    info!("fate-f64: Native CUDA accelerator enabled");
                    Some(acc)
                }
                Err(e) => {
                    warn!("fate-f64: Native CUDA accelerator init failed: {e}. Falling back to CPU.");
                    None
                }
            }
        } else {
            None
        };

        Self {
            entity_id,
            deformation_buffer: Vec::with_capacity(resource_limits.max_buffer_size),
            resource_limits,
            is_active: true,
            internal_space: InternalSpace::default(),
            signal_tick: 0,
            // 初始均匀分布，避免首次计算偏好时产生 NaN
            system_weights: [1.0; 4],
            system_counts: [0; 4],
            preference_history: Vec::new(),
            data_flowing: true, // 初始假设有数据要处理

            #[cfg(feature = "native_cuda")]
            gpu_accelerator,

            gpu_coherence_calls: Cell::new(0),
            gpu_coherence_skipped_cooldown: Cell::new(0),
            gpu_coherence_failures: Cell::new(0),
            last_gpu_coherence_wall_ms: Cell::new(0),
            last_gpu_pref_tick: Cell::new(0),
            last_snapshot_tick: 0,
        }
    }
    
    /// Receive deformation signal
    pub fn receive_signal(&mut self, mut signal: DeformationSignal) {
        if !self.is_active { return; }
        
        // 有数据流入，标记为 flowing
        self.data_flowing = true;
        
        self.signal_tick += 1;
        signal.signal_tick = self.signal_tick;
        
        // Update system accumulators
        let idx = match signal.source_system {
            SystemType::BlackHole => 0,
            SystemType::MeshBrain => 1,
            SystemType::MultiNova => 2,
            SystemType::BaseForce => 3,
        };
        self.system_weights[idx] += signal.weight();
        self.system_counts[idx] += 1;
        
        // Buffer management
        if self.deformation_buffer.len() >= self.resource_limits.max_buffer_size {
            // Remove oldest and update accumulators
            let old = self.deformation_buffer.remove(0);
            let old_idx = match old.source_system {
                SystemType::BlackHole => 0,
                SystemType::MeshBrain => 1,
                SystemType::MultiNova => 2,
                SystemType::BaseForce => 3,
            };
            self.system_weights[old_idx] -= old.weight();
            self.system_counts[old_idx] = self.system_counts[old_idx].saturating_sub(1);
        }
        
        self.deformation_buffer.push(signal);
    }

    /// Restore Fate accumulators from a persisted hub snapshot (cross-round session).
    pub fn seed_from_hub_snapshot(
        &mut self,
        prefs: &[f64],
        routed: [u64; 4],
        logical_time: Option<u64>,
    ) {
        let total_routed: u64 = routed.iter().sum();
        for i in 0..4 {
            let p = prefs.get(i).copied().unwrap_or(0.25).max(0.01);
            let cnt = routed[i].max(1) as f64;
            self.system_weights[i] = p * cnt;
            self.system_counts[i] = routed[i].max(1);
        }
        self.signal_tick = logical_time
            .filter(|&t| t > 0)
            .unwrap_or_else(|| total_routed.max(1));
        self.data_flowing = true;
    }

    /// Export full convergence state for manifold snapshot persistence.
    pub fn export_convergence_state(&self) -> FateConvergenceState {
        FateConvergenceState {
            preference_history: self.preference_history.clone(),
            system_weights: self.system_weights,
            system_counts: self.system_counts,
            signal_tick: self.signal_tick,
            data_flowing: self.data_flowing,
        }
    }

    /// Restore convergence state from manifold snapshot (preserves marginal history).
    pub fn import_convergence_state(&mut self, state: &FateConvergenceState) {
        self.preference_history = state.preference_history.clone();
        self.system_weights = state.system_weights;
        self.system_counts = state.system_counts;
        self.signal_tick = state.signal_tick;
        self.data_flowing = state.data_flowing;
    }
    
    /// Calculate system preferences (O(1) with accumulators)
    pub fn calculate_system_preferences(&self) -> SystemPreferences {
        // Fallback: current O(1) policy based on system accumulators only.
        let total: f64 = self.system_weights.iter().sum();
        if total.abs() < 1e-10 {
            return SystemPreferences::default();
        }

        // GPU path is optional and must be "real": only attempt when
        // (1) native CUDA accelerator exists, (2) GPU is enabled, and (3) we have enough samples.
        #[cfg(feature = "native_cuda")]
        {
            let fallback_affinities: Vec<f64> = self
                .system_weights
                .iter()
                .map(|w| w / total)
                .collect();

            if self.resource_limits.gpu_enabled
                && self.gpu_accelerator.is_some()
                && self.deformation_buffer.len() >= self.resource_limits.batch_size
            {
                // Fate-triggered, never repeated: GPU coherence may
                // run only once a full new batch of signals has
                // arrived since the last actual computation. Repeated
                // preference reads use the cheap accumulators.
                let batch = self.resource_limits.batch_size.max(1) as u64;
                let last_gpu = self.last_gpu_pref_tick.get();
                if last_gpu > 0 && self.signal_tick < last_gpu.saturating_add(batch) {
                    return SystemPreferences {
                        affinities: fallback_affinities,
                    };
                }
                // Degenerate input (zero direction vectors) carries
                // no coherence information; never spend GPU on it.
                let start0 = self
                    .deformation_buffer
                    .len()
                    .saturating_sub((self.resource_limits.batch_size * 4).max(1));
                if self.deformation_buffer[start0..]
                    .iter()
                    .all(|s| s.direction.iter().all(|&x| x == 0.0))
                {
                    return SystemPreferences {
                        affinities: fallback_affinities,
                    };
                }
                let now_ms = Self::now_wall_ms();
                let cooldown_ms = self.resource_limits.processing_interval_ms as u128;
                if self.last_gpu_coherence_wall_ms.get() > 0
                    && cooldown_ms > 0
                    && (now_ms.saturating_sub(self.last_gpu_coherence_wall_ms.get()) < cooldown_ms)
                {
                    // Too frequent -> respect resource control and stay on fallback.
                    self.gpu_coherence_skipped_cooldown.set(
                        self.gpu_coherence_skipped_cooldown.get() + 1,
                    );
                    return SystemPreferences {
                        affinities: fallback_affinities,
                    };
                }

                // GPU path is allowed now: mark last-call time and increment call counter.
                self.last_gpu_coherence_wall_ms.set(now_ms);
                self.gpu_coherence_calls.set(self.gpu_coherence_calls.get() + 1);

            // Safety: Native CUDA cosine similarity assumes fixed-length vectors equal to query.len().
            let dim = self.internal_space.dimensions.max(1);
            let alpha = 0.8; // keep fallback dominance when coherence is weak/degenerate

            // Limit work to a recent window to respect resource control.
            let max_samples = (self.resource_limits.batch_size * 4).max(1);
            let start = self.deformation_buffer.len().saturating_sub(max_samples);
            let slice = &self.deformation_buffer[start..];

            let mut vectors_by_sys: [Vec<Vec<f64>>; 4] = std::array::from_fn(|_| Vec::new());
            let mut weights_by_sys: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());

            for sig in slice {
                let idx = match sig.source_system {
                    SystemType::BlackHole => 0,
                    SystemType::MeshBrain => 1,
                    SystemType::MultiNova => 2,
                    SystemType::BaseForce => 3,
                };

                // Fixed-length vector (pad with zeros / truncate).
                let mut v = Vec::with_capacity(dim);
                for &x in sig.direction.iter().take(dim) {
                    v.push(x);
                }
                while v.len() < dim {
                    v.push(0.0);
                }

                vectors_by_sys[idx].push(v);
                weights_by_sys[idx].push(sig.weight());
            }

            let mut scores = [0.0f64; 4];
            for sys_idx in 0..4 {
                let vectors = &vectors_by_sys[sys_idx];
                let weights = &weights_by_sys[sys_idx];
                if vectors.is_empty() {
                    // Keep a small baseline so normalization doesn't collapse.
                    scores[sys_idx] = self.system_weights[sys_idx] * alpha;
                    continue;
                }

                let basis = self
                    .internal_space
                    .basis_vectors
                    .get(sys_idx)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; dim]);
                let mut basis_fixed = Vec::with_capacity(dim);
                for &x in basis.iter().take(dim) {
                    basis_fixed.push(x);
                }
                while basis_fixed.len() < dim {
                    basis_fixed.push(0.0);
                }

                // Compute weighted coherence via cosine similarity.
                let mut weighted_sim_sum = 0.0f64;
                let mut total_weight = 0.0f64;

                let b = self.resource_limits.batch_size.max(1);
                for (v_chunk, w_chunk) in vectors
                    .chunks(b)
                    .zip(weights.chunks(b))
                {
                    let sims_res = self
                        .gpu_accelerator
                        .as_ref()
                        .unwrap()
                        .batch_cosine_similarity(v_chunk, &basis_fixed);

                    let sims = match sims_res {
                        Ok(s) => s,
                        Err(e) => {
                            self.gpu_coherence_failures.set(self.gpu_coherence_failures.get() + 1);
                            warn!("fate-f64: GPU coherence compute failed: {e}. Falling back to accumulators.");
                            return SystemPreferences {
                                affinities: fallback_affinities,
                            };
                        }
                    };

                    for (j, &w) in w_chunk.iter().enumerate() {
                        if j >= sims.len() {
                            break;
                        }
                        weighted_sim_sum += sims[j] * w;
                        total_weight += w;
                    }
                }

                let coherence = if total_weight.abs() > 1e-12 {
                    weighted_sim_sum / total_weight
                } else {
                    0.0
                };

                // Map coherence [-1, 1] -> [0, 1], then blend with accumulator score.
                let coherence01 = ((coherence + 1.0) / 2.0).clamp(0.0, 1.0);
                let score = self.system_weights[sys_idx] * (alpha + (1.0 - alpha) * coherence01);
                scores[sys_idx] = score;
            }

            let total_score: f64 = scores.iter().sum();
            if total_score.abs() < 1e-10 {
                    // GPU computed but coherence degenerated -> fallback.
                return SystemPreferences {
                    affinities: fallback_affinities,
                };
            }

            info!(
                "fate-f64: GPU coherence used (samples={}, dim={}, cooldown_ms={})",
                slice.len(),
                dim
                ,
                self.resource_limits.processing_interval_ms
            );
            self.last_gpu_pref_tick.set(self.signal_tick);

            return SystemPreferences {
                affinities: scores.iter().map(|s| s / total_score).collect(),
            };
            }
        }
        
        SystemPreferences {
            affinities: self
                .system_weights
                .iter()
                .map(|w| w / total)
                .collect(),
        }
    }
    
    /// 记录当前偏好到历史 (用于收敛检测)
    pub fn record_preference_snapshot(&mut self) {
        // No new signals since the last snapshot: an epoch re-mark is
        // not new Fate evidence. Skip both the duplicate history entry
        // and the preference recomputation.
        if self.signal_tick == self.last_snapshot_tick && !self.preference_history.is_empty() {
            return;
        }
        self.last_snapshot_tick = self.signal_tick;
        let prefs = self.calculate_system_preferences();
        let snapshot: [f64; 4] = [
            prefs.affinities.get(0).copied().unwrap_or(0.0),
            prefs.affinities.get(1).copied().unwrap_or(0.0),
            prefs.affinities.get(2).copied().unwrap_or(0.0),
            prefs.affinities.get(3).copied().unwrap_or(0.0),
        ];
        self.preference_history.push(snapshot);
        
        // 保留最近 100 个快照
        if self.preference_history.len() > 100 {
            self.preference_history.remove(0);
        }
    }
    
    /// 标记一个 epoch 完成，此时才允许检查收敛
    /// 只有数据处理完毕后，才能判断是否收敛
    pub fn mark_epoch_complete(&mut self) {
        self.data_flowing = false;
        self.record_preference_snapshot();
    }
    
    /// 检查数据是否还在流入
    pub fn is_data_flowing(&self) -> bool {
        self.data_flowing
    }
    
    /// 代数形式的收敛检测 (Fate 自主判断)
    /// 前提条件：数据必须处理完毕 (data_flowing == false)
    /// 基于边际效应递减: 需要连续多次边际递减
    /// 纯代数比较，但需要足够的历史才能判断
    pub fn has_converged(&self) -> bool {
        // 如果数据还在流入，不能收敛
        // 必须等数据处理完毕才能判断
        if self.data_flowing {
            return false;
        }
        
        let len = self.preference_history.len();
        
        // 需要至少 5 个快照才能判断收敛趋势
        // (检查连续 3 次边际递减)
        if len < 5 {
            return false;
        }
        
        let h = &self.preference_history;
        
        // 计算最近 4 次变化
        let diffs: Vec<f64> = (1..5)
            .map(|k| {
                let idx = len - k;
                (0..4)
                    .map(|i| (h[idx][i] - h[idx-1][i]).abs())
                    .sum()
            })
            .collect();
        
        // diffs[0] = 最新变化, diffs[1] = 上一次变化, ...
        // 代数判断: 连续边际递减 (diff[n] < diff[n-1] < diff[n-2])
        // 且最新变化必须非常小 (趋近于稳定)
        let decreasing = diffs[0] < diffs[1] && diffs[1] < diffs[2];
        let stable = diffs[0] < diffs[3] * 0.5; // 最新变化小于初始变化的一半
        
        decreasing && stable && diffs[2] > 0.0
    }
    
    /// 检测当前模式 (代数形式)
    pub fn detect_convergence_pattern(&self) -> Option<&'static str> {
        let len = self.preference_history.len();
        if len < 3 {
            return None;
        }
        
        let h = &self.preference_history;
        
        let diff1: f64 = (0..4)
            .map(|i| (h[len-2][i] - h[len-3][i]).abs())
            .sum();
        
        let diff2: f64 = (0..4)
            .map(|i| (h[len-1][i] - h[len-2][i]).abs())
            .sum();
        
        // 不动点
        if diff1 == 0.0 && diff2 == 0.0 {
            return Some("fixed_point");
        }
        
        // 收敛: 边际递减
        if diff2 < diff1 {
            return Some("converging");
        }
        
        // 发散: 边际增加
        if diff2 > diff1 {
            return Some("diverging");
        }
        
        None
    }
    
    /// Batch process signals (GPU-friendly)
    pub fn create_batch(&self, max_size: usize) -> SignalBatch {
        let size = self.deformation_buffer.len().min(max_size);
        let mut batch = SignalBatch::with_capacity(size);
        
        for signal in self.deformation_buffer.iter().rev().take(size) {
            batch.push(signal);
        }
        
        batch
    }
    
    /// Evaluate system (for Fate autonomous control)
    pub fn evaluate_system(&self, system_index: usize) -> FateEvaluation {
        if self.signal_tick == 0 {
            return FateEvaluation {
                quality: 0.0,
                intensity: 0.0,
                is_bootstrap: true,
            };
        }
        
        let count = self.system_counts[system_index];
        if count == 0 {
            return FateEvaluation {
                quality: 0.0,
                intensity: 0.0,
                is_bootstrap: false,
            };
        }
        
        let avg_weight = self.system_weights[system_index] / count as f64;
        
        // Quality derived from accumulated weight performance
        let quality = (avg_weight * 10.0).min(1.0).max(0.0);
        
        // Intensity derived from signal density
        let density = count as f64 / self.signal_tick as f64;
        let intensity = density.min(1.0);

        // GPU path: compute direction coherence for this system and blend into quality.
        // This makes rules input (quality/intensity) observable and GPU-backed when enabled.
        #[cfg(feature = "native_cuda")]
        {
            if self.resource_limits.gpu_enabled
                && self.gpu_accelerator.is_some()
                && self.deformation_buffer.len() >= self.resource_limits.batch_size
            {
                // Same Fate-triggered batch gate as preference reads:
                // no new full batch since the last GPU computation,
                // or only degenerate zero-direction material -> CPU.
                let batch = self.resource_limits.batch_size.max(1) as u64;
                let last_gpu = self.last_gpu_pref_tick.get();
                let start0 = self
                    .deformation_buffer
                    .len()
                    .saturating_sub((self.resource_limits.batch_size * 4).max(1));
                if (last_gpu > 0 && self.signal_tick < last_gpu.saturating_add(batch))
                    || self.deformation_buffer[start0..]
                        .iter()
                        .all(|s| s.direction.iter().all(|&x| x == 0.0))
                {
                    return FateEvaluation {
                        quality,
                        intensity,
                        is_bootstrap: false,
                    };
                }
                let now_ms = Self::now_wall_ms();
                let cooldown_ms = self.resource_limits.processing_interval_ms as u128;
                if self.last_gpu_coherence_wall_ms.get() > 0
                    && cooldown_ms > 0
                    && (now_ms.saturating_sub(self.last_gpu_coherence_wall_ms.get()) < cooldown_ms)
                {
                    self.gpu_coherence_skipped_cooldown
                        .set(self.gpu_coherence_skipped_cooldown.get() + 1);
                    return FateEvaluation {
                        quality,
                        intensity,
                        is_bootstrap: false,
                    };
                }

                self.last_gpu_coherence_wall_ms.set(now_ms);
                self.gpu_coherence_calls.set(self.gpu_coherence_calls.get() + 1);
                self.last_gpu_pref_tick.set(self.signal_tick);

                let max_samples = (self.resource_limits.batch_size * 4).max(1);
                let start = self.deformation_buffer.len().saturating_sub(max_samples);
                let slice = &self.deformation_buffer[start..];

                let dim = self.internal_space.dimensions.max(1);
                let basis_vec = self
                    .internal_space
                    .basis_vectors
                    .get(system_index)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; dim]);
                let mut basis_fixed = Vec::with_capacity(dim);
                for &x in basis_vec.iter().take(dim) {
                    basis_fixed.push(x);
                }
                while basis_fixed.len() < dim {
                    basis_fixed.push(0.0);
                }

                let mut vectors: Vec<Vec<f64>> = Vec::new();
                let mut weights: Vec<f64> = Vec::new();
                for sig in slice {
                    let idx = match sig.source_system {
                        SystemType::BlackHole => 0,
                        SystemType::MeshBrain => 1,
                        SystemType::MultiNova => 2,
                        SystemType::BaseForce => 3,
                    };
                    if idx != system_index {
                        continue;
                    }

                    let mut v = Vec::with_capacity(dim);
                    for &x in sig.direction.iter().take(dim) {
                        v.push(x);
                    }
                    while v.len() < dim {
                        v.push(0.0);
                    }
                    vectors.push(v);
                    weights.push(sig.weight());
                }

                if !vectors.is_empty() {
                    let b = self.resource_limits.batch_size.max(1);
                    let mut weighted_sim_sum = 0.0f64;
                    let mut total_weight = 0.0f64;

                    for (v_chunk, w_chunk) in vectors.chunks(b).zip(weights.chunks(b)) {
                        let sims_res = self
                            .gpu_accelerator
                            .as_ref()
                            .unwrap()
                            .batch_cosine_similarity(v_chunk, &basis_fixed);
                        let sims = match sims_res {
                            Ok(s) => s,
                            Err(e) => {
                                self.gpu_coherence_failures
                                    .set(self.gpu_coherence_failures.get() + 1);
                                warn!(
                                    "fate-f64: GPU coherence compute failed in evaluate_system: {e}. Falling back."
                                );
                                break;
                            }
                        };

                        for (j, &w) in w_chunk.iter().enumerate() {
                            if j >= sims.len() {
                                break;
                            }
                            weighted_sim_sum += sims[j] * w;
                            total_weight += w;
                        }
                    }

                    if total_weight.abs() > 1e-12 {
                        let coherence = weighted_sim_sum / total_weight;
                        let coherence01 = ((coherence + 1.0) / 2.0).clamp(0.0, 1.0);
                        let alpha = 0.5;
                        let blended_quality = alpha * quality + (1.0 - alpha) * coherence01;

                        info!(
                            "fate-f64: GPU evaluate_system used (sys={}, samples={}, dim={})",
                            system_index,
                            vectors.len(),
                            dim
                        );

                        return FateEvaluation {
                            quality: blended_quality,
                            intensity,
                            is_bootstrap: false,
                        };
                    }
                }
            }
        }

        FateEvaluation {
            quality,
            intensity,
            is_bootstrap: false,
        }
    }
    
    /// Get resource usage
    pub fn get_resource_usage(&self) -> ResourceUsage {
        #[cfg(feature = "native_cuda")]
        let gpu_memory_mb = if self.gpu_accelerator.is_some() {
            // NativeCuda cosine similarity allocates query + a single vector buffers on GPU.
            // Estimate: 2 * dim * sizeof(f64).
            let dim = self.internal_space.dimensions.max(1) as u64;
            let bytes = 2 * dim * 8;
            bytes / 1_000_000
        } else {
            0
        };

        #[cfg(not(feature = "native_cuda"))]
        let gpu_memory_mb = 0;

        ResourceUsage {
            memory_bytes: self.deformation_buffer.len() * std::mem::size_of::<DeformationSignal>(),
            buffer_usage: self.deformation_buffer.len(),
            buffer_capacity: self.resource_limits.max_buffer_size,
            gpu_memory_mb,
        }
    }
    
    /// Get logical time
    pub fn get_logical_time(&self) -> u64 {
        self.signal_tick
    }
    
    /// Cleanup old signals
    pub fn cleanup_expired_signals(&mut self, max_age_ticks: u64) {
        let current = self.signal_tick;
        self.deformation_buffer.retain(|signal| {
            current.saturating_sub(signal.signal_tick) < max_age_ticks
        });
        
        // Recalculate accumulators after cleanup
        self.recalculate_accumulators();
    }
    
    fn recalculate_accumulators(&mut self) {
        self.system_weights = [0.0; 4];
        self.system_counts = [0; 4];
        
        for signal in &self.deformation_buffer {
            let idx = match signal.source_system {
                SystemType::BlackHole => 0,
                SystemType::MeshBrain => 1,
                SystemType::MultiNova => 2,
                SystemType::BaseForce => 3,
            };
            self.system_weights[idx] += signal.weight();
            self.system_counts[idx] += 1;
        }
    }
}

// ============================================================================
// GPU Acceleration (when feature enabled)
// ============================================================================

#[cfg(feature = "cuda")]
pub mod gpu {
    use candle_core::{Device, Tensor, DType};
    
    pub struct GpuAccelerator {
        device: Device,
    }
    
    impl GpuAccelerator {
        pub fn new() -> Result<Self, candle_core::Error> {
            let device = Device::cuda_if_available(0)?;
            Ok(Self { device })
        }
        
        pub fn is_cuda_available(&self) -> bool {
            matches!(self.device, Device::Cuda(_))
        }
        
        /// Batch matrix multiplication on GPU
        pub fn batch_matmul(&self, a: &[f64], b: &[f64], m: usize, n: usize, k: usize) 
            -> Result<Vec<f64>, candle_core::Error> 
        {
            let a_tensor = Tensor::from_slice(a, (m, k), &self.device)?.to_dtype(DType::F64)?;
            let b_tensor = Tensor::from_slice(b, (k, n), &self.device)?.to_dtype(DType::F64)?;
            let result = a_tensor.matmul(&b_tensor)?;
            result.to_vec2::<f64>().map(|v| v.into_iter().flatten().collect())
        }
        
        /// Batch cosine similarity on GPU
        pub fn batch_cosine_similarity(&self, vectors: &[Vec<f64>], query: &[f64]) 
            -> Result<Vec<f64>, candle_core::Error>
        {
            if vectors.is_empty() {
                return Ok(vec![]);
            }
            
            let dim = query.len();
            let n = vectors.len();
            
            // Flatten vectors
            let flat: Vec<f64> = vectors.iter().flatten().cloned().collect();
            
            let v_tensor = Tensor::from_slice(&flat, (n, dim), &self.device)?.to_dtype(DType::F64)?;
            let q_tensor = Tensor::from_slice(query, (dim, 1), &self.device)?.to_dtype(DType::F64)?;
            
            // Dot products
            let dots = v_tensor.matmul(&q_tensor)?;
            
            // Magnitudes (simplified - proper impl would normalize)
            let dots_vec: Vec<f64> = dots.to_vec2()?.into_iter().flatten().collect();
            
            Ok(dots_vec)
        }
    }
}

// ============================================================================
// CPU Parallel Operations (fallback)
// ============================================================================

pub mod cpu {
    use rayon::prelude::*;
    
    /// Batch cosine similarity (CPU parallel)
    pub fn batch_cosine_similarity(vectors: &[Vec<f64>], query: &[f64]) -> Vec<f64> {
        vectors.par_iter()
            .map(|v| cosine_similarity(v, query))
            .collect()
    }
    
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }
        
        dot / (mag_a * mag_b)
    }
    
    /// Batch weighted sum (CPU parallel)
    pub fn batch_weighted_sum(vectors: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
        if vectors.is_empty() {
            return vec![];
        }
        
        let dim = vectors[0].len();
        let total_weight: f64 = weights.iter().sum();
        
        if total_weight.abs() < 1e-10 {
            return vec![0.0; dim];
        }
        
        let mut result = vec![0.0; dim];
        for (v, &w) in vectors.iter().zip(weights.iter()) {
            for (i, &x) in v.iter().enumerate() {
                result[i] += x * w;
            }
        }
        
        result.par_iter().map(|x| x / total_weight).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fate_system_f64() {
        let limits = ResourceLimits::default();
        let mut fate = FateSystem::new("test".to_string(), limits);
        
        assert_eq!(fate.get_logical_time(), 0);
        
        let signal = DeformationSignal {
            source_system: SystemType::BlackHole,
            intensity: 0.5,
            direction: vec![1.0, 0.0, 0.0, 0.0],
            signal_tick: 0,
            quality_score: 0.8,
        };
        
        fate.receive_signal(signal);
        assert_eq!(fate.get_logical_time(), 1);
    }
    
    #[test]
    fn test_batch_operations() {
        let mut batch = SignalBatch::new();
        
        for i in 0..100 {
            let signal = DeformationSignal {
                source_system: SystemType::BlackHole,
                intensity: 0.5 + (i as f64 * 0.01),
                direction: vec![1.0, 0.0, 0.0, 0.0],
                signal_tick: i,
                quality_score: 0.8,
            };
            batch.push(&signal);
        }
        
        let weights = batch.compute_weights_parallel();
        assert_eq!(weights.len(), 100);
    }
    
    #[test]
    fn test_cpu_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((cpu::cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
        assert!((cpu::cosine_similarity(&a, &c) - 0.0).abs() < 1e-10);
    }
}
