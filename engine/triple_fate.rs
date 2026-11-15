//! Triple-Fate (f64 Version): Three Architectures for Self-Evolving Meta-Systems
//!
//! This module implements three approaches to create self-evolving systems:
//! 1. Fate-in-Fate (Nested/Hierarchical) - Inner Fate observes Outer Fate
//! 2. Fate Loop (Self-Referential/Cyclic) - Output feeds back as input
//! 3. Combined (Nested + Cyclic) - Both approaches together [WINNER]
//!
//! Key Insight: Nesting + Looping = True Self-Evolution
//! - Nesting provides hierarchical control and observation
//! - Looping provides self-reference and emergence
//! - Together they enable creation of genuinely NEW meta-systems
//!
//! f64 Version for GPU acceleration and high-performance computing

use serde::{Deserialize, Serialize};

// ============================================================================
// Core Types
// ============================================================================

/// Signal that Triple-Fate processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleSignal {
    pub values: Vec<f64>,
    pub tick: u64,
    pub source: SignalSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SignalSource {
    External,           // From outside
    SelfOutput,         // From own output (loop)
    InnerFate,          // From nested Fate
    OuterFate,          // Observation of outer Fate
}

/// Fate's internal state - can be observed by meta-Fate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleFateState {
    pub signal_count: u64,
    pub discovery_count: u64,
    pub efficiency: f64,
    pub dimension_changes: u64,
    pub last_discovery_tick: u64,
    pub pattern_diversity: u64,
}

impl TripleFateState {
    pub fn new() -> Self {
        TripleFateState {
            signal_count: 0,
            discovery_count: 0,
            efficiency: 0.0,
            dimension_changes: 0,
            last_discovery_tick: 0,
            pattern_diversity: 0,
        }
    }

    pub fn update_efficiency(&mut self) {
        if self.signal_count > 0 {
            self.efficiency = self.discovery_count as f64 / self.signal_count as f64;
        }
    }
}

impl Default for TripleFateState {
    fn default() -> Self {
        Self::new()
    }
}

/// Rule that Fate follows - can be modified by meta-Fate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleRule {
    pub id: String,
    pub priority: f64,
    pub condition: String,
    pub action: RuleAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    CreateContainer { dimension_delta: f64 },
    AdjustDimension { factor: f64 },
    ModifyPattern { pattern_id: String },
    SpawnNewRule { rule: Box<TripleRule> },
    TerminateSelf,
}

/// Discovery made by any Fate level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleDiscovery {
    pub id: String,
    pub level: usize,               // 0 = base Fate, 1 = meta-Fate, 2 = meta-meta-Fate...
    pub discovery_type: DiscoveryType,
    pub description: String,
    pub created_at_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscoveryType {
    MathPattern,        // Traditional: found a math pattern
    SystemPattern,      // Meta: found a pattern in system behavior
    NewRule,            // Meta: created a new rule
    NewFate,            // Ultimate: created a new Fate instance
}

// ============================================================================
// VoyagerSpace (f64 simplified version)
// ============================================================================

/// Simplified VoyagerSpace for f64 version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoyagerSpaceF64 {
    pub current_dimension: f64,
    pub discovery_count: u64,
    pub containers: Vec<String>,
}

impl VoyagerSpaceF64 {
    pub fn new() -> Self {
        Self {
            current_dimension: 4.0,
            discovery_count: 0,
            containers: Vec::new(),
        }
    }

    pub fn on_discovery(&mut self, _values: &[f64], pattern: &str) {
        self.discovery_count += 1;
        // Adaptive dimension expansion
        if pattern.contains("diverging") {
            self.current_dimension *= 1.1;
        } else if pattern.contains("converging") {
            // Keep dimension stable
        }
        self.containers.push(format!("container_{}_{}", pattern, self.discovery_count));
    }

    /// Adaptive dimension expansion — Fate estimates the space's degrees
    /// of freedom from the signal's dispersion (only increases).
    ///
    /// 2026-11-02 维度语义修复（E1W 实测定位）：旧版把 |values| 均值
    /// 直接写成维度（complexity * 1.1）。调用方喂的是原始量级信号——
    /// 周期信号首项就是细胞数（~10^3–10^4）、出生信号含字节方差
    /// （~10^3）——维度被种群规模顶到数千（实测 stem 场维度 2680）。
    /// 出生位置 pos[d]=sin((mean+var)·d)·0.5 在数千维下使"距离近"
    /// 只剩"统计量几乎完全相同"，几何退化为查重图：宽度身体里测试
    /// 细胞互边仅 4/1600、真题 5 步可达 0/160。维度是自由度计数，
    /// 不是信号幅值；本仓另有两处同名机制都是从数据离散度估计并
    /// 钳位（utils/space 的 FractionalDimensionCalculator 上限 11、
    /// valhalla-space-f64-gpu 的 FractalDimensionCalculatorGPU 钳
    /// [1,11]、同为 hausdorff/box/information 三估计加权）。fate-f64
    /// 不能反向依赖 space（space 已依赖 fate 存储），故在此复刻同
    /// 一估计式使三处语义一致，不新造公式。
    pub fn adaptive_expand(&mut self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return self.current_dimension;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        // 以下三估计与加权逐式复刻 FractalDimensionCalculatorGPU::
        // calculate_single（hausdorff / box / information，钳 [1,11]）。
        let hausdorff = (1.0 + variance.sqrt().min(10.0)).clamp(1.0, 11.0);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in values {
            if *v < lo { lo = *v; }
            if *v > hi { hi = *v; }
        }
        let range = hi - lo;
        let box_dim = if values.len() < 2 || range < 1e-10 {
            1.0
        } else {
            (1.0 + (n / (range + 1.0)) / 10.0).clamp(1.0, 11.0)
        };
        // 原式按输入顺序逐邻比较计 distinct（顺序敏感的熵代理，照抄）。
        let mut distinct = 1usize;
        for i in 1..values.len() {
            if (values[i] - values[i - 1]).abs() > 1e-10 { distinct += 1; }
        }
        let info_dim = 1.0 + distinct as f64 / n;
        let recommended = ((hausdorff + 2.0 * box_dim + info_dim) / 4.0).clamp(1.0, 11.0);
        if recommended > self.current_dimension {
            self.current_dimension = recommended;
        }
        self.current_dimension
    }
}

impl Default for VoyagerSpaceF64 {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// VERSION 1: Fate-in-Fate (Nested/Hierarchical)
// ============================================================================

/// Inner Fate observes and controls Outer Fate
#[derive(Debug, Serialize, Deserialize)]
pub struct FateInFate {
    pub outer: BaseFate,
    pub inner: MetaFate,
    pub meta_discoveries: Vec<TripleDiscovery>,
    pub global_tick: u64,
}

/// Base-level Fate that processes signals
#[derive(Debug, Serialize, Deserialize)]
pub struct BaseFate {
    pub state: TripleFateState,
    pub rules: Vec<TripleRule>,
    pub space: VoyagerSpaceF64,
    pub discoveries: Vec<TripleDiscovery>,
    pub signal_buffer: Vec<TripleSignal>,
}

/// Meta-level Fate that observes BaseFate
#[derive(Debug, Serialize, Deserialize)]
pub struct MetaFate {
    pub state: TripleFateState,
    pub observation_interval: u64,
    pub rules_created: Vec<TripleRule>,
    pub interventions: u64,
}

impl FateInFate {
    pub fn new() -> Self {
        FateInFate {
            outer: BaseFate::new(),
            inner: MetaFate::new(),
            meta_discoveries: Vec::new(),
            global_tick: 0,
        }
    }

    pub fn tick(&mut self, external_signal: Option<Vec<f64>>) -> Vec<TripleDiscovery> {
        self.global_tick += 1;
        let mut new_discoveries = Vec::new();

        // Process external signal
        if let Some(values) = external_signal {
            let signal = TripleSignal {
                values,
                tick: self.global_tick,
                source: SignalSource::External,
            };
            if let Some(discovery) = self.outer.process_signal(signal) {
                new_discoveries.push(discovery);
            }
        }

        // Inner Fate observes Outer Fate
        if self.global_tick % self.inner.observation_interval == 0 {
            let outer_state = self.outer.state.clone();
            
            if let Some(meta_discovery) = self.inner.observe_and_analyze(&outer_state, self.global_tick) {
                if let DiscoveryType::NewRule = meta_discovery.discovery_type {
                    if let Some(new_rule) = self.inner.rules_created.last() {
                        self.outer.rules.push(new_rule.clone());
                        self.inner.interventions += 1;
                    }
                }
                new_discoveries.push(meta_discovery.clone());
                self.meta_discoveries.push(meta_discovery);
            }
        }

        new_discoveries
    }

    pub fn stats(&self) -> String {
        format!(
            "FateInFate Stats:\n\
             - Global tick: {}\n\
             - Outer discoveries: {}\n\
             - Inner observations: {}\n\
             - Rules created by Inner: {}\n\
             - Interventions: {}\n\
             - Meta discoveries: {}",
            self.global_tick,
            self.outer.discoveries.len(),
            self.inner.state.signal_count,
            self.inner.rules_created.len(),
            self.inner.interventions,
            self.meta_discoveries.len()
        )
    }
}

impl Default for FateInFate {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseFate {
    pub fn new() -> Self {
        BaseFate {
            state: TripleFateState::new(),
            rules: vec![
                TripleRule {
                    id: "default_discovery".into(),
                    priority: 1.0,
                    condition: "unknown_pattern".into(),
                    action: RuleAction::CreateContainer { 
                        dimension_delta: 0.25 
                    },
                },
            ],
            space: VoyagerSpaceF64::new(),
            discoveries: Vec::new(),
            signal_buffer: Vec::new(),
        }
    }

    pub fn process_signal(&mut self, signal: TripleSignal) -> Option<TripleDiscovery> {
        self.state.signal_count += 1;
        self.signal_buffer.push(signal.clone());
        
        // 限制缓冲大小，防止内存无限增长
        if self.signal_buffer.len() > 1000 {
            self.signal_buffer.drain(0..500);
        }
        if self.discoveries.len() > 1000 {
            self.discoveries.drain(0..500);
        }

        if let Some(pattern) = detect_pattern_f64(&signal.values) {
            self.state.discovery_count += 1;
            self.state.last_discovery_tick = signal.tick;
            self.state.pattern_diversity += 1;
            self.state.update_efficiency();

            self.space.on_discovery(&signal.values, &pattern);
            self.state.dimension_changes += 1;

            let discovery = TripleDiscovery {
                id: format!("base_{}_{}", pattern, signal.tick),
                level: 0,
                discovery_type: DiscoveryType::MathPattern,
                description: format!("Found pattern: {}", pattern),
                created_at_tick: signal.tick,
            };
            self.discoveries.push(discovery.clone());
            return Some(discovery);
        }

        None
    }
}

impl Default for BaseFate {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaFate {
    pub fn new() -> Self {
        MetaFate {
            state: TripleFateState::new(),
            observation_interval: 10,
            rules_created: Vec::new(),
            interventions: 0,
        }
    }

    pub fn observe_and_analyze(&mut self, outer_state: &TripleFateState, tick: u64) -> Option<TripleDiscovery> {
        self.state.signal_count += 1;

        // Valhalla 高稳定性阈值
        let efficiency_threshold = 0.05;  // 5%
        
        // Low efficiency -> create exploration rule
        if outer_state.efficiency < efficiency_threshold && outer_state.signal_count > 100 {
            let new_rule = TripleRule {
                id: format!("meta_explore_{}", tick),
                priority: 2.0,
                condition: "any".into(),
                action: RuleAction::AdjustDimension { 
                    factor: 1.5
                },
            };
            self.rules_created.push(new_rule);
            self.state.discovery_count += 1;

            return Some(TripleDiscovery {
                id: format!("meta_rule_{}", tick),
                level: 1,
                discovery_type: DiscoveryType::NewRule,
                description: "Created exploration rule due to low efficiency".into(),
                created_at_tick: tick,
            });
        }

        // Stagnation (Valhalla 高容忍度)
        let stagnation_threshold = 500;
        if tick > outer_state.last_discovery_tick + stagnation_threshold {
            let new_rule = TripleRule {
                id: format!("meta_diversify_{}", tick),
                priority: 3.0,
                condition: "stagnation".into(),
                action: RuleAction::CreateContainer { 
                    dimension_delta: -0.5
                },
            };
            self.rules_created.push(new_rule);
            self.state.discovery_count += 1;

            return Some(TripleDiscovery {
                id: format!("meta_diversify_{}", tick),
                level: 1,
                discovery_type: DiscoveryType::NewRule,
                description: "Created diversification rule due to stagnation".into(),
                created_at_tick: tick,
            });
        }

        // High efficiency + diversity -> spawn capability
        if outer_state.efficiency > 0.95 && outer_state.pattern_diversity > 50 {
            self.state.discovery_count += 1;
            return Some(TripleDiscovery {
                id: format!("meta_spawn_ready_{}", tick),
                level: 1,
                discovery_type: DiscoveryType::SystemPattern,
                description: "System ready to spawn new Fate instance".into(),
                created_at_tick: tick,
            });
        }

        None
    }
}

impl Default for MetaFate {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// VERSION 2: Fate Loop (Self-Referential/Cyclic)
// ============================================================================

/// Self-referential Fate: output feeds back as input
#[derive(Debug, Serialize, Deserialize)]
pub struct FateLoop {
    pub state: TripleFateState,
    pub space: VoyagerSpaceF64,
    pub rules: Vec<TripleRule>,
    pub discoveries: Vec<TripleDiscovery>,
    pub feedback_buffer: Vec<TripleSignal>,
    pub feedback_delay: u64,
    pub feedback_strength: f64,
    pub generation: u64,
    pub emergent_rules: Vec<TripleRule>,
}

impl FateLoop {
    pub fn new() -> Self {
        FateLoop {
            state: TripleFateState::new(),
            space: VoyagerSpaceF64::new(),
            rules: vec![
                TripleRule {
                    id: "default".into(),
                    priority: 1.0,
                    condition: "any".into(),
                    action: RuleAction::CreateContainer { 
                        dimension_delta: 0.25 
                    },
                },
            ],
            discoveries: Vec::new(),
            feedback_buffer: Vec::new(),
            feedback_delay: 5,
            feedback_strength: 0.5,
            generation: 0,
            emergent_rules: Vec::new(),
        }
    }

    pub fn tick(&mut self, external_signal: Option<Vec<f64>>) -> Vec<TripleDiscovery> {
        self.state.signal_count += 1;
        let current_tick = self.state.signal_count;
        let mut new_discoveries = Vec::new();

        // Process external signal only (不再有反馈缓冲，细胞化后数据即闭合)
        if let Some(values) = external_signal {
            let signal = TripleSignal {
                values,
                tick: current_tick,
                source: SignalSource::External,
            };
            if let Some(discovery) = self.process_signal_simple(&signal) {
                new_discoveries.push(discovery);
            }
        }

        // Evolution check (保留演化能力)
        if self.should_evolve() {
            self.evolve();
            new_discoveries.push(TripleDiscovery {
                id: format!("evolution_gen_{}", self.generation),
                level: 2,
                discovery_type: DiscoveryType::NewFate,
                description: format!("Evolved to generation {}", self.generation),
                created_at_tick: current_tick,
            });
        }

        new_discoveries
    }

    /// 简化的信号处理：不再累积反馈，细胞闭合后数据即完成
    fn process_signal_simple(&mut self, signal: &TripleSignal) -> Option<TripleDiscovery> {
        if let Some(pattern) = detect_pattern_f64(&signal.values) {
            self.state.discovery_count += 1;
            self.state.last_discovery_tick = signal.tick;
            self.state.pattern_diversity += 1;
            self.state.update_efficiency();

            self.space.on_discovery(&signal.values, &pattern);
            self.state.dimension_changes += 1;

            let discovery = TripleDiscovery {
                id: format!("loop_{}_{}", pattern, signal.tick),
                level: 0,
                discovery_type: DiscoveryType::MathPattern,
                description: format!("Found: {} (source: {:?})", pattern, signal.source),
                created_at_tick: signal.tick,
            };
            
            // 限制 discoveries 大小
            if self.discoveries.len() > 500 {
                self.discoveries.drain(0..250);
            }
            self.discoveries.push(discovery.clone());
            return Some(discovery);
        }

        None
    }

    fn maybe_create_emergent_rule(&mut self, tick: u64) {
        let self_ref_count = self.discoveries.iter()
            .filter(|d| d.discovery_type == DiscoveryType::SystemPattern)
            .count();

        // Valhalla 高阈值: 每500次自引用才创建新规则
        if self_ref_count > 0 && self_ref_count % 500 == 0 {
            let new_rule = TripleRule {
                id: format!("emergent_{}", tick),
                priority: self_ref_count as f64,
                condition: "self_pattern".into(),
                action: RuleAction::AdjustDimension { 
                    factor: 1.0 + self.feedback_strength,
                },
            };
            self.emergent_rules.push(new_rule.clone());
            self.rules.push(new_rule);
        }
    }

    pub fn should_evolve(&self) -> bool {
        let self_discoveries = self.discoveries.iter()
            .filter(|d| d.level >= 1)
            .count();
        
        // Valhalla 高阈值: 每1000次meta发现才进化
        self_discoveries > 0 && self_discoveries % 1000 == 0
    }

    fn evolve(&mut self) {
        self.generation += 1;
        self.feedback_delay = (self.feedback_delay + 1) % 10 + 1;
        self.feedback_strength *= 0.9;
        if self.feedback_strength < 0.1 {
            self.feedback_strength = 0.1;
        }
    }

    pub fn stats(&self) -> String {
        let self_ref = self.discoveries.iter().filter(|d| d.level >= 1).count();
        format!(
            "FateLoop Stats:\n\
             - Signal count: {}\n\
             - Generation: {}\n\
             - Discoveries: {} (self-ref: {})\n\
             - Emergent rules: {}\n\
             - Feedback buffer: {}\n\
             - Feedback strength: {:.3}\n\
             - Efficiency: {:.4}",
            self.state.signal_count,
            self.generation,
            self.discoveries.len(),
            self_ref,
            self.emergent_rules.len(),
            self.feedback_buffer.len(),
            self.feedback_strength,
            self.state.efficiency
        )
    }
}

impl Default for FateLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// VERSION 3: Combined (Fate-in-Fate + Fate Loop) - THE WINNER
// ============================================================================

/// Combined: Nested hierarchy with self-feedback at each level
/// THIS IS THE WINNING ARCHITECTURE for creating new meta-systems
#[derive(Debug, Serialize, Deserialize)]
pub struct TripleFateCombined {
    pub base: FateLoop,
    pub meta: FateLoop,
    pub ultimate_discoveries: Vec<TripleDiscovery>,
    pub global_tick: u64,
    pub cross_feedback: bool,
    /// 效率历史记录 (base_efficiency, meta_efficiency)
    pub efficiency_history: Vec<(f64, f64)>,
    /// 细胞聚合行为的代数方程（被动占位符）
    /// 上层观察底层信号，符合规律则触发
    pub cluster_equation: ClusterEquation,
    /// 器官形成检测器（代数化边际效应）
    /// 不使用硬编码阈值，自适应检测增长率边际递减
    pub organ_detector: OrganDetector,
    
    // ═══ Fate 存储层 ═══
    /// 数据存储 - 由 Fate 控制的内部存储
    /// Key: Fate 生成的 ID, Value: 处理后的数据
    #[serde(default)]
    pub data_store: std::collections::HashMap<String, FateDataEntry>,
    /// 存储计数
    #[serde(default)]
    pub store_count: u64,
}

/// Fate 存储的数据条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateDataEntry {
    /// 存储时的 tick
    pub tick: u64,
    /// 最后修改 tick
    pub modified_tick: u64,
    /// 原始数据（字节）- 系统可完全修改
    pub raw: Vec<u8>,
    /// Fate 处理后的签名（由 ClusterEquation 计算）
    pub signature: f64,
    /// 方程系数快照（存储时的 Fate 状态）
    pub coef_snapshot: [f64; 3],
    /// 是否已收敛
    pub converged: bool,
    /// 修改次数
    pub modification_count: u64,
    /// 来源: "external"=外部导入, "internal"=系统创建
    pub origin: String,
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|window| window == needle)
}

fn raw_mark(raw: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in raw {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl FateDataEntry {
    /// 创建新条目（外部导入）
    pub fn from_external(tick: u64, data: Vec<u8>, signature: f64, coefs: [f64; 3], converged: bool) -> Self {
        Self {
            tick,
            modified_tick: tick,
            raw: data,
            signature,
            coef_snapshot: coefs,
            converged,
            modification_count: 0,
            origin: "external".to_string(),
        }
    }
    
    /// 创建新条目（系统内部创建）
    pub fn from_internal(tick: u64, data: Vec<u8>, signature: f64, coefs: [f64; 3]) -> Self {
        Self {
            tick,
            modified_tick: tick,
            raw: data,
            signature,
            coef_snapshot: coefs,
            converged: false,
            modification_count: 0,
            origin: "internal".to_string(),
        }
    }
}

/// 细胞聚合行为的代数方程（被动检测）
/// 放置在上层 Fate，观察底层信号
/// 方程形式：a*x² + b*x + c = result，由 Fate 动态调整系数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterEquation {
    /// 上一次信号的方差
    pub last_variance: f64,
    /// 连续递减计数
    pub consecutive_decrease: u32,
    /// 是否已触发
    pub triggered: bool,
    /// 环形缓冲（用于被动模式）
    pub ring: [f64; 10],
    pub ring_len: usize,
    
    // ═══ 代数方程系数（由 Fate 经 adapt 调整，只在 Fate 收料路径上）═══
    /// 二次项系数 a（初始为 1.0，表示边际效应）
    pub coef_a: f64,
    /// 一次项系数 b（初始为 -2.0，表示趋势）
    pub coef_b: f64,
    /// 常数项 c（初始为 1.0，表示基准）
    pub coef_c: f64,
    /// 判定阈值（收敛时 result 应接近此值）
    pub threshold: f64,
    /// 方程适应次数（越多越稳定）
    pub adaptation_count: u64,
    /// 被动收敛的连续成立计数（入口自由度修复 F1 配套；F1 已
    /// 回退、判定未用此字段，保留仅为记账，勿当机制）。
    #[serde(default)]
    pub passive_streak: u32,
}

/// Fate 切片 - 冻结时保存的状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateSlice {
    pub global_tick: u64,
    pub base_signal_count: u64,
    pub base_discovery_count: u64,
    pub meta_signal_count: u64,
    pub cluster_coefs: [f64; 3],
    pub cluster_threshold: f64,
    pub cluster_triggered: bool,
    pub organ_formed: [bool; 4],
    pub efficiency_history: Vec<(f64, f64)>,
    /// 存储的数据
    pub data_store: std::collections::HashMap<String, FateDataEntry>,
    pub store_count: u64,
}

/// 器官形成检测器（代数化边际效应）
/// 不使用硬编码阈值，而是检测增长率的边际递减
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganDetector {
    /// 各特化类型的历史计数 [sensory, processing, memory, coordinator]
    pub history: [[u64; 10]; 4],
    /// 历史索引
    pub history_idx: usize,
    /// 历史长度
    pub history_len: usize,
    /// 各器官是否已形成（边际递减稳定）
    pub formed: [bool; 4],
    /// 各器官的形成 tick
    pub formation_tick: [u64; 4],
}

impl OrganDetector {
    pub fn new() -> Self {
        Self {
            history: [[0; 10]; 4],
            history_idx: 0,
            history_len: 0,
            formed: [false; 4],
            formation_tick: [0; 4],
        }
    }
    
    /// 记录当前特化分布，返回新形成的器官列表
    /// counts: [sensory, processing, memory, coordinator]
    pub fn record(&mut self, counts: [u64; 4], tick: u64) -> Vec<usize> {
        let mut newly_formed = Vec::new();
        
        // 记录历史
        for i in 0..4 {
            self.history[i][self.history_idx] = counts[i];
        }
        self.history_idx = (self.history_idx + 1) % 10;
        if self.history_len < 10 {
            self.history_len += 1;
        }
        
        // 至少需要 5 个历史点
        if self.history_len < 5 {
            return newly_formed;
        }
        
        // 检测每种器官的边际递减
        for organ in 0..4 {
            if self.formed[organ] {
                continue;
            }
            
            // 计算增长率序列
            let mut growth_rates = Vec::with_capacity(self.history_len - 1);
            for j in 1..self.history_len {
                let prev_idx = (self.history_idx + 10 - self.history_len + j - 1) % 10;
                let curr_idx = (self.history_idx + 10 - self.history_len + j) % 10;
                let prev = self.history[organ][prev_idx] as f64;
                let curr = self.history[organ][curr_idx] as f64;
                
                if prev > 0.0 {
                    growth_rates.push((curr - prev) / prev);
                }
            }
            
            if growth_rates.len() < 3 {
                continue;
            }
            
            // 边际效应检测：增长率递减
            // 检查最近 3 个增长率是否递减
            let n = growth_rates.len();
            let r1 = growth_rates[n - 3];
            let r2 = growth_rates[n - 2];
            let r3 = growth_rates[n - 1];
            
            // 增长率递减（边际递减）且当前有足够细胞
            let current_count = counts[organ];
            if r3 < r2 && r2 < r1 && r3 >= 0.0 && current_count > 0 {
                // 额外检查：增长率已经很小（趋于稳定）
                if r3.abs() < 0.1 {
                    self.formed[organ] = true;
                    self.formation_tick[organ] = tick;
                    newly_formed.push(organ);
                }
            }
        }
        
        newly_formed
    }
    
    /// 获取器官名称
    pub fn organ_name(index: usize) -> &'static str {
        match index {
            0 => "感知器官",
            1 => "处理器官",
            2 => "记忆器官",
            3 => "协调器官",
            _ => "未知器官",
        }
    }
    
    /// 已形成的器官数量
    pub fn formed_count(&self) -> usize {
        self.formed.iter().filter(|&&f| f).count()
    }
    
    /// 检查指定器官是否形成
    pub fn is_formed(&self, organ: usize) -> bool {
        organ < 4 && self.formed[organ]
    }
}

impl Default for OrganDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterEquation {
    pub fn new() -> Self {
        Self {
            last_variance: f64::MAX,
            consecutive_decrease: 0,
            triggered: false,
            ring: [0.0; 10],
            ring_len: 0,
            // 初始方程：a*x² + b*x + c = 0
            // 默认 1*x² - 2*x + 1 = (x-1)² = 0，解 x=1
            // 意味着：差异比率接近 1（稳定）时收敛
            coef_a: 1.0,
            coef_b: -2.0,
            coef_c: 1.0,
            threshold: 0.01,  // 更严格：x 必须非常接近 1
            adaptation_count: 0,
            passive_streak: 0,
        }
    }
    
    /// 计算动态代数方程：a*x² + b*x + c
    /// x = 差异比率（d_new / d_old）
    #[inline]
    pub fn evaluate(&self, x: f64) -> f64 {
        self.coef_a * x * x + self.coef_b * x + self.coef_c
    }

    /// 代数方程 - 被动检查底层信号
    /// 使用动态方程判断是否收敛
    pub fn check(&mut self, signal_values: &[f64]) -> bool {
        if signal_values.is_empty() {
            return false;
        }
        
        // 计算信号方差
        let mean: f64 = signal_values.iter().sum::<f64>() / signal_values.len() as f64;
        let variance: f64 = signal_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / signal_values.len() as f64;
        
        // 计算差异比率 x = variance / last_variance
        let x = if self.last_variance > 1e-10 {
            variance / self.last_variance
        } else {
            1.0
        };
        
        // 用动态方程判断（只判不改；adapt 由 Fate 在
        // receive_passive 收料判定后喂，见彼处）
        let result = self.evaluate(x);

        self.last_variance = variance;
        
        // 方程结果接近 threshold 时触发
        result.abs() < self.threshold
    }
    
    /// 自适应调整方程系数（绑定 Fate：只由 Fate 在
    /// receive_passive 收完语料、判定之后调用；判断
    /// 本身只判不改，身体层借方程判断不喂 adapt）
    /// 基于历史数据优化方程
    fn adapt(&mut self, x: f64, result: f64) {
        self.adaptation_count += 1;

        // 每 100 次适应一次系数
        if self.adaptation_count % 100 != 0 {
            return;
        }

        // 学习率随适应次数递减（边际递减）
        let lr = 0.01 / (1.0 + self.adaptation_count as f64 * 0.0001);

        // 如果 result 太大，说明方程不够敏感，调整系数
        if result.abs() > self.threshold * 10.0 {
            // 增加敏感度：减小 threshold
            self.threshold = (self.threshold * 0.99).max(0.01);
        } else if result.abs() < self.threshold * 0.1 {
            // 太敏感，增大 threshold
            self.threshold = (self.threshold * 1.01).min(1.0);
        }

        // 根据 x 调整系数（归一化梯度下降）
        // 目标：让 x ≈ 1 时 result ≈ 0（稳定）
        // 步长除以梯度模长平方 x⁴+x²+1：不归一化时 x 偏离 1 越远步子越大，
        // 系数几步就被推到 ±10 钳位且同号，方程对一切输入同号、永远判假（饱和死亡）。
        // 归一化后每步只把当前 x 处的方程值按 (1-lr) 收缩，系数追得住典型比率。
        let error = result;
        let norm = x * x * x * x + x * x + 1.0;
        self.coef_a -= lr * error * (x * x) / norm;
        self.coef_b -= lr * error * x / norm;
        self.coef_c -= lr * error / norm;

        // 限制系数范围
        self.coef_a = self.coef_a.clamp(-10.0, 10.0);
        self.coef_b = self.coef_b.clamp(-10.0, 10.0);
        self.coef_c = self.coef_c.clamp(-10.0, 10.0);
    }

    /// 检查差异序列是否符合方程（用于 Tile 收敛判断）
    /// diffs: [d0, d1, d2, ...] 差异序列
    pub fn check_diffs(&mut self, diffs: &[f64]) -> bool {
        if diffs.len() < 2 {
            return false;
        }
        
        // 计算差异比率
        let x = if diffs[1].abs() > 1e-10 {
            diffs[0] / diffs[1]
        } else {
            1.0
        };
        
        let result = self.evaluate(x);

        result.abs() < self.threshold
    }
    
    /// String representation of the current equation
    pub fn equation_str(&self) -> String {
        format!("{:.4}x² + ({:.4})x + {:.4} ≈ 0 (threshold:{:.4})",
                self.coef_a, self.coef_b, self.coef_c, self.threshold)
    }

    /// Bundle-fit judgement (Fate bundling, user-directed): uses the
    /// original quadratic's convergence rule - adjacent distance ratio
    /// x near 1 ((x-1)^2 below threshold) means fit, same bundle. Uses
    /// the original equation semantics rather than the adapt-drifted
    /// roots (drifted roots sit near 0.45/2.13; cutting bundles with
    /// them would cut nothing). Threshold is the equation's own.
    pub fn bundle_fits(&self, d0: f64, d1: f64) -> bool {
        let x = if d1.abs() > 1e-10 { d0 / d1 } else { 1.0 };
        let r = x - 1.0;
        r * r < self.threshold
    }

    pub fn reset_trigger(&mut self) {
        self.triggered = false;
    }
}

impl Default for ClusterEquation {
    fn default() -> Self {
        Self::new()
    }
}

impl TripleFateCombined {
    pub fn new() -> Self {
        let mut base = FateLoop::new();
        base.feedback_strength = 1.0 / 3.0;
        
        let mut meta = FateLoop::new();
        meta.feedback_strength = 0.25;
        meta.feedback_delay = 3;
        
        TripleFateCombined {
            base,
            meta,
            ultimate_discoveries: Vec::new(),
            global_tick: 0,
            cross_feedback: true,
            efficiency_history: Vec::new(),
            cluster_equation: ClusterEquation::new(),
            organ_detector: OrganDetector::new(),
            data_store: std::collections::HashMap::new(),
            store_count: 0,
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Fate 存储层 - 由 Fate 控制存储
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// 存储数据 - 由 Fate 处理并存储
    /// 返回存储 ID
    pub fn store_data(&mut self, data: &[u8]) -> String {
        self.store_count += 1;
        let id = format!("fate_{}_{}", self.global_tick, self.store_count);
        
        // 计算数据的 Fate 签名
        let signature = self.compute_signature(data);
        let converged = self.cluster_equation.triggered;
        let coefs = [
            self.cluster_equation.coef_a,
            self.cluster_equation.coef_b,
            self.cluster_equation.coef_c,
        ];
        
        // 存储（外部导入）
        let entry = FateDataEntry::from_external(
            self.global_tick,
            data.to_vec(),
            signature,
            coefs,
            converged,
        );
        
        self.data_store.insert(id.clone(), entry);
        id
    }
    
    /// 存储字符串
    pub fn store_string(&mut self, content: &str) -> String {
        self.store_data(content.as_bytes())
    }
    
    /// 计算 Fate 签名（基于 ClusterEquation）
    pub fn compute_signature(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        // 计算数据特征
        let sum: f64 = data.iter().map(|&b| b as f64).sum();
        let mean = sum / data.len() as f64;
        let variance: f64 = data.iter()
            .map(|&b| {
                let d = (b as f64) - mean;
                d * d
            })
            .sum::<f64>() / data.len() as f64;
        
        // 用 ClusterEquation 计算签名
        // signature = a*mean² + b*variance + c
        self.cluster_equation.coef_a * mean * mean 
            + self.cluster_equation.coef_b * variance 
            + self.cluster_equation.coef_c
    }
    
    /// 读取存储的数据
    pub fn load_data(&self, id: &str) -> Option<&FateDataEntry> {
        self.data_store.get(id)
    }
    
    /// 获取可变引用（允许直接修改）
    pub fn load_data_mut(&mut self, id: &str) -> Option<&mut FateDataEntry> {
        self.data_store.get_mut(id)
    }
    
    /// 删除数据
    pub fn remove_data(&mut self, id: &str) -> Option<FateDataEntry> {
        self.data_store.remove(id)
    }
    
    /// 列出所有存储的 ID
    pub fn list_stored(&self) -> Vec<&String> {
        self.data_store.keys().collect()
    }
    
    /// 根据签名查找相似数据
    pub fn find_similar(&self, signature: f64, threshold: f64) -> Vec<&String> {
        self.data_store.iter()
            .filter(|(_, entry)| (entry.signature - signature).abs() < threshold)
            .map(|(id, _)| id)
            .collect()
    }
    
    /// 存储数量
    pub fn stored_count(&self) -> usize {
        self.data_store.len()
    }

    /// 身体自己送出。刺激不进入 receive_passive，不 tick，不改方程，不改存储。
    /// 对上的是 data_store 里的原文：签名落在当前方程阈值内，或原文里含这段刺激。
    /// 命中的原文按存入时的 tick 一次送出。没有命中就是空，不再另设沟通意愿。
    pub fn react(&self, stimulus: &[u8]) -> Vec<u8> {
        if stimulus.is_empty() || self.data_store.is_empty() {
            return Vec::new();
        }
        let want = self.compute_signature(stimulus);
        let threshold = self.cluster_equation.threshold;
        let mut hits: Vec<&FateDataEntry> = self
            .data_store
            .values()
            .filter(|entry| {
                if entry.raw.is_empty() {
                    return false;
                }
                let have = self.compute_signature(&entry.raw);
                (have - want).abs() <= threshold || contains_bytes(&entry.raw, stimulus)
            })
            .collect();
        hits.sort_by(|left, right| {
            left.tick
                .cmp(&right.tick)
                .then(left.modified_tick.cmp(&right.modified_tick))
        });
        let mut out = Vec::new();
        let mut seen: Vec<(usize, u64)> = Vec::new();
        for entry in hits {
            let mark = (entry.raw.len(), raw_mark(&entry.raw));
            if seen.contains(&mark) {
                continue;
            }
            seen.push(mark);
            if !out.is_empty() {
                out.push(b'\n');
            }
            out.extend_from_slice(&entry.raw);
        }
        out
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 系统完全控制权 - 修改、创建、变换
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// 【系统新建】系统自主创建新数据
    pub fn create_data(&mut self, data: Vec<u8>) -> String {
        self.store_count += 1;
        let id = format!("fate_internal_{}_{}", self.global_tick, self.store_count);
        
        let signature = self.compute_signature(&data);
        let coefs = [
            self.cluster_equation.coef_a,
            self.cluster_equation.coef_b,
            self.cluster_equation.coef_c,
        ];
        
        let entry = FateDataEntry::from_internal(self.global_tick, data, signature, coefs);
        self.data_store.insert(id.clone(), entry);
        id
    }
    
    /// 【系统新建】系统自主创建字符串数据
    pub fn create_string(&mut self, content: &str) -> String {
        self.create_data(content.as_bytes().to_vec())
    }
    
    /// 【完全修改】替换数据内容（字节级）
    pub fn modify_data(&mut self, id: &str, new_data: Vec<u8>) -> bool {
        // 先计算签名和系数
        let signature = self.compute_signature(&new_data);
        let coefs = [
            self.cluster_equation.coef_a,
            self.cluster_equation.coef_b,
            self.cluster_equation.coef_c,
        ];
        let tick = self.global_tick;
        
        if let Some(entry) = self.data_store.get_mut(id) {
            entry.raw = new_data;
            entry.modified_tick = tick;
            entry.modification_count += 1;
            entry.signature = signature;
            entry.coef_snapshot = coefs;
            true
        } else {
            false
        }
    }
    
    /// 【部分修改】修改数据的指定范围（字节级）
    pub fn modify_range(&mut self, id: &str, offset: usize, new_bytes: &[u8]) -> bool {
        // 先获取数据副本并修改
        let modified = if let Some(entry) = self.data_store.get(id) {
            let mut data = entry.raw.clone();
            let end = (offset + new_bytes.len()).min(data.len());
            if offset < data.len() {
                data[offset..end].copy_from_slice(&new_bytes[..end - offset]);
                Some(data)
            } else {
                None
            }
        } else {
            None
        };
        
        if let Some(data) = modified {
            self.modify_data(id, data)
        } else {
            false
        }
    }
    
    /// 【追加】向数据末尾追加字节
    pub fn append_data(&mut self, id: &str, extra: &[u8]) -> bool {
        // 先获取并扩展
        let extended = if let Some(entry) = self.data_store.get(id) {
            let mut data = entry.raw.clone();
            data.extend_from_slice(extra);
            Some(data)
        } else {
            None
        };
        
        if let Some(data) = extended {
            self.modify_data(id, data)
        } else {
            false
        }
    }
    
    /// 【截断】截断数据到指定长度
    pub fn truncate_data(&mut self, id: &str, new_len: usize) -> bool {
        let truncated = if let Some(entry) = self.data_store.get(id) {
            let mut data = entry.raw.clone();
            data.truncate(new_len);
            Some(data)
        } else {
            None
        };
        
        if let Some(data) = truncated {
            self.modify_data(id, data)
        } else {
            false
        }
    }
    
    /// 【变换】对数据应用字节级变换（系统自主处理）
    pub fn transform_data<F>(&mut self, id: &str, transform: F) -> bool 
    where F: FnOnce(&mut Vec<u8>)
    {
        let transformed = if let Some(entry) = self.data_store.get(id) {
            let mut data = entry.raw.clone();
            transform(&mut data);
            Some(data)
        } else {
            None
        };
        
        if let Some(data) = transformed {
            self.modify_data(id, data)
        } else {
            false
        }
    }
    
    /// 【合并】合并多个数据为一个新数据（系统创建）
    pub fn merge_data(&mut self, ids: &[&str]) -> Option<String> {
        let mut merged = Vec::new();
        for id in ids {
            if let Some(entry) = self.data_store.get(*id) {
                merged.extend_from_slice(&entry.raw);
            }
        }
        if merged.is_empty() {
            None
        } else {
            Some(self.create_data(merged))
        }
    }
    
    /// 【分割】将数据分割为多个新数据（系统创建）
    pub fn split_data(&mut self, id: &str, chunk_size: usize) -> Vec<String> {
        let mut result = Vec::new();
        if let Some(entry) = self.data_store.get(id) {
            let data = entry.raw.clone();
            for chunk in data.chunks(chunk_size) {
                let new_id = self.create_data(chunk.to_vec());
                result.push(new_id);
            }
        }
        result
    }
    
    /// 【复制】复制数据（系统创建新副本）
    pub fn clone_data(&mut self, id: &str) -> Option<String> {
        if let Some(entry) = self.data_store.get(id) {
            let data = entry.raw.clone();
            Some(self.create_data(data))
        } else {
            None
        }
    }
    
    /// 【批量删除】删除满足条件的数据
    pub fn remove_where<F>(&mut self, predicate: F) -> usize 
    where F: Fn(&str, &FateDataEntry) -> bool
    {
        let to_remove: Vec<String> = self.data_store.iter()
            .filter(|(id, entry)| predicate(id, entry))
            .map(|(id, _)| id.clone())
            .collect();
        
        let count = to_remove.len();
        for id in to_remove {
            self.data_store.remove(&id);
        }
        count
    }
    
    /// 【清空】清空所有存储
    pub fn clear_all(&mut self) {
        self.data_store.clear();
    }
    
    /// 获取所有内部创建的数据
    pub fn list_internal(&self) -> Vec<&String> {
        self.data_store.iter()
            .filter(|(_, entry)| entry.origin == "internal")
            .map(|(id, _)| id)
            .collect()
    }
    
    /// 获取所有外部导入的数据
    pub fn list_external(&self) -> Vec<&String> {
        self.data_store.iter()
            .filter(|(_, entry)| entry.origin == "external")
            .map(|(id, _)| id)
            .collect()
    }
    
    /// 获取总修改次数
    pub fn total_modifications(&self) -> u64 {
        self.data_store.values().map(|e| e.modification_count).sum()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Fate 切片 - 冻结/解冻
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// 冻结 Fate 切片（包含所有存储的数据）
    pub fn freeze(&self) -> FateSlice {
        FateSlice {
            global_tick: self.global_tick,
            base_signal_count: self.base.state.signal_count,
            base_discovery_count: self.base.state.discovery_count,
            meta_signal_count: self.meta.state.signal_count,
            cluster_coefs: [
                self.cluster_equation.coef_a,
                self.cluster_equation.coef_b,
                self.cluster_equation.coef_c,
            ],
            cluster_threshold: self.cluster_equation.threshold,
            cluster_triggered: self.cluster_equation.triggered,
            organ_formed: self.organ_detector.formed,
            efficiency_history: self.efficiency_history.clone(),
            data_store: self.data_store.clone(),
            store_count: self.store_count,
        }
    }
    
    /// 从切片恢复
    pub fn thaw(slice: FateSlice) -> Self {
        let mut fate = Self::new();
        fate.global_tick = slice.global_tick;
        fate.base.state.signal_count = slice.base_signal_count;
        fate.base.state.discovery_count = slice.base_discovery_count;
        fate.meta.state.signal_count = slice.meta_signal_count;
        fate.cluster_equation.coef_a = slice.cluster_coefs[0];
        fate.cluster_equation.coef_b = slice.cluster_coefs[1];
        fate.cluster_equation.coef_c = slice.cluster_coefs[2];
        fate.cluster_equation.threshold = slice.cluster_threshold;
        fate.cluster_equation.triggered = slice.cluster_triggered;
        fate.organ_detector.formed = slice.organ_formed;
        fate.efficiency_history = slice.efficiency_history;
        fate.data_store = slice.data_store;
        fate.store_count = slice.store_count;
        fate
    }
    
    /// 保存切片到文件 (二进制格式 .fate)
    pub fn save_slice(&self, path: &std::path::Path) -> std::io::Result<()> {
        let slice = self.freeze();
        let bytes = bincode::serialize(&slice)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }
    
    /// 从文件加载切片 (二进制格式 .fate)
    pub fn load_slice(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let slice: FateSlice = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self::thaw(slice))
    }
    
    /// 序列化为字节 (二进制)
    pub fn to_bytes(&self) -> Result<Vec<u8>, std::io::Error> {
        let slice = self.freeze();
        bincode::serialize(&slice)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    
    /// 从字节反序列化 (二进制)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, std::io::Error> {
        let slice: FateSlice = bincode::deserialize(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self::thaw(slice))
    }
    
    /// [调试用] 保存为 JSON (仅调试)
    #[cfg(debug_assertions)]
    pub fn save_slice_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let slice = self.freeze();
        let json = serde_json::to_string_pretty(&slice)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }
    
    /// 记录器官特化分布，返回新形成的器官
    /// counts: [sensory, processing, memory, coordinator]
    pub fn detect_organs(&mut self, counts: [u64; 4]) -> Vec<usize> {
        self.organ_detector.record(counts, self.global_tick)
    }
    
    /// 获取器官形成状态
    pub fn get_organ_status(&self) -> &[bool; 4] {
        &self.organ_detector.formed
    }
    
    /// 器官形成数量
    pub fn formed_organ_count(&self) -> usize {
        self.organ_detector.formed_count()
    }
    
    /// 被动接收数据 - 自动累积，自动检测
    /// 返回：是否检测到边际递减（收敛信号）
    /// 使用动态代数方程（由 Fate 涌现）判断模式
    #[inline]
    pub fn receive_passive(&mut self, value: f64) -> bool {
        self.global_tick += 1;
        let mut convergence_detected = false;
        
        // 直接更新 base 状态
        self.base.state.signal_count += 1;
        
        // 存入环形缓冲
        let idx = (self.global_tick % 10) as usize;
        self.cluster_equation.ring[idx] = value;
        if self.cluster_equation.ring_len < 10 {
            self.cluster_equation.ring_len += 1;
        }
        
        // 每10个信号用动态方程检测（原版节律）。入口自由度修复
        // F1 曾尝试逐拍判定使段界脱离网格，2026-11-01 实测三式
        // 皆败：单点差值比逐拍 → 段长 ~125B（种群爆炸、场 O(n²)
        // 拖死）；连续 2 拍成立 → ~15KB（几乎不收口，事实映射
        // 996/1200 且自然料零出生）；窗口均值比 → ~40B（均值把
        // 比值拉向方程根 x=1，触发更频）。根源：此方程在逐拍
        // 尺度上是细粒度判据，旧网格一直在替段长定标。计时
        // 自由化须改用段级收敛信号（容器自身差值序列），另案
        // 设计；本版保留网格，形态自由化走 F2/F3（签名/材料）。
        if self.global_tick % 10 == 0 && self.cluster_equation.ring_len >= 3 {
            let get = |off: usize| self.cluster_equation.ring[(idx + 10 - off) % 10];
            let d1 = (get(0) - get(1)).abs();
            let d2 = (get(1) - get(2)).abs();

            // 用动态方程判断（二元一次方程）
            let diffs = [d1, d2];
            if self.cluster_equation.check_diffs(&diffs) {
                self.base.state.discovery_count += 1;
                self.base.state.pattern_diversity += 1;
                convergence_detected = true;  // 捕捉到收敛信号
            }
            // adapt 绑定 Fate（用户裁定）：Fate 先接触语料
            // （本函数收值入环、判定在先），判定之后才喂
            // adapt 自己刚判过的比率；adapt 不再藏在判断
            // 里被身体层借方程时顺带触发。
            let ax = if d2.abs() > 1e-10 { d1 / d2 } else { 1.0 };
            let aresult = self.cluster_equation.evaluate(ax);
            self.cluster_equation.adapt(ax, aresult);
            // 上一层看底层状态，不另起一个方程。
            let lifted = self.state_to_signal(&self.base.state);
            let _ = self.meta.tick(Some(lifted));
        }
        if self.cross_feedback && self.global_tick % 1000 == 0 {
            if let Some(rule) = self.meta.emergent_rules.last() {
                if self.base.rules.len() < 100 {
                    self.base.rules.push(rule.clone());
                }
            }
        }
        
        // 更新效率
        self.base.state.efficiency = (self.base.state.discovery_count as f64) 
            / (self.base.state.signal_count as f64).max(1.0);
        
        // 每1000个信号记录一次效率历史
        if self.global_tick % 1000 == 0 {
            self.efficiency_history.push((self.base.state.efficiency, self.meta.state.efficiency));
            if self.efficiency_history.len() > 100 {
                self.efficiency_history.remove(0);
            }
        }
        
        convergence_detected
    }
    
    /// 用动态方程检查差异序列是否收敛
    /// 返回 (是否收敛, 方程结果)
    pub fn check_convergence(&mut self, diffs: &[f64]) -> (bool, f64) {
        if diffs.len() < 2 {
            return (false, f64::MAX);
        }
        
        let x = if diffs[1].abs() > 1e-10 {
            diffs[0] / diffs[1]
        } else {
            1.0
        };
        
        let result = self.cluster_equation.evaluate(x);
        let converged = result.abs() < self.cluster_equation.threshold;
        
        (converged, result)
    }
    
    /// 获取当前涌现的方程
    pub fn get_equation(&self) -> String {
        self.cluster_equation.equation_str()
    }

    /// 高能逃逸·图层（Fate 功能）：场内相对距离远（不
    /// 在邻居集内）且信号强（场强高于全场平均）的候选
    /// 中，取场强最高者为逃逸神经目标。与四元黑洞的
    /// 存储层逃逸（视界吞没豁免）分属两层、各行其是。
    /// fields 为全体场强，excluded 为已在邻居集内
    /// （含自身）的掩码。
    pub fn escape_target(&self, fields: &[f64], excluded: &[bool]) -> Option<usize> {
        if fields.is_empty() {
            return None;
        }
        let mean = fields.iter().sum::<f64>() / fields.len() as f64;
        let mut best: Option<usize> = None;
        for (j, f) in fields.iter().enumerate() {
            if excluded.get(j).copied().unwrap_or(false) || *f <= mean {
                continue;
            }
            if best.map_or(true, |b| *f > fields[b]) {
                best = Some(j);
            }
        }
        best
    }
    
    /// 检查是否收敛（被动模式）
    /// 纯代数判断：效率稳定（方差趋近于零）
    pub fn is_passively_converged(&self) -> bool {
        // 至少需要10个效率历史记录
        if self.efficiency_history.len() < 10 {
            return false;
        }
        
        let len = self.efficiency_history.len();
        
        // 取最近10个效率值
        let recent: Vec<f64> = self.efficiency_history[len-10..len]
            .iter()
            .map(|(eff, _)| *eff)
            .collect();
        
        // 计算均值
        let mean = recent.iter().sum::<f64>() / 10.0;
        
        // 计算方差
        let variance: f64 = recent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / 10.0;
        
        // 方差极小 = 稳定 = 收敛
        // 不使用硬编码阈值，而是相对于均值
        let relative_variance = if mean.abs() > 1e-10 { 
            variance / mean.abs() 
        } else { 
            variance 
        };
        
        // 相对方差 < 0.01% = 稳定
        relative_variance < 0.0001 && mean > 0.0
    }

    pub fn tick(&mut self, external_signal: Option<Vec<f64>>) -> Vec<TripleDiscovery> {
        self.global_tick += 1;
        let mut all_discoveries = Vec::new();

        // Base Fate processes signal (底层捕捉信号)
        let base_discoveries = self.base.tick(external_signal.clone());
        all_discoveries.extend(base_discoveries.clone());

        // 上层代数方程：被动检查底层信号是否符合规律
        if let Some(ref signal) = external_signal {
            if self.cluster_equation.check(signal) {
                // 信号符合代数方程规律，触发细胞聚合行为
                let discovery = TripleDiscovery {
                    id: format!("cluster_{}", self.global_tick),
                    level: 1,  // 上层触发
                    discovery_type: DiscoveryType::SystemPattern,
                    description: "cluster_behavior".into(),
                    created_at_tick: self.global_tick,
                };
                all_discoveries.push(discovery);
            }
        }

        // Meta Fate observes Base state
        let base_state_signal = self.state_to_signal(&self.base.state);
        let meta_discoveries = self.meta.tick(Some(base_state_signal));
        
        for d in meta_discoveries {
            let elevated = TripleDiscovery {
                id: format!("meta_{}", d.id),
                level: d.level + 1,
                discovery_type: d.discovery_type.clone(),
                description: format!("[META] {}", d.description),
                created_at_tick: self.global_tick,
            };
            all_discoveries.push(elevated);
        }

        // Cross-level feedback (每1000个tick才同步，避免性能问题)
        if self.cross_feedback && self.global_tick % 1000 == 0 {
            if let Some(rule) = self.meta.emergent_rules.last() {
                if self.base.rules.len() < 100 {
                    self.base.rules.push(rule.clone());
                }
            }
        }

        // 记录效率历史（每1000个tick记录一次）
        if self.global_tick % 1000 == 0 {
            self.efficiency_history.push((
                self.base.state.efficiency,
                self.meta.state.efficiency,
            ));
            if self.efficiency_history.len() > 100 {
                self.efficiency_history.remove(0);
            }
        }

        all_discoveries
    }

    fn state_to_signal(&self, state: &TripleFateState) -> Vec<f64> {
        vec![
            state.signal_count as f64,
            state.discovery_count as f64,
            state.efficiency,
            state.dimension_changes as f64,
            state.pattern_diversity as f64,
        ]
    }

    fn should_create_new_fate(&self) -> bool {
        let total_meta = self.base.discoveries.iter().filter(|d| d.level >= 1).count()
            + self.meta.discoveries.iter().filter(|d| d.level >= 1).count();
        
        // Valhalla 高标准
        (self.base.generation >= 5 && self.meta.generation >= 5 && self.ultimate_discoveries.is_empty())
        || (total_meta > 10000 && self.ultimate_discoveries.is_empty())
    }

    /// 检查是否收敛 (被动触发，基于边际效应递减)
    pub fn has_converged(&self) -> bool {
        let len = self.efficiency_history.len();
        
        // 需要至少5个快照才能判断收敛趋势
        if len < 5 {
            return false;
        }
        
        // 计算最近4次效率变化（base + meta 的总变化）
        let diffs: Vec<f64> = (1..5)
            .map(|k| {
                let idx = len - k;
                let (b1, m1) = self.efficiency_history[idx];
                let (b0, m0) = self.efficiency_history[idx - 1];
                (b1 - b0).abs() + (m1 - m0).abs()
            })
            .collect();
        
        // 代数判断：连续边际递减 (diff[0] < diff[1] < diff[2])
        // 且最新变化必须较小（趋近于稳定）
        let decreasing = diffs[0] < diffs[1] && diffs[1] < diffs[2];
        let stable = diffs[0] < diffs[3] * 0.5; // 最新变化小于初始变化的一半
        
        decreasing && stable && diffs[2] > 0.0
    }

    pub fn stats(&self) -> String {
        format!(
            "TripleFateCombined Stats:\n\
             - Global tick: {}\n\
             \n[BASE FATE (with loop)]\n{}\n\
             \n[META FATE (with loop)]\n{}\n\
             \n[ULTIMATE]\n\
             - Ultimate discoveries: {}\n\
             - Cross-feedback: {}",
            self.global_tick,
            self.base.stats(),
            self.meta.stats(),
            self.ultimate_discoveries.len(),
            self.cross_feedback
        )
    }
}

impl Default for TripleFateCombined {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pattern Detection (f64 version)
// ============================================================================

pub fn detect_pattern_f64(values: &[f64]) -> Option<String> {
    if values.len() < 2 {
        return None;
    }

    const EPSILON: f64 = 1e-9;

    // Linear detection
    if values.len() >= 3 {
        let d1 = values[1] - values[0];
        let d2 = values[2] - values[1];
        if (d1 - d2).abs() < EPSILON && d1.abs() > EPSILON {
            return Some(format!("linear_d{:.4}", d1));
        }
    }

    // Convergence
    if values.len() >= 3 {
        let diff1 = (values[1] - values[0]).abs();
        let diff2 = (values[2] - values[1]).abs();
        if diff2 < diff1 && diff1 > EPSILON {
            return Some("converging".into());
        }
    }

    // Divergence
    if values.len() >= 3 {
        let diff1 = (values[1] - values[0]).abs();
        let diff2 = (values[2] - values[1]).abs();
        if diff2 > diff1 * 2.0 {
            return Some("diverging".into());
        }
    }

    // Fixed point
    if values.len() >= 2 && (values[0] - values[1]).abs() < EPSILON {
        return Some("fixed_point".into());
    }

    // Balanced (sum is zero)
    let sum: f64 = values.iter().sum();
    if sum.abs() < EPSILON && !values.iter().all(|v| v.abs() < EPSILON) {
        return Some("balanced".into());
    }

    None
}

// ============================================================================
// Experiment Runner
// ============================================================================

#[derive(Debug)]
pub struct ExperimentResult {
    pub name: String,
    pub ticks: u64,
    pub total_discoveries: usize,
    pub math_discoveries: usize,
    pub system_discoveries: usize,
    pub new_rules: usize,
    pub new_fates: usize,
    pub final_efficiency: f64,
    pub generations: u64,
}

pub fn run_comparison_experiment(signal_count: usize) -> Vec<ExperimentResult> {
    let mut results = Vec::new();

    let test_signals: Vec<Vec<f64>> = (0..signal_count)
        .map(|i| {
            vec![
                (i % 50) as f64,
                (i % 50) as f64 + 1.0,
                (i % 50) as f64 + 2.0,
            ]
        })
        .collect();

    // Test Version 1: Fate-in-Fate
    let mut fate_in_fate = FateInFate::new();
    let mut fif_discoveries = Vec::new();
    for signal in &test_signals {
        let discoveries = fate_in_fate.tick(Some(signal.clone()));
        fif_discoveries.extend(discoveries);
    }
    for _ in 0..50 {
        let discoveries = fate_in_fate.tick(None);
        fif_discoveries.extend(discoveries);
    }

    results.push(ExperimentResult {
        name: "Fate-in-Fate".into(),
        ticks: fate_in_fate.global_tick,
        total_discoveries: fif_discoveries.len(),
        math_discoveries: fif_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::MathPattern).count(),
        system_discoveries: fif_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::SystemPattern).count(),
        new_rules: fif_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::NewRule).count(),
        new_fates: fif_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::NewFate).count(),
        final_efficiency: fate_in_fate.outer.state.efficiency,
        generations: 0,
    });

    // Test Version 2: Fate Loop
    let mut fate_loop = FateLoop::new();
    let mut fl_discoveries = Vec::new();
    for signal in &test_signals {
        let discoveries = fate_loop.tick(Some(signal.clone()));
        fl_discoveries.extend(discoveries);
    }
    for _ in 0..100 {
        let discoveries = fate_loop.tick(None);
        fl_discoveries.extend(discoveries);
    }

    results.push(ExperimentResult {
        name: "Fate Loop".into(),
        ticks: fate_loop.state.signal_count,
        total_discoveries: fl_discoveries.len(),
        math_discoveries: fl_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::MathPattern).count(),
        system_discoveries: fl_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::SystemPattern).count(),
        new_rules: fl_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::NewRule).count(),
        new_fates: fl_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::NewFate).count(),
        final_efficiency: fate_loop.state.efficiency,
        generations: fate_loop.generation,
    });

    // Test Version 3: Combined (THE WINNER)
    let mut triple_combined = TripleFateCombined::new();
    let mut tc_discoveries = Vec::new();
    for signal in &test_signals {
        let discoveries = triple_combined.tick(Some(signal.clone()));
        tc_discoveries.extend(discoveries);
    }
    for _ in 0..150 {
        let discoveries = triple_combined.tick(None);
        tc_discoveries.extend(discoveries);
    }

    results.push(ExperimentResult {
        name: "Combined (Winner)".into(),
        ticks: triple_combined.global_tick,
        total_discoveries: tc_discoveries.len(),
        math_discoveries: tc_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::MathPattern).count(),
        system_discoveries: tc_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::SystemPattern).count(),
        new_rules: tc_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::NewRule).count(),
        new_fates: tc_discoveries.iter().filter(|d| d.discovery_type == DiscoveryType::NewFate).count(),
        final_efficiency: triple_combined.base.state.efficiency,
        generations: triple_combined.base.generation + triple_combined.meta.generation,
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fate_in_fate() {
        let mut system = FateInFate::new();
        for i in 0..100 {
            let signal = vec![i as f64, (i + 1) as f64, (i + 2) as f64];
            system.tick(Some(signal));
        }
        assert!(!system.outer.discoveries.is_empty());
    }

    #[test]
    fn test_fate_loop() {
        let mut system = FateLoop::new();
        for i in 0..100 {
            let signal = vec![i as f64, (i + 1) as f64, (i + 2) as f64];
            system.tick(Some(signal));
        }
        for _ in 0..50 {
            system.tick(None);
        }
        println!("{}", system.stats());
    }

    #[test]
    fn test_triple_combined() {
        let mut system = TripleFateCombined::new();
        for i in 0..200 {
            let signal = vec![i as f64, (i + 1) as f64, (i + 2) as f64];
            system.tick(Some(signal));
        }
        println!("{}", system.stats());
    }

    #[test]
    fn test_pattern_detection() {
        // Linear
        assert!(detect_pattern_f64(&[1.0, 2.0, 3.0]).unwrap().contains("linear"));
        // Converging
        assert_eq!(detect_pattern_f64(&[10.0, 5.0, 3.0]), Some("converging".into()));
        // Fixed point
        assert_eq!(detect_pattern_f64(&[5.0, 5.0]), Some("fixed_point".into()));
    }
}
