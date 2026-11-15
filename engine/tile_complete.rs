//! Tile-Quattro Complete (f64 Version) - 四元系统并行架构
//!
//! 完整版本，从 valhalla-rational 移植
//! 自带: TripleFate (多层监控) + VoyagerSpace (自动维度)
//!
//! 架构:
//! - 四元系统并行处理
//! - 数据流 → Tile → Fate 监控
//! - VoyagerSpace 自动维度扩展
//! - **Fate 存储**: 数据由 Fate 处理并存储

use serde::{Deserialize, Serialize};
use std::path::Path;

// 系统存储 (旧版兼容)
pub use valhalla_storage::SystemStorage;

// 使用 fate-f64 的完整 Triple-Fate 架构
use fate_f64::{
    TripleFateCombined, TripleDiscovery, VoyagerSpaceF64, FateSlice,
};
use valhalla_metasystems_baseforce::BaseForceCoordinator;
use valhalla_metasystems_blackhole::{GravityField, GravityFieldConfig};
use valhalla_metasystems_meshbrain::{MeshBrainConfig, MeshBrainSystem};
use valhalla_metasystems_multinova::{MultinovaConfig, MultinovaProcessor};

// Space GPU 高维计算
use valhalla_space_f64_gpu::{SpaceGPU, FractalDimensionCalculatorGPU, HyperVector};

// ============================================================================
// 系统切片 - 用于冻结/解冻
// ============================================================================

/// Tile 系统切片
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileSlice {
    pub id: String,
    pub global_tick: u64,
    pub total_operations: u64,
    pub completed_tile_count: usize,
    /// 内嵌的 Fate 切片
    pub fate_slice: FateSlice,
}

// ============================================================================
// 配置
// ============================================================================

/// 四元瓦片配置 (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileQuattroConfig {
    pub tile_count: usize,
    pub parallel_threshold: f64,
    pub load_balance_enabled: bool,
    /// 小容器容量
    pub container_capacity: usize,
}

impl Default for TileQuattroConfig {
    fn default() -> Self {
        Self {
            tile_count: 4,
            parallel_threshold: 0.5,
            load_balance_enabled: true,
            container_capacity: 100,
        }
    }
}

// ============================================================================
// 小容器 (MicroTile) - 数据流入单元
// ============================================================================

/// 小容器 - 数据流入并局部收敛后成为有效 Tile（无冗余缓冲）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroTile {
    pub id: String,
    pub local_state: LocalTileState,
    pub created_tick: u64,
}

/// 局部 Tile 状态（固定数组，O(1)操作）
/// 收敛由上层 Fate 的 ClusterEquation 控制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalTileState {
    pub signal_count: u64,
    pub sum: f64,
    pub sum_sq: f64,
    /// 环形缓冲（供 Fate 分析）
    pub ring: [f64; 10],
    pub ring_idx: usize,
    pub ring_len: usize,
    pub converged: bool,
    /// 本段材料（入口自由度修复 F3，与 Stem 同构）：字节型整数
    /// 逐字保留供出生单位 content；非字节型不记。
    #[serde(default)]
    pub segment: Vec<u8>,
}

impl LocalTileState {
    pub fn new() -> Self {
        Self {
            signal_count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            ring: [0.0; 10],
            ring_idx: 0,
            ring_len: 0,
            converged: false,
            segment: Vec::new(),
        }
    }

    /// 接收数据（只收集，不判断收敛）
    /// 收敛由上层 Fate 控制
    #[inline]
    pub fn receive(&mut self, value: f64) {
        self.signal_count += 1;
        self.sum += value;
        self.sum_sq += value * value;

        // 环形写入（供 Fate 分析）
        self.ring[self.ring_idx] = value;
        self.ring_idx = (self.ring_idx + 1) % 10;
        if self.ring_len < 10 { self.ring_len += 1; }

        if value == value.trunc() && (0.0..=255.0).contains(&value) {
            self.segment.push(value as u8);
        }
    }
    
    /// 获取最近的差异序列（供 Fate 分析）
    pub fn get_diffs(&self) -> [f64; 3] {
        if self.ring_len < 4 {
            return [0.0, 0.0, 0.0];
        }
        let get = |off: usize| self.ring[(self.ring_idx + 10 - 1 - off) % 10];
        let d1 = (get(0) - get(1)).abs();
        let d2 = (get(1) - get(2)).abs();
        let d3 = (get(2) - get(3)).abs();
        [d1, d2, d3]
    }
    
    /// 由 Fate 设置收敛状态
    pub fn set_converged(&mut self, converged: bool) {
        self.converged = converged;
    }
    
    pub fn last_values(&self) -> Vec<f64> {
        (0..self.ring_len).map(|i| self.ring[(self.ring_idx + 10 - self.ring_len + i) % 10]).collect()
    }
    
    pub fn mean(&self) -> f64 {
        if self.signal_count == 0 { 0.0 } else { self.sum / self.signal_count as f64 }
    }
    
    pub fn variance(&self) -> f64 {
        if self.signal_count == 0 { 0.0 }
        else {
            let mean = self.mean();
            self.sum_sq / self.signal_count as f64 - mean * mean
        }
    }
}

impl Default for LocalTileState {
    fn default() -> Self {
        Self::new()
    }
}

impl MicroTile {
    pub fn new(id: String, _capacity: usize, tick: u64) -> Self {
        Self {
            id,
            local_state: LocalTileState::new(),
            created_tick: tick,
        }
    }
    
    /// 接收数据（只收集，不判断收敛）
    #[inline]
    pub fn receive_data(&mut self, data: f64) {
        self.local_state.receive(data);
    }
    
    /// 获取差异序列（供 Fate 分析）
    pub fn get_diffs(&self) -> [f64; 3] {
        self.local_state.get_diffs()
    }
    
    pub fn is_converged(&self) -> bool {
        self.local_state.converged
    }
    
    /// 由 Fate 设置收敛
    pub fn set_converged(&mut self, converged: bool) {
        self.local_state.set_converged(converged);
    }
}

// ============================================================================
// 四元信号和结果
// ============================================================================

/// 四元信号输入 (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadSignals {
    pub blackhole_distance: f64,
    pub meshbrain_intensity: f64,
    pub multinova_energy: f64,
    pub baseforce_coordination: f64,
}

/// 四元处理结果 (f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadResults {
    pub blackhole_gravity: f64,
    pub meshbrain_intelligence: f64,
    pub multinova_radiation: f64,
    pub baseforce_activity: f64,
    pub combined_output: f64,
}

// ============================================================================
// TileQuattro Complete - 完整的四元瓦片系统
// ============================================================================

/// 完整的四元瓦片系统 (f64)
/// 自带: TripleFate (多层监控) + VoyagerSpace (自动维度) + 系统存储
#[derive(Debug)]
pub struct TileQuattroComplete {
    pub id: String,
    pub config: TileQuattroConfig,
    /// 多层 Fate 系统
    pub triple_fate: TripleFateCombined,
    /// 自动维度空间
    pub space: VoyagerSpaceF64,
    /// 活跃的小容器
    pub active_tiles: Vec<MicroTile>,
    /// 已完成的 Tile 数据
    pub completed_tiles: Vec<CompletedTile>,
    pub total_operations: u64,
    pub global_tick: u64,
    /// 系统存储 - 完整的文件操作权限
    pub storage: Option<SystemStorage>,
    /// Space GPU 高维计算
    pub space_gpu: SpaceGPU,
    /// 四元在瓦片内部。只在 Fate 宣布收敛后走一次。
    pub blackhole: GravityField,
    pub meshbrain: MeshBrainSystem,
    pub multinova: MultinovaProcessor,
    pub baseforce: BaseForceCoordinator,
    /// 最近一次四元：黑洞、MeshBrain、MultiNova、BaseForce。
    pub last_channels: [f64; 4],
    /// 四元入账事件累计与上周期读数（死循环判定用，同 Stem）。
    pub quad_keep_events: u64,
    pub quad_keep_events_at_cycle: u64,
}

/// 已完成的 Tile（神经元节点）
/// 核心宗旨：自发形成类似神经网络的连接
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTile {
    pub id: String,
    pub mean: f64,
    pub variance: f64,
    pub signal_count: u64,
    pub created_tick: u64,
    pub completed_tick: u64,
    /// 数据特征（Tile 的"身份"）- 继承自 MicroTile
    pub data_signature: Vec<f64>,
    /// 能量（用于合并/分裂判断）
    pub energy: f64,
    
    // ═══ 神经网络属性 ═══
    /// 激活值
    pub activation: f64,
    /// 连接权重（自发形成）
    pub weights: Vec<f64>,
    /// 偏置
    pub bias: f64,
    /// 连接到其他 Tile 的 ID（自发形成的连接）
    pub connections: Vec<String>,
    /// 这个容器收口时组织起来的原文。骨架上的数不是原文。
    #[serde(default)]
    pub content: Vec<u8>,
}

fn keep_content(into: &mut Vec<u8>, extra: &[u8]) {
    if extra.is_empty() {
        return;
    }
    if into.len() >= extra.len() && into.windows(extra.len()).any(|window| window == extra) {
        return;
    }
    if !into.is_empty() {
        into.push(b'\n');
    }
    into.extend_from_slice(extra);
}

/// Tile 周期处理结果
#[derive(Debug, Clone)]
pub struct TileCycleResult {
    pub generation: u64,
    pub tile_count: usize,
    pub merges: usize,
    pub splits: usize,
    pub deaths: usize,
    pub avg_activation: f64,
    pub total_connections: usize,
    pub new_connections: usize,
    pub fate_links: usize,
}

/// 前向传播结果
#[derive(Debug, Clone)]
pub struct PropagationResult {
    pub propagated_tiles: usize,
    pub total_activation: f64,
    pub avg_activation: f64,
    pub new_connections: usize,
    pub total_connections: usize,
}

impl TileQuattroComplete {
    pub fn new() -> Self {
        Self::with_config(TileQuattroConfig::default())
    }
    
    pub fn with_config(config: TileQuattroConfig) -> Self {
        Self {
            id: format!("tile_complete_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() % 1_000_000),
            config,
            triple_fate: TripleFateCombined::new(),
            space: VoyagerSpaceF64::new(),
            active_tiles: Vec::new(),
            completed_tiles: Vec::new(),
            total_operations: 0,
            global_tick: 0,
            storage: None,
            space_gpu: SpaceGPU::new(),
            blackhole: GravityField::new(GravityFieldConfig::default()),
            meshbrain: MeshBrainSystem::new(MeshBrainConfig::default()),
            multinova: MultinovaProcessor::new(MultinovaConfig::default()),
            baseforce: BaseForceCoordinator::new(),
            last_channels: [0.0; 4],
            quad_keep_events: 0,
            quad_keep_events_at_cycle: 0,
        }
    }

    /// Fate 已经收下这个均值。四元只看这一次，不按字节累加。返回 BaseForce 合成的这一拍。
    fn quad_on_fate_close(&mut self, mean: f64) -> f64 {
        let distance = mean.abs().max(1e-6);
        let gravity = self.blackhole.calculate_gravity(distance);
        let mesh = self.meshbrain.generate_signal(mean, [0.0, 0.0, 0.0]);
        let nova = self.multinova.process_evolution(mean.abs());
        let mixed = self.baseforce.coordinate(gravity, mesh.intelligence_level, nova.radiation);
        self.last_channels = [gravity, mesh.intelligence_level, nova.radiation, mixed.total_activity];
        mixed.total_activity
    }

    pub fn quad_keep(&mut self, bytes: &[u8]) {
        self.quad_keep_events += 1;
        let gravity = self.blackhole.keep(bytes);
        let mesh = self.meshbrain.keep(bytes);
        let nova = self.multinova.keep(bytes);
        let mixed = self.baseforce.keep(bytes, gravity, mesh, nova);
        self.last_channels = [gravity, mesh, nova, mixed];
    }

    /// 四元内部重构·分解（2026-11-02 用户定向新增）：把第 index
    /// 条在 at 处切开，四份锁步执行——同一条目在四元里是四份
    /// 同文，任一份长度不符即整体拒绝，记账不散。
    pub fn quad_split_entry(&mut self, index: usize, at: usize) -> bool {
        let l = [self.blackhole.stored_get(index).map(|s| s.len()),
                 self.meshbrain.stored_get(index).map(|s| s.len()),
                 self.multinova.stored_get(index).map(|s| s.len()),
                 self.baseforce.stored_get(index).map(|s| s.len())];
        if l.iter().any(|x| *x != l[0]) { return false; }
        let a = self.blackhole.stored_split(index, at);
        let b = self.meshbrain.stored_split(index, at);
        let c = self.multinova.stored_split(index, at);
        let d = self.baseforce.stored_split(index, at);
        a && b && c && d
    }

    /// 四元内部重构·重组（同上，stored_join 的锁步版）。
    pub fn quad_join_entry(&mut self, index: usize) -> bool {
        let a = self.blackhole.stored_join(index);
        let b = self.meshbrain.stored_join(index);
        let c = self.multinova.stored_join(index);
        let d = self.baseforce.stored_join(index);
        a && b && c && d
    }

    /// 四元自主维护（2026-11-02 用户定向新增）：四元各用自己的
    /// 规则整理自家那份存储（黑洞吞视界内条目并塌缩最强条、
    /// 膜脑按涨落带分解/重组、新星爆发超阈条目、底力删逐字
    /// 重复）。周期末调用；四份自此按各自规则演化，入账同文
    /// 不再保证周期后同文（见 QUAD_SELF_REORG_20261102 预
    /// 注册）。只动四元 stored 账本，Fate 不审批。
    pub fn quad_reorganize(&mut self) -> usize {
        // 返回四元内部本趟消亡总数，供 Fate 检测死亡信号。
        self.blackhole.reorganize().removed
            + self.meshbrain.reorganize().removed
            + self.multinova.reorganize().removed
            + self.baseforce.reorganize().removed
    }

    /// 四元看当前容器。容器先留着，数学和 Voyager 还没走。
    pub fn quad_pulse(&mut self) -> Option<(f64, f64, f64)> {
        let tile = self.active_tiles.iter().find(|tile| !tile.is_converged() && tile.local_state.signal_count > 0)?;
        let mean = tile.local_state.mean();
        let variance = tile.local_state.variance();
        let activity = self.quad_on_fate_close(mean);
        Some((mean, variance, activity))
    }

    /// Voyager 看过数学的响应之后，这一拍写成 Tile。
    pub fn commit_pulse(&mut self, mean: f64, variance: f64, activity: f64, math_degree: f64) -> bool {
        let Some(tile_idx) = self.active_tiles.iter().position(|tile| !tile.is_converged() && tile.local_state.signal_count > 0) else {
            return false;
        };
        let tile = self.active_tiles.remove(tile_idx);
        self.space.adaptive_expand(&[mean, variance, activity, math_degree]);
        self.completed_tiles.push(CompletedTile {
            id: tile.id.clone(),
            mean,
            variance,
            signal_count: tile.local_state.signal_count,
            created_tick: tile.created_tick,
            completed_tick: self.global_tick,
            // 出生形态与 receive_data 路一致（入口自由度修复 F2/F3
            // 补齐）：commit_pulse 是 Body 的 ingest+promote 实际出生
            // 路，此前漏改。签名取容器环内 last_values()，content 取
            // 本段材料；注意 Body 的 Tile 侧每条只 ingest 一个条目
            // 均值，环内只有该值、本段材料为空——形态自由度在 Tile
            // 侧受喂入粒度所限，如实记账，不在此放大。
            data_signature: tile.local_state.last_values(),
            energy: mean.abs(),
            activation: 0.0,
            weights: Vec::new(),
            bias: mean,
            connections: Vec::new(),
            content: tile.local_state.segment.clone(),
        });
        self.total_operations += 1;
        true
    }

    /// 容器只攒数。收口由外面那一个 Fate 决定。
    pub fn ingest(&mut self, data: f64) {
        self.global_tick += 1;
        if self.active_tiles.iter().all(|tile| tile.is_converged()) {
            self.active_tiles.push(MicroTile::new(
                format!("tile_{}", self.global_tick),
                self.config.container_capacity,
                self.global_tick,
            ));
        }
        if let Some(tile) = self.active_tiles.iter_mut().find(|tile| !tile.is_converged()) {
            tile.receive_data(data);
        }
    }

    /// 全局 Fate 这一拍收口。四元看当前容器的均值。
    pub fn promote_from_fate(&mut self) -> bool {
        let Some((mean, variance, activity)) = self.quad_pulse() else {
            return false;
        };
        self.commit_pulse(mean, variance, activity, 0.0)
    }

    /// 周期和落盘都跟这一份全局 Fate，不再各写各的方程。
    pub fn follow_global_fate(&mut self, global: &TripleFateCombined) {
        self.triple_fate.global_tick = global.global_tick;
        self.triple_fate.cross_feedback = global.cross_feedback;
        self.triple_fate.cluster_equation = global.cluster_equation.clone();
        self.triple_fate.base.state = global.base.state.clone();
        self.triple_fate.meta.state = global.meta.state.clone();
        self.triple_fate.efficiency_history = global.efficiency_history.clone();
    }
    
    /// 带存储路径初始化
    pub fn with_storage(base_path: &Path) -> std::io::Result<Self> {
        let mut instance = Self::new();
        instance.init_storage(base_path)?;
        Ok(instance)
    }
    
    /// 初始化系统存储
    pub fn init_storage(&mut self, base_path: &Path) -> std::io::Result<()> {
        self.storage = Some(SystemStorage::new(base_path, "tile")?);
        Ok(())
    }
    
    /// 获取存储
    pub fn storage(&self) -> Option<&SystemStorage> {
        self.storage.as_ref()
    }
    
    /// 存储内容
    pub fn store(&self, name: &str, content: &str) -> std::io::Result<()> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .write(name, content)
    }
    
    /// 读取内容
    pub fn load(&self, name: &str) -> std::io::Result<String> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .read(name)
    }
    
    /// 删除内容
    pub fn remove(&self, name: &str) -> std::io::Result<()> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .delete(name)
    }
    
    /// 列出所有
    pub fn list_stored(&self) -> std::io::Result<Vec<String>> {
        self.storage.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Storage not initialized"))?
            .list_all()
    }
    
    /// 接收并存储内容
    pub fn receive_content(&mut self, content: &str) -> std::io::Result<String> {
        let id = format!("data_{}", self.global_tick);
        self.store(&id, content)?;
        let signal = content.len() as f64;
        self.receive_data(signal);
        Ok(id)
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Fate 存储层 - 委托给内部 triple_fate
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// 通过 Fate 存储数据
    pub fn fate_store(&mut self, data: &[u8]) -> String {
        self.triple_fate.store_data(data)
    }
    
    /// 通过 Fate 存储字符串
    pub fn fate_store_string(&mut self, content: &str) -> String {
        self.triple_fate.store_string(content)
    }
    
    /// 系统自主创建数据
    pub fn fate_create(&mut self, data: Vec<u8>) -> String {
        self.triple_fate.create_data(data)
    }
    
    /// 修改 Fate 存储的数据
    pub fn fate_modify(&mut self, id: &str, new_data: Vec<u8>) -> bool {
        self.triple_fate.modify_data(id, new_data)
    }
    
    /// 读取 Fate 存储的数据
    pub fn fate_load(&self, id: &str) -> Option<&fate_f64::FateDataEntry> {
        self.triple_fate.load_data(id)
    }
    
    /// 删除 Fate 存储的数据
    pub fn fate_remove(&mut self, id: &str) -> Option<fate_f64::FateDataEntry> {
        self.triple_fate.remove_data(id)
    }
    
    /// 列出 Fate 存储的所有 ID
    pub fn fate_list(&self) -> Vec<&String> {
        self.triple_fate.list_stored()
    }
    
    /// Fate 存储数量
    pub fn fate_count(&self) -> usize {
        self.triple_fate.stored_count()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 系统切片 - 冻结/解冻 (bincode 二进制)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// 保存系统切片 (.tile 二进制)
    pub fn save_slice(&self, path: &Path) -> std::io::Result<()> {
        let slice = TileSlice {
            id: self.id.clone(),
            global_tick: self.global_tick,
            total_operations: self.total_operations,
            completed_tile_count: self.completed_tiles.len(),
            fate_slice: self.triple_fate.freeze(),
        };
        let bytes = bincode::serialize(&slice)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }
    
    /// 从切片恢复
    pub fn load_slice(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let slice: TileSlice = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        let mut instance = Self::new();
        instance.id = slice.id;
        instance.global_tick = slice.global_tick;
        instance.total_operations = slice.total_operations;
        instance.triple_fate = fate_f64::TripleFateCombined::thaw(slice.fate_slice);
        Ok(instance)
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Space GPU 高维计算
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// 计算维度 (GPU 加速)
    pub fn space_calculate_dimension(&mut self, data: &[f64]) -> valhalla_space_f64_gpu::DimensionResult {
        self.space_gpu.process_and_expand(data)
    }
    
    /// 批量计算维度 (GPU 加速)
    pub fn space_calculate_batch(&mut self, data_batch: &[Vec<f64>]) -> Vec<valhalla_space_f64_gpu::DimensionResult> {
        self.space_gpu.process_batch(data_batch)
    }
    
    /// 发现未知模式
    pub fn space_on_discovery(&mut self, data: &[f64], pattern: &str) -> String {
        self.space_gpu.on_discovery(data, pattern)
    }
    
    /// 获取当前维度
    pub fn space_dimension(&self) -> f64 {
        self.space_gpu.get_dimension()
    }
    
    /// 获取 Space 状态
    pub fn space_status(&self) -> valhalla_space_f64_gpu::SpaceStatus {
        self.space_gpu.get_status()
    }
    
    /// 接收数据流
    /// 收敛由上层 Fate 的 ClusterEquation（二元一次方程）控制
    pub fn receive_data(&mut self, data: f64) -> TileFlowResult {
        self.global_tick += 1;
        let mut new_tiles = 0usize;
        let mut discoveries = Vec::new();
        
        // 找到或创建活跃的 MicroTile
        let tile_idx = if let Some(idx) = self.active_tiles.iter()
            .position(|t| !t.is_converged()) 
        {
            idx
        } else {
            let new_tile = MicroTile::new(
                format!("tile_{}", self.global_tick),
                self.config.container_capacity,
                self.global_tick,
            );
            self.active_tiles.push(new_tile);
            self.active_tiles.len() - 1
        };
        
        // 数据流入（只收集）
        self.active_tiles[tile_idx].receive_data(data);
        
        // ═══ Fate 被动捕捉收敛信号 ═══
        // Fate 用二元一次方程检测边际递减，返回是否收敛
        let converged = self.triple_fate.receive_passive(data);
        
        // 如果 Fate 捕捉到收敛信号
        if converged {
            let tile = self.active_tiles.remove(tile_idx);
            let mean = tile.local_state.mean();
            let variance = tile.local_state.variance();
            let signal_count = tile.local_state.signal_count;
            
            self.quad_on_fate_close(mean);
            let completed = CompletedTile {
                id: tile.id.clone(),
                mean,
                variance,
                signal_count,
                created_tick: tile.created_tick,
                completed_tick: self.global_tick,
                // 出生形态由 Fate 决定（入口自由度修复 F2/F3，
                // 与 Stem 同构）：签名 = Fate 窗口收口时刻的
                // last_values()，content = 本段材料；不再用
                // vec![mean, variance] + 空 content 的写死公式。
                data_signature: tile.local_state.last_values(),
                energy: mean.abs(),
                // 神经网络属性（初始为空，由 Fate 被动填充）
                activation: 0.0,
                weights: Vec::new(),
                bias: mean,
                connections: Vec::new(),  // 空连接，由 Fate 被动建立
                content: tile.local_state.segment.clone(),
            };
            
            self.completed_tiles.push(completed);
            new_tiles = 1;
            
            self.total_operations += 1;
        }
        
        TileFlowResult {
            tick: self.global_tick,
            data_received: data,
            tile_converged: converged,
            new_tiles,
            discoveries,
            active_tiles: self.active_tiles.len(),
            total_completed: self.completed_tiles.len(),
        }
    }
    
    /// 批量接收数据
    pub fn receive_batch(&mut self, data: &[f64]) -> TileBatchResult {
        let mut total_new_tiles = 0usize;
        let mut total_discoveries = Vec::new();
        let mut tiles_converged = 0usize;
        
        for &d in data {
            let result = self.receive_data(d);
            total_new_tiles += result.new_tiles;
            total_discoveries.extend(result.discoveries);
            if result.tile_converged {
                tiles_converged += 1;
            }
        }
        
        TileBatchResult {
            data_count: data.len(),
            new_tiles: total_new_tiles,
            tiles_converged,
            discoveries: total_discoveries,
            total_completed: self.completed_tiles.len(),
            fate_stats: self.triple_fate.stats(),
        }
    }
    
    /// 并行处理四元信号
    pub fn process_quad(&mut self, signals: QuadSignals) -> QuadResults {
        self.total_operations += 1;
        
        // 简化的四元处理
        let blackhole_gravity = 1.0 / (signals.blackhole_distance.abs() + 0.1);
        let meshbrain_intelligence = signals.meshbrain_intensity.tanh();
        let multinova_radiation = if signals.multinova_energy > 0.8 { 
            signals.multinova_energy * 1.5 
        } else { 
            signals.multinova_energy 
        };
        let baseforce_activity = (signals.baseforce_coordination + 1.0) / 2.0;
        
        let combined_output = (blackhole_gravity + meshbrain_intelligence 
                             + multinova_radiation + baseforce_activity) / 4.0;
        
        // 发送信号给 Fate
        let signal_values = vec![
            blackhole_gravity,
            meshbrain_intelligence,
            multinova_radiation,
            baseforce_activity,
        ];
        self.triple_fate.tick(Some(signal_values));
        
        QuadResults {
            blackhole_gravity,
            meshbrain_intelligence,
            multinova_radiation,
            baseforce_activity,
            combined_output,
        }
    }
    
    /// 获取当前维度
    pub fn get_dimension(&self) -> f64 {
        self.space.current_dimension
    }
    
    /// 获取系统状态
    pub fn get_status(&self) -> TileQuattroStatus {
        TileQuattroStatus {
            tile_id: self.id.clone(),
            total_operations: self.total_operations,
            completed_tiles: self.completed_tiles.len(),
            active_tiles: self.active_tiles.len(),
            dimension: self.space.current_dimension,
            fate_tick: self.triple_fate.global_tick,
            base_generation: self.triple_fate.base.generation,
            meta_generation: self.triple_fate.meta.generation,
        }
    }
    
    /// 纯 Tile 周期处理（第2周期+使用）
    /// 使用边际递减合并，与 StemCell 同样逻辑
    pub fn process_tiles_cycle(&mut self) -> TileCycleResult {
        self.total_operations += 1;
        let initial_count = self.completed_tiles.len();
        let mut splits = 0usize;
        let mut merges = 0usize;
        // 死亡判定为活力（见周期内）：四元消亡信号经
        // Fate 检测，无法更新且无法激活者死。
        let mut deaths = 0usize;
        
        // 1. Fate 感知整体状态
        let total_energy: f64 = self.completed_tiles.iter().map(|t| t.energy).sum();
        let avg_energy = if self.completed_tiles.is_empty() { 0.0 } 
                         else { total_energy / self.completed_tiles.len() as f64 };
        let _ = self.triple_fate.receive_passive(avg_energy);
        
        // 2. 合并：只在 Tile 数量较少时执行（避免 O(n log n) 排序）
        if initial_count < 100_000 {
            self.execute_cluster_merge();
            merges = initial_count.saturating_sub(self.completed_tiles.len());
        }
        
        // 3. Tile 演化：能量流转
        let total_energy: f64 = self.completed_tiles.iter().map(|t| t.energy).sum();
        let avg_energy = if self.completed_tiles.is_empty() { 0.0 } 
                         else { total_energy / self.completed_tiles.len() as f64 };
        
        // 分裂数据：能量, 数据特征, 连接（继承父 Tile）
        let mut new_tiles_data: Vec<(f64, Vec<f64>, Vec<String>, Vec<u8>)> = Vec::new();
        
        // 能量流与分裂恢复原版形态（bug 修复，与 Stem
        // 同规）：流率 ×0.1 强回流；分裂须能量 > 1.5×
        // 平均且本周期分裂数 < 10。判定先于 retain 预算。
        for tile in &mut self.completed_tiles {
            let energy_diff = avg_energy - tile.energy;
            tile.energy += energy_diff * 0.1;
        }
        let mut verdicts: Vec<bool> = Vec::with_capacity(self.completed_tiles.len());
        for tile in &self.completed_tiles {
            verdicts.push(tile.energy > avg_energy * 1.5);
        }
        let mut vi = 0usize;
        self.completed_tiles.retain_mut(|tile| {
            let split = verdicts[vi] && new_tiles_data.len() < 10;
            vi += 1;
            if split {
                tile.energy *= 0.5;
                new_tiles_data.push((tile.energy, tile.data_signature.clone(), tile.connections.clone(), tile.content.clone()));
            }
            true
        });
        
        // 添加分裂产生的新 Tile（继承父 Tile 的连接）
        splits = new_tiles_data.len();
        let tick = self.global_tick;
        let op = self.total_operations;
        for (i, (energy, parent_signature, parent_connections, parent_content)) in new_tiles_data.into_iter().enumerate() {
            let new_id = format!("split_{}_{}", op, i);
            self.completed_tiles.push(CompletedTile {
                id: new_id.clone(),
                mean: parent_signature.first().cloned().unwrap_or(0.0),
                variance: 0.0,
                signal_count: 0,
                created_tick: tick,
                completed_tick: tick,
                data_signature: parent_signature,
                energy,
                // 神经网络属性：继承部分连接
                activation: 0.0,
                weights: vec![0.1; parent_connections.len().min(4)],
                bias: 0.0,
                connections: parent_connections.into_iter().take(4).collect(),
                content: parent_content,
            });
        }
        
        // 连接在交互对话时才建立，训练时不建立

        // 4. 前向传播 - 信号沿连接传递
        let prop_result = self.propagate(avg_energy);

        // 四元内部先行维护：消亡数与本周期新进决定死
        // 循环信号——内部资料只减不新进即去活力，经
        // 下方 tick 送 Fate 检测（信号第 5 元，非死
        // 循环时为 0）。
        let quad_dying = self.quad_reorganize();
        let new_intake = self.quad_keep_events - self.quad_keep_events_at_cycle;
        self.quad_keep_events_at_cycle = self.quad_keep_events;
        let dead_loop = quad_dying > 0 && new_intake == 0;

        // 活力判死（用户裁定，与 Stem 同规）：死循环
        // 信号成立时才判；无法更新（出生后 signal_count
        // 仍为 0 且非本拍新生）且无法激活（本轮传播
        // 激活为 0）者死亡。
        if dead_loop {
            let before = self.completed_tiles.len();
            let now = self.global_tick;
            self.completed_tiles.retain(|t| !(t.signal_count == 0 && now > t.created_tick && t.activation == 0.0));
            deaths = before - self.completed_tiles.len();
        }

        // 本周期状态进 Fate 的多层 tick 供 Meta 观察
        // （与 Stem 的 [cells, avg, divisions, deaths] 同构；
        // Tile 侧为 [tiles, avg_energy, splits, deaths]；
        // 第 5 元为四元内部消亡信号，Fate 在此检测）。
        let state_signal = vec![
            self.completed_tiles.len() as f64,
            avg_energy,
            splits as f64,
            deaths as f64,
            if dead_loop { quad_dying as f64 } else { 0.0 },
        ];
        let _ = self.triple_fate.tick(Some(state_signal));

        // 更新维度
        let signal = vec![self.completed_tiles.len() as f64, avg_energy, splits as f64, merges as f64];
        self.space.adaptive_expand(&signal);

        TileCycleResult {
            generation: self.total_operations,
            tile_count: self.completed_tiles.len(),
            merges,
            splits,
            deaths,
            avg_activation: prop_result.avg_activation,
            total_connections: prop_result.total_connections,
            new_connections: prop_result.new_connections,
            fate_links: 0,
        }
    }
    
    /// 执行 Tile 聚合合并 - 边际递减判断相似度
    /// 与 StemCell 同样逻辑：排序 + 边际递减检测 + 删除重复保留差异
    fn execute_cluster_merge(&mut self) {
        let n = self.completed_tiles.len();
        if n < 4 {
            return;
        }
        
        // 1. 按 data_signature 排序（信号特征即身份）
        self.completed_tiles.sort_by(|a, b| {
            let sig_a = a.data_signature.first().unwrap_or(&0.0);
            let sig_b = b.data_signature.first().unwrap_or(&0.0);
            sig_a.partial_cmp(sig_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // 2. 计算相邻 Tile 的差异序列
        let mut diffs: Vec<f64> = Vec::with_capacity(n - 1);
        for i in 1..n {
            let diff = Self::compute_tile_diff(&self.completed_tiles[i - 1], &self.completed_tiles[i]);
            diffs.push(diff);
        }
        
        if diffs.len() < 3 {
            return;
        }
        
        // 3. 用边际递减标记待合并的 Tile
        use std::collections::HashSet;
        let mut to_merge: HashSet<usize> = HashSet::new();
        
        for i in 2..diffs.len() {
            let d0 = diffs[i];
            let d1 = diffs[i - 1];

            // 纯代数判定：相邻差异的比率交给 Fate 的簇方程（check_diffs），
            // 与 Stem 的 try_merge_into_similar 同一条方程路，
            // 不再用写死的 d0<d1<d2 顺序不等式（那对随机间隙也有 1/6 命中）。
            if d0 > 0.0 && self.triple_fate.cluster_equation.check_diffs(&[d0, d1]) {
                if i + 1 < n {
                    to_merge.insert(i + 1);
                }
            }
        }
        
        if to_merge.is_empty() {
            return;
        }
        
        // 4. 合并：删除重复部分，保留不同部分
        let merge_data: Vec<_> = to_merge.iter()
            .filter(|&&idx| idx > 0 && idx < self.completed_tiles.len())
            .map(|&idx| {
                let absorbed = &self.completed_tiles[idx];
                (idx - 1, absorbed.energy, absorbed.signal_count, 
                 absorbed.data_signature.clone(), absorbed.variance, absorbed.content.clone())
            })
            .collect();
        
        for (target_idx, energy, signals, sig, var, content) in merge_data {
            let target = &mut self.completed_tiles[target_idx];
            keep_content(&mut target.content, &content);
            
            // 能量合并
            target.energy += energy;
            target.signal_count += signals;
            
            // 特征合并：保留差异，删除重复
            for (i, &s) in sig.iter().enumerate() {
                if i < target.data_signature.len() {
                    target.data_signature[i] = (target.data_signature[i] + s) / 2.0;
                } else {
                    target.data_signature.push(s);
                }
            }
            
            target.variance = (target.variance + var) / 2.0;
        }

        let mut redirect: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for &idx in &to_merge {
            if idx > 0 && idx < self.completed_tiles.len() {
                redirect.insert(
                    self.completed_tiles[idx].id.clone(),
                    self.completed_tiles[idx - 1].id.clone(),
                );
            }
        }
        let resolve = |redirect: &std::collections::HashMap<String, String>, mut id: String| -> String {
            for _ in 0..8 {
                match redirect.get(&id) {
                    Some(next) if next != &id => id = next.clone(),
                    _ => break,
                }
            }
            id
        };
        let absorbed_edges: Vec<(usize, Vec<(String, f64)>)> = to_merge.iter().filter_map(|&idx| {
            if idx == 0 || idx >= self.completed_tiles.len() {
                return None;
            }
            let tile = &self.completed_tiles[idx];
            let edges = tile.connections.iter().enumerate().map(|(slot, peer)| {
                let weight = tile.weights.get(slot).copied().unwrap_or(0.1);
                (peer.clone(), weight)
            }).collect();
            Some((idx - 1, edges))
        }).collect();
        for (target_idx, edges) in absorbed_edges {
            if target_idx >= self.completed_tiles.len() {
                continue;
            }
            let survivor = resolve(&redirect, self.completed_tiles[target_idx].id.clone());
            let Some(dest) = self.completed_tiles.iter().position(|tile| tile.id == survivor) else {
                continue;
            };
            if to_merge.contains(&dest) {
                continue;
            }
            for (peer, weight) in edges {
                let peer = resolve(&redirect, peer);
                if peer == survivor {
                    continue;
                }
                let target = &mut self.completed_tiles[dest];
                if let Some(slot) = target.connections.iter().position(|id| id == &peer) {
                    if let Some(old) = target.weights.get_mut(slot) {
                        *old = old.abs().max(weight.abs());
                    }
                } else {
                    target.connections.push(peer);
                    target.weights.push(weight);
                }
            }
        }
        for tile in &mut self.completed_tiles {
            let self_id = resolve(&redirect, tile.id.clone());
            let mut kept: Vec<(String, f64)> = Vec::new();
            for (slot, peer) in tile.connections.iter().enumerate() {
                let peer = resolve(&redirect, peer.clone());
                if peer == self_id {
                    continue;
                }
                let weight = tile.weights.get(slot).copied().unwrap_or(0.1);
                if let Some((_, old)) = kept.iter_mut().find(|(id, _)| id == &peer) {
                    *old = old.abs().max(weight.abs());
                } else {
                    kept.push((peer, weight));
                }
            }
            tile.connections = kept.iter().map(|(id, _)| id.clone()).collect();
            tile.weights = kept.iter().map(|(_, weight)| *weight).collect();
        }
        
        // 5. 一次性移除所有标记的 Tile
        let mut current_idx = 0usize;
        self.completed_tiles.retain(|_| {
            let keep = !to_merge.contains(&current_idx);
            current_idx += 1;
            keep
        });
    }
    
    /// 计算两个 Tile 的信号差异
    fn compute_tile_diff(a: &CompletedTile, b: &CompletedTile) -> f64 {
        let mut diff = 0.0;
        
        // data_signature 差异
        let len = a.data_signature.len().min(b.data_signature.len());
        if len > 0 {
            for i in 0..len {
                diff += (a.data_signature[i] - b.data_signature[i]).abs();
            }
            diff /= len as f64;
        }
        
        // variance 差异
        diff += (a.variance - b.variance).abs();
        
        diff
    }
    
    /// ═══ Fate 被动建立连接 ═══
    /// ═══ Fate 被动建立连接 ═══
    /// 在演化周期（交互时）由 Fate 决定建立连接
    /// 高效实现：只处理连接少的 Tile，用相邻比较
    fn fate_passive_connect(&mut self) {
        let n = self.completed_tiles.len();
        if n < 2 {
            return;
        }
        
        // 收集需要建立连接的 Tile 对 (从, 到)
        let mut new_connections: Vec<(usize, usize)> = Vec::new();
        
        // 遍历连接少的 Tile（< 2 个连接）
        for i in 0..n {
            if self.completed_tiles[i].connections.len() >= 2 {
                continue;  // 已有足够连接
            }
            
            // 找相邻的 Tile（已按 data_signature 排序）
            let neighbors = [
                i.saturating_sub(1),
                (i + 1).min(n - 1),
            ];
            
            for &j in &neighbors {
                if i == j {
                    continue;
                }
                
                // 计算差异
                let diff_ij = Self::compute_tile_diff(
                    &self.completed_tiles[i], 
                    &self.completed_tiles[j]
                );
                
                // 用 Fate 方程判断是否连接（check_convergence 只读判定）
                // x = diff / (diff + 1)，避免除零；不再有阈值倍数这类写死放宽
                let (accepted, _) = self.triple_fate.check_convergence(&[diff_ij, diff_ij + 1.0]);
                
                if accepted {
                    new_connections.push((i, j));
                }
            }
        }
        
        // 建立连接（限制每次最多建立 100 个新连接）
        for (i, j) in new_connections.into_iter().take(100) {
            let id_j = self.completed_tiles[j].id.clone();
            let id_i = self.completed_tiles[i].id.clone();
            
            // i -> j
            if !self.completed_tiles[i].connections.contains(&id_j) {
                self.completed_tiles[i].connections.push(id_j);
                self.completed_tiles[i].weights.push(0.1);
            }
            
            // j -> i (双向)
            if !self.completed_tiles[j].connections.contains(&id_i) {
                self.completed_tiles[j].connections.push(id_i);
                self.completed_tiles[j].weights.push(0.1);
            }
        }
    }
    
    /// 查找相似的 Tile（用于自发连接）
    /// 使用边际递减判断相似度
    fn find_similar_tiles(&self, mean: f64, max_count: usize) -> Vec<String> {
        if self.completed_tiles.len() < 4 {
            return self.completed_tiles.iter().map(|t| t.id.clone()).collect();
        }
        
        let new_sig = mean;
        
        // 找到最接近的 Tile
        let mut candidates: Vec<(usize, f64)> = self.completed_tiles.iter()
            .enumerate()
            .filter_map(|(i, t)| {
                t.data_signature.first().map(|&sig| (i, (sig - new_sig).abs()))
            })
            .collect();
        
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        if candidates.len() < 4 {
            return candidates.iter()
                .take(max_count)
                .map(|(i, _)| self.completed_tiles[*i].id.clone())
                .collect();
        }
        
        // 使用动态方程筛选真正相似的（而不是硬编码 d1 < d2 < d3）
        let mut similar = Vec::new();
        
        for i in 0..candidates.len().min(max_count) {
            if i + 1 < candidates.len() {
                let d1 = candidates[i].1;
                let d2 = candidates[i + 1].1;
                
                // 用动态方程判断相似性
                // x = d1 / d2，如果 x ≈ 1（稳定），则相似
                // 不再用写死的简化式 (x-1)²<0.25，直接过身体自己的簇方程
                let x = if d2 > 1e-10 { d1 / d2 } else { 1.0 };
                let result = self.triple_fate.cluster_equation.evaluate(x);
                
                if result.abs() < self.triple_fate.cluster_equation.threshold {  // 方程判定为相似
                    similar.push(self.completed_tiles[candidates[i].0].id.clone());
                }
            } else {
                // 不够做方程检测，直接加入
                similar.push(self.completed_tiles[candidates[i].0].id.clone());
            }
            
            if similar.len() >= max_count {
                break;
            }
        }
        
        similar
    }
    
    /// 前向传播 - 信号沿连接传递
    pub fn propagate(&mut self, input_signal: f64) -> PropagationResult {
        let mut total_activation = 0.0;
        let mut propagated = 0usize;
        let mut new_connections = 0usize;
        
        // 训练时跳过复杂传播（Tile 太多）
        // 只在交互时（Tile < 10万）做完整传播
        if self.completed_tiles.len() > 100_000 {
            // 只计算平均激活
            for tile in &mut self.completed_tiles {
                tile.activation = (input_signal * tile.energy).tanh();
                total_activation += tile.activation;
            }
            
            let total_conns: usize = self.completed_tiles.iter()
                .map(|t| t.connections.len()).sum();
            
            return PropagationResult {
                avg_activation: total_activation / self.completed_tiles.len().max(1) as f64,
                total_activation,
                total_connections: total_conns,
                propagated_tiles: self.completed_tiles.len(),
                new_connections: 0,
            };
        }
        
        // 收集所有 Tile 的激活和连接信息
        let tile_ids: Vec<String> = self.completed_tiles.iter().map(|t| t.id.clone()).collect();
        let activations: std::collections::HashMap<String, f64> = self.completed_tiles.iter()
            .map(|t| (t.id.clone(), t.activation))
            .collect();
        // 全体平均激活（去硬编码后的相对判定基准）
        let mean_act = if activations.is_empty() {
            0.0
        } else {
            activations.values().map(|v| v.abs()).sum::<f64>() / activations.len() as f64
        };

        for tile in &mut self.completed_tiles {
            // 计算输入：来自连接的加权和
            let mut connection_input = 0.0;
            for (i, conn_id) in tile.connections.iter().enumerate() {
                if let Some(&conn_activation) = activations.get(conn_id) {
                    let weight = tile.weights.get(i).copied().unwrap_or(0.1);
                    connection_input += conn_activation * weight;
                }
            }
            
            // 加上外部输入信号
            let raw = input_signal * tile.energy + connection_input + tile.bias;
            
            // 激活函数 (tanh)
            tile.activation = raw.tanh();
            
            // 自适应权重调整（Hebbian-like）
            for (i, conn_id) in tile.connections.iter().enumerate() {
                if let Some(&conn_activation) = activations.get(conn_id) {
                    if i < tile.weights.len() {
                        // 同时激活则增强连接
                        let delta = tile.activation * conn_activation * 0.01;
                        tile.weights[i] = (tile.weights[i] + delta).clamp(-1.0, 1.0);
                    }
                }
            }
            
            // 自发形成新连接：高激活 Tile 尝试连接到其他高激活 Tile
            // （身体自己的激活线，与分裂/死亡的能量线同类——结构变化归
            // Tile/Stem 自身动力学。此前误把激活比率送进簇方程：tanh 饱和
            // 后激活近乎全等，比率≈1 必然过方程，每个 Tile 都连向第一个
            // 对端形成枢纽图，对外 BFS 因此毫无选择性。已按原版恢复。）
            // 去硬编码（2026-11-02 预注册）：删写死激活线 0.5/0.3
            // 与度数上限 8，改相对判定——激活高于全体平均即连。
            if tile.activation.abs() > mean_act {
                for other_id in &tile_ids {
                    if other_id != &tile.id && !tile.connections.contains(other_id) {
                        if let Some(&other_act) = activations.get(other_id) {
                            if other_act.abs() > mean_act {
                                tile.connections.push(other_id.clone());
                                tile.weights.push(0.1);
                                new_connections += 1;
                                break;  // 每次只添加一个新连接
                            }
                        }
                    }
                }
            }
            
            total_activation += tile.activation;
            propagated += 1;
        }
        
        PropagationResult {
            propagated_tiles: propagated,
            total_activation,
            avg_activation: if propagated > 0 { total_activation / propagated as f64 } else { 0.0 },
            new_connections,
            total_connections: self.completed_tiles.iter().map(|t| t.connections.len()).sum(),
        }
    }
    
    /// 获取 Fate 统计
    pub fn get_fate_stats(&self) -> String {
        self.triple_fate.stats()
    }
    
    /// 检查 Fate 是否收敛（被动模式）
    pub fn has_fate_converged(&self) -> bool {
        self.triple_fate.is_passively_converged()
    }
    
    /// 获取效率历史
    pub fn get_efficiency_history(&self) -> &[(f64, f64)] {
        &self.triple_fate.efficiency_history
    }
}

impl Default for TileQuattroComplete {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 结果类型
// ============================================================================

/// Tile 数据流结果
#[derive(Debug, Clone)]
pub struct TileFlowResult {
    pub tick: u64,
    pub data_received: f64,
    pub tile_converged: bool,
    pub new_tiles: usize,
    pub discoveries: Vec<TripleDiscovery>,
    pub active_tiles: usize,
    pub total_completed: usize,
}

/// 批量 Tile 结果
#[derive(Debug, Clone)]
pub struct TileBatchResult {
    pub data_count: usize,
    pub new_tiles: usize,
    pub tiles_converged: usize,
    pub discoveries: Vec<TripleDiscovery>,
    pub total_completed: usize,
    pub fate_stats: String,
}

/// 系统状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileQuattroStatus {
    pub tile_id: String,
    pub total_operations: u64,
    pub completed_tiles: usize,
    pub active_tiles: usize,
    pub dimension: f64,
    pub fate_tick: u64,
    pub base_generation: u64,
    pub meta_generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_micro_tile_convergence() {
        let mut tile = MicroTile::new("test".into(), 100, 0);
        
        for i in 0..20 {
            let value = 10.0 / (i as f64 + 1.0);
            tile.receive_data(value);
        }
        
        println!("Converged: {}", tile.is_converged());
    }
    
    #[test]
    fn test_data_flow() {
        let mut system = TileQuattroComplete::new();
        
        for i in 0..500 {
            let data = (i % 50) as f64 + (i as f64).sin();
            let result = system.receive_data(data);
            
            if result.tile_converged {
                println!("Tile converged at tick {}", result.tick);
            }
        }
        
        let status = system.get_status();
        println!("Final status: {:?}", status);
    }
    
    #[test]
    fn test_quad_processing() {
        let mut system = TileQuattroComplete::new();
        
        let signals = QuadSignals {
            blackhole_distance: 10.0,
            meshbrain_intensity: 0.5,
            multinova_energy: 0.8,
            baseforce_coordination: 0.3,
        };
        
        let results = system.process_quad(signals);
        println!("Results: {:?}", results);
        assert!(results.combined_output > 0.0);
    }
}
