//! Universal signal ingress — Valhalla is not limited to raw byte streams.
//!
//! Accepts text, bytes, f64 series, or JSON payloads and converts them to f64
//! signal sequences under an explicit resource budget before feeding Hub / Tile / StemCell.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{HubConfig, HubRequest, Priority, RequestType, ValhallaHub};
use crate::dl_front::{encode_modality, BodyFrontPorts, DlFrontConfig, ModalityInput};
use crate::triad_viz_types::{signature_hue, TileNeuronViz, TileSynapseViz};
use stem_cell::{CompleteCell, StemCellComplete};
use valhalla_tile_quattro::{CompletedTile, TileQuattroComplete};

// ---------------------------------------------------------------------------
// Resource budget
// ---------------------------------------------------------------------------

/// Hard limits for ingress experiments and production runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressBudget {
    /// Max number of logical items (lines, files, embedding vectors).
    pub max_items: usize,
    /// Total f64 scalars emitted across all items.
    pub max_signals_total: u64,
    /// Cap per single item after encoding.
    pub max_signals_per_item: usize,
    /// When encoding bytes/chars, emit every N-th element (1 = all).
    pub sample_stride: usize,
    /// Hub GPU / CPU resource percent (passed to HubConfig).
    pub resource_percent: u8,
}

impl Default for IngressBudget {
    fn default() -> Self {
        Self {
            max_items: usize::MAX,
            max_signals_total: u64::MAX,
            max_signals_per_item: 4096,
            sample_stride: 1,
            resource_percent: 70,
        }
    }
}

impl IngressBudget {
    /// Tight budget for controlled experiments.
    /// Larger budget for one-shot corpus incubation (all lines → Tile/Stem).
    pub fn incubation() -> Self {
        Self {
            max_items: 512,
            max_signals_total: 512_000,
            max_signals_per_item: 512,
            sample_stride: 1,
            resource_percent: 70,
        }
    }

    pub fn experiment() -> Self {
        Self {
            max_items: 64,
            max_signals_total: 32_768,
            max_signals_per_item: 512,
            sample_stride: 2,
            resource_percent: 70,
        }
    }

    /// Load overrides from environment:
    /// `VALHALLA_INGRESS_MAX_ITEMS`, `MAX_SIGNALS`, `MAX_SIGNALS_PER_ITEM`,
    /// `SAMPLE_STRIDE`, `RESOURCE_PERCENT`.
    pub fn from_env() -> Self {
        let mut b = Self::default();
        if let Ok(v) = std::env::var("VALHALLA_INGRESS_MAX_ITEMS") {
            if let Ok(n) = v.parse::<usize>() {
                b.max_items = n;
            }
        }
        if let Ok(v) = std::env::var("VALHALLA_INGRESS_MAX_SIGNALS") {
            if let Ok(n) = v.parse::<u64>() {
                b.max_signals_total = n;
            }
        }
        if let Ok(v) = std::env::var("VALHALLA_INGRESS_MAX_SIGNALS_PER_ITEM") {
            if let Ok(n) = v.parse::<usize>() {
                b.max_signals_per_item = n;
            }
        }
        if let Ok(v) = std::env::var("VALHALLA_INGRESS_SAMPLE_STRIDE") {
            if let Ok(n) = v.parse::<usize>() {
                b.sample_stride = n.max(1);
            }
        }
        if let Ok(v) = std::env::var("VALHALLA_INGRESS_RESOURCE_PERCENT") {
            if let Ok(n) = v.parse::<u8>() {
                b.resource_percent = n.clamp(1, 100);
            }
        }
        b
    }

    pub fn can_emit(&self, stats: &IngressStats, count: usize) -> bool {
        stats.items_processed < self.max_items
            && stats.signals_emitted + count as u64 <= self.max_signals_total
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngressStats {
    pub items_processed: usize,
    pub items_skipped: usize,
    pub signals_emitted: u64,
    pub hub_requests: u64,
    pub tile_ops: u64,
    pub stem_ops: u64,
    pub budget_exhausted: bool,
}

// ---------------------------------------------------------------------------
// Payload & encoding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IngressPayload {
    Text { content: String },
    Bytes { data: Vec<u8> },
    F64 { values: Vec<f64> },
    /// Arbitrary JSON — flattened numeric leaves to f64 sequence.
    Json { value: serde_json::Value },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingMode {
    /// One scalar per char (Unicode code point), stride-sampled.
    Char,
    /// Raw byte stream with stride sampling.
    ByteSample,
    /// Sentence-level statistics (5 dims) + optional char tail.
    TextStats,
    /// Use f64 values directly (clamped to finite).
    DirectF64,
    /// Map each f64 to [-1, 1] via tanh-like clamp: x.clamp(-1, 1).
    NormalizedF64,
    /// Pick encoder from payload kind.
    Auto,
}

impl EncodingMode {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "char" => Self::Char,
            "byte" | "byte_sample" => Self::ByteSample,
            "stats" | "text_stats" => Self::TextStats,
            "f64" | "direct_f64" => Self::DirectF64,
            "norm" | "normalized_f64" => Self::NormalizedF64,
            _ => Self::Auto,
        }
    }
}

/// Encode any payload into an f64 signal sequence, respecting per-item cap and stride.
pub fn encode_payload(
    payload: &IngressPayload,
    mode: EncodingMode,
    budget: &IngressBudget,
) -> Vec<f64> {
    let mode = if mode == EncodingMode::Auto {
        match payload {
            IngressPayload::F64 { .. } => EncodingMode::DirectF64,
            IngressPayload::Bytes { .. } => EncodingMode::ByteSample,
            IngressPayload::Json { .. } => EncodingMode::DirectF64,
            IngressPayload::Text { .. } => EncodingMode::Char,
        }
    } else {
        mode
    };

    let mut raw = match (payload, mode) {
        (IngressPayload::F64 { values }, _) => values.clone(),
        (IngressPayload::Json { value }, _) => flatten_json_numbers(value),
        (IngressPayload::Text { content }, EncodingMode::TextStats) => text_stats_vector(content),
        (IngressPayload::Text { content }, EncodingMode::ByteSample) => content
            .bytes()
            .map(|b| b as f64)
            .collect(),
        (IngressPayload::Text { content }, _) => content
            .chars()
            .map(|c| c as u32 as f64)
            .collect(),
        (IngressPayload::Bytes { data }, EncodingMode::TextStats) => {
            let lossy = String::from_utf8_lossy(data);
            text_stats_vector(&lossy)
        }
        (IngressPayload::Bytes { data }, _) => data.iter().map(|&b| b as f64).collect(),
    };

    if mode == EncodingMode::NormalizedF64 {
        for v in &mut raw {
            *v = v.clamp(-1.0, 1.0);
        }
    } else if mode == EncodingMode::Char {
        for v in &mut raw {
            *v = (*v / 65536.0).clamp(0.0, 1.0);
        }
    } else if mode == EncodingMode::ByteSample {
        for v in &mut raw {
            *v = (*v / 255.0).clamp(0.0, 1.0);
        }
    } else {
        for v in &mut raw {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }

    let stride = budget.sample_stride.max(1);
    if stride > 1 {
        raw = raw.into_iter().step_by(stride).collect();
    }

    if raw.len() > budget.max_signals_per_item {
        raw.truncate(budget.max_signals_per_item);
    }
    raw
}

fn text_stats_vector(text: &str) -> Vec<f64> {
    let len = text.len() as f64;
    let words = text.split_whitespace().count() as f64;
    let chars: Vec<char> = text.chars().collect();
    let avg_word_len = if words > 0.0 { len / words } else { 0.0 };
    let punctuation = chars.iter().filter(|c| c.is_ascii_punctuation()).count() as f64;
    let uppercase = chars.iter().filter(|c| c.is_uppercase()).count() as f64;
    vec![
        len / 100.0,
        words / 20.0,
        avg_word_len / 10.0,
        punctuation / 10.0,
        uppercase / 10.0,
    ]
}

fn flatten_json_numbers(value: &serde_json::Value) -> Vec<f64> {
    let mut out = Vec::new();
    flatten_json_rec(value, &mut out);
    out
}

fn flatten_json_rec(value: &serde_json::Value, out: &mut Vec<f64>) {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.push(f);
            }
        }
        serde_json::Value::Array(a) => {
            for v in a {
                flatten_json_rec(v, out);
            }
        }
        serde_json::Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for k in keys {
                if let Some(v) = map.get(k) {
                    flatten_json_rec(v, out);
                }
            }
        }
        _ => {}
    }
}

/// Which Valhalla body receives signals (single-system or triad).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyTarget {
    Triad,
    Hub,
    Tile,
    StemCell,
}

impl BodyTarget {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "hub" => Self::Hub,
            "tile" => Self::Tile,
            "stem" | "stemcell" | "stem_cell" => Self::StemCell,
            _ => Self::Triad,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriadFeedReport {
    pub mode: String,
    pub stats: IngressStats,
    pub hub_signals: u64,
    pub hub_routed: [u64; 4],
    pub hub_prefs: Vec<f64>,
    pub tile_completed: usize,
    pub tile_ops: u64,
    pub tile_dim: f64,
    pub stem_cells: usize,
    pub stem_connections: usize,
    pub stem_dim: f64,
    /// Extractable patch for direct model modification (Python/HF bridge).
    pub model_patch: ValhallaModelPatch,
    /// Multi-round Tile/StemCell optimization cycles executed.
    pub optimize_cycles: usize,
    /// Per-cycle structure snapshot after each process cycle.
    pub cycle_trace: Vec<CycleSnapshot>,
    /// MCQ Fate decompose/aggregate trace (`VALHALLA_MCQ_FATE_CYCLES=1`).
    #[serde(default)]
    pub mcq_fate_trace: Vec<crate::mcq_fate::McqFateCycleSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleSnapshot {
    pub cycle: usize,
    pub tile_completed: usize,
    pub tile_merges: u64,
    pub tile_splits: u64,
    pub tile_dim: f64,
    pub stem_cells: usize,
    pub stem_connections: usize,
    pub stem_merges: u64,
    pub stem_divisions: u64,
    pub stem_dim: f64,
}

/// Structure exported from Valhalla triad run — drives direct LLM weight / activation edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValhallaModelPatch {
    /// hub | tile | stemcell | triad
    pub body_source: String,
    pub fate_affinities: Vec<f64>,
    pub hub_routed: [u64; 4],
    /// Per-quad multipliers (~1.0) applied to transformer layer groups.
    pub quad_layer_scales: [f64; 4],
    pub tile_signatures: Vec<Vec<f64>>,
    pub stem_signatures: Vec<Vec<f64>>,
    /// Normalized f64 vector fused from active body structure.
    pub patch_vector: Vec<f64>,
    pub signals_emitted: u64,
    pub tile_count: usize,
    pub stem_cell_count: usize,
    /// Per-tile activation after neural propagate (aligned with tile_signatures order).
    #[serde(default)]
    pub tile_activations: Vec<f64>,
    #[serde(default)]
    pub tile_connection_count: usize,
    #[serde(default)]
    pub tile_avg_activation: f64,
    #[serde(default)]
    pub stem_organ_clusters: Vec<StemOrganCluster>,
    /// Averaged data_signature per organ cluster (aligned with stem_organ_clusters).
    #[serde(default)]
    pub stem_cluster_signatures: Vec<Vec<f64>>,
    #[serde(default)]
    pub stem_cluster_energies: Vec<f64>,
    /// DL / grinder encoder label when ingress used transformer front.
    #[serde(default)]
    pub ingress_encoder: String,
    /// Per-tile neural identity + internal weights (dashboard).
    #[serde(default)]
    pub tile_neurons: Vec<TileNeuronViz>,
    #[serde(default)]
    pub tile_synapses: Vec<TileSynapseViz>,
}

pub const PATCH_VECTOR_DIM: usize = 256;

/// High-connection cell cluster — runtime "organ" for export/decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemOrganCluster {
    pub cluster_id: usize,
    pub cell_ids: Vec<String>,
    pub cell_count: usize,
    pub avg_energy: f64,
    pub internal_edges: usize,
}

const VIZ_MAX_TILES: usize = 64;
const VIZ_WEIGHT_PREVIEW: usize = 8;

fn build_tile_neurons(tile: &TileQuattroComplete) -> Vec<TileNeuronViz> {
    tile.completed_tiles
        .iter()
        .take(VIZ_MAX_TILES)
        .enumerate()
        .map(|(i, t)| TileNeuronViz {
            index: i,
            id: t.id.clone(),
            activation: t.activation,
            bias: t.bias,
            weights: t.weights.iter().take(VIZ_WEIGHT_PREVIEW).copied().collect(),
            hue: signature_hue(&t.data_signature),
            connection_count: t.connections.len(),
        })
        .collect()
}

fn build_tile_synapses(tile: &TileQuattroComplete) -> Vec<TileSynapseViz> {
    let id_to_idx: HashMap<String, usize> = tile
        .completed_tiles
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.clone(), i))
        .collect();
    let mut edges = Vec::new();
    for (i, t) in tile.completed_tiles.iter().enumerate().take(VIZ_MAX_TILES) {
        for (wi, conn_id) in t.connections.iter().enumerate() {
            if let Some(&j) = id_to_idx.get(conn_id) {
                if i != j {
                    edges.push(TileSynapseViz {
                        from: i,
                        to: j,
                        weight: t.weights.get(wi).copied().unwrap_or(0.5),
                    });
                }
            }
        }
    }
    edges
}

fn stem_cluster_signature(stem: &StemCellComplete, members: &[String]) -> Vec<f64> {
    if members.is_empty() {
        return Vec::new();
    }
    let mut sig: Vec<f64> = Vec::new();
    let mut count = 0usize;
    for id in members {
        if let Some(c) = stem.cells.iter().find(|c| c.cell_id == *id) {
            count += 1;
            for (i, v) in c.data_signature.iter().enumerate() {
                if i >= sig.len() {
                    sig.push(*v);
                } else {
                    sig[i] += *v;
                }
            }
        }
    }
    if count > 0 {
        let n = count as f64;
        for x in &mut sig {
            *x /= n;
        }
    }
    sig
}

fn build_stem_organ_clusters(stem: &StemCellComplete, min_cells: usize) -> Vec<StemOrganCluster> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut adj: HashMap<String, HashSet<String>> = HashMap::new();
    let mut energy_map: HashMap<String, f64> = HashMap::new();

    for c in &stem.cells {
        energy_map.insert(c.cell_id.clone(), c.energy);
        adj.entry(c.cell_id.clone()).or_default();
        for (peer, strength) in &c.connections {
            if *strength > 0.01 {
                adj.get_mut(&c.cell_id).unwrap().insert(peer.clone());
                adj.entry(peer.clone()).or_default().insert(c.cell_id.clone());
            }
        }
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut clusters: Vec<StemOrganCluster> = Vec::new();
    let mut cluster_id = 0usize;

    for start in adj.keys() {
        if seen.contains(start) {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start.clone());
        seen.insert(start.clone());
        let mut members: Vec<String> = Vec::new();

        while let Some(id) = queue.pop_front() {
            members.push(id.clone());
            if let Some(neighbors) = adj.get(&id) {
                for n in neighbors {
                    if !seen.contains(n) {
                        seen.insert(n.clone());
                        queue.push_back(n.clone());
                    }
                }
            }
        }

        if members.len() < min_cells {
            continue;
        }

        let mut internal_edges = 0usize;
        for a in &members {
            if let Some(neighbors) = adj.get(a) {
                for b in neighbors {
                    if members.contains(b) {
                        internal_edges += 1;
                    }
                }
            }
        }
        internal_edges /= 2;

        let avg_energy = if members.is_empty() {
            0.0
        } else {
            members
                .iter()
                .filter_map(|id| energy_map.get(id))
                .sum::<f64>()
                / members.len() as f64
        };

        clusters.push(StemOrganCluster {
            cluster_id,
            cell_count: members.len(),
            cell_ids: members,
            avg_energy,
            internal_edges,
        });
        cluster_id += 1;
    }

    clusters.sort_by(|a, b| b.internal_edges.cmp(&a.internal_edges));
    clusters
}

/// Fuse structure from the active body into a patch vector for model surgery.
pub fn build_model_patch(
    body: BodyTarget,
    hub_prefs: &[f64],
    hub_routed: [u64; 4],
    tile: &TileQuattroComplete,
    stem: &StemCellComplete,
    stats: &IngressStats,
) -> ValhallaModelPatch {
    let tile_signatures: Vec<Vec<f64>> = if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
        tile
            .completed_tiles
            .iter()
            .map(|t| t.data_signature.clone())
            .collect()
    } else {
        Vec::new()
    };

    let stem_signatures: Vec<Vec<f64>> = if matches!(body, BodyTarget::StemCell | BodyTarget::Triad) {
        let mut stem_cells_sorted: Vec<_> = stem.cells.iter().collect();
        stem_cells_sorted.sort_by(|a, b| {
            b.energy
                .partial_cmp(&a.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stem_cells_sorted
            .into_iter()
            .take(32)
            .map(|c| c.data_signature.clone())
            .collect()
    } else {
        Vec::new()
    };

    let mut patch_vector = vec![0.0f64; PATCH_VECTOR_DIM];
    let mut idx = 0usize;

    match body {
        BodyTarget::Hub | BodyTarget::Triad => {
            let total = hub_routed.iter().sum::<u64>().max(1) as f64;
            for i in 0..4 {
                patch_vector[i] = hub_routed[i] as f64 / total;
                if let Some(&f) = hub_prefs.get(i) {
                    if f.is_finite() {
                        patch_vector[32 + i] = f;
                    }
                }
            }
        }
        _ => {}
    }

    if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
        for sig in &tile_signatures {
            for &x in sig {
                if x.is_finite() {
                    patch_vector[idx % PATCH_VECTOR_DIM] += x;
                    idx += 1;
                }
            }
        }
    }

    if matches!(body, BodyTarget::StemCell | BodyTarget::Triad) {
        for sig in &stem_signatures {
            for &x in sig {
                if x.is_finite() {
                    patch_vector[idx % PATCH_VECTOR_DIM] += x * 0.5;
                    idx += 1;
                }
            }
        }
    }

    if body == BodyTarget::Triad {
        for (i, &f) in hub_prefs.iter().take(4).enumerate() {
            if f.is_finite() {
                patch_vector[i] += f;
                patch_vector[PATCH_VECTOR_DIM - 4 + i] += f;
            }
        }
    }

    let norm: f64 = patch_vector.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
    for x in &mut patch_vector {
        *x /= norm;
    }

    let mut quad_layer_scales = [1.0f64; 4];
    match body {
        BodyTarget::Hub | BodyTarget::Triad if hub_prefs.len() >= 4 => {
            let mean = hub_prefs.iter().take(4).sum::<f64>() / 4.0;
            for (i, &p) in hub_prefs.iter().take(4).enumerate() {
                quad_layer_scales[i] = 1.0 + 0.05 * (p - mean);
            }
        }
        BodyTarget::Tile if !tile_signatures.is_empty() => {
            let means: Vec<f64> = tile_signatures
                .iter()
                .filter_map(|s| s.first().copied())
                .collect();
            if !means.is_empty() {
                let m = means.iter().sum::<f64>() / means.len() as f64;
                for (i, scale) in quad_layer_scales.iter_mut().enumerate() {
                    let v = means.get(i % means.len()).copied().unwrap_or(m);
                    *scale = 1.0 + 0.03 * (v - m);
                }
            }
        }
        BodyTarget::StemCell if !stem_signatures.is_empty() => {
            let energies: Vec<f64> = stem.cells.iter().map(|c| c.energy).collect();
            let max_e = energies.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
            for (i, scale) in quad_layer_scales.iter_mut().enumerate() {
                let e = energies.get(i % energies.len()).copied().unwrap_or(0.0);
                *scale = 1.0 + 0.04 * (e / max_e - 0.5);
            }
        }
        _ => {}
    }

    let tile_activations: Vec<f64> = if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
        tile.completed_tiles.iter().map(|t| t.activation).collect()
    } else {
        Vec::new()
    };
    let tile_connection_count: usize = if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
        tile.completed_tiles.iter().map(|t| t.connections.len()).sum()
    } else {
        0
    };
    let tile_avg_activation = if tile_activations.is_empty() {
        0.0
    } else {
        tile_activations.iter().sum::<f64>() / tile_activations.len() as f64
    };

    let stem_organ_clusters = if matches!(body, BodyTarget::StemCell | BodyTarget::Triad) {
        let min_cells = std::env::var("VALHALLA_STEM_ORGAN_MIN_CELLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        build_stem_organ_clusters(stem, min_cells)
    } else {
        Vec::new()
    };
    let stem_cluster_signatures: Vec<Vec<f64>> = stem_organ_clusters
        .iter()
        .map(|c| stem_cluster_signature(stem, &c.cell_ids))
        .collect();
    let stem_cluster_energies: Vec<f64> = stem_organ_clusters.iter().map(|c| c.avg_energy).collect();

    // Enrich tile signatures with post-propagate activation + connectivity
    let tile_signatures_enriched: Vec<Vec<f64>> = if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
        tile.completed_tiles
            .iter()
            .map(|t| {
                let conn_norm = t.connections.len() as f64 / 8.0;
                vec![
                    t.data_signature.first().copied().unwrap_or(t.mean),
                    t.data_signature.get(1).copied().unwrap_or(t.variance),
                    t.activation,
                    conn_norm,
                ]
            })
            .collect()
    } else {
        tile_signatures
    };

    ValhallaModelPatch {
        body_source: format!("{:?}", body).to_ascii_lowercase(),
        fate_affinities: hub_prefs.to_vec(),
        hub_routed,
        quad_layer_scales,
        tile_signatures: tile_signatures_enriched,
        stem_signatures,
        patch_vector,
        signals_emitted: stats.signals_emitted,
        tile_count: if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
            tile.completed_tiles.len()
        } else {
            0
        },
        stem_cell_count: if matches!(body, BodyTarget::StemCell | BodyTarget::Triad) {
            stem.cells.len()
        } else {
            0
        },
        tile_activations,
        tile_connection_count,
        tile_avg_activation,
        stem_organ_clusters,
        stem_cluster_signatures,
        stem_cluster_energies,
        ingress_encoder: String::new(),
        tile_neurons: if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
            build_tile_neurons(tile)
        } else {
            Vec::new()
        },
        tile_synapses: if matches!(body, BodyTarget::Tile | BodyTarget::Triad) {
            build_tile_synapses(tile)
        } else {
            Vec::new()
        },
    }
}

/// Frozen Tile/Stem structure after full-corpus incubation (fork per QA question).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriadIncubationSnapshot {
    pub completed_tiles: Vec<CompletedTile>,
    pub stem_cells: Vec<CompleteCell>,
    pub tile_total_operations: u64,
    pub hub_routed: [u64; 4],
    pub ingress_encoder: String,
    pub hub_prefs: Vec<f64>,
    #[serde(default = "default_fate_weight_mode_str")]
    pub fate_weight_mode: String,
    #[serde(default)]
    pub fate_routing_free: bool,
    #[serde(default = "default_fate_ingress_routing_str")]
    pub fate_ingress_routing: String,
    /// Grinder-signal affinity per corpus line index (Fate QA feedback; structure-derived).
    #[serde(default)]
    pub corpus_line_affinity: Vec<f64>,
    /// Fate logical time (signal_tick) after incubation / QA — proactive time unit.
    #[serde(default)]
    pub fate_signal_tick: u64,
    /// Manifold protocol tag — `valhalla-manifold-v1` when full Fate state is present.
    #[serde(default)]
    pub manifold_protocol: Option<String>,
    /// Fused patch vector (256-d) at export time.
    #[serde(default)]
    pub patch_vector: Vec<f64>,
    /// Full Fate convergence history (marginal stop survives fork).
    #[serde(default)]
    pub fate_convergence: Option<crate::FateConvergenceState>,
    /// Hub triple_fate storage slice.
    #[serde(default)]
    pub hub_triple_fate: Option<crate::FateSlice>,
    /// Composable structure fragments (modality / shard / corpus tracks).
    #[serde(default)]
    pub fragments: Vec<crate::fate_orchestrator::SnapshotFragment>,
}

fn default_fate_weight_mode_str() -> String {
    "fixed".into()
}

fn default_fate_ingress_routing_str() -> String {
    "quad_cycle".into()
}

/// Env override: if set, scale by align; else use E7 algebraic fn(align).
fn affinity_pos_from_align(align: f64) -> f64 {
    crate::fate_adaptive_algebra::FateAlgebraState::ambient().affinity_pos_delta(align)
}

fn affinity_neg_from_align(align: f64) -> f64 {
    crate::fate_adaptive_algebra::FateAlgebraState::ambient().affinity_neg_delta(align)
}

fn env_f64_override(env_key: &str, align: f64, algebraic: fn(f64) -> f64) -> f64 {
    std::env::var(env_key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|scale| scale * align)
        .unwrap_or_else(|| algebraic(align))
}

fn signal_cosine_local(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

pub struct TriadSession {
    pub body: BodyTarget,
    pub hub: ValhallaHub,
    pub tile: TileQuattroComplete,
    pub stem: StemCellComplete,
    pub budget: IngressBudget,
    pub stats: IngressStats,
    hub_routed: [u64; 4],
    ingress_encoder: String,
    /// Parallel to corpus line indices; persisted in incubation snapshot.
    corpus_line_affinity: Vec<f64>,
    /// Per-line / per-modality fragment tracks (manifold compose).
    pub fragment_tracks: Vec<crate::fate_orchestrator::SnapshotFragment>,
}

impl TriadSession {
    pub fn new(budget: IngressBudget) -> Self {
        Self::with_body(budget, BodyTarget::Triad)
    }

    pub fn with_body(budget: IngressBudget, body: BodyTarget) -> Self {
        let hub = ValhallaHub::with_config(HubConfig {
            resource_percent: budget.resource_percent,
            ..Default::default()
        });
        Self {
            body,
            hub,
            tile: TileQuattroComplete::new(),
            stem: StemCellComplete::new(),
            budget,
            stats: IngressStats::default(),
            hub_routed: [0; 4],
            ingress_encoder: String::new(),
            corpus_line_affinity: Vec::new(),
            fragment_tracks: Vec::new(),
        }
    }

    /// Record current session state as a composable fragment track.
    pub fn record_fragment(&mut self, id: &str, label: &str) {
        self.record_fragment_with_meta(id, label, crate::fate_orchestrator::FragmentCloseMeta::default());
    }

    pub fn record_fragment_with_meta(
        &mut self,
        id: &str,
        label: &str,
        meta: crate::fate_orchestrator::FragmentCloseMeta,
    ) {
        self.fragment_tracks
            .push(crate::fate_orchestrator::export_fragment(self, id, label, meta));
    }

    /// Feed raw payload through legacy char/f64 encoder (unchanged path).
    pub fn feed(&mut self, payload: IngressPayload, mode: EncodingMode) -> bool {
        if self.stats.budget_exhausted {
            return false;
        }
        if self.stats.items_processed >= self.budget.max_items {
            self.stats.budget_exhausted = true;
            return false;
        }

        let signals = encode_payload(&payload, mode, &self.budget);
        if signals.is_empty() {
            self.stats.items_skipped += 1;
            return true;
        }

        let remaining = self.budget.max_signals_total.saturating_sub(self.stats.signals_emitted);
        if remaining == 0 {
            self.stats.budget_exhausted = true;
            return false;
        }

        let take = signals.len().min(remaining as usize);
        let slice = &signals[..take];
        if take < signals.len() {
            self.stats.budget_exhausted = true;
        }

        self.emit_signals(slice);
        self.stats.items_processed += 1;
        !self.stats.budget_exhausted
    }

    /// Feed via DL front-end → three body ports (Hub / Tile / StemCell) before ingress.
    pub fn feed_modality(&mut self, input: ModalityInput, front_cfg: &DlFrontConfig) -> bool {
        if self.stats.budget_exhausted {
            return false;
        }
        if self.stats.items_processed >= self.budget.max_items {
            self.stats.budget_exhausted = true;
            return false;
        }

        let ports = encode_modality(&input, front_cfg);
        self.emit_body_ports(&ports);
        self.stats.items_processed += 1;
        !self.stats.budget_exhausted
    }

    /// Emit body-specific port signals (DL front output).
    pub fn emit_body_ports(&mut self, ports: &BodyFrontPorts) {
        self.emit_body_ports_with_line(ports, None);
    }

    /// Emit body ports with optional corpus line for semantic Hub routing.
    pub fn emit_body_ports_with_line(&mut self, ports: &BodyFrontPorts, line: Option<&str>) {
        self.hub
            .set_ingress_line(line.map(|s| s.trim().to_string()));
        self.emit_body_ports_inner(ports);
        self.hub.set_ingress_line(None);
    }

    fn emit_body_ports_inner(&mut self, ports: &BodyFrontPorts) {
        if !ports.encoder.is_empty() {
            self.ingress_encoder = ports.encoder.clone();
        }
        let feed_hub = matches!(self.body, BodyTarget::Hub | BodyTarget::Triad);
        let feed_tile = matches!(self.body, BodyTarget::Tile | BodyTarget::Triad);
        let feed_stem = matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad);

        if feed_hub && !ports.hub.is_empty() {
            let hub_cap = ports.hub.len().min(32);
            self.emit_signals_on_port(&ports.hub[..hub_cap], true, false, false);
        }
        // Transformer / DL ports: batch ingress preserves vector structure for Tile/Stem.
        if feed_tile && !ports.tile.is_empty() {
            let _ = self.tile.receive_batch(&ports.tile);
            let n = ports.tile.len() as u64;
            self.stats.tile_ops += n;
            self.stats.signals_emitted += n;
            if self.stats.signals_emitted >= self.budget.max_signals_total {
                self.stats.budget_exhausted = true;
            }
        }
        if feed_stem && !ports.stem.is_empty() {
            let _ = self.stem.receive_batch(&ports.stem);
            let n = ports.stem.len() as u64;
            self.stats.stem_ops += n;
            self.stats.signals_emitted += n;
            if self.stats.signals_emitted >= self.budget.max_signals_total {
                self.stats.budget_exhausted = true;
            }
        }
    }

    fn emit_signals_on_port(&mut self, signals: &[f64], to_hub: bool, to_tile: bool, to_stem: bool) {
        let remaining = self.budget.max_signals_total.saturating_sub(self.stats.signals_emitted);
        if remaining == 0 {
            self.stats.budget_exhausted = true;
            return;
        }
        let take = signals.len().min(remaining as usize);
        if take < signals.len() {
            self.stats.budget_exhausted = true;
        }
        let slice = &signals[..take];
        self.emit_signals_filtered(slice, to_hub, to_tile, to_stem);
    }

    fn emit_signals(&mut self, signals: &[f64]) {
        let feed_hub = matches!(self.body, BodyTarget::Hub | BodyTarget::Triad);
        let feed_tile = matches!(self.body, BodyTarget::Tile | BodyTarget::Triad);
        let feed_stem = matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad);
        self.emit_signals_filtered(signals, feed_hub, feed_tile, feed_stem);
    }

    fn emit_signals_filtered(
        &mut self,
        signals: &[f64],
        feed_hub: bool,
        feed_tile: bool,
        feed_stem: bool,
    ) {
        for (i, &v) in signals.iter().enumerate() {
            if feed_hub {
                let hub_intensity = if v.is_finite() {
                    v.clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let resp = self.hub.process(HubRequest {
                    request_id: format!("ing_{}", self.stats.hub_requests),
                    request_type: RequestType::Learning,
                    intensity: hub_intensity,
                    data: None,
                    priority: Priority::Normal,
                });
                let idx = route_index(&resp.message);
                self.hub_routed[idx] += 1;
                self.stats.hub_requests += 1;
            }

            if feed_tile {
                self.tile.receive_data(v);
                self.stats.tile_ops += 1;
            }

            if feed_stem {
                let stem_v = if v.abs() <= 1.0 {
                    v
                } else {
                    (v / 65536.0).clamp(-1.0, 1.0)
                };
                self.stem.receive_data(stem_v);
                self.stats.stem_ops += 1;

                if i > 0 && i % 64 == 0 {
                    self.stem.receive_data(0.0);
                }
            }

            self.stats.signals_emitted += 1;
        }
        if feed_stem {
            self.stem.receive_data(0.0);
        }
    }

    pub fn finalize(mut self, mode_label: &str) -> TriadFeedReport {
        self.finalize_with_cycles(mode_label, 1)
    }

    /// Run Tile/Stem optimize cycles without exporting patch (corpus incubation phase).
    pub fn run_optimize_cycles(&mut self, optimize_cycles: usize) {
        let cycles = optimize_cycles.max(1);
        if matches!(self.body, BodyTarget::Hub | BodyTarget::Triad) {
            self.hub.mark_epoch_complete();
        }
        let run_tile = matches!(self.body, BodyTarget::Tile | BodyTarget::Triad);
        let run_stem = matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad);

        for _ in 0..cycles {
            if run_tile {
                self.tile.process_tiles_cycle();
                if self.tile.has_fate_converged() {
                    break;
                }
            }
            if run_stem {
                self.stem.process_cells_cycle();
                let _ = self.stem.evolve(0.5);
                if self.stem.cells.len() >= 4 {
                    let _ = self.stem.evolve(0.35);
                }
                if self.stem.has_fate_converged() {
                    break;
                }
            }
        }
    }

    /// Emit DL ports to one body (compose: Stem text / Tile audio / Hub fate).
    pub fn emit_ports_target(&mut self, ports: &BodyFrontPorts, target: BodyTarget) {
        if !ports.encoder.is_empty() {
            self.ingress_encoder = ports.encoder.clone();
        }
        match target {
            BodyTarget::Hub => {
                if !ports.hub.is_empty() {
                    let hub_cap = ports.hub.len().min(32);
                    self.emit_signals_on_port(&ports.hub[..hub_cap], true, false, false);
                }
            }
            BodyTarget::Tile => {
                if !ports.tile.is_empty() {
                    let _ = self.tile.receive_batch(&ports.tile);
                    let n = ports.tile.len() as u64;
                    self.stats.tile_ops += n;
                    self.stats.signals_emitted += n;
                    if self.stats.signals_emitted >= self.budget.max_signals_total {
                        self.stats.budget_exhausted = true;
                    }
                }
            }
            BodyTarget::StemCell => {
                if !ports.stem.is_empty() {
                    let _ = self.stem.receive_batch(&ports.stem);
                    let n = ports.stem.len() as u64;
                    self.stats.stem_ops += n;
                    self.stats.signals_emitted += n;
                    if self.stats.signals_emitted >= self.budget.max_signals_total {
                        self.stats.budget_exhausted = true;
                    }
                }
            }
            BodyTarget::Triad => self.emit_body_ports(ports),
        }
    }

    /// Feed modality embedding to a single body target (non-traditional compose path).
    pub fn feed_modality_target(
        &mut self,
        input: ModalityInput,
        front_cfg: &DlFrontConfig,
        target: BodyTarget,
    ) -> bool {
        if self.stats.budget_exhausted || self.stats.items_processed >= self.budget.max_items {
            self.stats.budget_exhausted = true;
            return false;
        }
        let ports = encode_modality(&input, front_cfg);
        self.emit_ports_target(&ports, target);
        self.stats.items_processed += 1;
        !self.stats.budget_exhausted
    }

    /// Optimize cycles for one body only (per-shard closure in compose mode).
    pub fn run_optimize_cycles_target(&mut self, optimize_cycles: usize, target: BodyTarget) {
        let cycles = optimize_cycles.max(1);
        match target {
            BodyTarget::Hub => {
                self.hub.mark_epoch_complete();
            }
            BodyTarget::Tile => {
                for _ in 0..cycles {
                    self.tile.process_tiles_cycle();
                    if self.tile.has_fate_converged() {
                        break;
                    }
                }
            }
            BodyTarget::StemCell => {
                for _ in 0..cycles {
                    self.stem.process_cells_cycle();
                    let _ = self.stem.evolve(0.5);
                    if self.stem.cells.len() >= 4 {
                        let _ = self.stem.evolve(0.35);
                    }
                    if self.stem.has_fate_converged() {
                        break;
                    }
                }
            }
            BodyTarget::Triad => self.run_optimize_cycles(cycles),
        }
    }

    /// Structural closure metric for a body before/after shard feed (compose scoring).
    pub fn body_closure_metric(&self, target: BodyTarget) -> f64 {
        match target {
            BodyTarget::Tile => {
                let tiles = &self.tile.completed_tiles;
                if tiles.is_empty() {
                    self.stats.tile_ops as f64 * 1e-4
                } else {
                    tiles
                        .iter()
                        .map(|t| {
                            t.energy.abs()
                                + t.activation.abs() * 0.25
                                + t.connections.len() as f64 * 0.02
                        })
                        .sum::<f64>()
                        / tiles.len() as f64
                }
            }
            BodyTarget::StemCell => {
                let cells = &self.stem.cells;
                if cells.is_empty() {
                    self.stats.stem_ops as f64 * 1e-4
                } else {
                    cells
                        .iter()
                        .map(|c| c.energy.abs() + c.connections.len() as f64 * 0.01)
                        .sum::<f64>()
                        / cells.len() as f64
                }
            }
            BodyTarget::Hub => {
                let conv = if self.hub.has_fate_converged() { 0.35 } else { 0.0 };
                self.stats.hub_requests as f64 * 0.008 + conv
            }
            BodyTarget::Triad => {
                (self.body_closure_metric(BodyTarget::Tile)
                    + self.body_closure_metric(BodyTarget::StemCell))
                    * 0.5
            }
        }
    }

    /// Export patch report after optional optimize cycles (session remains usable).
    pub fn build_feed_report(&mut self, mode_label: &str, optimize_cycles: usize) -> TriadFeedReport {
        if optimize_cycles > 0 {
            self.run_optimize_cycles(optimize_cycles);
        }
        self.make_feed_report(mode_label, optimize_cycles.max(1))
    }

    pub fn incubation_snapshot(&self) -> TriadIncubationSnapshot {
        self.manifold_snapshot()
    }

    /// Full manifold export — Tile/Stem + Fate convergence + fragments.
    pub fn manifold_snapshot(&self) -> TriadIncubationSnapshot {
        let prefs = self.hub.get_fate_preferences();
        let mode_str = match self.hub.fate_weight_mode {
            crate::FateWeightMode::Fixed => "fixed",
            crate::FateWeightMode::Evaluated => "evaluated",
        };
        let report = self.make_feed_report("manifold_export", 0);
        let fragments = if self.fragment_tracks.is_empty() {
            vec![crate::fate_orchestrator::export_fragment(
                self,
                "main",
                "default",
                crate::fate_orchestrator::FragmentCloseMeta::default(),
            )]
        } else {
            let mut frags = self.fragment_tracks.clone();
            frags.push(crate::fate_orchestrator::export_fragment(
                self,
                "merged",
                "session_merged",
                crate::fate_orchestrator::FragmentCloseMeta::default(),
            ));
            frags
        };
        TriadIncubationSnapshot {
            completed_tiles: self.tile.completed_tiles.clone(),
            stem_cells: self.stem.cells.clone(),
            tile_total_operations: self.tile.total_operations,
            hub_routed: self.hub_routed,
            ingress_encoder: self.ingress_encoder.clone(),
            hub_prefs: prefs.affinities.clone(),
            fate_weight_mode: mode_str.into(),
            fate_routing_free: self.hub.fate_routing_free,
            fate_ingress_routing: match self.hub.ingress_routing {
                crate::IngressRoutingMode::RoundRobin => "round_robin".into(),
                crate::IngressRoutingMode::Semantic => "semantic".into(),
                crate::IngressRoutingMode::QuadCycle => "quad_cycle".into(),
                crate::IngressRoutingMode::HybridQuadSemantic => "hybrid_quad_semantic".into(),
            },
            corpus_line_affinity: self.corpus_line_affinity.clone(),
            fate_signal_tick: self.hub.fate_system().get_logical_time(),
            manifold_protocol: Some(crate::fate_orchestrator::MANIFOLD_PROTOCOL.into()),
            patch_vector: report.model_patch.patch_vector.clone(),
            fate_convergence: Some(self.hub.fate_system.export_convergence_state()),
            hub_triple_fate: Some(self.hub.triple_fate.freeze()),
            fragments,
        }
    }

    pub fn hub_routed_snapshot(&self) -> [u64; 4] {
        self.hub_routed
    }

    pub fn corpus_line_affinity_snapshot(&self) -> Vec<f64> {
        self.corpus_line_affinity.clone()
    }

    pub fn restore_hub_routed(&mut self, routed: [u64; 4]) {
        self.hub_routed = routed;
    }

    pub fn restore_corpus_line_affinity(&mut self, aff: Vec<f64>) {
        self.corpus_line_affinity = aff;
    }

    /// Fate-owned optimize — no Tile/Stem early break (Fate decides stop externally).
    pub fn run_optimize_cycles_fate_owned(&mut self, optimize_cycles: usize) {
        let cycles = optimize_cycles.max(1);
        if matches!(self.body, BodyTarget::Hub | BodyTarget::Triad) {
            self.hub.mark_epoch_complete();
        }
        let run_tile = matches!(self.body, BodyTarget::Tile | BodyTarget::Triad);
        let run_stem = matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad);

        for _ in 0..cycles {
            if run_tile {
                self.tile.process_tiles_cycle();
            }
            if run_stem {
                self.stem.process_cells_cycle();
                let _ = self.stem.evolve(0.5);
                if self.stem.cells.len() >= 4 {
                    let _ = self.stem.evolve(0.35);
                }
            }
        }
    }

    fn ingress_routing_from_str(s: &str) -> crate::IngressRoutingMode {
        if s.eq_ignore_ascii_case("semantic") {
            crate::IngressRoutingMode::Semantic
        } else if s.eq_ignore_ascii_case("quad_cycle") {
            crate::IngressRoutingMode::QuadCycle
        } else if s.eq_ignore_ascii_case("hybrid_quad_semantic") {
            crate::IngressRoutingMode::HybridQuadSemantic
        } else {
            crate::IngressRoutingMode::RoundRobin
        }
    }

    pub fn apply_snapshot_fate_policy(session: &mut TriadSession, snap: &TriadIncubationSnapshot) {
        let mode = if snap.fate_weight_mode.eq_ignore_ascii_case("evaluated") {
            crate::FateWeightMode::Evaluated
        } else {
            crate::FateWeightMode::Fixed
        };
        session.hub.set_fate_policy(
            mode,
            snap.fate_routing_free,
            Self::ingress_routing_from_str(&snap.fate_ingress_routing),
        );
        session.hub.seed_fate_from_snapshot(
            &snap.hub_prefs,
            snap.hub_routed,
            if snap.fate_signal_tick > 0 {
                Some(snap.fate_signal_tick)
            } else {
                None
            },
        );
        if let Some(ref conv) = snap.fate_convergence {
            session.hub.fate_system.import_convergence_state(conv);
        }
        if let Some(ref slice) = snap.hub_triple_fate {
            session.hub.triple_fate = crate::TripleFateCombined::thaw(slice.clone());
        }
    }

    /// 信号只进 Hub 上这一份叠层 Fate。Tile 和 Stem 跟上它，收口才长身体。
    pub fn observe_fate(&mut self, value: f64) -> bool {
        let closed = self.hub.triple_fate.receive_passive(value);
        self.tile.follow_global_fate(&self.hub.triple_fate);
        self.stem.follow_global_fate(&self.hub.triple_fate);
        if closed {
            self.tile.ingest(value);
            self.stem.ingest(value);
            self.tile.promote_from_fate();
            self.stem.promote_from_fate();
        }
        closed
    }

    /// Apply QA answer feedback into Fate + Tile/Stem passive loop.
    pub fn apply_qa_feedback(&mut self, question: &str, on_topic: bool) {
        let routed = self.hub_routed;
        self.hub.feedback_qa_outcome(question, on_topic, routed);
        let mut meta = crate::fate_adaptive_algebra::FateAlgebraMeta::default();
        meta.fragment_count = self.fragment_tracks.len();
        meta.pretrain_pass_rate = if on_topic { 1.0 } else { 0.0 };
        let state = crate::fate_adaptive_algebra::FateAlgebraState::from_meta(&meta, None);
        let intensity = if on_topic {
            state.qa_pass_intensity()
        } else {
            state.qa_fail_intensity()
        };
        self.observe_fate(intensity);
    }

    pub fn from_incubation_snapshot(
        budget: IngressBudget,
        body: BodyTarget,
        snap: TriadIncubationSnapshot,
    ) -> Self {
        let mut session = Self::with_body(budget, body);
        Self::apply_snapshot_fate_policy(&mut session, &snap);
        // Rehydrate before the tables are moved: a compact (deduped)
        // snapshot's fragments may carry only tile/cell ids.
        session.fragment_tracks = crate::fate_orchestrator::rehydrate_fragments(
            &snap.fragments,
            &snap.completed_tiles,
            &snap.stem_cells,
        );
        session.tile.completed_tiles = snap.completed_tiles;
        session.tile.total_operations = snap.tile_total_operations;
        session.stem.cells = snap.stem_cells;
        session.hub_routed = snap.hub_routed;
        session.ingress_encoder = snap.ingress_encoder;
        session.corpus_line_affinity = snap.corpus_line_affinity.clone();
        session
    }

    /// Structural corpus-line affinity from QA outcome (grinder cosine Q↔line, Fate adaptive).
    pub fn apply_corpus_retrieval_feedback(
        &mut self,
        corpus_idx: usize,
        question_sig: &[f64],
        line_sig: &[f64],
        line_text: &str,
        on_topic: bool,
    ) {
        while self.corpus_line_affinity.len() <= corpus_idx {
            self.corpus_line_affinity.push(0.0);
        }
        let align = signal_cosine_local(question_sig, line_sig).max(0.0);
        let delta = if on_topic {
            env_f64_override("VALHALLA_CORPUS_AFFINITY_POS", align, affinity_pos_from_align)
        } else {
            -env_f64_override("VALHALLA_CORPUS_AFFINITY_NEG", align, affinity_neg_from_align)
        };
        self.corpus_line_affinity[corpus_idx] =
            (self.corpus_line_affinity[corpus_idx] + delta).clamp(0.0, 1.0);
        self.hub
            .feedback_corpus_line_structural(line_text, align, on_topic);
        let passive = delta.abs();
        self.observe_fate(passive);
    }

    fn make_feed_report(&self, mode_label: &str, optimize_cycles: usize) -> TriadFeedReport {
        let prefs = self.hub.get_fate_preferences();
        let tstatus = self.tile.get_status();
        let sstatus = self.stem.get_status();
        let connections: usize = self.stem.cells.iter().map(|c| c.connections.len()).sum();
        let mut model_patch = build_model_patch(
            self.body,
            &prefs.affinities,
            self.hub_routed,
            &self.tile,
            &self.stem,
            &self.stats,
        );
        model_patch.ingress_encoder = self.ingress_encoder.clone();

        TriadFeedReport {
            mode: mode_label.to_string(),
            hub_signals: if matches!(self.body, BodyTarget::Hub | BodyTarget::Triad) {
                self.stats.hub_requests
            } else {
                0
            },
            stats: self.stats.clone(),
            hub_routed: self.hub_routed,
            hub_prefs: prefs.affinities.clone(),
            tile_completed: if matches!(self.body, BodyTarget::Tile | BodyTarget::Triad) {
                tstatus.completed_tiles
            } else {
                0
            },
            tile_ops: if matches!(self.body, BodyTarget::Tile | BodyTarget::Triad) {
                tstatus.total_operations
            } else {
                0
            },
            tile_dim: if matches!(self.body, BodyTarget::Tile | BodyTarget::Triad) {
                tstatus.dimension
            } else {
                4.0
            },
            stem_cells: if matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad) {
                sstatus.cell_count
            } else {
                0
            },
            stem_connections: if matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad) {
                connections
            } else {
                0
            },
            stem_dim: if matches!(self.body, BodyTarget::StemCell | BodyTarget::Triad) {
                self.stem.space.current_dimension
            } else {
                4.0
            },
            model_patch,
            optimize_cycles,
            cycle_trace: Vec::new(),
            mcq_fate_trace: Vec::new(),
        }
    }

    /// Run `optimize_cycles` rounds of Tile / StemCell processing before patch export.
    pub fn finalize_with_cycles(mut self, mode_label: &str, optimize_cycles: usize) -> TriadFeedReport {
        let cycles = optimize_cycles.max(1);
        let mut cycle_trace = Vec::with_capacity(cycles);
        for c in 0..cycles {
            self.run_optimize_cycles(1);
            if matches!(self.body, BodyTarget::Tile | BodyTarget::Triad) {
                let _ = self.tile.process_tiles_cycle();
            }
            let tstatus = self.tile.get_status();
            let connections: usize = self.stem.cells.iter().map(|c| c.connections.len()).sum();
            cycle_trace.push(CycleSnapshot {
                cycle: c,
                tile_completed: tstatus.completed_tiles,
                tile_merges: 0,
                tile_splits: 0,
                tile_dim: tstatus.dimension,
                stem_cells: self.stem.get_status().cell_count,
                stem_connections: connections,
                stem_merges: 0,
                stem_divisions: 0,
                stem_dim: self.stem.space.current_dimension,
            });
        }
        let mut report = self.build_feed_report(mode_label, optimize_cycles);
        report.cycle_trace = cycle_trace;
        report
    }
}

fn route_index(message: &str) -> usize {
    if message.contains("BlackHole") {
        0
    } else if message.contains("MeshBrain") {
        1
    } else if message.contains("MultiNova") {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_stats_fixed_width() {
        let p = IngressPayload::Text {
            content: "hello world".into(),
        };
        let v = encode_payload(&p, EncodingMode::TextStats, &IngressBudget::default());
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn f64_passthrough() {
        let p = IngressPayload::F64 {
            values: vec![0.1, -0.5, 3.0],
        };
        let v = encode_payload(&p, EncodingMode::DirectF64, &IngressBudget::default());
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn budget_caps_per_item() {
        let budget = IngressBudget {
            max_signals_per_item: 4,
            ..Default::default()
        };
        let p = IngressPayload::Text {
            content: "abcdefghij".into(),
        };
        let v = encode_payload(&p, EncodingMode::Char, &budget);
        assert!(v.len() <= 4);
    }
}
