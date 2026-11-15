//! Programmatic open-QA interface — ingress/egress via [`crate::fate_io`], native batch decode.
//!
//! Benchmarks (`multihop_bench`) load bundles and score; this module is the reusable API.

use crate::fate_io::{ensure_unified_runtime, fate_egress_body, fate_score_short_answer};
use crate::grinder::{default_port_cap, grind_text_hash};
use crate::native_qa::{
    run_native_qa_batch, NativeQaBatchItem, NativeQaBatchRequest, NativeQaQuestionSpec,
};

/// Stack flags for native open QA (maps to former fatewithv2 arms).
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenQaFlags {
    pub multihop_v2: bool,
    pub open_fate: bool,
}

impl OpenQaFlags {
    pub fn v2_only() -> Self {
        Self {
            multihop_v2: true,
            open_fate: false,
        }
    }

    pub fn fate_only() -> Self {
        Self {
            multihop_v2: false,
            open_fate: true,
        }
    }

    pub fn unified() -> Self {
        Self {
            multihop_v2: true,
            open_fate: true,
        }
    }
}

/// One open-QA call (corpus + question).
#[derive(Debug, Clone)]
pub struct OpenQaRequest {
    pub id: String,
    pub question: String,
    pub corpus: Vec<String>,
    pub incubation_cycles: usize,
    pub qa_cycles: usize,
    pub flags: OpenQaFlags,
}

impl OpenQaRequest {
    pub fn new(id: impl Into<String>, question: impl Into<String>, corpus: Vec<String>) -> Self {
        Self {
            id: id.into(),
            question: question.into(),
            corpus,
            incubation_cycles: 2,
            qa_cycles: 1,
            flags: OpenQaFlags::unified(),
        }
    }

    pub fn fast(mut self) -> Self {
        self.incubation_cycles = 1;
        self
    }

    pub fn with_flags(mut self, flags: OpenQaFlags) -> Self {
        self.flags = flags;
        self
    }
}

/// Native batch row + optional gold match.
#[derive(Debug, Clone)]
pub struct OpenQaResult {
    pub row: NativeQaBatchItem,
    pub matches_gold: Option<bool>,
}

impl OpenQaResult {
    pub fn answer(&self) -> &str {
        &self.row.answer
    }

    pub fn decode_path(&self) -> &str {
        &self.row.decode_path
    }
}

fn grind_ports_line(line: &str) -> crate::dl_front::BodyFrontPorts {
    grind_text_hash(line, default_port_cap())
}

/// Run one isolated native open-QA item (RelAddr feed + Fate/v2 inside `native_qa_batch`).
pub fn run_open_qa(req: &OpenQaRequest) -> OpenQaResult {
    if req.flags.open_fate && req.flags.multihop_v2 {
        ensure_unified_runtime();
    }
    let corpus_ports: Vec<_> = req.corpus.iter().map(|l| grind_ports_line(l)).collect();
    let q_ports = grind_ports_line(&req.question);
    let batch = NativeQaBatchRequest {
        body: "triad".into(),
        corpus: req.corpus.clone(),
        corpus_ports,
        questions: vec![NativeQaQuestionSpec {
            id: Some(req.id.clone()),
            question: req.question.clone(),
            score_type: "open".into(),
            question_ports: Some(q_ports),
            audio_features: None,
            prior_questions: None,
        }],
        incubation_cycles: req.incubation_cycles.max(1),
        qa_cycles: req.qa_cycles.max(1),
        ingress_mode: Some("grinder".into()),
        mcq_fate: false,
        pipeline_v4: true,
        multihop_v2: req.flags.multihop_v2,
        open_fate: req.flags.open_fate,
        multihop_fate: req.flags.open_fate,
        incubation_snapshot: None,
        corpus_fragments: None,
    };
    let resp = run_native_qa_batch(&batch);
    let row = resp.results.into_iter().next().unwrap_or(NativeQaBatchItem {
        id: req.id.clone(),
        answer: String::new(),
        score_type: "open".into(),
        decode_path: "empty".into(),
        tile_count: 0,
        stem_cells: 0,
        tile_connections: 0,
        tile_avg_activation: 0.0,
        stem_organ_clusters: 0,
        correct: false,
        mcq_fate_cycles: 0,
        mcq_fate_stopped: String::new(),
        mcq_decompose_score: 0.0,
        mcq_aggregate_score: 0.0,
        fog_evidence: None,
        fog_decision: None,
        retrieval_corpus_idx: None,
        query_fragments: None,
        query_fragment_scores: None,
        fate_grasp: None,
    });
    OpenQaResult {
        row,
        matches_gold: None,
    }
}

/// Run and score against reference (uses generic egress body compare).
pub fn run_open_qa_scored(req: &OpenQaRequest, gold: &str) -> OpenQaResult {
    let mut out = run_open_qa(req);
    out.matches_gold = Some(fate_score_short_answer(out.answer(), gold));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_presets() {
        assert!(OpenQaFlags::unified().multihop_v2 && OpenQaFlags::unified().open_fate);
        assert!(!OpenQaFlags::v2_only().open_fate);
    }

    #[test]
    fn score_uses_egress_body() {
        assert!(fate_score_short_answer(
            "[corpus:1] [fate:2@1] Yes",
            "Yes"
        ));
        assert_eq!(fate_egress_body("[corpus:1] Yes"), "Yes");
    }
}
