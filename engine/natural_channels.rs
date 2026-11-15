//! Natural internal channels — one question fans out through Hub / Tile / Stem / Triad,
//! then decode probes compete; no exclusive single-path gate.

use serde::{Deserialize, Serialize};

use crate::dl_front::{BodyFrontPorts, DlFrontConfig};
use crate::pipeline::{self, PipelineConfig, PipelineOutcome};
use crate::signal_ingress::{IngressBudget, TriadFeedReport, TriadSession};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPick {
    pub id: String,
    pub score: f64,
    pub decode_path: String,
}

#[derive(Debug, Clone)]
pub struct NaturalChannelOutcome {
    pub answer: String,
    pub decode_path: String,
    pub report: TriadFeedReport,
    pub picks: Vec<ChannelPick>,
    pub pipeline_outcome: Option<PipelineOutcome>,
    pub mcq_fate_meta: Option<(usize, String, f64, f64)>,
}

pub fn natural_channels_enabled() -> bool {
    match std::env::var("VALHALLA_NATURAL_CHANNELS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    }
}

/// Pulse question into all Triad bodies — information flows naturally, no preset body route.
pub fn pulse_question_natural(
    session: &mut TriadSession,
    question_ports: &BodyFrontPorts,
    qa_cycles: usize,
    audio_features: Option<&[f64]>,
    front: &DlFrontConfig,
) {
    crate::free_multimodal_fate::free_pulse(session, question_ports);
    crate::multimodal::feed_audio(session, audio_features, front);
    session.run_optimize_cycles_fate_owned(qa_cycles.max(1));
    if session.hub.is_data_flowing() {
        session.hub.mark_epoch_complete();
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn keyword_overlap(question: &str, text: &str) -> f64 {
    let q: std::collections::HashSet<String> = question
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_string())
        .collect();
    if q.is_empty() {
        return 0.0;
    }
    let t = text.to_lowercase();
    let hit = q.iter().filter(|w| t.contains(w.as_str())).count();
    hit as f64 / q.len() as f64
}

fn score_answer_candidate(
    answer: &str,
    decode_path: &str,
    question: &str,
    score_type: &str,
    bonus: f64,
) -> f64 {
    if answer.starts_with("No confident match") || decode_path.contains("reject") {
        return 0.05;
    }
    if decode_path.contains("abstain") {
        return 0.08;
    }
    let body = crate::fate_adaptive_algebra::strip_all_citations(answer)
        .trim()
        .to_string();
    let mut s = bonus + keyword_overlap(question, &body) * 0.35;
    if score_type == "mcq" && body.contains("ANSWER:") {
        s += 0.15;
    }
    // Fate structure shorts (Yes/no/Agree/Similar) are valid — don't only reward long spans.
    if score_type == "open" {
        if decode_path.contains("multihop_fate") {
            s += 0.12;
        } else if body.len() > 8 && body.len() < 400 {
            s += 0.1;
        }
    }
    s.clamp(0.0, 1.0)
}

/// Ground Fate structure shorts the same way triad_memory grounds retrieval idx.
fn fate_structure_grounding(
    question: &str,
    _fate_path: &str,
    corpus_refs: &[crate::multihop_fate::CorpusLineRef],
    line_aff: &[f64],
) -> f64 {
    if corpus_refs.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f64;
    for line in corpus_refs {
        let aff = line_aff.get(line.idx).copied().unwrap_or(0.0);
        let g = keyword_overlap(question, &line.text);
        best = best.max(aff * 0.55 + g * 0.25);
    }
    best
}

fn hub_channel_weight(report: &TriadFeedReport) -> f64 {
    let prefs = &report.hub_prefs;
    if prefs.len() < 4 {
        return 0.25;
    }
    prefs.iter().copied().fold(0.0_f64, f64::max) / prefs.iter().sum::<f64>().max(1e-9)
}

/// Run all internal decode channels on the live session and pick the best outcome.
#[allow(clippy::too_many_arguments)]
pub fn resolve_natural_channels(
    qa_session: &mut TriadSession,
    snap: &crate::signal_ingress::TriadIncubationSnapshot,
    question: &str,
    score_type: &str,
    question_ports: &BodyFrontPorts,
    memories: &[crate::native_qa::MemoryEntry],
    line_aff: &[f64],
    line_aff_passive: bool,
    mode_label: &str,
    qa_cycles: usize,
    mode: &str,
    dl: bool,
    front: &DlFrontConfig,
    prior: &[String],
    multihop_v2: bool,
    open_fate: bool,
    use_pipeline_v4: bool,
    use_mcq_fate: bool,
    use_live: bool,
    query_selection: bool,
    budget: IngressBudget,
) -> NaturalChannelOutcome {
    let mut picks: Vec<ChannelPick> = Vec::new();
    let mut best_answer = String::new();
    let mut best_path = String::new();
    let mut best_score = -1.0_f64;
    let mut best_report = qa_session.build_feed_report(mode_label, qa_cycles);
    let mut pipeline_outcome: Option<PipelineOutcome> = None;
    let mut mcq_fate_meta: Option<(usize, String, f64, f64)> = None;

    let pipe_cfg = PipelineConfig::default();
    let hub_w = hub_channel_weight(&best_report);

    // Channel: Triad memory / structure decode
    let base_report = qa_session.build_feed_report(mode_label, qa_cycles);
    let (mem_ans, _top, mem_path, ridx) = crate::native_qa::decode_from_report(
        question,
        score_type,
        memories,
        &base_report,
        &budget,
        mode,
        dl,
        front,
        prior,
        line_aff,
        line_aff_passive,
        multihop_v2,
    );
    let mut mem_score = score_answer_candidate(&mem_ans, &mem_path, question, score_type, 0.25 + hub_w * 0.2);
    if let Some(idx) = ridx {
        if let Some(&aff) = line_aff.get(idx) {
            mem_score += aff * 0.55;
        }
        if let Some(m) = memories.iter().find(|m| m.corpus_idx == Some(idx)) {
            mem_score += keyword_overlap(question, &m.text) * 0.25;
        }
    }
    // Yes/No / relation questions: long retrieval spans must not beat Fate shorts.
    let qlow = question.to_ascii_lowercase();
    let yesno_like = qlow.starts_with("does ")
        || qlow.starts_with("do ")
        || qlow.starts_with("is ")
        || qlow.starts_with("are ")
        || qlow.contains("agree or disagree")
        || qlow.contains("similar way")
        || qlow.contains("similar reason");
    if yesno_like && mem_path.contains("open_retrieval") {
        mem_score *= 0.45;
    }
    mem_score = mem_score.clamp(0.0, 1.0);
    picks.push(ChannelPick {
        id: "triad_memory".into(),
        score: mem_score,
        decode_path: mem_path.clone(),
    });
    if mem_score > best_score {
        best_score = mem_score;
        best_answer = mem_ans.clone();
        best_path = mem_path.clone();
        best_report = base_report.clone();
    }

    // Channel: Hub — Fate preference weighted retrieval bias
    let hub_bonus = env_f64("VALHALLA_CHANNEL_HUB_WEIGHT", 0.18);
    let hub_score = (hub_w + keyword_overlap(question, &mem_ans) * 0.5).clamp(0.0, 1.0);
    picks.push(ChannelPick {
        id: "hub_routing".into(),
        score: hub_score,
        decode_path: format!("hub_prefs/{mem_path}"),
    });
    if hub_score + hub_bonus > best_score && !mem_ans.starts_with("No confident") {
        best_score = hub_score + hub_bonus;
        best_answer = mem_ans;
        best_path = format!("hub/{mem_path}");
        best_report = base_report;
    }

    // Channel: Pipeline v4 (decompose → recompose)
    if use_pipeline_v4 {
        let outcome = if use_live || query_selection {
            pipeline::run_question_pipeline_on_session(
                qa_session,
                question,
                score_type,
                question_ports,
                &pipe_cfg,
                front,
                mode_label,
                None,
            )
        } else {
            pipeline::run_question_pipeline(
                snap,
                question,
                score_type,
                question_ports,
                &pipe_cfg,
                front,
                mode_label,
                None,
            )
        };
        let pipe_report = outcome.report.clone();
        let (pipe_ans, pipe_path, pipe_score) = if score_type == "mcq" {
            let (dec_ans, _, dec_path, _) = crate::native_qa::decode_from_report(
                question,
                score_type,
                memories,
                &pipe_report,
                &budget,
                mode,
                dl,
                front,
                prior,
                line_aff,
                line_aff_passive,
                multihop_v2,
            );
            let (ans, path) =
                pipeline::merge_mcq_pipeline_decode(&dec_ans, &dec_path, &outcome.recompose.mcq_option_affinity);
            let sc = score_answer_candidate(&ans, &path, question, score_type, 0.35)
                + outcome.mcq_fate.as_ref().map(|m| m.aggregate_score * 0.2).unwrap_or(0.0);
            (ans, path, sc)
        } else {
            let hint = outcome.decode_hint.clone();
            let (a, _, _, _) = crate::native_qa::decode_from_report(
                question,
                score_type,
                memories,
                &pipe_report,
                &budget,
                mode,
                dl,
                front,
                prior,
                line_aff,
                line_aff_passive,
                multihop_v2,
            );
            let path = format!("pipeline_v4_{score_type}");
            let sc = score_answer_candidate(&a, &path, question, score_type, 0.32)
                + outcome.recompose.open_structure_score * 0.15;
            (a, outcome.decode_hint.clone(), sc)
        };
        picks.push(ChannelPick {
            id: "pipeline_v4".into(),
            score: pipe_score,
            decode_path: pipe_path.clone(),
        });
        if pipe_score > best_score {
            best_score = pipe_score;
            best_answer = pipe_ans;
            best_path = pipe_path;
            best_report = pipe_report;
        }
        if let Some(ref mf) = outcome.mcq_fate {
            mcq_fate_meta = Some((
                mf.cycles_executed,
                mf.stopped_reason.clone(),
                mf.decompose_score,
                mf.aggregate_score,
            ));
        }
        pipeline_outcome = Some(outcome);
    } else if score_type == "mcq" && use_mcq_fate {
        let outcome = crate::mcq_fate::run_mcq_fate_pipeline(
            qa_session,
            question,
            question_ports,
            mode_label,
            front,
        );
        mcq_fate_meta = Some((
            outcome.cycles_executed,
            outcome.stopped_reason.clone(),
            outcome.decompose_score,
            outcome.aggregate_score,
        ));
        let (a, _, p, _) = crate::native_qa::decode_from_report(
            question,
            score_type,
            memories,
            &outcome.report,
            &budget,
            mode,
            dl,
            front,
            prior,
            line_aff,
            line_aff_passive,
            multihop_v2,
        );
        let sc = score_answer_candidate(&a, &p, question, score_type, 0.28) + outcome.aggregate_score * 0.25;
        picks.push(ChannelPick {
            id: "mcq_fate".into(),
            score: sc,
            decode_path: p.clone(),
        });
        if sc > best_score {
            best_score = sc;
            best_answer = a;
            best_path = "mcq_fate_cycle".into();
            best_report = outcome.report;
        }
    }

    // Channel: Open Fate structure compose
    if open_fate && score_type == "open" {
        let corpus_refs: Vec<crate::multihop_fate::CorpusLineRef> = memories
            .iter()
            .filter_map(|m| {
                m.corpus_idx.map(|idx| crate::multihop_fate::CorpusLineRef {
                    idx,
                    text: m.text.clone(),
                    signals: m.signals.clone(),
                })
            })
            .collect();
        if !corpus_refs.is_empty() {
            if let Some((fate_ans, fate_path)) = crate::multihop_fate::decode_open_fate(
                question,
                &corpus_refs,
                line_aff,
                &best_report.hub_prefs,
                &question_ports.embedding,
                multihop_v2,
            ) {
                let mut sc =
                    score_answer_candidate(&fate_ans, &fate_path, question, score_type, 0.38);
                // Align scoring interface with triad_memory grounding.
                sc = (sc
                    + fate_structure_grounding(question, &fate_path, &corpus_refs, line_aff))
                    .clamp(0.0, 1.0);
                picks.push(ChannelPick {
                    id: "open_fate".into(),
                    score: sc,
                    decode_path: fate_path.clone(),
                });
                if sc > best_score {
                    best_score = sc;
                    best_answer = fate_ans;
                    best_path = fate_path;
                }
            }
        }
    }

    // Tile / Stem channel bonuses — structure activation from report
    let tile_act = best_report.model_patch.tile_avg_activation;
    let stem_n = best_report.model_patch.stem_cell_count;
    if tile_act > 0.05 {
        picks.push(ChannelPick {
            id: "tile_structure".into(),
            score: (tile_act * 0.4).min(0.35),
            decode_path: best_path.clone(),
        });
    }
    if stem_n > 0 && !best_answer.starts_with("No confident") {
        let stem_score = (stem_n as f64 / 32.0).min(0.25) + best_score * 0.05;
        picks.push(ChannelPick {
            id: "stem_span".into(),
            score: stem_score,
            decode_path: best_path.clone(),
        });
    }

    let channel_ids: String = picks
        .iter()
        .filter(|p| p.score > 0.1)
        .map(|p| p.id.as_str())
        .collect::<Vec<_>>()
        .join("+");
    best_path = format!("natural/{channel_ids}/{}", best_path);

    NaturalChannelOutcome {
        answer: best_answer,
        decode_path: best_path,
        report: best_report,
        picks,
        pipeline_outcome,
        mcq_fate_meta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_channels_default_on() {
        std::env::remove_var("VALHALLA_NATURAL_CHANNELS");
        assert!(natural_channels_enabled());
    }

    #[test]
    fn score_rejects_low_confidence() {
        let s = score_answer_candidate(
            "No confident match in incubated memory.",
            "open_reject_low_score",
            "What PIN?",
            "open",
            0.5,
        );
        assert!(s < 0.2);
    }
}
