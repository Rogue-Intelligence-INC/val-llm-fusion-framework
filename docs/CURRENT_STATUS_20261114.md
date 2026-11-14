# Valhalla 现状巩固报告（20261114）

一键复现：`powershell -File reproduce.ps1`（full）/ `--quick` 快速档。
引擎：valhalla_base 1.1.0 · 协议 valhalla-base-v1 · 解码头 mistral24-valhalla-s0.08（16K ctx）
总判定：**PASS**

| 套件 | 实测 | 基线 | 编造 | 判定 |
|---|---|---|---|---|
| wire smoke (QA/extend/restore) | - | SMOKE PASS | 0 | PASS |
| probe_fate_gateway | - | exit 0 | 0 | PASS |
| probe_corpus_switch | - | exit 0 | 0 | PASS |
| multimodal (valhalla arm) | 9/9 | >=9/9 | 0 | PASS |
| extreme targeted 806 (hybrid) | 12/12 | >=12/12 | 0 | PASS |
| extreme targeted 3006 (hybrid) | 12/12 | >=12/12 | 0 | PASS |
| extreme targeted 10006 (hybrid) | 12/12 | >=12/12 | 0 | PASS |
| enhancement ceiling 1.5M tok (hybrid) | 17/21 | >=16/21 | 0 | PASS |
| BOOST adapt v2 (spread+operator+discipline) | 20/21 | >=19/21 | 0 | PASS |

倍率口径（工件派生）：工作语料 14,945,110 token = 原生窗口 912×（千倍级）、全上下文死亡点 331×、每题触达语料 ~1/83,000、会话线缆 3.42MB 恒定。

已知边界（不因复现通过而改变）：LoCoMo-15 0/15 不赢；多跳残留 hop_employer（实体消歧，失败模式为弃权）；矛盾事实静默选后写；proactive 发声未实现；特征抓取为同文件再识别。

服务编排记录：decode head: already healthy；gateway: already healthy；engine: already healthy
