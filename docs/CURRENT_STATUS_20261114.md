# Valhalla 现状巩固报告（20261114）

一键复现：`powershell -File reproduce.ps1`（full）/ `--quick` 快速档。
引擎：valhalla_base 1.1.0 · 协议 valhalla-base-v1 · 解码头 mistral24-valhalla-s0.08（16K ctx）
总判定（上午复现）：**PASS**；当日续报（机制落地 + 本地版极限）见文末。

| 套件 | 实测 | 基线 | 编造 | 判定 |
|---|---|---|---|---|
| wire smoke (QA/extend/restore) | - | SMOKE PASS | 0 | PASS |
| probe_fate_gateway | - | exit 0 | 0 | PASS |
| probe_corpus_switch | - | exit 0 | 0 | PASS |
| multimodal (valhalla arm, Fate-native) | 6/9 | >=6/9 | 0 | PASS |
| extreme targeted 806 (hybrid) | 12/12 | >=12/12 | 0 | PASS |
| extreme targeted 3006 (hybrid) | 12/12 | >=12/12 | 0 | PASS |
| extreme targeted 10006 (hybrid) | 12/12 | >=12/12 | 0 | PASS |
| enhancement ceiling 1.5M tok (hybrid) | 17/21 | >=16/21 | 0 | PASS |
| BOOST adapt v2 (spread+operator+discipline) | 20/21 | >=19/21 | 0 | PASS |

倍率口径（工件派生）：工作语料 14,945,110 token = 原生窗口 912×（千倍级）、全上下文死亡点 331×、每题触达语料 ~1/83,000、会话线缆 3.42MB 恒定。

已知边界（不因复现通过而改变）：LoCoMo-15 0/15 不赢；多跳残留 hop_employer（实体消歧，失败模式为弃权）；矛盾事实静默选后写；proactive 发声未实现；特征抓取为同文件再识别。

服务编排记录：decode head: already healthy；gateway: already healthy；engine: already healthy

## 当日续报（2026-11-14 晚）：机制落地 + 本地版极限

上午基线（hybrid 臂）保持可复现，但产品形态当日已变更：

| 项 | 晚终态 | 工件 |
|---|---|---|
| 形态修复 | 头部闭合、片段增量卸载、分段 id 种子（部署形态下"块"此前不存在） | 锚点 `20261114-optimized-BC`（引擎 `C3BEE261…7FD2`） |
| 裁决门 | Fate 簇方程自控（`bundle_fits` 读活方程）；校准常数门已按裁定删除 | `archive/20261114-fate-gate-anchor`（`2B9C509E…C1FC1`） |
| 66 轮对话 | 核心 24/24、陷阱 10/12（出口毒丸清洗后）、编造 0 | `dialogue_optBC2_lines10006.json` |
| 本地版极限（新） | 引擎直连 312 轮：核心 90/90 三段平直、编造 0/312、零漂移、零轮换；陷阱 36/60（失败定性穷尽） | `valhalla-local/docs/EXTREME_LIMITS_LOCAL_312TURNS_20261114.md` |
| 真极限 | 摄入吞吐：每轮 0.5s→~12s（会话 ~400 行），引擎 pulse 成本曲线 | 同上逐轮记录 |
| 定性裁定 | **机制，非优化**（五条判别见全面总结） | `docs/COMPREHENSIVE_SUMMARY_20261114.md` |

已知边界更新：Fate 簇方程在本机仍为新生状态（适应 0–3 次），边界案
布局敏感属训练状态问题；实体-属性绑定与干扰题场读沉默仍在；出口
毒丸纪律目前仅服务器层（本地版 premise 8/8 复述毒丸行）。其余边界
（LoCoMo 同题 0/15、proactive 未实现、多跳/聚合天花板）与上午表述
一致，不因新结果改变。

## 当日续报（2026-11-15）：双墙拆除 + Fate 被动抓取 + 定位变更

| 项 | 终态 | 工件 |
|---|---|---|
| 快照去重（治本） | tile/cell 全局一份 + fragment id 引用；492 行 37.23MB → 0.94MB；627 行整体摄入无墙 | `valhalla-local` 引擎新构建 |
| 问答延迟墙 | 判前 adapt 定位并拆除（逐行全量导出 + 整库回喂）；560 行 ~155s/题 → 0.4–0.6s/题 | `WALL_CAUSE…` §5 |
| Fate 被动抓取 | 本地 `VALHALLA_QA_PASSIVE_FATE=1`：只读激活量作亲和、判后喂 `receive_passive` 训方程；探针已撤 | `COMPREHENSIVE_SUMMARY` §7.2/§12 |
| 极限复测 | 312 轮核心 90/90、编造 0、总时长 **1.23 分钟**（上轮 20.5 分钟） | `valhalla-local/docs/EXTREME_LIMITS_LOCAL_312TURNS_20261114.md`（已按新工件重生成） |
| 循环自优化审计 | 四层均已实现；缺后台回路与问答侧增益（只查未改） | `COMPREHENSIVE_SUMMARY` §12.1 |
| **定位变更** | **Fate 原生长时记忆基质**：不检索不生成、以自身方程裁决答/沉默、随试用时长自我训练；不再与记忆 API 同题竞分 | `COMPREHENSIVE_SUMMARY` §12 |

已知边界再更新：整体摄入 QA 在大块会话上抓取正确但 margin 门沉
默（n_start 型）属"法官未训练"的现形，随判后学喂养与时长摊平；
服务器部署版未同步本轮引擎（仍为 `20261114-optimized-BC` 口径）。
