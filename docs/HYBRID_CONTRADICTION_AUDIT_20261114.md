# Hybrid 矛盾审计（2026-11-14）

背景：owner 指出 hybrid 政策（弱放行+多轨证据+transformer 口头化）
来自 valhalla-transformer《13_HYBRID_UNIFIED_PRODUCT》，不是 owner
的设计。owner 架构：Fate 与 Valhalla 框架回答，transformer 不参与
决策。本审计逐条对着代码查 hybrid 与该设计的矛盾。

## 矛盾清单

1. **门的权威是假的**：`hybrid.py` 旧注释称 "Fate remains the only
   gate"，但 `resolve` 的 weak-emit 路径在 Fate abstain 时凭词法
   证据+transformer 验证照样开口——最终决定权在词法轨+transformer。
   `app.py:359` 注释 "Fate is the authority on whether to speak:
   honor FOG" 的下一行就是这个覆盖调用，注释与行为互相矛盾。
2. **field_gate 名实不符**：名为 Fate 场强门，实现是 Python IDF
   词法分数与均值比较；引擎的场/亲和（corpus_line_affinity、
   field_strength）从未被读。
3. **证据不是 Fate 选的**：build_evidence 里 Fate 只给 1 行，其余
   由词法轨补齐；Fate 的 native 答案被降格为"证据/兜底"，交给
   transformer 重新裁决——回答权实际在 transformer。
4. **transformer 有否决权**：emit 路径上 transformer 返回
   "I don't know" 会被原样当作答案输出（实测 swap2/premise），
   等于下游否决了 Fate 的 emit。
5. **覆盖路径即攻击面**：§11.7 实测注入毒翻的正是 weak-emit
   （3 题 AUR-999）；Fate 原生从未中招。覆盖 abstain 的口子
   就是毒进来的口子。
6. **两套回答权威叠床架屋**：引擎本身会原生回答（fate egress、
   multihop_fate 实体抽取），网关算完又丢给 transformer 重答；
   `internal.chat_config` 里 decode=hybrid 与 follow_up_decode=
   native_follow_up_aware 并存，同一系统两处"谁说了算"不一致。
7. **弃权决策被外包**：mistral_head 的 SYSTEM 明文指示 transformer
   "证据不足就说不知道"——把弃权判断写进了提示词，而这正是
   Fate 代数（margin vs δ/(dim+r)）该独占的决策。

## 处置（2026-11-14 已实施）

- 产品路径（app.py 聊天/媒体问答/特征抓取）改用
  `hybrid.resolve_fate_answer`：Fate emit → Valhalla native 答案
  原文；abstain → 沉默文本；transformer 不再被咨询。特征抓取
  直接返回被抓取的语料行原文。
- `hybrid.resolve` 保留为测试对照臂（在案可复测），文件头已
  标注其政策来源与本清单，不再冒充产品设计。
- 代价预期：native 答案是结构碎片，口头化质量与多事实合成
  会掉分（引擎直连 native 臂极端档 4/12 为先行参照）；
  换来的是裁决权单一、注入面收敛、答案可溯源到 Fate 读数。

## 实测（2026-11-14，网关多模电池）

- native 碎片原文版：**3/9**——Fate 抓对行（idx 全对）但出口
  读数是 `[fate:18@2] Project Aurora` 式实体碎片；
- 改为返回 Fate 抓取行原文（框架事实本体，仍是零 transformer）：
  **6/9**（q3/q4/q6/q7/q8/q9 过；特征抓取返回语料行反而最干净）；
- 剩余 3 题全是 Fate 侧问题、与 transformer 无关：
  q1/q2 抓对行（idx 0/3）但代数弃权（margin 未过 δ）、
  q5 需两条事实而 top-1 只给一条——即待办的 fate_grasp top-k
  场序出口（BOOST 仓 §8）要解决的正是这两类。
- 对照：transformer-hybrid 政策同电池 9/9。那 3 分买的是
  裁决权外包+注入面，owner 已裁定不买。
