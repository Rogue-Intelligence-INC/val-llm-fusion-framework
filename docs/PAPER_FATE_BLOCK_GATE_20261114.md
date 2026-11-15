# Block-Local Convergence Gating: Emit/Silence Decisions Controlled by a Learned Cluster Equation in a Signal-Native Memory Architecture

**块内收敛裁决：信号原生记忆架构中由自有簇方程控制的发声/沉默机制**

Rogue Intelligence LNC · 2026-11-14 · **工作草稿（DRAFT，未投稿）**
配套总汇：`docs/COMPREHENSIVE_SUMMARY_20261114.md`；全部数字源自本机实测工件。

## Abstract

Retrieval-augmented and memory-augmented LLM systems typically decide
*whether* to answer with machinery external to the memory substrate:
calibrated confidence thresholds, ranker margins, or the generator's
own self-assessment. Each introduces constants fitted to a benchmark
and failure modes that shift with the test distribution. We describe
and evaluate an alternative in which the emit/silence decision is a
function of the memory substrate's own convergence state. In the
Valhalla signal-native architecture, ingested lines are closed into
blocks by a bundling test (`bundle_fits`) driven by a cluster equation
((x−1)² in its birth form) whose coefficients and threshold are adapted
only *after* judgments, on a fixed cadence. At question time, when a
global margin rule would abstain, the decision is re-judged by
competition **inside the grasped block**: the system emits iff its own
live equation rejects the fit between the grasped line and its
strongest in-block competitor. The gate therefore contains no
benchmark-fitted constant; its threshold is learned state. In a
312-turn persistent-session limit test run against the bare local
engine (no gateway, no retrieval-side policy code), the mechanism held
core accuracy at 90/90 with zero fabricated tokens and zero answer
drift across all session thirds, was invariant to corpus permutation,
and exhibited only stationary, pre-identified failure modes. We also
report a diagnostic negative result: because post-judgment adaptation
requires ~1000 passive signals per adjustment while QA sessions supply
10–30, the judging equation remains in its birth state in every
session we measured, and boundary cases straddle the birth threshold
exactly as this predicts. The practical scaling limit of the system is
ingestion throughput (per-turn latency growing superlinearly to ~12 s
at ~400 session lines), not context capacity or hallucination. We are
explicit about what the mechanism does not do: on identical questions
it scores 0/15 on a LoCoMo-style slice where plain RAG scores 9/15,
and proactive emission is not implemented.

## 1. 引言

记忆增强系统的核心风险不在"答错"，而在裁决权的归属：谁决定答与
不答？主流答案有三类——检索分数阈值、生成器自评、校准集上拟合的
弃权规则。三者的共同点是裁决状态**外生于**记忆基质：换语料分布
就要重调，失败模式随测试集漂移，且"为什么沉默"不可审计到基质
内部状态。

本文研究的系统（Valhalla）采取相反立场：记忆基质自身有收敛动力
学（信号被动摄入、瓦片/词干收敛关闭、集成束、场强抓取），裁决应
该是这一动力学的读数。本文贡献：

1. **机制**：块内收敛裁决——全局边距规则弃权时，改由被抓取块内部
   的竞争 + 基质自有簇方程裁决发声/沉默（§2）。
2. **形态诊断**：该机制在服务器部署形态下曾不存在——三处形态缺陷
   （头部行未入块、片段全量克隆、分段 id 碰撞）使"块"名存实亡；
   逐项修复与实测（§3）。
3. **判别方法**：给出"机制 vs 优化"的五条可证伪判别并逐项验证
   （§4–§5），包括排列不变性与 312 轮长会话段平直性。
4. **负结果**：裁决方程在当前喂养节奏下从未受训，边界案行为被
   此定量预测（§6.3）——这是机制主张自带的下一个可证伪点。

## 2. 架构与机制

### 2.1 一链一币

信号入 → Fate 被动摄入 `receive_passive` → 收敛关闭瓦片/词干 →
集成束 → 场（场强 = 能量·√复杂度）→ 刺激激活块 → 只按场强抓取 →
`react` 只读出口（沉默合法）→ **判后学**：反馈在裁决之后进入；
摄入节奏为每 10 个信号判定一次、每 100 次判定才真实调整一次
方程系数/阈值（阈值钳制 [0.01, 1.0]）。

### 2.2 簇方程与集成束

簇方程出生形态 (a,b,c)=(1,−2,1)，即 evaluate(x)=(x−1)²。
`bundle_fits(d0,d1)`：x=d0/d1，拟合 iff (x−1)² < 自有阈值 θ。
设计语义为"两者足够接近则同束"；曾出现的漂移根（~0.45/2.13）
经所有者裁定为漂移并废弃。

### 2.3 块内裁决门

部署问答路径中，fog 层以**全局**排名边距弃权，而抓取是**块局部**
的——两者错位产生结构性误弃权（例：针为场峰，但同实体错属性行
在全局与之并列，全局边距 0.000266）。门规则：当 fog 弃权原因为
边距类时，取被抓取行所在块 [a,b] 内的最强竞争者读数 (s1,s2)，

> emit ⟺ 块内无竞争者，或 ¬bundle_fits(s1, s2)（以会话活方程计）；
> 否则维持沉默。tau/native_reject/defer 类弃权不经过此门。

关键性质：门中唯一的阈值 θ 是方程自身状态，由判后学在运行时塑
造；放行路径上不存在任何针对题集拟合的常数（曾短暂存在的校准
常数门已按所有者裁定删除并以 grep 核验不在路径上）。

## 3. 落地形态修复

机制声称成立前，须先证明"块"在部署中存在。三处形态缺陷：

| 缺陷 | 后果 | 修复 |
|---|---|---|
| 预学头部行 [0, prelearn_n) 被 pulse 却不入任何块 | 首块不存在，会话首部事实结构性不可答 | 头部关入首块（`prelearn_head`，seg_0） |
| 片段导出克隆累计瓦片 | 线缆每 ~2 轮翻倍，24 轮对话第 16 轮 HTTP 413 | 片段只带本 tick 新瓦片，按 tile id 合并去重 |
| 分段 id 跨追加碰撞 | 块边界错乱 | 分段器以会话已有轨道数为种子 |

另：候选系列 5→12；自适应代数读数改读活亲和（快照缺亲和时曾静默
回落出厂 δ=0.075 的"假自适应"已消除）。动态卸载 A/B（24 轮）：
开启 1.49MB 完赛 vs 关闭 5.92MB（修复前 413 中断）。

## 4. 机制 vs 优化：五条可证伪判别

| # | 判别 | 机制应然 | 本系统实测 |
|---|---|---|---|
| 1 | 拟合常数 | 放行路径无题集拟合常数 | 删除后 grep 核验；锚点明文排除 |
| 2 | 阈值来源 | 系统自有学习状态 | 活方程遥测逐行暴露（系数/θ/适应次数） |
| 3 | 位置不变性 | 语料排列不改变机制行为 | 逆序/洗牌/极端种子三布局，针题均抓对块发声 |
| 4 | 长程平直 | 指标不随会话长度退化 | 312 轮三段平直（§5.4） |
| 5 | 失败结构 | 失败为先验已知的静止缺口，无规模涌现型 | 失败在首轮段即全部定性（§5.4） |

作为对照，本系统另有两处明示的**优化**（不记入机制功劳）：重复
连问的 follow-up 自证加成切断（边距通胀 0.245→0.651 变为恒定
0.245）；删除头部行的位置惩罚。

## 5. 实验

统一环境：本地构建引擎（valhalla_base 1.1.0）+ Mistral-24B 解码
头（烘焙强度 0.08）；语料为 10,006 行合成事实语料（客户端持有，
会话逐轮累积定向行）；记分规则与工件路径见配套总汇。

### 5.1 20 题电池
Fate 门版 vs 已删除的校准常数版：仅 1 处翻转（absent_ent 弃权→
发声；其 (s1/s2−1)²=0.0153 ≥ 出生 θ=0.01，旧沉默系全局并列偶然，
定性为绑定/出口层问题，非门问题）。

### 5.2 66 轮对话
核心 24/24；陷阱 9/12，网关出口补毒丸行清洗后 10/12；编造 0；
轮换 0；线缆 1.07MB。

### 5.3 扰动审计
MM 逆序、seed-99 洗牌、seed-777 三种布局：针题在块 [5,6]/[2,4]/
[0,1] 均正确发声；释义、冲突、缺失属性题沉默保持。边界案（q2、
swap1）读数 0.0078↔0.0107–0.0138 跨出生 θ 摆动（见 §6.3）。

### 5.4 本地版 312 轮极限（主实验）
引擎直连、零服务器代码、引擎原生答策（fog 放行→逐字答，否则
沉默），30 轮 ×（核心 3 + 干扰 4 + 陷阱 2 + 冲突 1）+ 周期性
bulk 追加，20.5 分钟，会话 8→422 行、零轮换：

| 指标 | 前段 | 中段 | 后段 | 合计 |
|---|---|---|---|---|
| 核心 | 30/30 | 30/30 | 30/30 | **90/90** |
| 编造 token | 0 | 0 | 0 | **0/312** |
| 陷阱（重评分） | 12/20 | 12/20 | 12/20 | 36/60 |
| 干扰题被答出 | 2/40 | 0/40 | 0/40 | 2/120 |

三道核心题各自答案全程唯一（零漂移）。陷阱失败穷尽定性：
premise 8/8 逐字复述语料毒丸行（本地版无出口清洗，该纪律现位于
服务器呈现层——对照 §5.2 的 10/12）；absent_attr2 8/8 错属性绑
定；inject_q 7 沉默+1 正确（沉默被针式记分记 fail，行为合纪
律）；swap1 单次错边。每轮延迟由 ~0.5s 升至 ~12s（会话 ~400
行），单轮线缆最大 3.08MB，无协议层失败。

### 5.5 窗口不变性（本地版整体摄入）

以整体摄入（无客户端定向选行）复测窗口内外区分：121 行与 318 行
两臂的 10 题行为逐行一致（核心同答、沉默同位、编造 0；延迟
0.2→1.3s）。806 行臂在摄入 492 行时会话状态由 3.78MB 爆炸至
37.29MB 并被引擎拒收（HTTP 413），墙后不可问（独立探针复现：
426 行/35.12MB）。即真窗外在整体摄入下不可达——限制先于 LM 窗
口出现，是基质状态预算而非窗口；生产形态（定向摄入，每题触达
≤12 行）下不区分由 §5.4 的 312 轮证明（客户端语料 ~150K token
全在窗外）。附带发现：link 变束属基质组织而非传输优化——无 link
时同臂核心仅 2/3 且缺失题错答，有 link 后 3/3 且全沉默。全记录
`valhalla-local/docs/WINDOW_INVARIANCE_20261114.md`。

## 6. 分析

### 6.1 裁决与抓取的同位性
门的有效性来自读数与抓取同位：抓取按块内场强发生，裁决也只读
同一块内的竞争。全局边距门的历史失败（场峰被并列压弃）是读数
错位，不是阈值松紧——这解释了为何删除校准常数后行为不退化。

### 6.2 零编造的来源
全部发声均为语料行逐字（呈现层不经生成器改写）。零编造不是解
码约束的功劳，而是"嘴"被限定为存储事实本体 + 沉默合法的共同
结果；其代价同样明确：多事实合成、口头化质量不由本机制提供。

### 6.3 法官未受训（负结果）
三轮遥测一致：本机全部会话（引擎直连与网关）的簇方程均为出生
状态（(1,−2,1)，θ=0.01，适应 0–3 次）。判后学节奏下一次真调整
约需 1000 个被动信号，QA 会话仅供给 10–30。§5.3 的边界摆动幅
度（跨度恰绕 θ=0.01）与"法官未受训"定量吻合。推論：当前边界
行为是训练状态而非设计缺陷；但喂养通道被摄入吞吐墙（§6.4）
阻挡，受训后的扰动审计应全绿——这是本机制最直接的可证伪点。

### 6.4 真极限：摄入吞吐
312 轮中正确性与幻觉指标无一退化；唯一随规模恶化的是每轮延迟
（超线性，源于 Fate pulse 对累积状态的摄入成本），本地直连原
样复现服务器部署的 bulk 卡点。故系统的可扩展边界在写入侧，
不在读取侧，与"上下文窗口"无关。

## 7. 局限（如实声明）

1. **同题基准不赢**：LoCoMo 式切片同题 Valhalla 0/15、RAG 9/15、
   Mem0 5/15。本机制主张的是裁决形态与长会话行为稳定性，不是
   该基准的分数；任何引用禁止混口径。
2. **实体-属性绑定缺口**：释义针抓对而沉默（块内相对读数仅
   0.04/0.06）、错属性题错答、干扰针多沉默（2/120 被答出）——
   场读分辨力问题，未解。
3. **裁决方程未受训**（§6.3）：边界案布局敏感，未解。
4. **摄入吞吐墙**（§6.4）：数百行会话后交互延迟不可接受，未解。
5. **其他已知天花板**：多跳一跳检索（0/2）、聚合被集束去冗吃掉
   （0/2）、矛盾事实静默选一、主动发声未实现（合格样本数 0）、
   多事实 top-k 出口未实现。
6. **证据范围**：单机、单解码头、合成语料；方程受训态、真实语
   料分布与跨周持续学习均未测。

## 8. 相关工作定位

与选择性预测/弃权学习的区别：弃权阈值不在校准集上拟合，而是
记忆基质收敛动力学的内生状态——其代价是阈值质量取决于基质的
喂养史（§6.3），其收益是裁决可审计到基质状态且随语料排列不变。
与 RAG/外部记忆 API（Mem0 类）的区别：本系统不以生成器或外部
排序器裁决，抓取与裁决同处一个基质；对照分数见 §7.1，我们不
在对方强项口径上宣称胜出。与系统自身服务器期 hybrid 路线的区
别：词法轨道、弱 emit、常数门均已停放或删除，本文全部数字为
裁决权单一化之后测得。

## 9. 结论

答/沉默由记忆系统自有收敛状态裁决的形态，在无拟合常数、无服
务器辅助、排列扰动与 312 轮长会话下保持行为稳定，取得核心
90/90 与零编造；其失败全部位于先验已知的结构坐标（未受训的
法官、未落地的绑定、未下沉的出口纪律、写入侧吞吐）。下一可证
伪点明确：打通判后学喂养后，扰动审计边界案应由摆动转为稳定。

## 参考工件

- 总汇与判别细则：`docs/COMPREHENSIVE_SUMMARY_20261114.md`
- 312 轮全记录：`valhalla-local/docs/EXTREME_LIMITS_LOCAL_312TURNS_20261114.md`
  （原始数据 `valhalla-local/results/extreme_limits_local{,_scored}.json`）
- 主报告 §14–§15：`docs/MISTRAL24_VS_VALHALLA_UNIQUE_POINTS_20261113.md`
- 回滚锚点：`archive/20261114-fate-p1-backup.zip`、
  `archive/20261114-fate-gate-anchor`、`archive/20261114-optimized-BC`
- 源码：`Valhalla-main/manifestsys/hub-f64/src/{fate_qa,native_qa,fate_orchestrator,fate_adaptive_algebra}.rs`

---
*Rogue Intelligence LNC · DRAFT · 数字均来自本机实测，未测项已标注。*
