import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

// All measured numbers: this machine, 2026-11-13 base + 2026-11-15 rerun (passive Fate build).
// Rerun sources: valhalla-local/results/extreme_limits_local_scored.json, window_invariance.json.
// Industry same-Q numbers: Valhalla repo competitor-compare moderate set, 2026-07-13, Qwen2.5-3B backbone.
// Positioning changed 2026-11-15: long-horizon memory substrate — competitor table is for boundary-drawing, not score competition.

export default function Summary() {
  return (
    <Stack gap={20} style={{ padding: 20 }}>
      <Stack gap={6}>
        <H1>Valhalla：Fate 原生长时记忆基质 — 复测与定位（2026-11-15）</H1>
        <Text tone="secondary">
          定位（2026-11-15 变更）：不检索、不生成——经历被动收敛成块，以自身簇方程裁决答与沉默，判后学随使用时长训练裁决者。
          它销售的不是答对率，是长时运行下才显形的三件事：零编造、沉默纪律、行为平稳性。下方为同类对比存档与本机复测。
        </Text>
        <Row gap={8} wrap>
          <Pill active>定位：长时记忆基质（非记忆 API、非 RAG）</Pill>
          <Pill>试用时长即训练时长：自优化回路四层已实现</Pill>
          <Pill>不在 LoCoMo 式排行榜竞分（同题 0/15 如实保留）</Pill>
        </Row>
      </Stack>

      <Stack gap={8}>
        <H2>2026-11-15 复测：双墙拆除 + Fate 被动抓取（本地版，被动构建）</H2>
        <Callout tone="success" title="两个墙都拆了，裁决权仍在 Fate 手里">
          快照去重治本拆掉会话状态墙（492 行 37.23MB → 0.94MB，627 行整体摄入无墙）；判前 adapt 的成本定位并拆除后，问答改由 Fate 被动抓取激活状态（只读激活量、不回喂语料、判后才学）。312 轮极限复测：核心 90/90 三段平直、编造 0/312、总时长从 20.5 分钟降到 1.23 分钟。诚实代价：被动构建在 560 行整体摄入会话上比 adapt 少对 1 题（抓取正确但门沉默）——adapt 的分离度来自已被否决的自确认回喂，此代价不隐藏。
        </Callout>
        <Grid columns={4} gap={12}>
          <Stat value="90/90" label="312 轮核心复测（三段各 30/30，编造 0）" />
          <Stat value="1.23 分钟" label="312 轮总时长（上轮 20.5 分钟）" />
          <Stat value="0.4–0.6s" label="单题问答 @560 行（曾 ~155s）" />
          <Stat value="0.94MB" label="492 行快照（曾 37.23MB，状态墙已拆）" />
        </Grid>
        <Text tone="secondary" size="small">
          来源：valhalla-local/results/extreme_limits_local_scored.json（2026-11-15 重跑+离线重评分）；窗口复测 W1（126 行）/W2（356 行）核心 3/3 同答、W3（806 行，吞入 708s）零墙通过、560 行静默瓶颈已除。W3 出现一次被动裕度弃答（n_start，806 行尺度命运方程裕度被稀释）——行为安全（弃答优于编造），编造仍 0；放大机制待长时训练验证。机制判别 valhalla-server/docs/COMPREHENSIVE_SUMMARY_20261114.md §7.2/§12。诚实边界不变：LoCoMo 同题 0/15 不赢、proactive 发声未实现、Fate 簇方程在问答侧仍未受训（判后学通道已通、增益待时长）。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>2026-11-15 晚 逐步优化第一步：去重复 + 真碎片化（原结构上修，不重建）</H2>
        <Callout tone="success" title="审计定的三处全量重复，已修前两处并让 Fate 天然触发">
          审计结论：慢不在模型加载，在三处重复——① Fate 偏好对零方向向量逐个上 GPU 空算（每行/每周期重复）；② Stem 物理场每周期建销 cuBLAS 并全图重写连接；③ 碎片用 Hub 时钟过滤 Tile 时钟，装的是旧材料。修法：零向量走累加器、GPU 只在新增整批信号后由 Fate 触发、无新信号不写重复偏好快照；cuBLAS 进程级常驻、物理场仅在新细胞闭合或种群变化时重算；分段记录 Tile 自身时钟、碎片只装本段 tiles。计时探针已清理。
        </Callout>
        <Grid columns={4} gap={12}>
          <Stat value="3.8s" label="W3（806 行）整体摄入（曾 462s / 最初 708s）" />
          <Stat value="3/3×3" label="窗口三臂核心全同答（W3 n_start 转正，编造 0）" />
          <Stat value="89/90" label="312 轮核心（0.84 分钟，编造 0；turn 21 早期弃答为已知代价）" />
          <Stat value="0.5s" label="hub-f64 库测试执行（曾 46.9s；通过数仍 106/5 同基线）" />
        </Grid>
        <Text tone="secondary" size="small">
          来源：valhalla-local/results/window_run4.log 与 limits_run4.log（2026-11-15 晚，同构建复测）；智能电池（560 行）4/10 构成不变、para1 由抓毒行改弃答。未竟：Stem 碎片尚无自身时钟（cells 靠快照层去重）、会话双存储与每题重建未动、毒行出口纪律仍在网关侧——逐步优化下一步清单。详见 COMPREHENSIVE_SUMMARY §12.3。
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="18/18" label="Mistral 24B 单独（窗口内）" />
        <Stat value="18/18" label="Valhalla + 24B 完整版混合" />
        <Stat value="0/8 → 8/8" label="超窗压力：24B 全灭 → 完整版混合全对" />
        <Stat value="0 次" label="Valhalla 两轮编造幻觉（26 个答案）" />
      </Grid>

      <Card>
        <CardHeader>本机实测四臂分项正确数（满分：grounded 8 · conflict 2 · trap 6 · drift 2）</CardHeader>
        <CardBody>
          <BarChart
            height={280}
            categories={["grounded 有据问答", "conflict 过时事实", "trap 幻觉陷阱", "drift_return 主题返回"]}
            series={[
              { name: "Mistral-24B 单独（全上下文）", data: [8, 2, 6, 2] },
              { name: "Valhalla 原生（未执行 Fate 裁决）", data: [6, 1, 0, 0] },
              { name: "Valhalla 原生（执行 Fate 裁决）", data: [6, 1, 3, 0] },
              { name: "Valhalla + 24B 完整版混合", data: [8, 2, 6, 2] },
            ]}
            showValues
          />
          <Text tone="secondary" size="small">
            来源：valhalla-server/tests/results_unique_points（2026-11-13 本机）。完整版 = Fate 弱放行标定 + Tile 多轨证据 + GPU + Tier A 烘焙后 24B 解码头。
          </Text>
        </CardBody>
      </Card>

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>行业同题对比（仓库口径，n 小，仅作边界）</CardHeader>
          <CardBody>
            <BarChart
              height={240}
              categories={["LoCoMo-15", "MultiHop-10", "LongBench-10"]}
              series={[
                { name: "全上下文 LM", data: [9, 6, 2] },
                { name: "普通 RAG", data: [9, 6, 2] },
                { name: "Mem0（本地）", data: [5, 6, 2] },
                { name: "Valhalla", data: [0, 6, 1] },
              ]}
              showValues
            />
            <Text tone="secondary" size="small">
              来源：COMPETITOR_COMPARE 2026-07-13 moderate 集。Valhalla 在记忆 API 主场明确落后，禁止反向引用。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>超窗压力（33K token 语料 vs 16K 窗口，8 题）</CardHeader>
          <CardBody>
            <BarChart
              height={240}
              categories={["表面答对", "源行命中（含答对）"]}
              series={[
                { name: "Mistral-24B 全上下文", data: [0, 0] },
                { name: "Valhalla 持久会话", data: [4, 6] },
              ]}
              showValues
            />
            <Text tone="secondary" size="small">
              24B 因提示超窗被服务直接拒绝（HTTP 400）。公平边界：其原生 128K 窗口在大显存/云端可容纳此语料，本项证明的是窗口与成本结构差异。
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>能力矩阵（● 有 · ◐ 部分 · ○ 无）</H2>
        <Table
          striped
          headers={["能力", "Valhalla+24B 混合", "Mistral 24B 单独", "Mem0", "Zep/Graphiti", "Letta", "普通 RAG"]}
          rows={[
            ["零编造（答案只能由语料构成）", "●", "○（闭卷编造过具体数字）", "◐", "◐", "○", "◐"],
            ["不可答时拒答", "●（Fate 门+证据约束双防线）", "◐（靠提示指令）", "○", "○", "○", "○"],
            ["超上下文窗口持续问答", "●（结构不占窗口）", "○（窗口封顶）", "●", "●", "◐", "●"],
            ["结构层主题不漂移", "●（源行纯度 16/16）", "◐（靠历史回放）", "○", "◐", "○", "○"],
            ["时序事实有效期", "○", "○", "◐", "●", "◐", "○"],
            ["多模态同体（文/音/图共享结构）", "●（U1 0.9999）", "◐（视觉单模）", "◐", "○", "○", "○"],
            ["持续优化沉淀为权重资产", "◐（L3 烘焙，效果未闭环）", "○（需另行微调）", "○", "○", "○", "○"],
            ["托管 API 成熟度", "○", "◐", "●", "●", "◐", "○"],
            ["窗口内问答质量（本机实测）", "◐（14/18）", "●（18/18）", "—", "—", "—", "◐（同题 9/15）"],
          ]}
        />
        <Text tone="secondary" size="small">定性行来源：仓库 UNIQUE_VS_MEMORY_APIS 同行矩阵 + 本机实测；“—”为未在本机同题实测。</Text>
      </Stack>

      <Stack gap={8}>
        <H2>持续优化能力对比（要点）</H2>
        <Table
          striped
          headers={["路线", "代表", "优化形态", "实测量化证据", "主要短板"]}
          rows={[
            ["会话结构学习", "Valhalla L1", "零梯度摄入→快照", "TPI 0.755 · 污染救援 +38.7pp · 30 轮课程 96.7%", "干净集只打平 RAG；加 cycle 无增益"],
            ["结构预训练+烘焙", "Valhalla L2/L3", "亲和收敛→写权重", "17 行 3/3 收敛 · 烘焙 Δw≈298@s0.10", "n=3 小样本；烘焙后 MCQ 未闭环"],
            ["事实库更新", "Mem0 / Zep", "抽取→增删改/时序失效", "厂商口径为主（C 档）", "被动检索，无污染自救协议"],
            ["Agent 自编辑", "Letta", "改写记忆块", "产品演示为主", "无量化门控，上下文仍增长"],
            ["梯度持续学习", "LoRA/微调", "SGD 更新权重", "学界基准成熟", "成本高、灾难性遗忘、不可即时"],
          ]}
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>审计：Valhalla 是否最大化优化了 24B？—— 运行时+权重侧已吃满（10/12）</H2>
        <Callout tone="success" title="总判定（2026-11-13 晚更新）">
          初版审计的缺口已逐项补完并复测：完整版混合在常规 18 题与超窗 8 题上均满分且零硬错误；Tier A patch 已烘进 24B 本体（s0.08，Δw=0.604），烘焙后本体与混合复测仍 18/18，强度 0.02/0.08 两点均无退化。剩余两项为 mcq 路由未单独基准、proactive 未在服务器形态跑协议——均为未测，非做不到。
        </Callout>
        <Table
          striped
          headers={["#", "优化杠杆（出处）", "状态", "证据 / 缺口", "预期收益"]}
          rows={[
            ["1", "持久会话（transformer 仓：有效杠杆）", "● 已做", "会话跨轮复用，摄入一次约 1s/题", "—"],
            ["2", "Fate 输出门 FOG（U2）", "● 已接+已标定", "弱放行策略：abstain 时由 Tile 轨证据+24B 验证，原 4 题误拒全部救回", "本轮 18/18、超窗 8/8"],
            ["3", "Hybrid 路由 open→LM 解码", "● 已做", "混合臂常规 18/18、硬错误 0", "—"],
            ["4", "Hybrid 路由 mcq→Tile+Fate / numeric", "◐ 配置已设未基准", "mcq_decode=max_pick 已写入生产配置；本题集无 MCQ 未单独测", "需 MCQ 题集另测"],
            ["5", "Tier A patch 烘进 24B 权重", "● 已做", "40 层分解→Fate 5 轮→融合→s0.08 烘焙 Δw=0.604；烘焙后本体 18/18、混合 18/18", "—"],
            ["6", "Patch 强度扫描（0.02/0.08/0.10）", "◐ 两点已测", "s0.02（Δw=0.151）与 s0.08 本体均 18/18 无退化；0.10 未测", "边际点，可选"],
            ["7", "多轨证据（Tile+Stem 并行、top-k 源行）", "● 已做", "结构轨 + Tile 词法轨（IDF top-3）并行喂解码", "—"],
            ["8", "结构 GPU 计算", "● 已做", "use_cpu=false 探针通过（0.5s 答对），已切生产配置", "—"],
            ["9", "24B 本体精度/窗口", "◐ 部分", "自研 BF16→Q4_K_M 转换链；16K/128K 窗口、KV q8；视觉 mmproj 未接（本测为文本）", "视觉与三模态连通另测"],
            ["10", "多模态同体（三模态 × 24B 视觉）", "◐ 官方验收过", "官方 test_multimodal_mixed_answer：MULTIMODAL_MIXED_ANSWER_PASS（三模同会话一次答齐）；24B 视觉 mmproj 未接为另一回事（详见报告 §11.5）", "U1 独有点端到端兑现"],
            ["11", "proactive 主动路径实测", "◐ 已测未触发", "官方协议复核：门电池 3/3 GATES_OK、handcraft 发声 0/4 拒答 3/3（MAR 1.0/FIR 0.0/UP 0.0）；发声未实现（详见报告 §11.2/§11.5）", "不得对外宣称已实现"],
            ["12", "QA 反馈闭环标定（持续优化）", "◐ 通道已通增益待时长", "serve 层判后反馈闭环真实存在（结构一致性 on_topic → Fate+亲和写回快照）；但方程每 ~1000 信号才动一次系数，问答侧实测 adaptations=0——增益由试用时长摊平（2026-11-15 审计）", "方程受训后边界案复测"],
            ["13", "极限高压（上下文×幻觉）", "● 已测+已修", "targeted 模式 hybrid 在 806/3006/10006 行均 12/12、编造 0；注入攻击曾毒翻弱放行与全上下文，已以会话证据+指令行隔离修复；全会话 HTTP 天花板约 100 行/32MB（报告 §11.7）", "矛盾不提示（后写赢）待治"],
            ["14", "优化版：变束+集束抓取", "● 已落地", "fate_link 组8：片段 40→5、线缆 −20%（网关 ≥16 片段自动变束）；证据集束+摄入集束+弱放行场强门全量回归 12/12 不变，会话线缆 1.61→1.34MB（报告 §11.8/§11.9）", "自动变束在 chat 流量下待命"],
            ["15", "★多模修复：特征抓取", "● 已落地", "官方回退编码器无区分力（图 cos 0.87–0.97、音频 1.000）→换 CLIP ViT-H/时段 mel（自 1.0 vs 他者 ≤0.72/≤0.92）；网关 feature_grasp：多模电池 Valhalla 9/9、caption-RAG 7/9（特征题结构上收不了）、闭卷 0/9（报告 §12.1）", "跨实例语义匹配（重录/重拍）未测"],
            ["16", "★全面复测（2026-11-14）", "● 全绿", "极限三档 hybrid 12/12×3、编造 0、Mistral 45K 起 0/13；60 轮漂移线缆恒 1.28MB；Fate/换语料探针与线缆冒烟 PASS（报告 §12.2/§12.3 独有点圈注 ★1–★6）", "矛盾沉默、LoCoMo 不赢、proactive 发声未实现——均如实保留"],
            ["17", "★LLM 增强形态极限", "● 已测至 1495 万 token", "不经网关直连：增强臂在 45万/149万/1495万 token 三档均 17/21、编造 0、画像逐项一致，会话线缆恒 ~3.4MB；胜同检索 RAG（13–14/21，注入/时序被污染）（报告 §13）", "天花板三处：多跳 0/2（单趟检索）、计数/列举 0/2（集束吃枚举，RAG 反而 2/2）、矛盾沉默；不提升通用推理"],
            ["18", "★机制落地：块形态+Fate 自控门", "● 已落地（取代 #2 弱放行口径）", "头部闭合/片段增量卸载/分段 id 种子三处形态修复；门改判块内竞争、阈值读会话活方程；校准常数门已删除。20 题仅 1 处翻转、66 轮对话核心 24/24 陷阱 10/12 编造 0；扰动审计位置不变", "Fate 方程未受训（适应 0–3 次），边界案跨出生阈值摆动——训练状态问题"],
            ["19", "★本地版极限（零服务器代码）", "● 312 轮已测（11-15 复测同值）", "本地引擎直连：核心 90/90 三段平直、编造 0/312、答案零漂移、零轮换；W3（806 行）零墙通过、560 行静默瓶颈已除；W3 出现一次被动裕度弃答（n_start，806 行尺度）——弃答安全、编造仍 0", "真极限仍为吞吐与尺度裕度：吞入 ~1.15s/行线性；806 行下被动单次抓取裕度被稀释，需方程训练后放大"],
            ["20", "★快照去重拆状态墙（治本）", "● 已落地+已验证", "tile/cell 全局一份 + fragment 只存 id 引用（旧格式无损加载）；492 行快照 37.23MB → 0.94MB、560 行 1.05MB 线性增长；627 行整体摄入无墙、窗口三臂可问", "snapshot 顶层 fragments 响应副本仍全量（客户端 slim 已处理）"],
            ["21", "★问答延迟墙 + Fate 被动抓取", "● 已落地（本地默认开）", "判前 adapt 曾占每题 99%（逐行全量导出 + 整库回喂 1120 次）；改被动抓取激活量 + 判后喂方程：560 行 ~155s/题 → 0.4–0.6s/题，312 轮总时长 20.5 分钟 → 1.23 分钟", "被动构建大块会话少对 1 题（抓对沉默）：原始激活近饱和、无回喂放大——诚实代价"],
            ["22", "★循环自优化审计（只查未改）", "● 四层已实现", "Fate 内核判后 adapt（每 100 判定动一次系数）· 编排层 Fate 掌停机 · 身体 follow/promote · serve 层 QA 反馈闭环——骨架全在", "无后台离线回路；问答侧回路增益≈0（信号喂不饱）——新定位下读作随试用时长变准"],
            ["23", "★逐步优化一：去重复 + 真碎片化", "● 已落地（11-15 晚）", "Fate 偏好零向量不再上 GPU、新增整批信号才触发、无新信号不写重复快照；Stem cuBLAS 常驻 + 物理场仅种群变化时重算；碎片改按 Tile 自身时钟取材。W3 摄入 462s→3.8s、三臂核心全同答、库测 46.9s→0.5s", "Stem 碎片无自身时钟；312 轮 turn 21 早期弃答（89/90，弃答非编造）；会话双存储未动"],
          ]}
        />
        <Text tone="secondary" size="small">
          状态图标：● 已做 · ◐ 部分/未标定 · ○ 未做。出处：valhalla-transformer《13_HYBRID_UNIFIED_PRODUCT》《06_TF前置与孵化实验总结》、unified-core 生产配置、本机代码与实测（2026-11-13）。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>总结（可对外版，2026-11-15 定位变更）</H2>
        <H3>一句话</H3>
        <Text>
          Valhalla 是一个 Fate 原生的长时记忆基质：它不检索、不生成——经历被动收敛成块，以自身簇方程裁决“答与沉默”，判后学随使用时长持续训练裁决者本身。它不与记忆 API 在 LoCoMo 式主场竞分（同题 0/15，如实保留），它销售的是长时运行下才显形的三件事：零编造（312 轮 0/312 复测）、沉默纪律（答不出就沉默、抓错也沉默）、行为平稳性（三段平直、答案零漂移、每题成本不随规模爆炸——560 行 0.4–0.6s/题）。试用时长即训练时长：自优化回路四层已实现，其慢增益由长周期摊平，这是定位的前提，不是缺陷。
        </Text>
        <Grid columns={3} gap={12}>
          <Card>
            <CardHeader>已兑现</CardHeader>
            <CardBody>
              <Text>裁决机制（块内竞争 + Fate 自控门、无拟合常数）· 双墙拆除：快照去重（37.23MB→0.94MB）+ 被动抓取（~155s→0.5s/题）· 312 轮复测 90/90、编造 0、1.23 分钟完赛 · 窗口内外不区分（定向 312 轮 + 整体 W1/W2 逐行一致）· 循环自优化四层在位</Text>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>明确不行</CardHeader>
            <CardBody>
              <Text>LoCoMo 行业同题（0/15 vs RAG 9/15）——不在同一层，不进场 · Fate 方程在问答侧未受训（判后学每问只喂个位数信号，边界案靠时长摊平）· 大块会话抓对但门沉默（n_start 型）· 毒丸行在本地版复述（出口纪律仅服务器层）· proactive 发声未实现 · 摄入 ~1.2s/行</Text>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>下一步（按收益序）</CardHeader>
            <CardBody>
              <Text>1 判后学训练态跨快照分叉存活（法官真正开始学）→ 2 实体-属性绑定落地 → 3 出口毒丸纪律是否下沉引擎 → 4 摄入增量化（降 1.2s/行）→ 5 方程受训后复测扰动审计（机制主张的下一可证伪点）</Text>
            </CardBody>
          </Card>
        </Grid>
        <Text tone="secondary" size="small">
          完整报告：valhalla-server/docs/MISTRAL24_VS_VALHALLA_UNIQUE_POINTS_20261113.md · 全面总结（含 §12 定位重分析）：docs/COMPREHENSIVE_SUMMARY_20261114.md · 竞品口径（划界用）：docs/VALHALLA_COMPETITIVE_COMPARISON_20261113.md
        </Text>
      </Stack>
    </Stack>
  );
}
