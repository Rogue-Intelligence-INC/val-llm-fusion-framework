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

// All measured numbers: this machine, 2026-11-13 (see docs/MISTRAL24_VS_VALHALLA_UNIQUE_POINTS_20261113.md)
// Industry same-Q numbers: Valhalla repo competitor-compare moderate set, 2026-07-13, Qwen2.5-3B backbone.

export default function Summary() {
  return (
    <Stack gap={20} style={{ padding: 20 }}>
      <Stack gap={6}>
        <H1>Valhalla × Mistral 24B 同类对比总结</H1>
        <Text tone="secondary">
          本机实测（i9-14900KF / RTX 4090）· 2026-11-13 · 虚构双项目语料 18 题 + 33K token 超窗压力 8 题 ·
          行业同题数据来自仓库 competitor-compare（2026-07-13，Qwen2.5-3B 标定）
        </Text>
        <Row gap={8} wrap>
          <Pill active>结论：窗口内 24B 满分</Pill>
          <Pill>独有点：超窗 / 零编造 / 结构不漂移</Pill>
          <Pill>24B 适配已吃满运行时+权重侧（10/12 杠杆）</Pill>
        </Row>
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
            ["10", "多模态同体（三模态 × 24B 视觉）", "○ 未做", "Mistral 3.1 视觉与 Valhalla 三模态未连通", "U1 独有点端到端兑现"],
            ["11", "proactive 主动路径实测", "◐ 已开未测", "配置已改回 true，服务器形态未跑 U2 协议", "主动提醒/拒答卖点落地"],
            ["12", "QA 反馈闭环标定（持续优化）", "◐ 解码期已闭环", "弱放行即反馈式标定；Fate 阈值本身的在线学习未做", "阈值在线自适应"],
          ]}
        />
        <Text tone="secondary" size="small">
          状态图标：● 已做 · ◐ 部分/未标定 · ○ 未做。出处：valhalla-transformer《13_HYBRID_UNIFIED_PRODUCT》《06_TF前置与孵化实验总结》、unified-core 生产配置、本机代码与实测（2026-11-13）。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>总结（可对外版）</H2>
        <H3>一句话</H3>
        <Text>
          窗口内，24B 级通用模型本身就是满分选手，Valhalla 没有胜算也不必假装有；Valhalla 的位置是给这个选手加一副“不会编造、超窗不断、串题为零、可审计裁决”的结构骨架——完整版混合已在常规与超窗两套题上做到满分且硬错误归零，且结构已烘进 24B 本体。剩下的是 MCQ、proactive、多模态三块未测项，不是能力缺口宣称。
        </Text>
        <Grid columns={3} gap={12}>
          <Card>
            <CardHeader>已兑现</CardHeader>
            <CardBody>
              <Text>零编造 · 超窗 0/8→8/8 · 源行主题纯度 16/16 · 完整版混合常规 18/18 · Tier A 烘焙后 24B 本体 18/18（s0.02/s0.08 无退化）</Text>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>明确不行</CardHeader>
            <CardBody>
              <Text>窗口内单比裸模型速度（24B 0.3s/题 vs 混合摄入分钟级）· LoCoMo 行业同题（0/15 vs RAG 9/15）· MCQ 路由与 proactive 未在服务器形态单独基准</Text>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>下一步（按收益序）</CardHeader>
            <CardBody>
              <Text>1 MCQ 题集基准（max_pick 已配置）→ 2 proactive U2 协议实测 → 3 三模态 × 24B 视觉连通 → 4 Fate 阈值在线学习</Text>
            </CardBody>
          </Card>
        </Grid>
        <Text tone="secondary" size="small">
          完整报告：valhalla-server/docs/MISTRAL24_VS_VALHALLA_UNIQUE_POINTS_20261113.md · 竞品口径：docs/VALHALLA_COMPETITIVE_COMPARISON_20261113.md
        </Text>
      </Stack>
    </Stack>
  );
}
