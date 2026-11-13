# Val-LLM 拆解融合框架

`valhalla-server` —— 把任意 LLM 按层拆解、经 Fate 体系适配、与文/声/图多模态自由融合、再烘焙回权重并以结构会话服务的框架。

按较新的 `valhalla-unified-core`（2026-07）模式部署 Valhalla。桌面端和手机端不改。

## 部署模式

- **ValhallaBase 是本体，LM 只是解码头。** 网关在本进程内解密 `valhalla_base.exe`，以 `serve` 跑在内部端口 8787（`valhalla-base-v1`），烘焙模型作为 `VALHALLA_LM_MODEL` 挂上。
- **生产参数**：`decode=hybrid`、`follow_up_decode=native_follow_up_aware`、`fate_ingress_routing=quad_cycle`、`fate_qa_feedback=true`、`VALHALLA_DYNAMIC_CORPUS_FRAGMENTS=1`。open 走 native，mcq 走 LM 头。
- **会话不隔离**：同一模型的多轮请求共用一个 Fate 会话，网关把返回的 session 带回下一轮。
- **Fate 裁决优先**：每题的 `fog_decision`（emit/abstain/defer）由引擎给出，网关必须执行——判 abstain 时返回“根据已摄入的资料无法确定。”，不再把检索文本硬发给用户；裁决记录在响应的 `valhalla.fog_decision` 里。
- **24B 混合解码头（valhalla-transformer 路线）**：Fate 放行后，可由 Mistral-24B 只根据 Valhalla 检索到的证据行生成干净答案（`valhalla_server/mistral_head.py`，llama.cpp 服务在 18080，环境变量 `VALHALLA_MISTRAL_URL` / `VALHALLA_DECODE_HEAD=mistral`）。解码头不看全语料、无权推翻 Fate 的 abstain；不可达时回退原生文本。

## 生产管线（POST /v1/pipeline）

1. 分解：一个 LM 按层深四分（Q0→Hub、Q1→Tile、Q2→Stem、Q3→Hub），`valhalla_model_fate` 多轮适配。
2. 融合：同一样本的文本/音频/图像端口在**同一个** Triad 会话里交错脉冲，不隔离、不预设路由（`valhalla_trimodal_incubate`，`valhalla-free-multimodal-fusion-v3`）。
3. 烘焙：结构写入基础模型生成新检查点，原模型不动；产物带模型卡和 `.gitattributes`（finalize 模式），命名 `*-Unified-Core-v1-test-s{strength}`。
4. 加密：引擎和权重在磁盘上都是 VALENC1（`runtime/*.enc`、`data/vault/<model>/model.safetensors.enc`），只在本进程缓存里解密，退出时清除。

## 模型处理自动化（modelops）

Valhalla 对 LM 的整条处理链已程式化为服务端作业，一次提交全自动跑完：

**校验 → 分解（按层深四分）→ Fate 适配 → 三模态融合 → Tier A 烘焙 → GGUF 转换 → 量化 → 部署解码头 → 验证（有据题+陷阱题）→ 注册**

- API：`POST /v1/modelops`，参数 `model_id`、`base_model`（HF 模型目录）、`strength`（默认 0.08）、`quant`（默认 Q4_K_M）、可选 `samples`（烘焙语料，缺省用通用样本，绝不自动用调用方测试数据）、`deploy`（默认 true）；进度查 `GET /v1/jobs/<id>`。
- 解码头注册表：`GET /v1/decode-heads`（含每个头的权重变化、Fate 适配记录、GGUF 哈希）；`POST /v1/decode-heads/<id>/deploy` 一键切换线上解码头（llama-server 由服务端托管）。
- CLI：`python -m valhalla_server modelops <head_id> --base <模型目录> [--strength 0.08] [--quant Q4_K_M] [--no-deploy]`
- 大模型走分片级烘焙（`bake_mistral`，其余分片硬链接）；小模型自动走 `bake_runner`。产物 GGUF 与 HF 目录落在基础模型同级目录，命名 `<base>-Valhalla-s<strength>`。

2026-11-13 已用此链自动重跑 Mistral-24B 全流程并通过部署后验证（job `mistral24-auto-v1`）。

## 归档

2026-11-13 全部实测与优化成果已归档：[`archive/20261113-mistral24-fullopt/`](archive/20261113-mistral24-fullopt/ARCHIVE_INDEX.md)（另有同名 zip），含报告、逐题原始数据、日志、分解/融合工件与模型哈希索引。

## 竞品与持续优化对比

详细指标对比（同题实测 / 内部协议 / 厂商自报三档口径，含持续优化能力专章）：[`docs/VALHALLA_COMPETITIVE_COMPARISON_20261113.md`](docs/VALHALLA_COMPETITIVE_COMPARISON_20261113.md)

Mistral 24B 本机实测（幻觉 / 上下文 / 主题漂移独有点裁定）：[`docs/MISTRAL24_VS_VALHALLA_UNIQUE_POINTS_20261113.md`](docs/MISTRAL24_VS_VALHALLA_UNIQUE_POINTS_20261113.md)，原始数据在 `tests/results_unique_points/`。

## 接口（http://127.0.0.1:8780）

- `GET /health`
- `GET /v1/runtime` 引擎加密状态
- `GET /v1/models` 已发布模型及链接
- `POST /v1/pipeline` 生产一个模型，返回 `job_id`；`GET /v1/jobs/<id>` 看进度
- `POST /v1/models/<id>/chat/completions` 调用（ValhallaBase 本体 + 该烘焙模型作解码头）
- `POST /v1/chat/completions` 同上，`model` 字段选模型
- `POST /v1/infer` 原始 `valhalla-base-v1` 透传到内部运行时

设 `VALHALLA_SERVER_KEY` 后，除 `/health` 外都要 `Authorization: Bearer <key>`。

```bat
启动.bat
"C:\Ai\Valhalla-main\.venv-llm\Scripts\python.exe" -m valhalla_server build unified-core-v1
```
