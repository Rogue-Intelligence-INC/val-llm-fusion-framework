# 归档：Mistral 24B × Valhalla 完整优化与实测（2026-11-13）

> 归档编号：20261113-mistral24-fullopt · 机器：i9-14900KF / 128GB / RTX 4090（Windows）
> 一句话：24B 下载实测 → Fate 接线修复 → Hybrid 混合解码 → 运行时吃满（18/18）→ Tier A 烘焙进 24B 权重（复测仍 18/18，超窗 8/8）。

## 内容索引

| 目录 | 内容 |
|---|---|
| `docs/` | 竞品对比报告、Mistral 实测报告（含 §7 Fate 修复、§8 Hybrid 适配、§9 审计、§10 吃满复测）、对比总结画布源码 |
| `results/` | 每轮原始数据 JSON（语料、问题、逐题原始答案、延迟、评分）：首轮 / Fate 修复后 / 混合 / 完整版 / 超窗 / 烘焙后 |
| `logs/` | 各阶段运行日志：bench_full、bench_patched_hybrid、bench_patched_solo、bench_v3、stress_hybrid、fate_probe、m24_stage13、m24_bake、m24_sweep |
| `artifacts/` | 24B 分解计划（40 层）、三模态融合导出（model_patch）、烘焙语料 |
| `models/` | 服务器模型注册表快照、s0.08 / s0.02 烘焙清单（Δw=0.604 / 0.151） |

## 关键结果（可回溯到 results/ 与 logs/）

| 测试 | 结果 |
|---|---|
| Mistral-24B 单独（窗口内 18 题） | 18/18；闭卷陷阱编造 1/6 |
| Valhalla 原生（未执行 Fate 裁决） | 7/18，硬错误 11 |
| Valhalla 原生（执行 Fate 裁决） | 10/18，硬错误 1 |
| Hybrid v1（24B 解码头） | 14/18，硬错误 0 |
| **完整版混合（弱放行+多轨+GPU）** | **18/18，硬错误 0** |
| **完整版 + Tier A 烘焙后 24B** | **18/18**（本体单独亦 18/18，patch 无退化） |
| 超窗压力（33K token，完整版） | **8/8**（24B 单独 0/8 超窗被拒） |
| 强度扫描 s0.02 / s0.08 | 本体均 18/18 |

## 模型工件（不在本包内，体积过大；按哈希校验）

| 工件 | 路径 | SHA256 |
|---|---|---|
| 原始 Q4_K_M（unsloth） | `C:\Ai\models\Mistral-Small-3.1-24B-Instruct-2503-GGUF\...-Q4_K_M.gguf` | `6D670773C3908584349D41A5048D1472226B593C881FD394E8AC196C802E81E2` |
| 烘焙后 Q4_K_M（s0.08，当前解码头） | `C:\Ai\models\Mistral-Small-3.1-24B-Valhalla-s0.08-Q4_K_M.gguf` | `D7BD899102C2387B9CC6794C097BD29E4D41CA12E519D9A8C9F6131A7C93AC8C` |
| 烘焙后 HF 目录（s0.08 / s0.02） | `C:\Ai\models\Mistral-Small-3.1-24B-Valhalla-s0.08` / `-s0.02` | 见 `models/valhalla_manifest_*.json` |
| BF16 原始权重 | `C:\Ai\models\Mistral-Small-3.1-24B-Instruct-2503` | ModelScope `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |

## 复现

自动化后（见服务器 README 的 modelops 一节）：

```powershell
python -m valhalla_server modelops mistral24-auto-v1 --base C:\Ai\models\Mistral-Small-3.1-24B-Instruct-2503 --strength 0.08
python tests\run_unique_points.py --hybrid-only
python tests\run_stress_hybrid.py
```

## 追加（2026-11-13 晚）：官方协议复测

拆解→融合→多模态输出＋主动输出，用仓库官方测试代码复测（报告 §11.5）：

| 工件 | 位置 |
|---|---|
| 官方多模态验收 PASS 报告 | `results/multimodal_mixed_answer.json`（源：Valhalla-main reports） |
| 官方 proactive 门电池 3/3 报告 | `results/proactive_fate_v2_test.json` |
| 官方 handcraft 电池明细（发声 0/4、拒答 3/3） | `results/results_handcraft.json` |
| 官方 FOG 输出门报告（FOG_V2_ALGEBRAIC_OK，覆盖检查未过） | `results/fate_output_gate_v2_test.json` |
| 官方测试日志 ×4 | `logs/official_{mm,proactive,handcraft,fog}.log` |
| 驱动与透明桩（缺失模块调用即报错） | `results/run_official.py`、`run_handcraft.py`、`artifact.py`、`grind.py`、`run_strict_incubation_experiment.py` |

fair 电池不可运行：checkout 缺 `artifact.py`、`run_strict_incubation_experiment.py`、`incubation_benchmark_suite.json`（全机不存在），已在报告中如实标注。

*Rogue Intelligence LNC.*
