# 引擎源码同步件（ValhallaBase / Fate）

本目录是 Valhalla 引擎现行源码的同步副本，2026-11-15 自
`C:\Ai\Valhalla-main` 同步，覆盖本地版后续全部已落地优化（此前
服务器部署仍停在 2026-11-14 的 optimized-BC 构建）。文件与源
逐一对应（改名仅为避免同名冲突）：

| 本目录 | Valhalla-main 源 |
|---|---|
| `fate_orchestrator.rs` | `manifestsys/hub-f64/src/fate_orchestrator.rs` |
| `native_qa.rs` | `manifestsys/hub-f64/src/native_qa.rs` |
| `multihop_fate.rs` | `manifestsys/hub-f64/src/multihop_fate.rs` |
| `signal_ingress.rs` | `manifestsys/hub-f64/src/signal_ingress.rs` |
| `fate_qa.rs` | `manifestsys/hub-f64/src/fate_qa.rs` |
| `fate_adaptive_algebra.rs` | `manifestsys/hub-f64/src/fate_adaptive_algebra.rs` |
| `valhalla_base.rs` | `manifestsys/hub-f64/src/valhalla_base.rs` |
| `hub_lib.rs` | `manifestsys/hub-f64/src/lib.rs` |
| `open_qa_interface.rs` | `manifestsys/hub-f64/src/open_qa_interface.rs` |
| `fate_system_lib.rs` | `systems/metasystems/fate-f64/src/lib.rs` |
| `triple_fate.rs` | `systems/metasystems/fate-f64/src/triple_fate.rs` |
| `stemcell_complete.rs` | `manifestsys/stem-cell/src/stemcell_complete.rs` |
| `tile_complete.rs` | `manifestsys/tile-quattro/src/tile_complete.rs` |

## 已同步的优化（含实测归因）

- **碎片作用域脉冲**（`fate_orchestrator`）：摄入收敛只在 Fate
  密度段收口时跑，不再逐行全流形收敛；
- **Fate 偏好去重复**（`fate_system_lib`）：零方向跳过 GPU、
  按新信号批次触发、偏好快照去重；
- **Stem 物理场 Fate 触发**（`stemcell_complete`）：cuBLAS 常驻
  复用、仅种群变化时重算场；
- **Tile 自身时钟碎片窗口**（`fate_orchestrator`）：片段窗口改用
  Tile 时钟（此前误用 Hub 时钟）；
- **快照去重**（`signal_ingress`）：片段只存 tile/cell id，
  492 行快照 37.23MB → 0.94MB；
- **被动 Fate QA**（`multihop_fate` / `native_qa`）：问题脉冲后
  只读抓取，判后学（`receive_passive` 在裁决之后）；
- **Fate 块门**（`native_qa` / `fate_qa`）：抓取由会话活簇方程
  `bundle_fits` 判定，无标定常数。

效果（2026-11-15 本地实测）：窗口 W3 摄入 462s → 3.8s、
窗口不变性核心跨臂逐题一致、312 轮极限核心 89–90/90、编造恒 0。

## 部署

网关的引擎供给是自动的：`valhalla_server/runtime.py` 在
`Valhalla-main\target\release\valhalla_base.exe` 比
`runtime/valhalla_base.exe.enc` 新时自动重封。构建命令：

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
cargo build -p hub-f64 --release --bin valhalla_base   # 工作区 C:\Ai\Valhalla-main\manifestsys
```

2026-11-15 已按此部署：服务器一键复现（`reproduce.ps1 --quick`）
新旧引擎门禁结果一致 —— OVERALL PASS（multimodal Fate-native
6/9、extreme targeted 806 12/12、编造 0）。
