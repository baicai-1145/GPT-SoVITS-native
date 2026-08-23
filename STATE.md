# STATE.md — 任务状态板

> **唯一写者：决策者 pi**。执行者只读本文件，通过回报建议状态变更。
> 状态机: `TODO → CLAIMED → IN_PROGRESS → REVIEW → DONE`（旁路 `BLOCKED`）

## 使用说明（决策者）

- 分配任务时: 状态改 `CLAIMED`，所有者填执行者名，分支填 `task/<id>`
- 执行者回报 DONE 证据后: 自查 DoD → 改 `REVIEW` → 验收通过改 `DONE` 并合并
- 新增任务卡追加到对应 Phase 表格，ID 连续编号
- 本文件每次变更单独 commit: `state: <摘要>`

## Phase A — 地基（串行）

| ID | 任务 | 产出 | 所有者 | 状态 | 分支 | 备注 |
|---|---|---|---|---|---|---|
| A1 | CMake 骨架 + 目录结构 + 编译选项(`-mcpu=apple-m4 -O3`) | 可构建空工程 + tests 框架 | exec-a | DONE | task/A | 含 .gsv 头部解析单测 |
| A2 | `.gsv` 读取器（mmap、目录解析、64B 对齐校验） | `src/runtime/gsv_loader` + 单测 | exec-a | DONE | task/A | 对照 tools/convert.py docstring 规范 |
| A3 | NEON 内核层 v1: GEMV(fp16存储+fp32累加)/rmsnorm/rope/silu/softmax | `src/kern/*` + 数值单测 | exec-a | DONE | task/A | 层0出口: cos=1.0/maxrel=1.2e-6 ✅ 单测直接对 golden 层快照 |
| A4 | Accelerate 封装: sgemm/hgemm 薄包装 + 线程池(QoS 分簇) | `src/kern/accel*` | exec-a | DONE | task/A | 本机无cblas_hgemm→f16升位回退已定论 |

**出口条件**: A1–A4 全 DONE；用 A2+A3 手工跑通一个 AR 层并对上 layers_first 快照。✅ 已达成并合入 main(e4b5b57)。Phase B 开工。

## Phase B — 三路并行

| ID | 任务 | 依赖 | 所有者 | 状态 | 分支 | 备注 |
|---|---|---|---|---|---|---|
| B1 | M1-fp32步: prefill(Accelerate) | A4 | 未分配 | TODO | — | |
| B2 | M1-fp32步: decode GEMV 循环 + KV cache(fp32) + top-k 贪心（fp16 化为后续独立任务） | A3,A4 | 未分配 | TODO | — | 对 pairs G1/G2 |
| B3 | M2: enc_p + quantizer | A4 | 未分配 | TODO | — | |
| B4 | M2: flow + dec(im2col→GEMM) + WAV 写出 | A4 | 未分配 | TODO | — | G3 只用稳定锚点对 |
| B5 | M3: jieba DAG/HMM 移植 + 词典 trie | — | exec-txt | DONE | task/B5 | 已合入 main；验收=204覆盖句+7504模糊句 diff 全空；trie.bin 由 tools/export_jieba_trie.py 再生不入库 |
| B6 | M3: WordPiece tokenizer + G2PW 推理(复用 kern BERT) | A3 | 未分配 | TODO | — | |
| B7 | M3: pypinyin 表 + 数字/符号规则 + symbols2 映射 | B5,B6 | 未分配 | TODO | — | diff 驱动 |
| B8 | Transformer 编码栈原生(roberta-large 24L×1024d + G2PW BERT-base 12L×768d 共用 kernel) | A4 | 未分配 | TODO | — | 对 pairs bert_out/bert_feat_1024 过 G1；供 B6 与韵律特征使用 |

## Phase C/D/E — 集成与后段（任务卡由决策者在 B 启动后细化）

| ID | 任务 | 依赖 | 所有者 | 状态 | 备注 |
|---|---|---|---|---|---|
| C1 | HuBERT/SV 编码器 + 参考缓存 | A4 | 未分配 | TODO | 对 refs bundle |
| C2 | CLI 全链路串接 | 全 B+C1 | 决策者 | TODO | |
| D1 | bench harness(powermetrics/吃满率采样) | C2 | 未分配 | TODO | |
| D2 | 流水线重叠(AR‖SoVITS 双缓冲)+线程调优 | D1 | 未分配 | TODO | §5 度量达标 |
| E1 | KV fp16 开关评估 | C2 | 未分配 | TODO | |
| E2 | AR int8 权重+int8 KV | E1 | 未分配 | TODO | A/B 听感 |

## 阻塞/风险登记

| 时间 | 内容 | 影响 | 处置 |
|---|---|---|---|
