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
| B1 | M1-fp32步: prefill(Accelerate) | A4 | exec-ar | DONE | task/B12 | G1 63/63 | |
| B2 | M1-fp32步: decode GEMV 循环 + KV cache(fp32) + top-k 贪心（fp16 化为后续独立任务） | A3,A4 | exec-ar | DONE | task/B12 | 18966步全程一致; 采样口径=惩罚后(见CALIBRATION) |
| B3 | M2: enc_p + quantizer | A4 | exec-sov | DONE | task/B34 | G1 14 hooks 全过 | |
| B4 | M2: flow + dec(im2col→GEMM) + WAV 写出 | A4 | exec-sov | DONE | task/B34 | G3 新口径(h_dec fixture 锚点) 6对 mel_rel≤2.4e-5 |
| B5 | M3: jieba DAG/HMM 移植 + 词典 trie | — | exec-txt | DONE | task/B5 | 已合入 main；验收=204覆盖句+7504模糊句 diff 全空；trie.bin 由 tools/export_jieba_trie.py 再生不入库 |
| B6 | M3: WordPiece tokenizer + G2PW 推理(复用 kern BERT) | A3,B8 | exec-sov | DONE | task/B34 | 我独立复验 tok439/439+conv365/365; 注入=终章任务(exec-txt 进行中) |
| B7 | M3: pypinyin 表 + 数字/符号规则 + symbols2 映射 | B5,B6 | exec-txt | DONE | task/B5 | part1+part2 全部达成, 随 B6/B10/B9 综合交付, pairs 65/65 全通 |
| B8 | Transformer 编码栈原生(roberta-large 24L×1024d + G2PW BERT-base 12L×768d 共用 kernel) | A4 | exec-sov | DONE | task/B34 | 自建fixtures全过(cos=1.0); position_ids=arange口径(CPUFast自实现BertEmb); get_bert_feature取hidden[-3][0][1:-1] |

## Phase C/D/E — 集成与后段（任务卡由决策者在 B 启动后细化）

| ID | 任务 | 依赖 | 所有者 | 状态 | 备注 |
|---|---|---|---|---|---|
| B10 | M3: 英文g2p+en_norm+LangSegmenter混排 | B9 | exec-txt | DONE | task/B5 | pairs 63/65(余2=G2PW类); rep_map正则key怪癖已复现 |
| B9 | M3: TextFrontend 运行时编排(分句+组装) | B5,B7 | exec-txt | DONE | task/B5 | pairs parity 61/65; 真实口径=prompt_ids++分段ids(短句补。), phone_units假设证伪 |
| C1 | HuBERT/SV 编码器 + 参考缓存 | A4 | exec-ar | DONE | task/B12 | 6/6 refs 全链 cos=1.0; convert.py WN bug 修复+hubert 重转 |
| C2 | CLI 全链路串接 | 全 B+C1 | exec-ar(授权公共地带) | DONE | task/C2 | 1468步位级复现; CLI冒烟亲测3.42s wav; RTF=3.75记录; M4 达成 |
| D1 | bench harness(powermetrics/吃满率采样) | C2 | exec-txt | DONE | task/B5 | bench三脚本+reporter已就绪; 待安静窗口执行正式定标 |
| D2 | 流水线重叠(AR‖SoVITS 双缓冲)+线程调优 | D1 | exec-ar(原exec-txt prep) | DONE | task/D2 | AR‖SoVITS三阶段重叠调度+QoS线程分簇; 串行/重叠逐样本一致; M5 达成 |
| E1 | KV fp16 开关评估 | C2 | exec-sov | DONE | task/B34 | 裁决B: --fp16默认=kv-only(与fp32逐位一致,G1 65/65); FMLAL gemv为实验开关(62/65,激活舍入雪崩归M6); 抽测1.77×待安静窗口定标 |
| E2-AR | AR Decode 全 GEMV fp16 化数值对齐 (解决4/65雪崩) | E1 | exec-ar | DONE | task/E2-AR | 混合精度: WQKV/WOut/FFN-W1走FMLAL, Logits/FFN-W2保fp32; --fp16-all下65/65 B12全过(含1500步雪崩对); 真fp16计算 |
| E2-SOV | SoVITS 端到端 fp16 算子(真FMLAL GEMM) | E1 | exec-sov | REVIEW | task/E2-SOV | 重做达标: 真fp16计算/RSS-36%/G3过(mel_rel≤0.0144); 但手写FMLAL比AMX sgemm慢2.9x→待裁决: 仅可作--sovits-fp16选装开关合入, 默认保持fp32 |
| E2-ENC | BERT/RoBERTa/HuBERT 编码栈 fp16 化 | E1 | exec-txt | DONE | task/E2-ENC | view零拷贝+真FMLAL+生命周期修复; 初判REJECTED系误判(短句基线vs长句对照无效): 受控A/B证fp16与fp32输出token级一致(同句均910)且load 4.6s→170ms → 已reapply(f754269); 调节链fp16安全性与AR采样语义问题拆分至E4 |
| E3 | AR int8 权重+int8 KV (原E2) | E2-AR | 未分配 | TODO | A/B 听感 |
| E6 | SoVITS热点消灭二轮DONE合入: bias-fold快径ones列bug修复(main潜伏bug!)+ConvT相位amx_batch_run接入; 配对A/B SoVITS段 -62%(1645→627ms) | E5-P2 | exec-sov | REVIEW→DONE待定标 | task/E6 | G3全过/单测过/off不变; 绝对值待安静窗口D1定标复测 |
| E9 | kern 批量多GEMM单次派发 API(amx_batch_run, phase-图调度, prepare钩子就地im2col) | E6 | exec-ar | DONE | task/E9 | amx_batch_run+AmxBatchNode+prepare钩子实现; bench: C场景prep流水1.10x, bitwise_diff=0; test amx_batch_matches_sequential PASS |
| E7 | 加载方差治理: 根因=USB盘73MB/s+双布局冗余63%+页缓存挤占; --slim(-63%字节,wav逐位一致)+RDADVISE预读 | E5-P2 | exec-ar | DONE | task/E7 | 已合并; slim权重已部署内置NVMe ~/gsv-weights(load 380ms达标); 后续: convert.py直产slim(C方案)归M0工具链 |
| E10-MHA | MHA batched sgemm(320→58ms实测) — 已合入 | E9 | exec-sov-b35 | DONE | task/E6 | 热态配对-9%; 剩余attn_out 29+rel 24标量路径低优先 |
| E10-S2 | S2排查一轮DONE合入: im2col dil>1 NEON化(100→25ms), 全局voc 900→~770ms; S2 224→157ms | E10-MHA | exec-sov-b37 | DONE(一轮) | task/E6 | 确诊根因=per-node prepare依赖; 后续在E10-K2 |
| E10-K2 | kern amx_chain_run per-tile prepare依赖(行波推进, math/prepare overlap) | E10-S2 | exec-ar-b36 | CLOSED | task/E10-K2 | K2闭包税裁决后K3重试: 安静窗口736ms反输per-node29ms→裁决不合并, per-node为工程最优 |
| E10-MEM | 多段segfault修复(s_hi off-by)+内存治理: panel ping-pong(18→6)+张量合并(21→12)+预算守卫 | E10-K2 | exec-sov-b37 | DONE | task/E6 | 19s段8.5GB必爆→6.1GB跑通; 位级一致+golden 6/6+长句679ms反超707; 残余=conv1d thread_local缓冲(补刀卡≤5GB目标, 2.5GB门经重算判定为此结构不可达) |
| E11-1 | AR attention KV扫描 NEON 4-lane 树形归约 | main | exec-ar-b36 | DONE | task/E11-1 | 2.4x内核/数值1e-9; G1/G2 63对全过 |
| E11-2 | prefill 大GEMM走AMX(wqkv/w2) | E11-1 | exec-ar-b36 | DONE | task/E11-2 | 254→155ms(-39%安静口径/当前带宽环境-18%); W1雪崩敏感保守留FMLAL |
| E11-4 | decode GEMV全核(P+E)派发 | E11-2 | exec-ar-b36 | CLOSED | task/E11-4 | 验收无收益(5.76vs5.82不可区分)默认关; 根因=GEMV轮转86%贴4核墙+串行链, E核FMLAL拖尾 |
| E11-5 | prefill QK^T/PV 走AMX GEMM(FlashAttention CPU子集) | E11-2 | exec-ar-b36 | CLAIMED | task/E11-5 | 靶: 短句prefill 114→≤105或长句285→≤260; K=32薄K形状允许负结论交付 |
| E8 | encoder DenseF16→AMX后端切换(bert.ffn形状bench 2.07x; 当前仅66ms, 低优先级) | E5 | 未分配 | TODO | | 待E6后视余量 |
| E8 | BERT栈FMLAL→AMX: 密集层AmxPanel预打包+形状分流 | E5-P2 | exec-ar-b36+决策者合并 | DONE | task/E8 | roberta长句345→50ms(负载下75ms); wav md5位级一致fa78ef01; ctest 7/7 |
| E4 | 采样器对齐python: topk_sample(k=15/pen1.35)+--sample旋钮; greedy保默认 | C2 | exec-ar-b36+决策者接线 | DONE | task/E4 | 验收5/5自然eos(12045→620ms -95%); 贪心位级不变fa78ef01 |
| FE-AUTO-1 | 前端auto切分位级移植(LangSegmenter空参口径): fasttext lid C++推理+数字归属规则+punctuation兜底; zh/en片直落现有G2P, ja/ko片显式降级告警; fixture langsegment_auto.json全过 | E4 | 未分配 | TODO | 移植重量: split_lang逻辑(~200行)+fasttext(~150行lid推理, 模型已在CPUFast); ja/ko G2P拆FE-AUTO-2/3 |
| E5 | AMX 指令直接编程: fp16×fp16→fp32 矩阵协处理器后端(压榨CPU终极手段) | E2-SOV,E2-ENC | exec-ar | DONE(phase1) | task/E5 | 已合入(main默认OFF,-DGSV_AMX_GEMM=ON); 本机复验bench: amxpp 9/10形状反超sgemm 1.04-2.07x, cos=1.0全PASS; 实锤手写fmlal慢5-10x→E2-SOV的fmlal方案作废 |
| E5-P2 | AMX接线: sovits conv按形状分流(M<64/T<64/dil>1→sgemm GEMV, 大块→amxpp+panel直写im2col), w_f16单副本, --amx开关默认关 | E5 | exec-sov | DONE | task/E5-P2 | 已合入; 本机复验: --amx热缓存RTF=0.508, off/on wav cos=0.999903(int16 maxdiff=2); G3全过(mel_rel最差0.00414); 内存裁决: --amx=速度模式(+120MB panel), 默认=省内存fp32 |

## 阻塞/风险登记

| 时间 | 内容 | 影响 | 处置 |
|---|---|---|---|
| 15:17-15:27 | tests/golden 内容被清（refs/pairs/fixtures 全丢，根因未定位；weights 无损） | G3/锚点基线一度失效 | 已重导出恢复（65 pairs+56 refs）；CALIBRATION.md 从 git 恢复。**新政策：全员禁止 git clean/清理 ignored 目录；tests/golden 为决策者领地** |
| 同日 | exec-sov 实验证实：v2Pro dec 每进程重采样 noise → 历史端到端 wav 跨进程不可复现（同 codes 下 mel_rel=0.83） | G3 锚点口径变更 | 决策者批准：G3 改为对照 fixture 内 h_dec hook（固定 noise 输入），历史 wav 降为仅记录；落实见 CALIBRATION 后续补录 |
