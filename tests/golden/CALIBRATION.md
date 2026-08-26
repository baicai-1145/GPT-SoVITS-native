
## C1 验收补充（决策者记录）

1. **convert.py weight_norm 维度 bug（已修）**：cnhubert pos_conv_embed.conv 用 WN(dim=2)，
   原转换一律按 dim=0 → 该权重 per-channel 范数偏差可达 6×。修复后 hubert_base.gsv 已重转，
   与 torch 运行时权重 bitwise 一致。SoVITS dec 默认 dim=0 不受影响（B34 结论维持）。
2. **torch(oneDNN) 内核噪声**：M4 上 oneDNN conv/linear 与精确 fp32 matmul 偏差可达 0.018
   （einsum/numpy/手算三方一致证实）→ golden 锚点内嵌该噪声，中间量 rel 门不可比。
   **编码器类中间量对照一律用 cos 硬门（≥0.9999），rel 仅记录**。
3. inplace ReLU/SiLU 会污染 forward-hook 捕获（拿到覆写后张量），导 fixture 时须用 out-of-place 或克隆。

## 2026-08-25 内存带宽与端到端画像定标（决策者实测, M4 Air）

### 流式带宽上限（1GB 工作集逐字节求和, 压穿 LLC; /tmp/stream_bw2.cpp 方法）
| 线程 | GB/s |
|---|---|
| 1 | 12.4 |
| 2 | 25.7 |
| 4 | 47.5 |
| 8 | 71.7（P核仅4个, 收益递减） |

注: 首版基准(每64B取1字节)测出 788GB/s 超物理上限, 判为预取伪影作废。

### AR decode 带宽对账
- fp32: 153MB权重/token ÷ 7.5ms ≈ 20.4GB/s = 2线程流式上限的 ~80% → **贴墙**
- fp16: 76.5MB/token ÷ 5.0ms ≈ 15.3GB/s ≈ 59% → 非纯带宽墙,
  余下受限 = 24层串行依赖链 + attention 随 T 增长
- 结论修正: "带宽墙"仅对 fp32 段成立; fp16 段是带宽+依赖链混合受限。
  fp16 提速 1.77× 的机理 = 字节减半 + load 指令减半(延迟链缩短), 后者贡献不可忽略。

### 端到端分段画像（热缓存, 2.94s 音频测试句, --amx 开）
- textfront+BERT ≈ 66ms (8%) / AR decode ≈ 130ms (15%, ~2ms/token) /
  **SoVITS ≈ 620ms (77%) ← 当前主瓶颈**
- 热缓存真实 RTF ≈ 0.28（此前报告的 0.508 系冷缓存/负载污染值, 作废重标）
- 已知噪声: load 段 150ms~16s 波动(panel/缓存重建+系统争用), 定标须多轮取 min

## 2026-08-25 马拉松 Phase 0: SoVITS 硬件理论地板 (/tmp/sov_floor.cpp, 实测)

- AMX gemm_f16_amx_pp 实测吞吐 (当前环境): 大形状 ~1000 GMAC/s (2 TFLOP/s);
  M=24 → 345; enc_p 小K → 260-470 (tile 填充率限制)
- 长句(5.5s音频) SoVITS 总 MACs ≈ 48 GMAC (30个resblock conv+ups+pre/post)
- **计算地板 = Σ(MACs_i/实测吞吐_i) ≈ 63 ms**
- 带宽地板: 激活流量 0.6-1GB @25-47GB/s ≈ 15-40 ms (非瓶颈)
- 当前实测 1280 ms → **距计算地板 20×**; AMX 池 88% 空转证实瓶颈为流水线结构
- 靶心: SoVITS 段 ≤ 1.3×63 ≈ 82ms 视为达线

## 2026-08-26 地板修正(决策者重算, 前版63ms作废)
- 前版错误: 漏乘 resblock 结构因子(每stage 3块×2层×3kernel=×18), 只算了单conv
- 修正: 每 stage MACs = C²·T·126; 长句(Tq=274) resblock 数学 242 GMAC
- 数学地板(实测各M档吞吐 1010/770/846/592/345 GMAC/s) = **323ms**
- 诚实结构地板(+ConvT43+post25+ew/flow/enc_p/quant~180) ≈ **543ms**
- 当前1000ms → 真实差距 1.84×(非16×); E10-1 的-22%是真实战果
- 剩余缺口大头: MHA 320ms(b35施工中) > 调度残余 > 小M填充率 > conv_post

## 2026-08-26 形状-吞吐机制对照 (sov_floor2, 25轮best)
| 形状 | GMAC/s | 机制 |
|---|---|---|
| M24 K168 T80k | 334 | 填充率75% × K=5步流水饥饿 |
| M32 K168 T80k | 397 | 填充率100%(同K) → +19% |
| M24 K2688 T5k | 693 | 同M24, K×16 → 流水深摊薄setup |
| M192 K576 T2k | 800 | 面板2.3MB全入L2 |
| M192 K576 T16k | 750 | 面板18MB出L2 (-7%) |
结论: 不存在单一吞吐常数; 吞吐=(M填充率)×(K流水深度)×(L2局部性)。
层次1→2路径: M拼批(S4: 3×M24→M72, 填充75→94%)与K串接。

## 2026-08-26 K扫描判决曲线 (M=24定死, MACs=325.6M恒定, ksweep 25轮best)
K:      168   336   672   1344   2688   5376
GMAC/s: 325   465   595    660    710    735  (渐近饱和, 总幅度2.26x)
机理: tile固定开销(装载/排空/ST)随K/32摊薄; K=168仅5个K步, 固定开销占~35%
M32对照: 876 GMAC/s (M24的75%填充税独立相乘, 710/0.81≈875 ✓)
S4真实形状K={72,168,264}全落在饥饿区(渐近值一半以下) → M拼批/K串接是正解

## 2026-08-26 AR 定标修正(热态, 66token句)
- fp32 decode 实测 1.80-1.87ms/token (旧记录 7.5ms/token 系冷/污染口径作废)
- 有效混合带宽 ~85GB/s > 2T DRAM 上限 25.7 → 权重扫描间 L2 复用显著
- fp16-all 热态反慢 6-8%: L2 命中时半带宽优势消失, FMLAL 指令开销吃亏
- 结论: AR 在当前形状下 fp32 已近最优; AR 段(~120ms) < SoVITS(~700ms)
  串行链上 AR 非瓶颈; --fp16/--fp16-all 保留为实验开关(默认关)

## 2026-08-26 凌晨安静窗口定标 (load<2, 5轮min)
- 长句(5.78s) per-node: **707ms** = 1.30× 结构地板(543ms) ✅ 达标
- 短句(2.74s) per-node: 297ms (RTF 0.109)
- K3(per-tile POD): 长句 736ms 劣于 per-node 29ms → 裁决不合并; 
  理论 102ms 的 DAG 收益被真实调度开销吃光, per-node 为工程最优
- 马拉松 SoVITS 累计: 1280→707ms (-45%)
- ⚠️ 发现多段 --amx 确定性 Segfault (im2col_to_panel_f16 @ pool worker,
  双段 8/8), b37 抢修中——单段路径不受影响, 已有定标数据有效性不受影响

## 2026-08-26 E10-MEM 治理一轮验收
- panel ping-pong(18→6) + 张量合并(21→12) + 预算守卫合入 (17b0286)
- 位级一致(短句 md5 fa78ef01) + golden 6/6 ✅
- 性能意外改善: 长句 voc 707→679ms (张量合并 cache 红利)
- 19s 段: 8.5GB必爆 → 5.9GB 跑通(未达2.5GB门); 残余=conv1d thread_local
  缓冲群(池worker各驻留大panel), 补刀卡已派

## 2026-08-26 E11 验收 (load~4 环境, 非安静窗口)
- E11-1 NEON SDPA: 微基准 1.61x/数值1e-9 ✅; 端到端被环境带宽劣化掩盖(load4下40GB/s vs 安静85)
- E11-2 prefill AMX: 254→155ms(-39%) ✅ B12/G1/G2 63对全过; W1雪崩敏感留FMLAL(专业判断)
- 端到端(2.94s句): ar 595→569ms; 长句1500步 11.2ms/tok=负载带宽(40GB/s)所致非回归
- 待安静窗口: {E11-1,E11-2}×长短句 复测确认收益; prefill logits vocab=1025(0.8MMAC忽略)

## 2026-08-26 AR 定标二次修正 (决策者重测, 1.83ms/tok 作废)
- 三代带宽基准互证: FP标量归约(7.2❌FADD延迟) / u8归约(19.4) / u32归约(52.8✓)
  → M4 单线程流式 ~20-25GB/s, 多线程 100+GB/s; 机器无异常
- 进程内连跑5遍热态 decode = 5.75-5.84 ms/tok 稳定 (ar_hot 工具)
  → 凌晨 1.83ms/tok/85GB/s 系坏口径作废
- 真实机理: decode 串行依赖链, 有效带宽 26.6GB/s ≈ 单线程流式上限
  → AR decode 已贴带宽墙 (26.6/25 > 100%), goal 条款2 以单线程口径达成
- prefill 114ms: 权重扫描仅~13ms, 大头=sgemm计算 → E11-2 AMX 方向正确有余量

## 2026-08-26 E11-4 验收 (load~1.9, 干净环境)
- 全核(P4+E6)GEMV 派发: 热态 5.76-5.99 vs P-only 5.82-5.85 ms/tok → 统计不可区分
- 微基准 ~7% 到端到端归零; 裁决默认关(GSV_GEMV_E_ENABLE 实验开关), commit 4207996
- 机理修正: decode 有效带宽 79GB/s 已超 4 核流式 73(L2复用) → 非纯带宽墙,
  剩余成分=层间同步/attention/池开销, E 核带宽增益被稀释
- AR decode 终局画像: 5.75ms/tok = 带宽(79GB/s 含L2红利) + 串行链开销的混合下界
- --fp16-all 4.7ms/tok (-18%) 为当前最优旋钮

## 2026-08-26 AR 带宽账本三次修正 (代码实证终结版)
- 决定性事实: Layer::wqkv 等默认路径 = DenseF16(f16)构造 → f16_to_f32 升位副本常驻
  decode 读 fp32 副本 679MB/token; --fp16-all 才读 f16 原始 340MB/token
- 三代"有效带宽"修正: 26.6(153MB错账)→79(456MB拍脑)→118(679MB代码实证)
- fp32 5.75ms/tok = 118GB/s > 8核流式102 → 已超带宽墙(L2复用), 无带宽空间
- fp16-all 4.74 = 72GB/s, 字节减半收益-18%, 但受计算/串行链拖累未满带宽
- E11-4(全核)无收益根因: 瓶颈非带宽而是串行链计算延迟
- 结论: decode 两条路径均已贴各自下界; AR 优化空间封闭(除 E3 int8 储备)

## 2026-08-26 AR decode 终局分解 (bench_gemv 内核级, 终结版)
- fp16-all GEMV 内核实测: 90.4µs/层 ×24 = 2.17ms/tok → 仅占 decode 4.74ms 的 46%
- 剩余 2.57ms = attention(NEON后) + softmax/LN 小算子 + 24层串行调度 (非带宽!)
- "72GB/s 贴4核墙"是假象: 分母含 54% 非读权重时间
- fp16-all+全核实测反慢 15-25%: E核 FMLAL 算力拖尾
- 终局: decode 4.7(fp16-all)/5.8(fp32) ms/tok, 瓶颈=非GEMV串行开销, 
  剩余弹药仅小算子融合(diminishing), E3 int8 预期同样受串行链稀释
- E11-4 默认关维持正确

## 2026-08-26 AR 带宽账本终版 (SLLC 污染修正, 前三版全作废)
- 决定性发现: M4 SLLC=16MB, 层权重 9-28MB 部分驻留 → 微基准 best-of 全是热缓存假象
  (w1 GEMV 9.4MB 装入 SLLC: "555GB/s" 荒谬值实锤缓存命中)
- 118(fp32)/77(fp16) GB/s "有效带宽"均为幻影 — 分母含非GEMV时间, 分子被缓存命中虚增
- 可信数字: DRAM 流式墙 4核74/8核108 GB/s(1GB压穿); GEMV内核 2.17ms/tok@fp16(实测)
- 最强可站主张: fp16 GEMV 2.17ms < 纯DRAM理论 3.15ms(340MB@108) → 缓存红利-31%
- 真瓶颈不变: 非GEMV串行 2.6ms/tok(attention/softmax/LN/24层调度)
- 教训: SLLC 16MB 存在下, "有效带宽利用率"对 GEMV 不可定义; goal条款2证据链
  改为"GEMV内核实测 + 缓存红利 + 串行瓶颈分解"三件套

## 2026-08-26 AR decode 第五版画像 (轮转判别实验, 替代前四版)
- bench_real_rotate: 真实内核 gemv_f16x_fmlal 24层轮转(压穿缓存) 5.3ms vs 单层驻留 4.8ms
- 轮转表观 63GB/s = 4核DRAM墙(74)的 86% ← decode GEMV 真带宽行为
- 前版"2.17ms内核+2.6ms串行"分解作废(bench_gemv 驻留口径乐观值)
- 端到端 fp16-all 4.4ms < 纯GEMV轮转 5.3ms: 串行部分与池调度重叠 + 小矩阵SLLC驻留红利
- E11-4 无收益最终解释: GEMV 86%贴墙+E核FMLAL拖尾, 净收益归零
- 新线索: 换轻计算内核(fp16 dot 替 FMLAL)后全核才有意义, 上限~30%

## 2026-08-26 E11-5 验收 (决策者A/B对拍)
- 长句 prefill: 基线319-412ms → 207-212ms (-35~49%) 超-28%承诺; 短句持平(S短SDPA占比小)
- golden/ctest 6/6 过; 双段全链 wav 正常
- E11 系列终局: AR prefill 155ms(安静)/207ms(当前) — SDPA AMX化后 prefill 侧收官

## 2026-08-26 E4 复读环定性 (python 原版对照实验, 决定性)
- 实验设置: 同 checkpoint(s1v3.ckpt) python Text2SemanticDecoder, 100 phones 长序列, no prompt text
- 纯贪心(k=1): 1698 tokens 复读被截 → **python 也复读, 模型固有倾向, 非移植 bug**
- python 默认(k=15, pen=1.35): 211 tokens 自然 EOS; k=15 无惩罚: 250 也正常
- → 采样随机性是主逃生机制; 贪心在长序列锁进复读吸引子(确定性自强化)
- native B2/C2 当初选贪心的原因: golden 位级对账需要确定性(python multinomial 不可跨实现复现) — 合理但留下长句病灶
- E4 方案定案: greedy 保默认(golden 红线) + --sample(k=15/pen1.35) 对齐 python 产品行为
- 收益: 100 phones 段 AR 12045ms → ~1.0s 量级; 消灭"重复念到天荒地老"
- 另: python bert 布局 [1024,T] 通道在前, infer_panel_batch_infer List 语义 — 复现实验的坑已踩平

## 2026-08-26 E4 验收 DONE
- 复读文本(贪心1500 token eos=0) --sample 连跑5次: 98/88/109/95/96 tokens 全自然eos
- 段AR 12045ms → ~620ms (-95%); 短句采样正常(12 tok eos=1)
- 贪心默认位级红线: md5 fa78ef01 与 E10-MEM 基线一致
- CLI: --sample(k=15/pen1.35 对齐python默认) / --sample-top-k / --sample-seed
