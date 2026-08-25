
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
