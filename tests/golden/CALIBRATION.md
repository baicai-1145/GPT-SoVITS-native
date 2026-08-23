# 精度门定标报告 (M0)

基线: fp32 / seed=42 / top_k=1 贪心 / cut0 单段。子集: 步数≤400 按分位取样。

## 关键事实

1. **五个模型源权重全部是 fp16 存储** → torch fp32 推理本身就是「fp16 存储 + fp32 计算」，
   与本引擎第一版设计**同构**。「权重存储噪声地板」精确为零。
2. **唯一真·fp32 数值** = 加载时 weight_norm 融合产物（SoVITS dec 164 张 + HuBERT 1 张）。
   实测二次舍入到 fp16 会显著劣化音质 → `.gsv` 中此类张量只存 fp32（convert.py 已实现）。
3. **音频返回格式是 int16 刻度**（TTS.py `audio*32768`），golden 已按此记录，比较时先 /32768。
4. **存在按 pair 的解码器不稳定**: R-probe（仅换线程数，AR token 逐位一致）下部分超短生成
   （12–15 步，prompt 文本错配早停所致）波形 mel_rel 达 0.44~0.82；≥79 步的 pair 全部精确稳定。
   这是模型对 OOD 输入的自身敏感区，不是引擎可控行为；此类 pair 不作 G3 锚点。
5. bf16 autocast 包络过宽（末步 logits cos 0.44~0.99），无门槛标定价值，弃用。

## R-probe 实测（确定性地板）

| pair (steps) | AR logits | mel_rel |
|---|---|---|
| vo_BZLQ001_6_hutao_06__s0 (12) | 逐位一致 | 0.441 ⚠️不稳定 |
| vo_DPEQ002_6_hutao_27__s0 (15) | 逐位一致 | 0.000 |
| vo_BZLQ001_4_hutao_02__s9 (22) | 逐位一致 | 0.000 |
| vo_HTLQ001_3_hutao_16__s0 (79) | 逐位一致 | 0.000 |
| vo_ZBLQ001_13_hutao_19__s0 (306) | 逐位一致 | 0.000 |
| （前轮）vo_BZLQ001_6_hutao_02__s0 (15) | 逐位一致 | 0.819 ⚠️ |
| （前轮）vo_BZLQ001_6_hutao_03__s0 (12) | 逐位一致 | 0.610 ⚠️ |

## 定标后的门槛

| 门 | 数值 | 适用范围 | 状态 |
|---|---|---|---|
| **G1 张量级** | cos ≥ 0.9999 且 max-rel ≤ 1e-3 | 编码器输出/AR 隐藏态/SoVITS 中间量 | provisional，M1 复核 |
| **G2 解码级** | token 一致率 ≥ 98%；长度比 ∈ [0.8,1.25]；无雪崩 | 固定种子贪心全集 | 定稿 |
| **G3 音频级** | mel 相对变化 ≤ 5% | 仅 R-probe mel_rel=0 的锚点对 | provisional，M1 复核 |

规则说明: torch 自身确定性地板为 0（AR 逐位、稳定对音频逐位），故门槛无法从「波动地板」导出，
改由「设计噪声上界」推导: 我们的 kernel 为 fp16 激活 + fp32 累加，单点舍入 ~5e-4，
24 层放大后按 1e-3 收口；G2 由 AR 的贪心路径稳定性直接要求；G3 取听感透明量级。

## B12 验收补充（决策者记录, native 引擎对照结论）

1. **golden logits/tokens 口径 = repetition penalty 施压后的状态**：
   CPUFast `logits_to_probs` 用 `scatter_` 就地改写了导出 hook 持有的存储，
   故 pairs 里的 logits_first8/logits_last/tokens 均为惩罚后值。native 引擎按同构口径实现。
2. **penalty 语义 = 每步对唯一 token 集合去重覆盖**（gather/scatter），非逐出现累乘
   （step0 反推证实：出现 7 次的 prompt token 只压一次）。M6 int8 评估沿用此口径。
3. native fp32 步实测：G1 63/63 pairs 全过（cos=1.000000000，max-rel≈6e-7，门 1e-3）；
   G2 稳定对 27/27、短对记录 36/36 agree=1.0/lratio=1.0；总 18966 步与 golden 全程一致。
   → 证明「fp16存储+fp32计算」同构假设成立，G1 provisional 门槛可复核收紧。
4. 2 个 pair 缺输入字段（vo_BZLQ001_4_hutao_02__s1、vo_BZLQ001_6_hutao_02__s0）不参与对照。
