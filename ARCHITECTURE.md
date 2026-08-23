# GPT-SoVITS-native 架构设计 v1

> 目标平台: Apple MacBook Air M4 (4P+6E, ~120GB/s DRAM, 无风扇)
> 硬约束: 纯 CPU，零 GPU/Metal/ANE，零 Python 运行时依赖，不用 torch/onnx/ggml，CLI 交付
> 性能北极星: **尽可能吃满 CPU**（P/E 核利用率、DRAM 带宽占比、NEON/SME 有效算力），RTF 仅作参考
> 版本锁定: v2Pro 家族（v2Pro 与 v2ProPlus 共享代码路径，仅权重不同；本仓库现有 s2Gv2ProPlus.pth 作为参考实现）

## 1. 模型规格（从 ckpt 实测）

| 模型 | 参数量 | 结构 | 调用时机 |
|---|---|---|---|
| 文本前端 | — | jieba 分词 + G2PW 多音字(BERT) + pinyin 规则 → symbols2 词表(732) | 每句 |
| BERT 韵律 | chinese-roberta-wwm-ext-large ≈330M | 24层 × 1024d | 每句 |
| HuBERT | chinese-hubert-base ≈95M | 12层 × 768d, 输出 25Hz 语义特征 | 参考音频一次 |
| SV 说话人 | eres2netv2w24s4ep4 ≈25M (ckpt 103MB fp32) | CNN | 参考音频一次 |
| **AR/T2S** | s1v3 **77.6M 实测** | 24层 × 512d × 16头, FFN 2048, fused QKV (1536×512), 词表 732音素 + 1025语义, EOS=1024, top_k=5 | 每 token 自回归 |
| **SoVITS** | s2Gv2ProPlus **99.9M 实测** | enc_p(235t) + dec(290t, weight_norm 需融合 g·v/‖v‖) + flow(124t) + enc_q(103t) + ref_enc(18t) + quantizer/ssl_proj/sv_emb/ge_to512/prelu | 每 mel 帧 |

两个 ckpt 内嵌 `config` 键，转换工具直接读取，不硬编码维度。

## 2. 总体架构

```
gsv-native (单二进制 C++20, 链接 Accelerate)
├── textfront/    原生中文前端: 分词→G2PW多音字→拼音→音素ID + BERT韵律特征
├── encoder/      HuBERT + SV 说话人编码（参考音频结果磁盘缓存）
├── ar/           T2S 引擎: prefill(Accelerate GEMM) + decode(手写 NEON GEMV) + KV cache
├── sovits/       enc_p / flow / dec 全 CPU 解码（im2col→GEMM + fused epilogue）
├── kern/         手写内核层（见 §3）
└── runtime/      mmap 权重加载、QoS 分簇线程池、段级流水线、WAV 输出

tools/
├── convert.py    离线一次性转换 .pth/.ckpt → .gsv 格式（Python 只存在于工具链，
│                 不进运行时；torch 仅开发机需要）
└── bench/        微基准 + 吃满率采样(powermetrics/asitop)
```

### 数据流与流水线（吃满 CPU 的核心设计）

```
E核:   下一段文本前端 + BERT        (utility QoS, 后台)
P核0-1: AR 生成第 N+1 段语义 token   ← 双缓冲段队列 →
P核2-3: VITS 合成第 N 段波形         (user-initiated QoS)
主线程: WAV 流式写盘
```

原理: AR decode 是 **DRAM 带宽瓶颈**，VITS 是 **L2 内算力瓶颈**——资源互补，
纯 CPU 上重叠效率反而高于异构方案。首包 = 第一段 AR 完成 → 立即开合成。

### 缓存

- 参考音频 → {HuBERT 特征, SV 嵌入}: 按 (路径+mtime+配置) 哈希缓存 `~/.cache/gsv-native/`
- 权重: 单文件 `.gsv`（自定义布局, 64B 对齐）, mmap 零拷贝启动
- 音色切换 = 切换参考缓存指针，不重算编码器

## 3. 计算内核层（三层分工）

| 层 | 技术 | 用途 |
|---|---|---|
| L1 手写 NEON intrinsics | `-mcpu=apple-m4` | AR decode GEMV（int8 权重→寄存器内 dequant fp16→FMA，epilogue 融合 rmsnorm/silu）、KV-cache attention（自定义按 head 分页连续布局）、rope/softmax/gelu、im2col、逐层 elementwise |
| L2 Accelerate | vDSP_mmul / BNNS | 一切大 GEMM：AR prefill、VITS im2col 后的卷积 GEMM、BERT/HuBERT。M4 上自动调度到 SME 矩阵单元 |
| L3 SME 实验分支 | FMOPA outer-product（未文档化，用户态可执行已被 Jena/tzakharko 验证） | 二期验证 int8 GEMV/GEMM 是否胜过 L1/L2 组合，赢了才合入 |

数值策略（两步走）:
- 第一步（M1/M2 实现期）: **全 fp32 计算与累加**，权重加载 fp32 段，目标 = 与 torch golden
  最大程度对齐（同源数值、仅归约顺序差异）；KV cache fp32
- 第二步（fp16 化）: 权重切 fp16 段（源值即 fp16，逐位等价零噪声）+ FMLAL 扩展精度乘法
  （无 fp16 中间舍入点）+ KV cache fp16 开关；每次切换过三道门后才合入
- 硬件前提（本机实测）: FEAT_FP16+FEAT_I8MM；NEON fp16 FMA 双倍吞吐；FMLAL 可直接累进 fp32

### 精度依据（M0 定标结论，详见 tests/golden/CALIBRATION.md）

- 五模型源权重全为 fp16 存储 → torch fp32 基线本身就是「fp16存储+fp32计算」，本设计与 golden 同构；
  R-probe 实测跨线程 AR logits 逐位一致 → 权重存储层零额外噪声
- WN 融合产物是真·fp32，二次舍入会劣化音质 → .gsv 中此类张量只存 fp32
- 存在按 pair 的解码器不稳定（超短生成+prompt 错配），与引擎无关但影响 golden 锚点选择

### 精度一致性定义（"与 torch 一致"的可测试语义）

逐位一致不可达（BLAS 归约顺序不同），采用三道门，数值已于 M0 锁定（tests/golden/gates.json）：
- G1 张量级: cos-sim ≥ 0.9999 且 max-rel ≤ 1e-3（provisional，首次 native 对照后复核）
- G2 解码级: token 序列一致率 ≥ 98%，长度比 ∈ [0.8,1.25]，无雪崩
- G3 音频级: 稳定锚点对上 mel 相对变化 ≤ 5%
- 阶段二量化（已批准）: AR 权重 int8 + per-channel scale + KV int8；VITS 逐层 A/B 听感
- 内核纪律: softmax/rmsnorm 统计量只用 fp32（fp16 尾数 10 位，长链必炸）；不用 bf16

## 4. 线程模型

- decode 阶段 AR 仅 2 线程：带宽墙下更多线程会把负载挤上 E 核导致行为劣化（eclecticlight 实测结论）
- prefill/VITS/BERT 可扩到全部 P 核
- QoS 绑簇：实时链路 user-initiated → P 核；后台编码 utility → E 核
- 无风扇热约束：bench 必须报告 30 分钟持续负载下的频率曲线，降频后仍需满足吃满率目标

## 5. "吃满 CPU" 的验收度量（bench 工具输出）

1. 各核利用率（powermetrics 采样）：合成期间 P 核 ≥ 90%，E 核 ≥ 50%
2. DRAM 带宽：AR decode 达到理论峰值的 ≥ 60%（int8 后 ≥ 70%）
3. GEMM 有效算力：prefill 与 VITS 阶段对照 Accelerate 峰值 ≥ 80%
4. RTF、首包延迟、每段 token 速率：仅记录，不设门槛
5. 数值正确性永远优先于以上所有指标

## 6. 文本前端原生重写清单

| 组件 | 方案 |
|---|---|
| 分词 | jieba DAG+HMM 算法移植，词典从 CPUFast 导出为二进制 trie；golden 对照 CPUFast 分词输出 |
| 多音字 | **G2PW 复用我们自己的 BERT kernel 跑**（它本身就是 BERT-base）；WordPiece tokenizer 移植；phrase_overrides pickle 直接二进制读入 |
| 拼音/规则 | compact_pypinyin 表 + 数字/符号/拉丁转写规则逐条移植 chinese2.py |
| 词表 | symbols2 (732) 编译期为常量表 |

## 7. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M0 | convert.py + golden 数值基线导出（torch 记录各模型关键中间张量）+ 三道门数值定标 | .gsv 加载成功; golden 集合可复现; G1/G2/G3 门槛数值锁定 |
| M1 | AR 引擎: 先全 fp32 对齐 golden，再切 fp16（存储/FMLAL/KV）；prefill(Accelerate) + 手写 NEON decode GEMV + KV cache | fp32 步: G1/G2 达标; fp16 步: 三道门复验; decode 吞吐记录 |
| M2 | SoVITS 全链路: enc_p/quantizer/flow/dec + weight_norm 融合 | 端到端出 wav, 听感与 torch 版无差异 |
| M3 | 文本前端原生: 分词/G2PW/pinyin/规则 | 与 CPUFast 输出逐句 diff 为空（测试集 ≥200 句） |
| M4 | 编码器 + 缓存 + CLI 整合 | CLI 一条命令完成 文本+参考音频→wav |
| M5 | 吃满率优化: 流水线重叠/线程调优/bench harness | §5 度量全达标 |
| M6 | int8 量化阶段二 | A/B 听感通过; AR decode 带宽占比提升至 ≥70% |

## 8. 风险表

| 风险 | 对策 |
|---|---|
| G2PW 原生化（WordPiece/位置编码细节） | M1 先有 BERT kernel，M3 移植 tokenizer 有 golden 对照 |
| jieba 移植分词不一致 | 以 CPUFast 输出为唯一 golden，diff 驱动修复 |
| MBA 热节流吃不满 | §4 线程数保守化 + bench 报告频率曲线，必要时暴露 `--threads` 给用户 |
| SME 未文档化无 ABI 承诺 | 只在 L3 实验分支，运行时 AMXVER/SME 探测失败自动回退 |
| v2Pro vs v2ProPlus 权重差异 | 引擎家族化设计，维度全部来自 ckpt 内嵌 config |

## 9. 决策记录

已迁至 `AGENTS.md`「已定决策」清单（唯一维护点），此处不再重复。
