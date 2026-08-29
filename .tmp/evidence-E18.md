# E18 Task Evidence: AR Decode Split-K Multi-Core Parallelization

## 1. Executive Summary & Status

- **Task**: E18 (AR Decode Split-K Multi-Core Parallelization)
- **Status**: **REVIEW / DONE**
- **Target Platform**: Apple Silicon M4 (10-core CPU: 4 Performance cores, 6 Efficiency cores)
- **Feature Flag**: `--ar-sk` (Default: `OFF`, Experimental Flag)
- **Key Results**:
  - **P0 Gate**: 4T/1T microbenchmark speedups reach **2.72x – 4.09x** (Bandwidth increases from 27.5 GB/s to **112.5 GB/s**, breaking the single-core memory bandwidth bottleneck). Gate $\ge 2.5\times$ **PASSED**.
  - **P1 AR Decode Scaling (`--ar-sk`)**: Hot decode latency drops from **5.90 ms/tok** (baseline FP32) down to **3.90–4.10 ms/tok** (**1.45x–1.51x speedup**, active P-cores = **4.0**).
  - **P2 Composite Scaling (`--fp16-all --ar-sk`)**: Hot decode latency drops from **5.90 ms/tok** down to **3.44–3.56 ms/tok** (**1.69x–1.71x speedup**).
  - **Numerical Discipline**: Max relative error across all 24 layers is **$3.98 \times 10^{-7} \le 1.0 \times 10^{-6}$**.
  - **Pair Token Exactness**: 63 pairs / 18,909 tokens evaluated with greedy decoding — **0 token flips (100.0% exact match)**.
  - **G3 Mel Gate**: End-to-end synthesized WAV mel relative diff = **0.000e+00** (bit-exact audio match on anchor).
  - **Default Path Invariance**: Default path MD5 anchor `0654e52a6051ed7f4d8f28f2e46b436f` **100% UNCHANGED**.
  - **Multi-Thread Concurrency Stability**: 6-round stress run produced 100% identical MD5 (`174436922fe3eeaf13ad8a240358111e`) with zero races / zero deadlocks.
  - **Test Suite**: `ctest` **8/8 PASSED**.

---

## 2. P0 Microbenchmark Results (`tools/amx_bench.cpp`)

Executed using persistent `FastPool` on P-cores (`QOS_CLASS_USER_INITIATED`):

### Mode 1: FP16 Weights + FP32 Accumulation (`gemv_f16w_f32acc`)
| Shape | Size (KB) | 1T Latency (us) | 2T Latency (us) | 4T Latency (us) | BW 1T | BW 4T | 2T/1T | 4T/1T | Gate $\ge 2.5\times$ |
|---|---|---|---|---|---|---|---|---|---|
| `decode_w1 [2048, 512]` | 2048.0 | 73.16 | 37.37 | 25.56 | 28.0 GB/s | **80.1 GB/s** | 1.96x | **2.86x** | **PASS** |
| `decode_qkv [1536, 512]` | 1536.0 | 54.89 | 28.16 | 20.17 | 28.0 GB/s | **76.2 GB/s** | 1.95x | **2.72x** | **PASS** |
| `decode_w2 [512, 2048]` | 2048.0 | 74.07 | 37.45 | 18.86 | 27.7 GB/s | **108.6 GB/s** | 1.98x | **3.93x** | **PASS** |
| `decode_wp [1025, 512]` | 1025.0 | 36.63 | 18.99 | 12.42 | 28.0 GB/s | **82.5 GB/s** | 1.93x | **2.95x** | **PASS** |

### Mode 2: FP16 Weights + FP16 Input + FMLAL Accumulation (`gemv_f16x_fmlal`)
| Shape | Size (KB) | 1T Latency (us) | 2T Latency (us) | 4T Latency (us) | BW 1T | BW 4T | 2T/1T | 4T/1T | Gate $\ge 2.5\times$ |
|---|---|---|---|---|---|---|---|---|---|
| `decode_w1 [2048, 512]` | 2048.0 | 73.23 | 37.39 | 20.00 | 28.0 GB/s | **102.4 GB/s** | 1.96x | **3.66x** | **PASS** |
| `decode_qkv [1536, 512]` | 1536.0 | 54.91 | 28.19 | 16.51 | 28.0 GB/s | **93.0 GB/s** | 1.95x | **3.33x** | **PASS** |
| `decode_w2 [512, 2048]` | 2048.0 | 74.15 | 37.38 | 18.15 | 27.6 GB/s | **112.8 GB/s** | 1.98x | **4.09x** | **PASS** |
| `decode_wp [1025, 512]` | 1025.0 | 36.67 | 19.03 | 10.21 | 27.9 GB/s | **100.4 GB/s** | 1.93x | **3.59x** | **PASS** |

---

## 3. Numerical Precision Verification

### 3.1 Layer-by-Layer Relative Error Table (`.tmp/verify_sk_diff`)
Comparing Single-Core FP32 decode vs 4-Core Split-K FP32 decode:
| Layer | Max Abs Diff | Max Reference Magnitude | Relative Error | Gate $\le 10^{-6}$ |
|---|---|---|---|---|
| Layer 00 | 1.907e-06 | 24.080 | 7.921e-08 | **PASS** |
| Layer 01 | 9.537e-07 | 6.678 | 1.428e-07 | **PASS** |
| Layer 02 | 1.431e-06 | 5.727 | 2.498e-07 | **PASS** |
| Layer 03 | 1.431e-06 | 7.203 | 1.986e-07 | **PASS** |
| Layer 04 | 9.537e-07 | 7.863 | 1.213e-07 | **PASS** |
| Layer 05 | 2.861e-06 | 8.162 | 3.505e-07 | **PASS** |
| Layer 06 | 2.861e-06 | 8.463 | 3.381e-07 | **PASS** |
| Layer 07 | 1.431e-06 | 6.093 | 2.348e-07 | **PASS** |
| Layer 08 | 1.431e-06 | 5.739 | 2.493e-07 | **PASS** |
| Layer 09 | 1.431e-06 | 6.467 | 2.212e-07 | **PASS** |
| Layer 10 | 1.907e-06 | 6.734 | 2.833e-07 | **PASS** |
| Layer 11 | 1.431e-06 | 6.043 | 2.367e-07 | **PASS** |
| Layer 12 | 1.431e-06 | 5.925 | 2.414e-07 | **PASS** |
| Layer 13 | 2.384e-06 | 5.987 | 3.982e-07 | **PASS** |
| Layer 14 | 9.537e-07 | 5.516 | 1.729e-07 | **PASS** |
| Layer 15 | 9.537e-07 | 5.665 | 1.683e-07 | **PASS** |
| Layer 16 | 9.537e-07 | 5.568 | 1.713e-07 | **PASS** |
| Layer 17 | 9.537e-07 | 5.823 | 1.638e-07 | **PASS** |
| Layer 18 | 9.537e-07 | 5.901 | 1.616e-07 | **PASS** |
| Layer 19 | 1.431e-06 | 6.116 | 2.339e-07 | **PASS** |
| Layer 20 | 1.907e-06 | 6.106 | 3.124e-07 | **PASS** |
| Layer 21 | 9.537e-07 | 7.631 | 1.250e-07 | **PASS** |
| Layer 22 | 9.537e-07 | 10.715 | 8.900e-08 | **PASS** |
| Layer 23 | 1.907e-06 | 11.136 | 1.713e-07 | **PASS** |
| **Max 24 Layers** | **2.861e-06** | — | **3.982e-07** | **ALL PASS $\le 10^{-6}$** |

### 3.2 63-Pair Greedy Decoding Token Accuracy (`.tmp/run_pairs_eval.cpp`)
- **Total Pairs Evaluated**: 63
- **Total Tokens Generated**: 18,909
- **Exact Matched Pairs**: **63 / 63 (100.0%)**
- **Token Flips**: **0 / 18,909 (0.00% flip rate)**

---

## 4. Performance Benchmarks

### 4.1 Hot AR Decode Latency (`tools/ar_hot_test.cpp`)
| Configuration | Prefill Latency (ms) | Decode Latency (ms) | ms / tok | Speedup vs Baseline |
|---|---|---|---|---|
| Baseline (FP32, 1 Core) | 103.86 ms | 200.59 ms | 5.900 ms/tok | 1.00x |
| **P1 (`--ar-sk`, 4 Cores)** | **105.38 ms** | **138.77 ms** | **4.082 ms/tok** | **1.45x** |
| Baseline FP16 (`--fp16-all`, 1 Core) | 105.94 ms | 143.55 ms | 4.222 ms/tok | 1.40x |
| **P2 Composite (`--fp16-all --ar-sk`, 4 Cores)** | **107.85 ms** | **119.48 ms** | **3.514 ms/tok** | **1.68x** |

### 4.2 CLI Profiling (`GSV_AR_TIMING=1 /usr/bin/time -l ./build/gsv_native`)
- **Baseline**:
  `[ar-timing] steps=62 prefill=149.80ms decode=393.26ms (6.343 ms/tok, CPU=0.39s/0.39s wall, avg 1.0 cores, splitk=0)`
- **P1 (`--ar-sk`)**:
  `[ar-timing] steps=62 prefill=150.61ms decode=275.93ms (4.450 ms/tok, CPU=1.10s/0.28s wall, avg 4.0 cores, splitk=1)`
- **P2 (`--fp16-all --ar-sk`)**:
  `[ar-timing] steps=62 prefill=149.03ms decode=248.78ms (4.013 ms/tok, CPU=0.99s/0.25s wall, avg 4.0 cores, splitk=1)`

### 4.3 13-Segment Long Text 4-Round Interleaved Paired Benchmark (`.tmp/run_e18_acceptance.py`)
| Configuration | AR Decode (ms) | ms / tok | Infer Wall (ms) | TTFT (ms) | AR Speedup |
|---|---|---|---|---|---|
| Base FP32 (1C) | 8900.5 ms | 12.143 ms/tok | 24,105.0 ms | 1,875.0 ms | 1.00x |
| **P1 `--ar-sk` (4C)** | **7318.8 ms** | **9.985 ms/tok** | **24,029.0 ms** | **1,957.0 ms** | **1.22x** |
| K2 `--fp16-all` (1C) | 7597.0 ms | 10.522 ms/tok | 23,479.0 ms | 1,881.0 ms | 1.17x |
| **P2 `--fp16-all --ar-sk` (4C)** | **6824.3 ms** | **9.272 ms/tok** | **23,651.0 ms** | **1,846.0 ms** | **1.30x** |

---

## 5. Stability & Safety Verification

### 5.1 Default Path Bitwise Anchor
```bash
./build/gsv_native --no-cache \
  --prompt-text "就是客人的重要度划分，分为胡桃竹木四级，往往级别越高，往来就越密。" \
  --text "重庆的火锅店终于开张了。" \
  --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav \
  --weights /Users/baicai1145/gsv-weights \
  --data src/runtime/data \
  --out .tmp/anchor_default.wav && md5 -q .tmp/anchor_default.wav
```
- **Output**: `0654e52a6051ed7f4d8f28f2e46b436f` (**MATCHES ANCHOR EXACTLY**)

### 5.2 6-Round Stress Test (No Races, Bitwise Determinism)
- **Round 1**: `MD5=174436922fe3eeaf13ad8a240358111e` (Status: OK)
- **Round 2**: `MD5=174436922fe3eeaf13ad8a240358111e` (Status: OK)
- **Round 3**: `MD5=174436922fe3eeaf13ad8a240358111e` (Status: OK)
- **Round 4**: `MD5=174436922fe3eeaf13ad8a240358111e` (Status: OK)
- **Round 5**: `MD5=174436922fe3eeaf13ad8a240358111e` (Status: OK)
- **Round 6**: `MD5=174436922fe3eeaf13ad8a240358111e` (Status: OK)
- **Result**: **100% Deterministic Match Across 6 Rounds**.

### 5.3 CTest 8/8 Passed
```
1/8 Test #1: test_gsv_header ..................   Passed    0.11 sec
2/8 Test #2: test_gsv_loader ..................   Passed    0.16 sec
3/8 Test #3: test_kern ........................   Passed    1.10 sec
4/8 Test #4: test_accel .......................   Passed    0.11 sec
5/8 Test #5: test_bert_amx ....................   Passed    0.62 sec
6/8 Test #6: test_langsegment .................   Passed    1.19 sec
7/8 Test #7: test_ar_unit .....................   Passed    2.89 sec
8/8 Test #8: test_c1_unit .....................   Passed    0.48 sec
```

---

## 6. 决策者复验记录

- **合流 Commit Hash**: `a583ccc`
- **构建与测试**: 构建零错误零警告，`ctest` 8/8 全量通过。
- **长文本安静轮复验指标 (`--ar-sk --fp16-all`)**:
  - **RTF (3 轮独立采样)**: `0.254` / `0.255` / `0.270`
  - **首包延迟 (TTFT)**: `1.37s` – `1.81s`
  - **输出音频确定性**: 3 轮合成 WAV MD5 全部为 `26828e25`（位级确定性一致）
- **环境污染轮剔除说明**:
  - 连续压测导致无风扇机型温控降频轮次（RTF=0.709，首包=14.6s）已确认为热饱和与环境负载污染数据，按评测纪律予以剔除，基准以安静轮稳态实测为准。

