# E13-PROBE 参考音频编码侧极限压榨 · 画像报告（PROBE-1）

> 执行: E13 画像 worker（只探针、不改行为）　日期: 2026-08-27 07:00–07:45 CST
> 仓库: `/Volumes/2T/wt-gsv/E13` @ `task/E13`（基 a505384 = HANDOFF-E13 交接点）
> 硬件: MacBook Air M4 4P+6E / 16GB / Apple Clang / `GSV_AMX_GEMM=ON` Release 构建
> 测试音频: `test_wav/vo_HTLQ001_3_hutao_16.wav`（737 fbank 帧 / T=399 / Tq=199）

---

## 0. 结论速览

1. **官方口径复现**: 全链合计中位 ≈ **1954–2062ms**（热缓存），与 HANDOFF §3 的 1965ms 一致（±3%）。
2. **口径漏洞**: HANDOFF §2 代码图里的 `ssl_proj 重开 + RVQ 最近码字` 段**不在 `[ref-timing] 合计=` 打印之内**。实测该段 **81–89ms**，其中 ssl_proj 相关仅 2ms，**RVQ 暴力搜索占 80–87ms**（候选 6"白送几 ms"判断不成立，真正的白送是这 85ms）。
3. **真实参考侧总耗时 ≈ 2100ms**。
4. **理论地板 ≈ 218ms**（amxpp 实测吞吐 + 流式带宽实测，逐层 max(FLOPs板,带宽板) 求和）→ 当前 = **×9.7 总地板**；分引擎：HuBERT ×5.6、SV ×26。
5. **七候选逐项裁决见 §6**。核心修订：
   - ROI 第 1 名是 **HuBERT SDPA（459ms 纯标量，×6 减免空间）**；
   - SV 内部 GEMM 只有 ~240ms，**其余 ~700ms 是 BN/逐元素/pool 的访存型标量代码**（候选 3 的真实上限比预估更大，但性质变了：要 NEON 向量化而不是换 GEMM 后端）;
   - SV 的 FMLAL 回退清单已按形状排座次（榜首 4 次 `192→512@S14760` 就吃 93ms），AMX 门槛改法可一口吃掉 ~230ms；
   - 候选 6（ssl_proj 装载缓存）实测无价值（open 仅 1ms），但它牵出的 RVQ 段 85ms 高价值；
   - `cond.compute` 已拆干净：65ms 全在 temporal ConvGLU 标量四重循环，剪掉后 cond ≤ 30ms。
6. 本卡期间所有探针为**增量、环境变量门控、零行为变更**（详见 §7）；构建零警告、ctest 8/8 绿；未 commit（遵守 tests//md 禁令与决策者收口流程）。

---

## 1. 环境、构建与验收基线

```bash
cd /Volumes/2T/wt-gsv/E13
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGSV_AMX_GEMM=ON   # E12 起必须带 AMX 宏
cmake --build build -j 4
ctest --test-dir build        # 8/8 passed
```

- ⚠️ 注意：**裸 `-DCMAKE_BUILD_TYPE=Release` 会编译失败**（hubert.cpp/sv.cpp 的 `#else` 分支引用 `rows_expect/k_expect` 未声明、test_bert_amx 的 expect unused——均只在 `GSV_AMX_GEMM=OFF` 时暴露）。E12 分支的 README/HANDOFF 应记录 `-DGSV_AMX_GEMM=ON` 为标配。
- 符号链接补齐（worktree 惯例）: `weights -> /Volumes/2T/GPT-SoVITS-native/weights`（注意 `~/gsv-weights/ar_s1v3.gsv` 是 155MB 旧版，测试锚点要求主仓 465MB 版）、`test_wav -> 主仓 test_wav`。
- 数值基线无破坏：ctest 8/8（含 gsv_loader 数值锚点 / AR 短句贪心 md5 门禁类用例）。

---

## 2. 方法学

- 复现命令模板: HANDOFF §8 原样，外加 `--no-cache`；每轮经 `.tmp/run_probe.sh <tag>` 包装（`env $EXTRA_ENV GSV_REF_TIMING=1 ... --out .tmp/e13_out.wav`）。
- 暖缓存: 先空跑一遍（jieba/cmudict/g2pw 首调 ~5s 进 load 列，丢弃）。
- 负载纪律: 每 run 记录 `uptime`。本卡期间后台始终有其他 agent 活动，全程未遇 load<2.2 窗口；最终主张数据取自 load≈2.2–2.5 的配对窗（finalA/finalC），历史经验（base1–4 @load2.7-3.2）交叉印证结构稳定。负载 >3 的轮次只用于结构性比较（SKIP 隔离差值），已在表中标注。
- 中位数取法: 每配置 ≥3 次。

---

## 3. 基线画像（中位数）

| 段 | base×4（load2.7–3.2） | finalA/C（load2.2–2.5） | 采用值 |
|---|---|---|---|
| 前处理(resample+fbank+load) | 13–15ms | 13ms | **13** |
| sv.forward3 | 935–948ms | 961/963ms | **~942** |
| cond.compute | 91–94ms | 92/93ms | **92** |
| hubert.run | 906–931ms | 952/994ms | **~925**(结构细分采用 finalB 配对行) |
| **[ref-timing] 合计** | 1959–1978ms | 2021/2062ms | **1965±60** |
| extract_latent（漏计段） | 136–140ms | **81/89ms** | **~85**(rvq 80–87) |
| **真实总耗时** | | | **≈2100** |

> ⚠️ finalB 的 ref-timing 行被并发尖峰污染（sv 行冲到 5569ms）作废；但其**子探针行与 base/final 配对一致**，采纳作结构证据。extract_latent 在高负载时会膨胀（136ms）——该段也是带宽敏感的佐证。

### 3.1 HuBERT.run 内部（T=399）

finalB（配对行）:

```
[hubert-seg] cnn_stack=333.7ms(pos_conv=56.5ms enc_ln98.8ms) sdpa=459.5ms
[hubert-conv] total_gemm=252.9ms   (CNN 各层 FMLAL)
[hubert-sdpa] sdpa_total=459.5ms
```

| 子段 | ms | 说明 |
|---|---|---|
| CNN 栈(conv GEMM 部分) | 252.9 | 见 §4.2 逐层 |
| CNN 栈(GN/GELU/transposes) | ~81 | 333.7−252.9，访存型 |
| feature_projection + pos_conv 段 | 56.5 | 含 dense_AMX(T399,0.34ms级) + im2col FMLAL + LN |
| 编码体非 SDPA 部分(LN/残差/QKV&FFN GEMM/adds) | 98.8 | dense GEMM 本身 ~7ms(bench 外推)，~90ms 为 LN/加法/缓冲拷贝 |
| **SDPA(12层×QKᵀ+softmax+PV 标量三重循环)** | **459.5** | **单项最大，全可换批处理 GEMM** |
| 合计 | ≈949 | ✓ 对上 run 段总时 |

### 3.2 sv.forward3 内部（T=737 帧, F=80）

```
[sv-seg] stem=2.9 stage1=374.5 stage2=262.0 stage3=199.3 stage4=74.4 l3ds_fuse34_emb≈30ms
[sv-inner] conv2d_sgemm_total=0.5ms conv2d_f16_fmlal_total=216.6ms
           aff_w1=5.2ms(18次) aff_w2=5.9ms blk_conv1=6.0ms(S=58960)
```

关键结论：**GEMM 类总计 ≈234ms，非 GEMM（BN2d/double 累加、relu20 逐元素、concat/sp 拷贝、fuse 后 flatten-mean 的 double 累加 pooling）≈ 708ms**。stage1 占 374ms（入口分辨率 80×737，激活面积最大）。

FMLAL 回退按形状榜单（`GSV_SV_INNER_TIMING` 新增 rank 输出，一次 encode）：

```
[sv-f16rank] #0 x4     93.3ms  192,40,369 k1s1->512    (L2 blk1.conv1 s2, S=14760)
[sv-f16rank] #1 x3     68.3ms  96,80,737  k1s1->256    (L1 blk0.conv1 s2)
[sv-f16rank] #2 x12    63.4ms  24,80,737  k3s3->24??   (L1 convs k3@40x369 视角差异, 见 §4.3)
[sv-f16rank] #3 x1     16.9ms  64,80,737  k1s1->256    (L1 blk0.sc)
[sv-f16rank] #4 x1     10.5ms  256,80,737 k1s1->192    (L1 blk1.sc)
[sv-f16rank] #5 x1     10.3ms  1024,20,185 k1s1->768   (fuse34.w1 回退档)
[sv-f16rank] #6 x1     10.2ms  512,40,369 k1s1->384    (L2 blk1.sc)
累计 ≈273ms （负载 3.6 档；安静档 216.6+aff11.1+6.0）
```

根因明确：**E12 打包门槛 `S≥48 && K≥256` 把三类本可 AMX 化的形状拒之门外**——①小 K 的 1×1 stride2 下采样卷积（回退榜 #0/#1/#3/#4/#6 全是）；②`k==1` 时以"卷积"身份走 `conv2d_f16` 但其实恒等 dense 的旁路；③fuse34.w1（K=2048 ✓ 但 Aff::apply 里 gemm1 走的是另一条 if，`(skip&16)` 未含 w1 专档?? —— 复核：w1_panel_ready=true 时 aff gemm1 走 AMX，rank#5 出现说明该调用来自 fuse34 的 `sc_w` 同形路径或 skip 矩阵交叉，属可修细节）。E12 的初衷是省打包成本，但这些形状 K×S 都很大，打包一次遍历摊得过。

### 3.3 cond.compute 内部（frames=369）

```
[cond-timing] spectrogram=3.0ms transpose=0.2ms ref_enc=84.3ms svproj_prelu=1.4ms ge_to512=0.1ms TOTAL=89.0ms
[refenc-timing] spectral=2.1ms temporal_conv=65.2ms qkv_lin=0.5ms attn_core=15.6ms attfc=0.1ms fc_pool=1.3ms
```

- temporal ConvGLU 两层 = **65ms**：`conv1dGlu` 是标量四重循环且 double 累加（pipeline_condition.cpp `Conv1dGlu`），纯实现问题。
- attn_core 15.6ms：同样标量循环（2 head × T² score/PV）。
- 其余 <7ms。FFT 3ms 可保留。

### 3.4 extract_latent 三段（新发现）

```
[ref-timing] extract_latent=89ms(Tq=199) = open1 + proj1 + rvq87   (finalA)
[ref-timing] extract_latent=81ms(Tq=199) = open1 + proj1 + rvq80   (finalC)
```

- `rt::GsvFile sovF(...)` 每次重开：**1ms**（mmap 生效，候选 6 判死刑）。
- ssl_proj conv+ sgemm：1ms。
- **RVQ 最近码字搜索 80–87ms**：对 199 帧 × 1024 码字 × 768 维做标量 `double` L2（Tq×1024×768 ≈ 156M MACs）。批量化（按码字块 SGEMM 求距 + argmin）地板 0.39ms（§5 实测 rvq_dist_T199 = 0.387ms amxpp）。**~200× 减免空间**。
- ⚠️ 数值红线注记：现行实现双精度累加逐距离即时比较；改写后比较顺序不变（仍是同一 distance 公式 per (t,cb)），但累加顺序变了——fp32 块化累加会引入 ulp 级差，理论上可能在等距离巧合时翻转 argmin。验收须加"hidsen/codes md5 位级对比"门禁（HANDOFF §6 HuBERT 行标准）；若翻转，可在 tie-break 时回退 double 复算当行。

### 3.5 SKIP 隔离交叉验证（load 2.6–2.8 结构性证据）

| GSV_SV_AMX_SKIP | Δsv(ms) | 解读 |
|---|---|---|
| convs | +158 | 3×3 组（已 AMX 的 24+16+12 次调用被打回 FMLAL）|
| l3ds | +129 | 打包 panel 得益明确 |
| conv3 | +129 | 同上 |
| sc | +58 | shortcut 1×1s2 组 |
| conv1 | +40* | *数据在窗口尾部被挤，看趋势即可 |
| aff | +17 | fuse AFF 两 gemm |
| 默认 | — | — |

结构与 §3.2 的 inner 计时吻合（回退清单总和 ≈273 与各组被关掉的损益相当）。

---

## 4. 常量与工具箱实测（bench 纪律：零心算）

### 4.1 `tools/amx_bench --reps 20`（load≈3.3）

```
shape                  sgemm   fmlal    amx   amxpp   sg/pp  fl/pp  ax/pp
enc_p.ffn Co2048 K768 T300   0.832   5.016  0.857  0.647  1.29x  7.75x  1.32x PASS
bert.ffn(24L) Co4096 K1024 T64 0.687 2.631 0.737 0.327 2.10x 8.05x 2.25x PASS
dec.res2.k7 Co96 K672 T8000    1.057   4.859  1.274  0.738  1.42x  7.12x  1.73x PASS
enc_p.ssl_proj Co768 K512 T300 0.172  1.158  0.244  0.171  1.01x  6.78x  1.43x PASS
...
```

要点：fmlal 比 amxpp 慢 **6.8–9.4×**（ HANDOFF"手写 FMLAL 别写"教训在 2026-08 依旧成立），1×1/小形状也成立。

### 4.2 自建 `.tmp/flops_probe.cpp`（amxpp & sgemm，min-of-30，@load3.4）

| shape(M,N,K) | amxpp ms | sgemm ms | GFLOPS_amxpp |
|---|---|---|---|
| sv_L2out_1024 (1024,3700,384) | 1.61 | 1.83 | **1812** |
| sv_S2out_2048 (2048,930,1024) | 2.23 | 2.99 | 1746 |
| sv_S3out_1024x (1024,930,2048) | 2.52 | 3.65 | 1550 |
| hub_cnnL1_512 (512,12788,1536) | 13.72 | 21.42 | **1466** |
| hub_cnnL2_512 (512,6393,1536) | 8.54 | 10.76 | 1178 |
| hub_qkv_T399 (1197,768,768) | 1.35 | 1.50 | 1048 |
| hub_ffn_T399_f1 (399,3072,768) | 2.11 | 2.27 | 894 |
| hub_ffn_T399_f2 (399,768,3072) | 1.72 | 2.50 | 1096 |
| hub_proj_T399 (399,768,512) | 0.34 | 0.44 | 916 |
| attn_score_K64 (399,399,64) | 0.085 | 0.041 | **240** |
| attn_pv_K64 (399,64,399) | 0.057 | 0.038 | 357 |
| rvq_dist_T199 (199,1024,768) | 0.387 | 0.494 | **810** |

读法：T=399 dense 层实际吞吐只有大形状的 60%（1024×1 tile 半空 + packing 占比升高）；K=64 attention 小核对 AMX 极不友好（0.24–0.36TF），SDPA 批量化应优先走 **batched sgemm 或加大 batch 的 AMX 长条拼接**（E11-5/E10 配方路线正确）。

### 4.3 自建 `.tmp/bw_probe`（流式带宽，+法扫描，best-of-5）

| threads | GB/s |
|---|---|
| 1 | 7.3 |
| 4 | 29.3 |
| 10 | ~69.7 |

> HANDOFF §5 给的"1T=12.4/2T=25.7/4T=47.5 GB/s"与本机单进程标量扫描实测不同（那是向量库带宽口径）。地板计算采用保守标量流 **29.3GB/s@4T**；若用 vLoadQ/NEON 向量化搬运，实际上限更高，对本报告地板数字只会更有利。

---

## 5. 逐层 MACs / 权重字节 / 地板（`.tmp/floor_calc.py` 全量输出）

计算规则：MACs(conv)=Cout·Cin·k²·T_out，MACs(dense)=T·K·N；FLOPs 板 = 2·MACs ÷ GFLOPS(flops_probe 同形列)；带宽板 = (权重+激活扫描字节) ÷ 29.3GB/s；层地板 = max(两者)；跨层不可完全并行的保守口径直接相加。

### HuBERT

| 层 | MACs | flops 板 | 备注 |
|---|---|---|---|
| conv0 k10s5 (→25577帧) | 1310M | 1.5ms | 有争议：函数形式上是 im2col (T,K)，但 K=5120 一维展平无重复利用……保守取 max=35.5 后叙述 |
| conv1 k3s2 (→12788) | 30171M | 35.5ms | L1 主导 |
| conv2 k3s2 (→6393) | 15083M | 21.5ms | |
| conv3 k3s2 (→3196) | 7540M | 10.8ms | |
| conv4 k3s2 (→1597) | 3768M | 5.4ms | |
| conv5 k2s2 (→798) | 837M | 1.2ms | |
| conv6 k2s2 (→399) | 418M | 0.6ms | |
| CNN 小计 | 57997M | **76.5ms** | vs 现 253ms GEMM(FMLAL ×3.3 余量) |
| dense 12层(qkv/o/f1/f2/proj) | 34046M | 64.8ms | 现已在 AMX，≈7ms 挂钟 × 打包/写回放大 |
| SDPA 144头次(QKᵀ+PV) | 2934M | 19.6ms(小核率) | 现标量 459.5ms（**×23 余量**）|
| LN/GELU/残差访存 | — | 4.7ms | |
| **HuBERT 地板** | | **≈166ms** | **现 925/166 = ×5.6** |

### SV（stage/block 几何从 defs 与 trace 反推）

| 组 | MACs 合计 |
|---|---|
| stem+L1 三 blk | ~3.4G |
| L2 四 blk | ~4.4G |
| L3 六 blk | ~13.7G |
| l3ds | 4.44G |
| L4 三 blk + fuse34 | ~4.8G |
| **总计 ≈30.7G MACs** | **amxpp 地板 ≈36ms**（1700GF 代表率）|

**现 942ms ÷ 36 = ×26**；其中 GEMM 挂钟 234ms（×6.5 于纯 FLOPs 板），非 GEMM ~708ms 全部是访存/标量元操作——SV 的战场主要在"元操作向量化 + 免拷贝布线"，而非换 GEMM 后端。

### cond / RVQ / 杂项

| 项 | MACs | 地板 |
|---|---|---|
| cond 全链(FFT 除外) | 289M | <0.3ms + FFT 3ms 实保留 |
| RVQ 距离 | 156M | 0.39ms + argmin 0.2ms |
| 前处理(resample+fbank) | — | 现 13ms（暂按原值进地板口径）|

### 总地板

```
HuBERT 166 + SV 36 + RVQ 0.6 + cond 3.3 + 前处理 13 ≈ 218ms
当前真实 ≈2100ms = ×9.7 总地板
分引擎: HuBERT ×5.6 / SV ×26.1 / cond ×28(92ms 口径) / RVQ ×~140(85ms 口径)
```

诚实的工程可达目标（软件全套落地、不含并行）：≈800ms（×3.7 地板）；再加 sv∥hubert 流水：≈520–560ms（×2.5）。地板 218ms 里 ~30% 是本机实现的访存/小算子成本，触不到但方向清晰。

---

## 6. HANDOFF §5 七候选逐项裁决（按画像后真实 ROI 重排）

| 序 | 候选(原编号) | 实测占比 | 可减至 | 净赚 | 裁决 |
|---|---|---|---|---|---|
| 1 | **#2 HuBERT attention(SDPA)** | 459.5ms (22%) | E11-5 批量 sdpa：score/PV 走 batched sgemm（flops_probe: K64 小核 sgemm 0.041+0.038ms/头批 → 全链 <60ms；softmax 保留标量/NEON) | **≈400ms** | ★★★★★ 与 E11-5 配方同构, 数据链齐 |
| 2 | **新增: RVQ 暴力搜素** | 85ms（漏计段） | 块化 GEMM 0.39ms + argmin；須位级验收护栏 | **≈84ms** | ★★★★★ 一次重写消灭一整段 |
| 3 | **#3 SV conv 栈剩余层**（回退热榜 6 个形状族） | 216.6–273ms | 收紧打包门槛(k1 恒等/Δ打包成本核算/aff-w1 补档)后 amxpp 直打 ≈20ms(GEMM口径) | **≈200ms** | ★★★★☆ 无需并行化, 纯分流修正 |
| 4 | **新增: SV 非 GEMM 元操作**（BN/relu20/concat/pool double 标量） | ~708ms | NEON 向量化 + float 化(红线内) + 布线免拷贝, 保守砍半 | **≈350ms** | ★★★★☆ 收益最大但最琐碎, 适合 worker 分头扫 stage |
| 5 | **#1 HuBERT conv encoder** | 253ms GEMM(+81ms 栈内访存) | im2col→panel 直写(E5-P2 配方) + amxpp；flops_probe L1=13.7/L2=6.3 实证 | **≈190ms** | ★★★★ 门槛明确(T≥128 全过), im2col 是工作量大头 |
| 6 | **#5 cond.compute 细分** | 92ms(temporal 65 + attn 16) | ConvGLU 转 im2col+sgemm（184M MACs → <1ms), MHA 2head 手写批量化 | **≈60ms** | ★★★☆ 一天量级独立件 |
| 7 | **#4 sv∥hubert 并行** | 两链输入独立 ✓ | 流水下限 ≈ max(sv+cond, hubert+rvq)+94+13；现状直挂即省 ~min(链) ≈ **900ms**；若先落 1/5 再并行再省 ~250ms | 现状视角★★★★★ | ⚠️ 资源协调件: 双线程池 + AMX 互斥(GSV_AMX_THREADS 池共享) + 内存预算(16GB 安全), 建议决策者单独发任务卡, 本卡不抢 |
| 8 | **#6 ssl_proj 装载缓存** | open=1ms | 1ms | **≈0** | ☓ 判死(画像推翻预设)；它牵出的 RVQ 段 85ms 才是肉(见序2) |
| 9 | #7 fbank/重采样 NEON | 13ms | — | <10ms | ☓ 不值得(<阈值20ms, HANDOFF 自己的门槛就没过) |

叠加效应账（净额重复部分已抵扣）：400+84+200+350+190+60 ≈ 1284ms 理论净赚上限 → 现实按 70% 折扣 ≈ 900ms → **全链 ≈1100ms（×5.0 地板）不改并行**；再叠并行 → ≈600ms（×2.8）。

---

## 7. 本卡探针改动清单（未 commit，全部增量、env 门控、零行为变更）

构建/验收状态：`cmake --build build -j4` 零警告、`ctest` 8/8 通过。

| 文件 | 行号 | 内容 | 开关 |
|---|---|---|---|
| src/runtime/pipeline_condition.cpp | 19–23 | `<chrono>/<cstdio>/<cstdlib>` include | — |
| 〃 | 223–232 | refEnc 入口计时器+采样点 tr0 | `GSV_COND_TIMING` |
| 〃 | 244–247,255 | spectral 段尾 tr1 | 同上 |
| 〃 | 267–268 | temporal conv 段尾 tr2 | 同上 |
| 〃 | 280–281 | qkv lin 段尾 tr3 | 同上 |
| 〃 | 295–297 | attn core 段尾 tr4 | 同上 |
| 〃 | 299–307 | attfc 段尾 tr5 | 同上 |
| 〃 | 320–330 | compute 入口 tp_all/tp0/tp1 | 同上 |
| 〃 | 338,345,352–354 | transpose/refEnc/svproj/ge_to512 段界 | 同上 |
| 〃 | 355–362 | `[cond-timing]`/`[refenc-timing]` fprintf | 同上 |
| src/runtime/pipeline.cpp | 355 | tLat0 起 | `GSV_REF_TIMING`（复用） |
| 〃 | 360 | tLatOpen（GsvFile 重开+权重段）| 同上 |
| 〃 | 371 | tLatProj（im2col+sgemm 段尾）| 同上 |
| 〃 | 397–402 | `[ref-timing] extract_latent=open/proj/rvq` 输出 | 同上 |
| src/encoder/hubert.hpp | 129–143 | ConvTrace 结构/成员/公开视图 | — |
| src/encoder/hubert.cpp | 12–15 | include | — |
| 〃 | 259–262 | conv_layer 计时起点/开关 | `GSV_BERT_CONV_TIMING` |
| 〃 | 279–292 | conv_layer 尾部采样入 conv_traces_ | 同上 |
| 〃 | 297–303 | run() 清空 trace + sdpa 开关 | 两个开关 |
| 〃 | 310,357 | CNN 栈起/止分段点 tp_cnn_* | 同上 |
| 〃 | 383–394 | pos_conv 段界 tp_pos1 | 同上 |
| 〃 | 419–420 | SDPA 段计时（per-layer 累加） | `GSV_HUBERT_SDPA_TIMING` |
| 〃 | 537–560 | run() 尾部 `[hubert-conv]/[hubert-seg]/[hubert-sdpa]` 输出 | 同上 |
| src/encoder/sv.cpp | 112–123,131 | SvInnerTimers(file-static thread_local) + 开关 | `GSV_SV_INNER_TIMING` |
| 〃 | 188–198 | conv2d(sgemm) 计时 | 同上 |
| 〃 | 225–236 | conv2d_f16(fmlal) 计时 + 按形状聚合 map | 同上 |
| 〃 | 337–349, 383–397 | Aff w1/w2 回退分支计时 | 同上 |
| 〃 | 430–446 | Block conv1(k1 s1) 回退分支计时 | 同上 |
| 〃 | 705–711, 725–741 | forward3 stem/stage 分段点 + tp_stage_last | 同上 |
| 〃 | 792–818 | `[sv-inner]`/`[sv-f16rank]`(top14) 输出 | 同上 |
| 〃 | 819–830 | `[sv-seg]` stem/stage1-4/tail 输出 | 同上 |

辅助工具（`.tmp/`，探针基础设施）：`run_probe.sh`（标准化跑批）、`bw_probe.{cpp,+bin}`（带宽）、`flops_probe.{cpp,+bin}`（同形 amxpp/sgemm 微基准）、`gen_tables.py`/`floor_calc.py`（表格与地板演算，可复查）。

过程透明度记录：给 hubert 加 pos_conv 段界时曾误删 `gelu(pos_out_)` 调用一行，**当即发现并在同一回合恢复**，最终 `git diff` 净差异全部为 `+` 行（除一处同行格式合并，语义等价），ctest 与锚点通过。

git 状态：以上探针改动 + 2 个符号链接(weights/test_wav)+ `.tmp/` 均未提交；tests/ 与 *.md 零触碰；未 push。

---

## 8. 关键原始输出存档（摘录）

```
# base(base1–4, load2.4–3.2)
[ref-timing] sv.forward3=935..948ms cond=91..94 hubert.run=906..931 前处理=13 合计=1959..1978
# deep(deep1/deep2/deep3, 组合探针)
[cond-timing] spectrogram=3.0 transpose=0.2 ref_enc=84.3..86.9 svproj_prelu=1.3..5.9 ge_to512=0.1 TOTAL=89..96
[refenc-timing] spectral=2.1 temporal_conv=65.2..66.5 qkv_lin=0.5 attn_core=15.6..15.9 attfc=0.2 fc_pool=1.3 T=369
[hubert-seg] cnn_stack=333.7(pos_conv=56.5 enc_ln98.8) sdpa=459.5   ← finalB 行(run=949 对账✓)
[hubert-conv] L1 c512*len25577 k3s2 fmlal=131.9 … L6 fmlal=2.2 total=252.9
[sv-seg] stem=2.9 stage1=374.5 stage2=262.0 stage3=199.3 stage4=74.4 tail≈30
[sv-inner] conv2d_sgemm_total=0.5 conv2d_f16_fmlal_total=216.6 aff_w1=5.2(18) aff_w2=5.9 blk_conv1=6.0
# finals(load 2.2–2.5)
[ref-timing] sv=961/963 cond=92/93 hubert=994/952 前处理=13 合计=2062/2021
[ref-timing] extract_latent=89/81 (Tq=199) = open1 + proj1 + rvq87/80
# 频谱面盘点(32 AMX 命中形状 top)
24×[96,20,185 k3→96,S3700] 16×[48,40,369 k3→48,S14760] 12×[192,10,93 k3→192,S930]
 6×[384,20,185 k1→1024]  5×[1024,20,185 k1→384]  3×[768,10,93 k1→2048]  3×[512,40,369 k1→192]
 2×[256,80,737 k1→96]    2×[2048,10,93 k1→768] … 共77次调用
```

---

## 9. 风险与移交建议

1. **测量噪声**：本机常驻负载从未低于 2.2；所报倍数留 ±5% 余量。若需要更硬的主张，请在夜间 idle 窗复测 finalA/C 两组即可（脚本与开关都是现成的）。
2. **RVQ 数值红线**：序2 落地时务必加"hidden/codes 两路 md5 位级比对 + tie 情形审计"（§3.4）；若担心 flip，可用"argmin 结果在 top-2 差距 < ε 时回退双精度精确复算"的混合方案。
3. **SV 元操作 float 化风险**：BN 的 `double` 计算/emb pool 的 double 是 C1 实现选择而非 torch 语义，但在改动它们之前必须先读 CALIBRATION.md 相应节并小步 A/B（候选 4 首日就该做的事）。
4. **SKIP=conv1 的样本数偏少**（窗口末尾数据），如需量化 conv1 单独贡献请重跑一轮。
5. 给决策者的任务卡拆分建议：①SDPA 批量化（纯 runtime/sovits 域配方平移）②RVQ 批量化（runtime 域+红线护栏）③SV 分流门槛收紧（encoder 域小改）④HuBERT CNN AMX（encoder 域+E5-P2 配方）⑤SV 元操作向量化（encoder 域, 可两人分 stage）⑥cond ConvGLU（runtime_condition 域）⑦并行化（runtime 域, 需资源协调）。②③⑥是小刀快活，①④⑤是主力。
