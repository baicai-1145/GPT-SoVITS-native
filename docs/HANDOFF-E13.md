# E13 交接文档：参考音频编码侧极限压榨（独立马拉松卡）

> 写给：接手本卡的独立执行 agent（预算 ~10-15 小时）
> 写者：决策者（pi orchestrator） 日期：2026-08-27
> 仓库：/Volumes/2T/GPT-SoVITS-native（main @ 37b50a5）

## 0. 一句话任务

把"参考音频 → 可用条件"的编码全链（SV 说话人 + HuBERT 语义 + cond 条件 + 前处理）从实测 **1965ms** 压到**可证明的硬件极限**（AMX 算力地板或内存带宽地板），数值纪律不破。

## 1. 硬约束（违反即返工，无商量）

1. **纯 CPU**：禁 Metal/GPU/ANE/CoreML；**AMX 属 CPU 矩阵扩展，允许且是主武器**
2. 数值红线（详见 §6）：默认路径（旗关）位级不变；HuBERT 输出喂 AR 调节链必须逐位一致
3. 第三方库不引入（现有一切够用）；运行时零 Python
4. 内存纪律：编译 `-j 4`、单测串行、大缓冲及时释放（本机 16GB，曾因批处理 OOM）
5. STATE.md 只读（决策者唯一写者）；你只 commit 自己领地文件，**tests/ 与 *.md 不许 commit**（回报时留给决策者收）
6. git：从 main 拉分支 `task/E13`，本地小步 commit，禁 push

## 2. 你要优化的东西（代码地图）

全部在 `src/runtime/pipeline.cpp` 的参考构建段（搜 `GSV_REF_TIMING`）：

```
wav 加载 → 重采样(32k/16k) → normalize
 ├─ SV 路:   a16→kaldi_fbank_80 → SvEngine::forward3(fbank,737帧) → svEmb[20480]     [src/encoder/sv.{hpp,cpp}]
 ├─ cond 路: cond_.compute(a32, svEmb) → SoVITS 条件(spec→ref_enc→ge/ge_text)        [src/runtime/pipeline_condition.cpp]
 └─ HuBERT 路: wav16k+9600零 → HubertEngine::run → hidden[T=399,768]
                → ssl_proj conv(k2s2) → RVQ 最近码字 → prompt_semantic(199 codes)     [src/encoder/hubert.{hpp,cpp}]
```

注意：`ssl_proj` 权重每次 encode 都 `GsvFile` 重开重读（pipeline.cpp ~L318）——白送的优化点。

## 3. 当前画像（2026-08-27，负载 2.2，热页缓存，--no-cache 口径）

| 段 | 耗时 | E12 前 | 状态 |
|---|---|---|---|
| sv.forward3 | 935ms | 1629ms | E12 只切了 K≥256/S≥48 的 conv；小形状层仍旧路径 |
| hubert.run | 925ms | 3115ms | dense 层已 AmxPanel；conv encoder + attention 未动 |
| cond.compute | 92ms | 92ms | 未画像细分 |
| 前处理(重采样+fbank) | 13ms | — | 已小 |
| **合计** | **1965ms** | 4928ms | |

**探针**（都在 main，环境变量门控）：
- `GSV_REF_TIMING=1`：pipeline 层分段
- `GSV_SV_AMX_TRACE=1`：SV 每 conv 的 AMX 命中形状
- `GSV_SV_AMX_DUMP=<n>` / `GSV_SV_AMX_SKIP=conv1|convs|conv3|sc|aff|l3ds`：SV 层级隔离
- `GSV_BERT_LAYERS_TIMING=1`：bert 层分解（HuBERT 复用同栈时参考）
- AR 侧 `GSV_AR_TIMING=1`

**计时纪律**：系统负载 >3 的数字只看结构不看绝对值；每次实验先空跑一遍暖懒加载（jieba/cmudict/g2pw 首调 ~5s 会污染 text 列）；安静窗口（load<2）的配对数据才是主张依据。

## 4. 武器库（全部在 main，先读再用，不要重造）

| 配方 | 位置 | 适用 |
|---|---|---|
| AmxPanel 预打包 + `gemm_f16_amx_pp` | `src/kern/gemm_f16_amx.hpp` | 一切 dense GEMM |
| E8 bert 配方（形状分流 T≥48/K≥256、QKV 三联单批、装载期打包） | `src/bert/bert_ops.hpp` + task/E8 diff | transformer dense |
| E5-P2 conv 配方（im2col+panel 直写、M<64/T<64/dil>1/k==1 回退 sgemm） | `src/sovits/` + task/E5-P2 diff | conv 栈 |
| E10 MHA batched sgemm / 批量图 / im2col NEON | `src/sovits/` | attention |
| E11-5 prefill SDPA AMX（QK^T/PV GEMM） | task/E11-5 | attention 大头 |
| `tools/amx_bench.cpp` | 十形状 × {sgemm,fmlal,amx,amxpp} 对照 | 任何形状先 bench 再选路径 |
| AMX 线程纪律 | **AMX 与 cblas 同线程互斥（SIGILL）**；AMX 调用落专用线程池，worker 长驻 AMX_SET 永不碰 cblas | 并行化前必读 |

**历史教训（别再踩）**：
- 手写 FMLAL 比 sgemm 慢 5-10×（bench 实锤）——别写 FMLAL，直接 AMX 或 sgemm
- AMX tile 累加序 vs FMLAL 有 ulp 级差：在 SV 这种连续链会放大到 1e-3——可接受但要证据（见 §6）
- 跨进程 wav corr 对比**永远无效**（v2Pro dec noise 每进程随机，已两次踩坑）——数值差验收只用 mel 包络相对差或固定 noise 重放

## 5. 优化候选（按预判 ROI 排序；每项先画像证实再动手）

1. **HuBERT conv encoder**（前置 CNN，输入帧率高，conv FLOPs 占比未画像——先测）
2. **HuBERT attention**（T=399 的 MHA；E11-5 配方直接抄）
3. **SV conv 栈剩余层**（E12 分流门槛外的中小形状；可能需要 batched/panel 化而不是逐层）
4. **sv ‖ hubert 并行**（两者输入独立！a16cond 与 wav16k 互不依赖；4P+6E 上双线程池，注意 AMX 线程纪律——两个引擎都用 AMX 时要共享 AMX 池或串行化 AMX 段）
5. **cond.compute 细分**（92ms 未拆过；spec/ref_enc 各多少）
6. ssl_proj 权重装载期缓存（§2 尾注，白送几 ms~十几 ms）
7. fbank/重采样 NEON（若画像显示 >20ms 再做）

**理论地板算法**（bench 纪律：常量禁止心算，用 amx_bench/微基准拿数）：
- FLOPs 地板：逐层算 MACs（conv: Cout×Cin×k²×T_out；dense: T×K×N）÷ 实测 AMX 峰值吞吐（amx_bench 里同形状的 amxpp 列）
- 带宽地板：权重/激活扫描字节 ÷ 实测流式带宽（1T=12.4 / 2T=25.7 / 4T=47.5 GB/s @本机）
- 取 max 为该层地板，全链求和 → 总地板；你的成绩要能表达为"×N×地板"并有数据链

## 6. 数值验收红线

| 对象 | 红线 | 证据方法 |
|---|---|---|
| 默认路径（无 --amx-enc） | 位级不变 | ctest 8/8 + c2_pairs_run + 短句贪心 wav md5 `fa78ef013b28f855e5fc37d5f26b9e21` |
| HuBERT hidden / RVQ codes | **逐位一致**（喂 AR 调节链） | 同 wav dump hidden/codes 两路径 md5 |
| svEmb | 允许 ulp→1e-3 级 | 向量 corr ≥0.999 且下游 wav mel 包络相对差 ≤0.005 |
| cond | 同 svEmb 口径 | mel 包络 |
| 端到端听感 | 无退化 | A/B wav 生成好留给决策者/用户 afplay |

**开关形态**：全部优化收在 `--amx-enc` 旗后（默认关）；如需更细档位用 `GSV_AMX_ENC_SUB=nohub|nosv|<你的新档>` 扩展，CLI help 同步。

## 7. 工作节奏与回报

- 开工三件事：① `git checkout -b task/E13`（基于 main 最新）② 空跑一遍 CLI 暖缓存拿干净基线画像 ③ 把 §5 候选逐项画像排序（预期第一小时全花在画像上，别急着写码）
- 每 2-3 小时一个里程碑 commit（可编译、单测绿），防一把梭
- 完成定义：构建零警告、ctest 全绿、位级证据链齐、每项优化附 amx_bench/微基准数据链、总耗时表达为"×N 实测地板"
- 回报格式（给决策者）：`{task_id: E13, status: DONE|BLOCKED, evidence: [命令+结果], files_changed, risks, next}`——BLOCKED 必附最小复现

## 8. 环境

- 硬件：MacBook Air M4（4P+6E，16GB），macOS，Apple Clang
- 权重：`~/gsv-weights`（hubert 531MB / sv 306MB / bert 1.9GB / ar 444MB / sovits 460MB，mmap）
- 测试参考音频：`test_wav/vo_HTLQ001_3_hutao_16.wav`（3s 胡桃，737 fbank 帧 / T=399）
- 复现命令模板：
  ```
  GSV_REF_TIMING=1 ./build/gsv_native --amx --amx-bert --amx-enc --no-cache \
    --prompt-text "就是客人的重要度划分，分为胡桃竹木四级，往往级别越高，往来就越密。" \
    --sample --text "重庆的火锅店终于开张了。" \
    --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out /tmp/x.wav
  ```
- CPUFast python 参考：`/Volumes/2T/GPT-SoVITS-CPUFast`（G2PWModel/字典等）
- corsix/amx 汇编宏参考：`/Volumes/2T/ref/amx/amx`（aarch64.h 可直接抄）
- worktree 惯例：`/Volumes/2T/wt-gsv/<名字>`，别在主仓直接开分支干

## 9. 已知未解问题（可选加分项，不是本卡范围）

- RefCache 键不含内容哈希（用户已裁定路径缓存非解，但你若顺手改成 sha256(wav bytes) 内容键也算造福）
- FE-AUTO-1.5（auto 混排逐片 BERT）、FE-AUTO-2/3（ko/ja G2P）——别碰，是别人的卡
