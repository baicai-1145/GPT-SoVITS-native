# E13 终局验收（FINAL-ACCEPT）— F1 差分门禁 + F2 安静窗终标

> 执行: PROBE worker（只跑不改码）　日期: 2026-08-27 12:10–12:35 CST
> 合流态: `/Volumes/2T/wt-gsv/E13` @ `e39251e`（HUB/SV/MIX 三卡已收工；ctest 8/8 ✓，本机重建零警告）
> 基线: `/Volumes/2T/wt-gsv/E13-F2BASE`（临时 worktree @ `04d4b26` "E13 probe 分段计时探针"，独立构建，产物 12:12）
> 测试音频: `test_wav/*.wav` N=56（主仓共享清单）；模板文本与 HUB/MIX 树证据链一致

---

## F1 合流态 N=56 codes/wav 差分门禁

### 方法
- 单变量对照：OFF 臂 `--amx --amx-bert --no-cache` vs ON 臂 `--amx --amx-bert --amx-enc --no-cache`；
  共同固定 `--sample --sample-seed 7`、同一 prompt/text（沿用 t14_mel.sh / e13_accept.sh 口径）。
- codes 层：E13-MIX 门2 同款 `GSV_RVQ_DUMP` 前 `Tq*8B` 截取 md5。
- 激活性检查（T14 失效探针坑）：ON 臂 `sdpa_total=54.0/52.9/53.6/53.2ms` —— AMX SDPA **确认激活**
  （基线臂同探针 = 444ms NEON，差异即后端切换的物理证据，非 324/445ms 的失效形态）。

### a) codes 层差分段清单：**8 段**（原始文件 `.tmp/f1work/codes_diff.txt`）

| # | 段 | Tq | codes(OFF) | codes(ON=合流态全旗) |
|---|---|---|---|---|
| 1 | vo_BZLQ001_6_hutao_03 | 207 | 00d15bda… | 0907c297… |
| 2 | vo_DPEQ002_6_hutao_34 | 114 | 20476ac2… | d958446c… |
| 3 | vo_EQHDJ401_15_hutao_09 | 130 | 093121c0… | d0fd88fe… |
| 4 | vo_EQHDJ402_17_hutao_10 | 261 | 680add9d… | 4a652b78… |
| 5 | vo_EQHDJ403_16_hutao_13 | 182 | fb5b1c29… | 6916d409… |
| 6 | vo_EQHDJ403_22_hutao_01 | 256 | 9fe0f1d6… | 8d8003c9… |
| 7 | vo_EQZYJ001_3_hutao_09 | 239 | d02ef2a3… | fde850d3… |
| 8 | vo_HTLQ001_3_hutao_20 | 152 | 6fbf8716… | fc61c25c… |

交叉锚点（跨树逐位一致性）：
- **合流态 ON 臂 56/56 与 HUB 树 `t14_gate_allon.md` 全表 codes md5 完全一致**——三卡 merge 未移动任何编码位型。
- OFF 臂值均在 HUB 树既有基线表中出现（如 EQHDJ401_15 `093121c0`=T14 记录的无旗基线；HTLQ001_3_20 `6fbf8716`=t12_codes_baseline 表原值），谱系闭合。

### 谱系对账（对任务书预期 "~9 段 = E12 固有 4 + T12 层差"）

| 口径 | 数量 | 说明 |
|---|---|---|
| T14: vs dense-only 部署基线 | 9 段 | T14 §3 主表口径 |
| T14 尾注: vs 无旗位级基线 | **48/56 相同 ⇒ 8 段差** | 本实验恰为此口径（OFF 臂 = 无旗） |
| 本次实测 vs 无旗 | **8 段** | 上表；全部 ⊆ T14 九段清单，**零清单外新段** |

两个"T14 有而本次无"的成员（EQHDJ401_10_hutao_16 / EQHDJ403_2_hutao_08，均属 E12 固有 4 段集）在合流态下
`codes(OFF)=codes(ON)`（`c11646b2…` / `19ef92da…`，双臂双跑稳定，且等于 T14 allon 表原值）——即全开 AMX 链
在这两段的 codes 回归到无旗同值，按定义不进本次差分；它们相对 dense-only 的翻转仍是 T14 已记录事实，
不构成本次观察缺口。预期 9 vs 实测 8 之差完全由基准选择（dense-only vs 无旗）解释，无异常。

### b) wav 层差分

- md5 层：56/56 全部不同。为单变量设计的物理预期结果——OFF 撤掉整个 `--amx-enc` 使 cond/ge 走 ulp~1e-3 口径差，
  经 v2Pro 解码噪声与 AR logits 进采样链后样本级必异（HANDOFF §4"跨进程 corr 无效"同机理）；md5 相等不是有效门。
- 内容级判定（mel 门, MelSpectrogram(32000,2048/2048/640,128), rel=L1/mean ≤0.005, t14_mel.sh 公式）：
  **48 PASS / 8 FAIL**（`.tmp/f1_mel_result.md`）。FAIL 集 **≡ codes 差集（1:1 重合，无一例外）**：

| 段 | mel_rel | corr | 归因 |
|---|---|---|---|
| BZLQ001_6_hutao_03 | 1.549 | −0.002 | codes 翻转级联（E13 旗后路径组）|
| DPEQ002_6_hutao_34 | 1.616 | 0.026 | 同上 |
| **EQHDJ401_15_hutao_09** | **2.307** | −0.003 | **E12 固有翻转级联（T14 P0 已知, 波及 main dense）** |
| EQHDJ402_17_hutao_10 | 1.003 | 0.000 | codes 翻转级联 |
| EQHDJ403_16_hutao_13 | 1.693 | 0.017 | 同上 |
| EQHDJ403_22_hutao_01 | 1.147 | 0.272 | 同上 |
| **EQZYJ001_3_hutao_09** | **0.766** | 0.210 | **E12 固有翻转级联（T14 P0 已知）** |
| HTLQ001_3_hutao_20 | 2.152 | 0.020 | codes 翻转级联 |

> 判读：任务书"wav 层仅 4 段 E12 固有差"的下限口径在 mel 定级层表现为 8 段（codes 差的全镜像）：
> 其中 2 段是 T14 已判定的 E12 固有内容级失败（P0 在案）；另 6 段为"E13 各卡旗后整体 OFF/ON"这一更宽单变量下的
> 采样级联首测（此前 T14 只做过 A=T12off-dense vs C=全开的窄对照, 9/9 字节相同）。此 6 段的 codes 差本身
> ⊆ T14 九段谱系, 属已知微扰的下游表现, 非 merge 引入的新异常; 但其 mel 内容级FAIL 是本口径首次定级,
> **提请决策者知情**: 波及面与 T14 P0 同类（随 `--amx-enc` 旗存在）, 默认路径不受影响。

---

## F2 安静窗终标（交错配对 ×3 中位）

- 安静窗守望：`f2_waiter.sh` 于 **12:27:39 触发（load1m=2.39 < 2.5）**，三轮即时负载 2.36/2.18/2.31（1m 均值 3.3–3.5 为早前尖峰衰减尾，immediate 值为准）。
- 侧序交替：f2a base→merged / f2b merged→base / f2c base→merged（脚本随机起点+显式参数混合）。
- 原始日志：`.tmp/f2_{base,merged}_f2{a,b,c}.log`（每轮含完整 uptime + ref-timing + sdpa 探针行）。

| 段(ms) | 基线 04d4b26（中位, n=3） | 合流态 e39251e（中位, n=3） | Δ | 加速 |
|---|---|---|---|---|
| sv.forward3 | 917 (904–918) | **426** (418–447) | −491 | 2.15× |
| cond.compute | 91 | **11** | −80 | 8.3× |
| hubert.run | 915 (912–923) | **326** (325–327) | −589 | 2.81× |
| └ sdpa 子段 | 444.0 | **53.2** | −391 | 8.3× |
| 前处理(resample+fbank) | 13 | 13 | — | — |
| **[ref-timing] 合计** | **1937** (1920–1944) | **776** (770–796) | −1161 | **2.50×** |
| extract_latent（漏计段） | 80 (80–81) | **3** | −77 | 26.7× |
| └ rvq 子段 | 78–79 | **2** | −76.5 | ~39× |
| **真实参考侧总耗时** | **2017** | **779** | −1238 | **2.59×** |

### 总地板倍数（地板 218ms 口径, 见 PROBE 报告 §5）

| 口径 | 基线 04d4b26 | 合流态 e39251e |
|---|---|---|
| 官方合计 | 1937/218 = **×8.9** | 776/218 = **×3.56** |
| 含 extract_latent | 2017/218 = **×9.25** | 779/218 = **×3.57** |

E13 卡族全程战绩：交接时真实口径 ≈2103ms（×9.7 地板）→ 合流后 **779ms = ×3.57 地板**。

---

## 环境记录

- 合流态构建：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGSV_AMX_GEMM=ON && cmake --build build -j4`
  （零警告；⚠ 该宏必需，OFF 时 E12 起 encoder 代码无法编译）；ctest 8/8 passed @e39251e。
- 基线构建：临时 worktree `git worktree add /Volumes/2T/wt-gsv/E13-F2BASE 04d4b26`（同配置，构建零警告）；
  两树均符号链接 weights/test_wav 至主仓。验收完成后 worktree 可按惯例移除。
- 探针资产复用：codes dump=pipeline 内建 `GSV_RVQ_DUMP`（MIX 门2 法）; mel=t14_mel.sh 公式;
  SDPA 激活监视=`GSV_HUBERT_SDPA_TIMING`（04d4b26 自带）。
- 原始输出索引：F1 `.tmp/f1_gate_result.txt`(汇总) + `.tmp/f1work/{codes_diff,wav_diff}.txt` + `d_{on,off}_*.bin`(56×2)
  + `log_{on,off}_*.txt`; mel 定级 `.tmp/f1_mel_result.md`; F2 六日志见上; 附加稳定性三方检验记录见本报告 F1 节内嵌。
- 未动任何源码/worktree 提交; F2BASE worktree 由 `git -C GPT-SoVITS-native worktree add` 创建, 决策者可自行 prune。
