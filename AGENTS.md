# GPT-SoVITS-native 项目级指令（AGENTS.md）

本文件由 pi 在本仓库的每次会话自动加载。完整架构设计与决策记录见 `ARCHITECTURE.md`——改设计先改文档，再改代码。

## 项目定位

GPT-SoVITS v2Pro 家族（参考实现 s2Gv2ProPlus.pth）推理引擎，针对 Apple M4 CPU 极致优化。
单二进制 C++20 + Accelerate。**第一版 = fp16 权重存储 + 全程 fp32 计算与累加**
（FMLAL 扩展精度乘积，无 fp16 中间舍入；KV cache fp32 起步），
精度一致性按 ARCHITECTURE.md §3 的 G1/G2/G3 三道门验收。

## 硬约束（违反即返工）

1. **纯 CPU**：禁止引入 Metal / GPU / ANE / CoreML 依赖
2. **运行时零 Python**：Python 仅允许存在于 `tools/` 下的离线工具（convert.py、golden 导出）
3. **版本锁 v2Pro 家族**：所有维度从 `.gsv` 头部 config 读取，禁止硬编码模型形状
4. **数值纪律**：计算与累加一律 fp32（FMLAL 的扩展精度乘积视同 fp32）；权重可存 fp16；KV cache 第一版 fp32；禁用 bf16
5. **Golden 纪律**：任何内核改动必须跑 `tools/golden/` 对照通过后才能合入
6. **Bench 纪律**：性能主张必须附 `tools/bench/` 实测数据，禁止无证据优化；
   常量禁止心算，用工具算或写成表达式
7. 第三方库默认不引入；确需引入先在会话中说明理由并更新 ARCHITECTURE.md

## 计算分工（勿偏离）

| 计算 | 引擎 |
|---|---|
| AR decode GEMV、KV attention、逐层小算子 | 手写 NEON intrinsics（`-mcpu=apple-m4`，FMLAL 宽化累加） |
| 大 GEMM（prefill、VITS 卷积 im2col 后、BERT/HuBERT） | Accelerate（vDSP/BNNS/cblas_hgemm，自动上 SME） |
| SME FMOPA | 仅实验分支 `experimental/sme/`，运行时探测失败须回退 |

线程：decode 阶段 AR ≤2 线程（带宽墙）；VITS/prefill 可扩至全部 P 核；
实时链路 QoS user-initiated 绑 P 核，后台编码 utility 绑 E 核。

## 目录约定

```
pretrained_models/   原始官方权重（只读，不修改不删除）
tools/               convert.py、golden 导出/对照脚本、bench（Python 允许区）
weights/             .gsv 转换产物（生成物；由 tools/convert.py 从 pretrained_models 重建）
src/                 C++20 源码（kern/ ar/ sovits/ encoder/ textfront/ runtime/）
tests/               数值对照与端到端测试
experimental/        未合入主线的实验分支（如 SME）
```

## 当前状态

里程碑 M0–M6 定义见 ARCHITECTURE.md §7。完成一项更新本节：

已完成: 权重转换（2024 记录：五模型 → weights/*.gsv，fp32+fp16 双段，
回读逐张量校验全过；SoVITS 164 组 + HuBERT 1 组 weight_norm 已融合；
BERT 的 position_ids(int64) 已跳过）。M0 其余项（golden 导出、精度门定标）未开始。

- [x] M0 权重转换 + golden 基线 + 精度门定标 ✅
      （五模型 .gsv；golden: refs×56 + pairs×65；门槛见 tests/golden/CALIBRATION.md）
      （增强补录 ✅: AR 输入 phones/prompt_tokens/bert_feat_1024、24层快照 layers_first+layers_laststep、
       参考 wav16k；确定性回归检查通过 59/59，漂移对 s1+hutao_02_s0 已剔除并记录）
- [ ] M1 AR 引擎 fp16 版（prefill + NEON GEMV decode + KV cache）
- [ ] M2 SoVITS 全链路出声
- [ ] M3 文本前端全原生（jieba 移植 + G2PW 复用自家 BERT kernel + pinyin 规则）
- [ ] M4 编码器缓存 + CLI 整合
- [ ] M5 吃满率优化（§5 度量达标）
- [ ] M6 AR int8 量化阶段二

## 已定决策（勿重开讨论）

1. 语言 C++20；交付形态 CLI
2. GPU/Metal/ANE 永久排除；运行时零 Python
3. 第一版 = fp16 权重存储 + 全 fp32 计算（v1.0 修订）；"与 torch 一致" = G1/G2/G3 达标（bitwise 不可达已论证）
4. 阶段二 AR int8 已批准；VITS 量化需逐层 A/B 听感
5. 性能北极星 = CPU 吃满率（P核≥90%、E核≥50%、带宽占比、GEMM 效率），RTF 仅记录
6. 文本前端全原生重写，golden 以 CPUFast 输出为准
