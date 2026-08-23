# GPT-SoVITS-native 项目指令（多 Agent 协作版 v2）

本文件由每个 pi 会话（决策者或执行者）在本仓库启动时自动加载。
**权威链**: `ARCHITECTURE.md`（设计/决策记录）> 本文件（流程/协作）> 会话口头指示。
冲突时按权威链裁决；改设计必须先改 ARCHITECTURE.md。

## 角色定义

| 角色 | 职责 | 禁止 |
|---|---|---|
| **决策者**（orchestrator） | 任务分解与分配；`STATE.md` **唯一写者**；验收与合并；冲突裁决；对外汇报 | 直接大范围写执行者领地内的代码 |
| **执行者**（executor） | 在**认领的目录边界内**实现 + 自验 + 小步 git 提交 + 按格式回报 | 越界修改他人目录；自行开工 STATE.md 之外的任务；重开已定决策 |

新会话快速上手：① 读 ARCHITECTURE.md §1–§3 → ② 读本文件 → ③ 查 `STATE.md` 找分配给自己的任务 → ④ 无任务则向决策者索要，**不要自行开工**。

## 硬约束（所有角色，违反即返工）

1. **纯 CPU**：禁止引入 Metal / GPU / ANE / CoreML 依赖
2. **运行时零 Python**：Python 仅允许存在于 `tools/` 离线工具
3. **版本锁 v2Pro 家族**：维度从 `.gsv` config 读取，禁止硬编码模型形状
4. **数值纪律（两步走）**：第一步全 fp32 实现并与 torch 对齐；第二步切 fp16（FMLAL 扩展精度视同 fp32、KV fp16 开关）。禁用 bf16。当前阶段以任务卡标注为准
5. **Golden 纪律**：内核/引擎改动必须过 `tests/golden` 对照（门槛见 `tests/golden/gates.json` 与 CALIBRATION.md）
6. **Bench 纪律**：性能主张附实测数据；常量禁止心算
7. 第三方库默认不引入；确需引入先报决策者，更新 ARCHITECTURE.md 后才可用

## 并行协作协议

### 目录所有权（执行者的领地边界）

| 目录 | 归属任务 | 说明 |
|---|---|---|
| `src/kern/` | A 阶段产出后归 AR/SoVITS 共享 | 只读使用；改动须走决策者 |
| `src/ar/` | M1 执行者 | |
| `src/sovits/` | M2 执行者 | |
| `src/textfront/` | M3 执行者 | 含词典/规则数据 |
| `src/encoder/` | M4 执行者 | HuBERT/SV |
| `src/runtime/`、`src/main.cpp`、CMake/构建脚本 | 决策者或集成执行者 | 公共地带 |
| `tools/*.py` | 对应任务执行者可加新文件；改公共 convert.py 须报备 | |
| `ARCHITECTURE.md`、`AGENTS.md`、`STATE.md`、`tests/golden/` | **决策者专属** | 执行者只读 |

### 任务状态机（STATE.md）

`TODO → CLAIMED → IN_PROGRESS → REVIEW → DONE`（旁路: `BLOCKED`）
- 只有决策者写状态；执行者在回报中建议状态变更
- REVIEW 的唯一入口条件 = 完成定义（DoD）全部自验通过

### 完成定义（DoD，每个任务逐条自查后才能报 REVIEW）

1. 构建零错误零警告（`cmake --build build` 全量通过）
2. 新增代码有对应单测且通过（tests/ 下）
3. 适用项 golden 对照达标：AR 任务对 G1/G2；SoVITS 对 G1/G3；文本前端对照 CPUFast 输出 diff 为空
4. git 提交历史干净（小步、信息可读、只含本任务文件）
5. 回报证据清单（见下）

### 回报格式（执行者 → 决策者）

```
{task_id, status: DONE|BLOCKED, evidence: [命令+结果摘要], files_changed: [...],
 risks: [...], next: 建议下一步}
```
BLOCKED 必须附最小复现信息。禁止只说"做完了"。

### Git 纪律

- 每个执行者在独立分支/worktree 工作：`task/<id>`；公共基线在 `main`
- 执行者仅允许**本地 commit** 到自己的分支；**禁止提交测试脚本（tests/）与文档（*.md）**——这两类由决策者在验收合并时统一提交；**禁止 push**（本项目无远程）
- 合并权在决策者；合并前抽查 DoD 第 1–3 条
- 大文件永不入库（.gitignore 已配置，勿改动）

## 任务分解基线（详细任务卡由决策者按此拆解进 STATE.md）

```
Phase A 地基（串行，单人）: CMake 骨架 + .gsv 读取器 + 张量/错误处理基建 +
          NEON 内核层第一版(GEMV/rmsnorm/rope/silu/softmax) + tests 框架
Phase B 三路并行:
  B-AR  (M1): prefill(Accelerate) + decode GEMV + KV cache(fp32) + 贪心采样
              验收: G1/G2 达标(对 pairs/*.pt)
  B-TXT (M3): jieba 移植 + G2PW(复用 kern 的 BERT 计算) + pinyin 规则
              验收: 与 CPUFast 输出 diff 为空(≥200 句)
  B-SOV (M2): enc_p/quantizer/flow/dec(im2col→Accelerate) + WAV 输出
              验收: 用 golden 输入端到端出 wav 过 G3
Phase C 集成（M4）: encoder/HuBERT+SV + 缓存 + CLI 串接全链路
Phase D 性能（M5）: 流水线重叠/线程调优/bench harness（可多执行者分头测）
Phase E 量化（M6）: KV fp16 开关评估 → AR int8 权重 + KV int8
```

依赖规则: Phase A 未 DONE 前，B 路执行者只能做不依赖骨架的部分（如词典数据准备、算子数值验证脚手架）。B-AR 与 B-SOV 共享 kern 层的 API 变更须经决策者协调。

## 当前状态

里程碑定义见 ARCHITECTURE.md §7：

- [x] M0 转换 + golden + 定标 ✅（含增强补录：AR 输入/24 层快照/wav16k；漂移对已剔除）
- [ ] M1 AR 引擎（fp16 存储 + fp32 计算）
- [ ] M2 SoVITS 全链路出声
- [ ] M3 文本前端全原生
- [ ] M4 编码器缓存 + CLI 整合
- [ ] M5 吃满率优化
- [ ] M6 AR int8 阶段二

## 已定决策（勿重开讨论）

1. 语言 C++20；交付 CLI
2. GPU/Metal/ANE 永久排除；运行时零 Python
3. 精度路线两步走：先全 fp32 对齐 torch，再切 fp16（ARCHITECTURE.md §3）；"一致" = G1/G2/G3 达标（bitwise 不可达已论证）
4. KV fp16 开关与 AR int8 同属阶段二；VITS 量化需逐层 A/B 听感
5. 性能北极星 = CPU 吃满率，RTF 仅记录
6. 文本前端全原生重写，golden 以 CPUFast 输出为准
7. 多 Agent 协作采用本文件的目录所有权 + STATE.md 状态机模式；仓库用 git，main 为受保护基线
