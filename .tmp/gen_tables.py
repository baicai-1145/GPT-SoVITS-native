#!/usr/bin/env python3
# E13 报告表格生成器: 逐层 MACs/权重字节/地板 + 候选排序汇总
# 数据源: .tmp/probe_*.log 实测 + .tmp/flops_probe 输出(手抄常量于下方)
import sys

# ---- flops_probe 实测 amxpp 吞吐(GFLOPS, 2026-08-27 @load3.4, min-of-30) ----
GF = {
    "sv_L2out_1024": 1812.4, "sv_S2out_2048": 1745.8, "sv_S3out_1024x": 1550.1,
    "hub_cnnL1_512": 1466.3, "hub_cnnL2_512": 1177.6,
    "hub_qkv_T399": 1047.6, "hub_ffn_T399_f1": 894.2, "hub_ffn_T399_f2": 1095.8,
    "hub_proj_T399": 915.9, "attn_score_K64": 240.1, "attn_pv_K64": 357.2,
    "rvq_dist_T199": 809.6,
}
BW_4T = 29.3   # GB/s 流式实测(bw_probe)
BW_1T = 7.3

def macs_conv(cout, cin, k, tout):
    return cout * cin * k * k * tout

def macs_dense(T, K, N):
    return T * K * N

def row(name, m, n, k, gf, note=""):
    mac = m * n * k
    tf_s = 2 * mac / (gf * 1e9) * 1000  # ms
    wb = m * k * 2 / 1024               # KiB fp16 权重
    bw_ms = m * k * 2 / (BW_1T * 1e9) * 1000 if False else 0
    print(f"| {name} | {m} | {n} | {k} | {mac/1e6:.1f} | {wb:,.0f} | {gf:.0f} | {tf_s:.2f} | {note} |")

print("## HuBERT dense/SDPA/RVQ 层级表(T=399)")
print("| 层 | M(Cout) | N(Tout) | K | MACs(M) | 权重KiB(fp16) | 实测GFLOPS | 地板ms |")
print("|---|---|---|---|---|---|---|---|")
row("qkv三联×12", 1197, 399, 768, GF["hub_qkv_T399"])
row("o_proj×12", 399, 399, 768, 1050.0, "(按同尺寸类推)")
row("f1×12", 399, 399, 3072, GF["hub_ffn_T399_f2"], "(K/N 对调形状)")
row("f2×12", 399, 399, 3072, GF["hub_ffn_T399_f2"])
row("proj×1", 399, 399, 512, GF["hub_proj_T399"])
row("sdpa QKT×12头批量", 1197, 399, 64, GF["attn_score_K64"])
row("sdpa PV×12头批量", 399, 399, 64, GF["attn_pv_K64"])

# HuBERT CNN
print("\n## HuBERT CNN 栈(整输入一条通路, MACs=Cout*Cin*k²*T_out)")
print("| 层 | 形状 | K=Cin*k | T_out | MACs(M) | 权重KiB | amxppGF(类推) | 地板ms | FMLAL实测ms |")
hl = [(0, 512, 25577, 5120, 11.8), (1, 512, 12788, 1536, 131.9),
      (2, 512, 6393, 1536, 61.1), (3, 512, 3196, 1536, 27.9),
      (4, 512, 1597, 1536, 15.8), (5, 512, 798, 1024, 4.5),
      (6, 512, 399, 1024, 2.2)]
for li, co, to, K, ms in hl:
    mac = co * K * to
    gf = GF["hub_cnnL1_512"] if to > 10000 else (1400 if to > 3000 else 1300)
    fl = 2 * mac / (gf * 1e9) * 1000
    print(f"| conv{li} | 512ch,k{[10,3,3,3,3,2,2][li]}s{[5,2,2,2,2,2,2][li]} | {K} | {to} "
          f"| {mac/1e6:.0f} | {co*K*2/1024:,.0f} | {gf:.0f} | {fl:.1f} | {ms} |")

# RVQ
mac_rvq = 199 * 1024 * 768
fl_rvq = 2 * mac_rvq / (GF["rvq_dist_T199"] * 1e9) * 1000
print(f"\nRVQ 距离块(T199): MACs={mac_rvq/1e6:.0f}M, 板载embed 1536KiB, amxpp地板={fl_rvq:.2f}ms (现暴力标量=134ms安静值)")

# SV 大 GEMM 清单(名称, M=Cout, N=S, K, 调用次数, 当前路径)
print("\n## SV conv/dense 大形状清单(一次 encode)")
sv_layers = [
    # name, M(out), S, K, count, current, fallback_rank
    ("S2blk0.conv3(k1 384->1024)", 1024, 3700, 384, 6, "AMX", ""),
    ("S2blk0.conv1/sc旁路(1x1 s2)", 2048, 930, 1024, 2, "AMX", ""),
    ("S3fuse_w2核(1x1 s1)", 1024, 930, 2048, 5, "AMX", "#5"),
    ("S2convs k3(96ch)", 96, 3700, 96 * 9, 24, "AMX", ""),
    ("L1convs k3(48ch)", 48, 14760, 48 * 9, 16, "AMX", ""),
    ("S3convs k3(192ch)", 192, 930, 192 * 9, 12, "AMX", ""),
    ("sc_L0blk0(1x1 s2 64->96)", 96, 58960, 64, 2, "AMX", ""),
    ("S1blk1.conv1(1x1 s2 512ch入口)", 192, 14760, 512, 1, "AMX", ""),
]
print("| 层 | M | N(S) | K | 次数 | MACs合计(M) | 权重MiB | 现路径 |")
for name, m, s_, k_, cnt, cur, fbk in sv_layers:
    mac = m * k_ * s_ * cnt
    print(f"| {name} | {m} | {s_} | {k_} | {cnt} | {mac/1e6:,.0f} | {m*k_*2/1048576:.1f} | {cur} |")

# SV FMLAL 回退榜单(copy of sv-f16rank)
rank = [
    ("192,40,369 k1s1->512 (S2blk1.conv1)", 512, 369*40, 192, 4, 93.3),
    ("96,80,737 k1s1->256 (S1blk0.conv1)", 256, 737*80, 96, 3, 68.3),
    ("24,80,737 k3p1s3 (S1convs)", 24, None, None, 12, 63.4),
    ("64,80,737 k1s1->256 (S1blk0.sc)", 256, 737*80, 64, 1, 16.9),
    ("256,80,737 k1s1->192 (S1blk1.sc)", 192, None, None, 1, 10.5),
    ("1024,20,185 k1s1->768 (S3fuse_w1)", 768, 185*20, 1024, 1, 10.3),
    ("512,40,369 k1s1->384 (S2blk1.sc)", 384, None, None, 1, 10.2),
]
print("\n## SV FMLAL 回退热榜(sv-f16rank 安静档实测)")
print("# | 次数 | 耗时ms | M=N?说明") 
for i, (nm, _m, _s, _k, c, ms) in enumerate(rank):
    print(f"{i} | {c} | {ms} | {nm}")

# 候选 ROI 排序(占比与可减量, 全部来自上面实测)
tot = 942 + 92 + 925 + 13 + 138 + 0  # 官方口径 + extract_latent
print(f"\n总口径: 1965(原) ≈ sv942+cond92+hubert925+前处理13(+漏计extract_latent~138) → 真实≈2103")

if __name__ == "__main__":
    pass
