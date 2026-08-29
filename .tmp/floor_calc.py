#!/usr/bin/env python3
# E13 地板计算 v2: 程序化枚举全链各层(几何从代码推导), 地板=max(FLOPs板, 带宽板)
# 吞吐常量: .tmp/flops_probe min-of-30 实测; 带宽: .tmp/bw_probe 实测
BW = 29.3        # GB/s 4线程流式实测(bw_probe, 32-128MB 区间稳定值)
GF_BIG = 1700.0  # 大形状 amxpp 实测代表值(1500-2000 区间中位, 见 flops_probe)
GF_T399DENSE = 1050.0  # T=399 dense 实测区间(895-1100)代表值
GF_ATTN = 300.0  # K=64 attention 小核实测区间(240-357)保守值
MS = lambda macs, gf: 2.0 * macs / (gf * 1e9) * 1000.0

# ---------- HuBERT ----------
print("== HuBERT ==")
# CNN: 输入 127894 样本(3s@16k + 9600零补齐后实际47xxx+9600→实测len127894来自L0输出25577*5-2)
conv_len_in = 127894
dims_ks = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
lin = conv_len_in
cnn_macs = cnn_floor = 0.0
for i, (k, s) in enumerate(dims_ks):
    tout = (lin - k) // s + 1
    lin = tout
    K = 512 * k if i > 0 else 1 * k
    cin = 1 if i == 0 else 512
    mac = 512 * cin * k * k * tout
    fl = MS(mac, GF_BIG if tout > 8000 else 1400.0)
    bw_fl = ((512 * cin * k * k + tout * 512) * 2 / (BW * 1e9) * 1000) + tout * cin * k * k * 2 / (BW * 1e9) * 1000
    cnn_macs += mac
    cnn_floor += max(fl, bw_fl)
    print(f"  conv{i} k{k}s{s}: T_out={tout} MACs={mac/1e6:.0f}M flop板={fl:.1f}ms")
T = 399
H = 768
I = 3072
dense = [
    ("qkv三联", 1197 * H * H), ("out_proj", T * H * H),
    ("f1", T * H * I), ("f2", T * I * H), ("proj", T * 512 * H),
]
d_sum = sum(m for _, m in dense)
for n_, m_ in dense:
    print(f"  {n_}(单层量): {m_/1e6:.0f}M MACs -> x12={m_*12/1e6:.0f}M")
dense_total = d_sum * 12 - T * 512 * H * 11  # proj 只 1 次
f_dense = MS(dense_total, GF_T399DENSE)
sdpa_macs = 2 * (12 * 12 * T * T * 64)
f_sdpa = MS(sdpa_macs, GF_ATTN)
print(f"  dense合计(x12层+proj): {dense_total/1e6:.0f}M MACs -> {f_dense:.1f}ms @AMX")
print(f"  SDPA(QKT+PV)x144头次: {sdpa_macs/1e6:.0f}M MACs -> {f_sdpa:.1f}ms @K64小核率")
layernorm_gelu_bytes = (12 * (T * H * 8 * 2) + (25577 + 12788) * 512 * 4)  # 读写字节粗估
hubert_floor = cnn_floor + f_dense + f_sdpa + layernorm_gelu_bytes / (BW * 1e9) * 1000
print(f"  >> HuBERT 地板 ≈ {hubert_floor:.0f}ms (CNN {cnn_floor:.0f} + dense {f_dense:.1f} + sdpa {f_sdpa:.1f} + LN/GELU访存{layernorm_gelu_bytes/(BW*1e9)*1000:.1f})")

# ---------- SV ----------
print("== SV ==")
Fb, Tb = 80, 737
def geo(kh, kw, s, pad, h, w):
    return (h + 2 * pad - kh) // s + 1, (w + 2 * pad - kw) // s + 1

def conv(name, cout, cin, kh, kw, stride, pad, h, w, cnt=1, hit=True):
    oh, ow = geo(kh, kw, stride, pad, h, w)
    S = oh * ow
    mac = cout * cin * kh * kw * S * cnt
    if not hit:
        print(f"  [fallback] {name}: S={S} MACs={mac/1e6:.0f}M")
        return mac, S, h, w
    wb = cout * cin * kh * kw * 2
    ab = cnt * (wb + S * cin * kh * kw * 2)
    fl = MS(mac, GF_BIG)
    bf = ab / (BW * 1e9) * 1000
    print(f"  {name}: S={S}x{cnt} MACs={mac/1e6:.0f}M 板={max(fl,bf):.1f}ms(flop{fl:.1f}/bw{bf:.1f}) 权重{wb/1048576:.1f}MiB")
    return mac, S, h, w

sv_macs = 0.0
sv_floor_list = []
# stem conv1 3x3 p1 s1, fp32 sgemm 已 0.5ms 内
m1, S, hh, ww = conv("stem", 64, 1, 3, 3, 1, 1, Fb, Tb)
h, w = Fb, Tb
sv_macs += m1
# 阶段几何: defs {3,1,False},{4,2,False},{6,2,True},{3,2,True}; widths: L1=48,L2=96,L3=192,L4=384? 由trace反推
# trace: L1 convs是48ch(S=14760 即 40x369); L2 convs 96ch(S=3700=20x185); L3 convs 192ch(S=930=10x93)
widths = {1: 48, 2: 96, 3: 192}
geo_map = {1: (40, 369), 2: (20, 185), 3: (10, 93)}
nblk = {1: 3, 2: 4, 3: 6}
# L1: blk0(stride1, no-sc? 有sc由rank3 'S1blk0.sc' 可知有shortcut), 每 blk: conv1(k1)+scale个convs+conv3(+sc)
cnt_dbg = {}
for L in (1, 2, 3):
    wp = widths[L]
    for b in range(nblk[L]):
        cin_blk = wp if b > 0 else (64 if L == 1 else widths[L - 1])
        scale = 3 if L <= 2 else 6
        co1 = wp * scale
        hh_, ww_ = geo_map[L]
        S_blk = hh_ * ww_
        first = (L == 1 and b == 0)
        mac_c1 = co1 * cin_blk * S_blk
        mac_cv = wp * wp * 9 * S_blk * scale
        mac_c3 = wp * co1 * S_blk
        sv_macs += mac_c1 + mac_cv + mac_c3
        if first:
            mac_sc = wp * cin_blk * (40 * 737 if L == 1 else S_blk)  # L1blk0 入口在 80x737 s2→40x369
            sv_macs += mac_sc
        cnt_dbg[f"L{L}b{b}"] = (mac_c1 + mac_cv + mac_c3) / 1e6
# l3ds 3x3 s2 p1: 2048·1024·9·S(10x93->5x47?) 用 5x46=230: trace显示s2输出S=233?? 实测435? 直接按公式:
oh3, ow3 = geo(3, 3, 2, 1, 10, 93)
mac_l3ds = 2048 * 1024 * 9 * oh3 * ow3
sv_macs += mac_l3ds
print(f"  l3ds: ({oh3}x{ow3})={oh3*ow3} MACs={mac_l3ds/1e6:.0f}M")
# fuse34(w1/w2 是 inter=512, ch=2048 的 1x1, 作用在 layer4 输出 5x47=235? 由dim: h,w 保持10x93)
oh4, ow4 = geo(1, 1, 2, 0, 10, 93)
S4 = oh4 * ow4
mac_f34 = (512 * 2 * 2048 * S4) + (2048 * 512 * S4) * 1  # w1[512,4096], w2[2048,512]
sv_macs += mac_f34
# L4 block(3块, width 384, exp 1536, 无 aff)
for b in range(3):
    cin_b = 384 if b > 0 else 1536
    mac_c1 = 1536 * cin_b * S4
    mac_cv = 384 * 384 * 9 * S4 * 3
    mac_c3 = 384 * 1536 * S4
    sv_macs += mac_c1 + mac_cv + mac_c3
for k_, v_ in sorted(cnt_dbg.items()):
    print(f"    {k_}: {v_:.0f}M")
sv_floor = MS(sv_macs, GF_BIG)
print(f"  SV 总 MACs ≈ {sv_macs/1e6:.0f}M -> amxpp 地板 ≈ {sv_floor:.0f}ms @1700GF")

# ---------- cond ----------
tc_mac = 2 * (256 * 128 * 5 * 369)          # temporal conv x2 层
spec_lin = 369 * (704 * 128 + 128 * 128)     # sp0+sp3
att_lin = 369 * (128 * 128 * 4)              # qkv fc
attn_core = 2 * 369 * 369 * 128              # 单头温度×2 head 视作 d=64x2 合计同 128
fc_pool = 369 * 128 * 1024
svproj = 20480 * 1024
mac_cond = tc_mac + spec_lin + att_lin + attn_core + fc_pool + svproj
print(f"== cond == 全链 MACs≈{mac_cond/1e6:.0f}M -> 板 {MS(mac_cond, GF_BIG)/1000:.2f}ms(+FFT 3ms 保留)")

# ---------- RVQ ----------
rvq = 199 * 1024 * 768
print(f"RVQ: {rvq/1e6:.0f}M MACs -> {MS(rvq, GF['rvq' if False else 1] if False else 809.6):.2f}ms @810GF")

tot_floor = hubert_floor + sv_floor + MS(rvq, 809.6) / 1000 + 3 + 13
print(f"\n== 总地板(逐项求和) ≈ hubert{hubert_floor:.0f} + sv{sv_floor:.0f} + rvq0.4 + cond3+fft3 + 前处理13 ≈ {tot_floor:.0f}ms")
cur = 2103
print(f"当前真实值 ≈{cur}ms = x{cur/tot_floor:.1f} 总地板")
print(f"HuBERT: 925/{hubert_floor:.0f} = x{925/hubert_floor:.1f}")
print(f"SV:     942/{sv_floor:.0f} = x{942/sv_floor:.1f}")
