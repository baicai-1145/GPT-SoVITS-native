// sv.hpp — 说话人编码器 eres2netv2w24s4ep4 推理引擎(C1)
// 对照 CPUFast GPT_SoVITS/sv.py + eres2net/{ERes2NetV2,fusion}.py。
//
// 拓扑(架构常量与权重形状互相校验; .gsv 无 model_config → 全部自描述):
//   conv1[64,1,3,3]/s1/p1 → bn1 → ReLU(Hardtanh(0,20))
//   layer1: 3×BasicBlock(plain) planes=64  width=24  scale=4 exp=4 stride=1
//   layer2: 4×BasicBlock(plain) planes=128 width=48              stride=2
//   layer3: 6×BasicBlock(AFF)   planes=256 width=96              stride=2
//   layer4: 3×BasicBlock(AFF)   planes=512 width=192             stride=2
//   forward3 分支: layer3_ds(1024→2048,k3,s2,p1) → fuse34=AFF(2048,r=4) →
//   flatten(C×F) → mean(T) → sv_emb[C*F]=[20480](seg_1/TSTP pool 不参与 forward3)
//
// 布局约定: 激活一律通道主 [C][H][W](H=频率维, W=时间维); 卷积经 im2col +
// sgemm('N','T') 直接产出通道主输出(零转置)。BN 按 eval 语义逐算子实现
// (running stats + affine, eps=1e-5), 不折叠进卷积 —— 与 torch 逐 op 数值路径一致。
// 激活 ReLU 实为 Hardtanh(0,20): 上界 20 必须钳位。
#pragma once

#include "kern/gemv_fmlal.hpp"
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"
#endif

#include <cstddef>
#include <vector>

namespace gsv::rt {
class GsvFile;
}

namespace gsv::encoder {

// E12: 进程级 SV AMX 使能(--amx-enc 与 HuBERT 共用开关; 装载前设置)。
bool& amx_sv_enabled();

class SvEngine {
 public:
  explicit SvEngine(const rt::GsvFile& f);

  // fbank[frames,80] 行主 → sv_emb[C*F](=20480); 返回维度数。
  size_t forward3(const float* fbank, size_t frames);

  // 中间量只读视图(forward3 结束后、下次调用前有效)
  const std::vector<float>& conv1_out() const { return o_conv1_; }
  const std::vector<float>& layer_out(int l) const { return o_layer_[l]; }  // l=1..4
  const std::vector<float>& fuse34_out() const { return o_fuse_; }
  const std::vector<float>& emb_out() const { return emb_; }

  // 内部结构(单测/harness 只读; 嵌套类型需公开以便自由函数加载器命名)
  struct Bn {
    std::vector<float> g, b, mean, var;
    void apply(float* x, int c, size_t s) const;  // [c, s] 每通道一行
  };
  struct Aff {  // fusion.AFF: local_att(cat[x,ds]) → 门控融合
    const uint16_t* w1 = nullptr;  // [inter,2C] 1x1 卷积权重(fp16 直读)
    const uint16_t* w2 = nullptr;  // [C,inter] 1x1 卷积权重(fp16 直读)
    std::vector<uint16_t> w2_own;  // 无 f16 段时的一次性量化副本(fp16 常驻, 拥有权)
    std::vector<float> b1, b2;  // 两个 1x1 卷积的 bias(local_att Conv2d bias=True!)
#if defined(GSV_AMX_GEMM)
    kern::AmxPanel w1_panel, w2_panel;  // E12: 装载期预打包(1x1=dense GEMM)
    bool w1_panel_ready = false, w2_panel_ready = false;
#endif
    Bn bn1, bn2;
    int inter = 0, ch = 0;
    void apply(const float* x, const float* ds, float* out, int h, int w,
               std::vector<uint16_t>& xh, std::vector<uint16_t>& xh2
#if defined(GSV_AMX_GEMM)
               , std::vector<uint8_t>& amx_scratch
#endif
               );
    // scratch / 对拍
    std::vector<float> cat_, att_, att_t_;
    std::vector<float> last_out;  // 最近一次 apply 的输出 xo
  };
  struct Block {
    bool aff = false;
    int width = 0, scale = 0, exp_planes = 0;
    int stride = 1;
    const uint16_t* conv1_w = nullptr;  // 1x1 权重摊平 [Co,Ci] (fp16 直读)
    const uint16_t* conv3_w = nullptr;  // 1x1 权重摊平 [exp,Ci] (fp16 直读)
    std::vector<const uint16_t*> convs_w;  // 3x3 权重摊平 [width,width*9] (fp16 直读)
#if defined(GSV_AMX_GEMM)
    // E12: 预打包面板 — conv1/conv3/shortcut 为 1x1(=dense); convs_w(3x3)按
    // 形状分流(S>=kAmxEncMinRows 且 K=in_c*9>=kAmxEncMinK 才打)。
    kern::AmxPanel conv1_panel, conv3_panel, sc_panel;
    std::vector<kern::AmxPanel> convs_panels;
    std::vector<bool> convs_panels_ready;   // 与 convs_w 一一对应
    bool conv1_panel_ready = false, conv3_panel_ready = false, sc_panel_ready = false;
#endif
    std::vector<Bn> bns;
    Bn bn1, bn3;
    std::vector<Aff> fuses;                       // scale-1 个(AFF 块才有)
    bool has_shortcut = false;
    const uint16_t* sc_w = nullptr;  // 1x1 权重摊平 (fp16 直读)
    Bn sc_bn;
    void apply(const float* in, int c_in, int h_in, int w_in, SvEngine& eng);
    std::vector<float> last_out;  // 最近一次 apply 的输出(对拍用)
  };

  // 对拍视图
  const std::vector<float>& block_out_view(int l, int i) const {
    return stages_[static_cast<size_t>(l)][static_cast<size_t>(i)].last_out;
  }
  const Block& block_ref(int l, int i) const {
    return stages_[static_cast<size_t>(l)][static_cast<size_t>(i)];
  }

 private:
  void load_block(int l, int i, bool expect_aff);

  const rt::GsvFile* f_;
  std::vector<float> conv1_w_;  // [64, 9] stem conv (fp32 常驻, 仅 576 参)
  Bn bn1_;
  const uint16_t* l3ds_w_ = nullptr;   // [2048, 1024*9] fp16 直读
#if defined(GSV_AMX_GEMM)
  kern::AmxPanel l3ds_panel_;          // E12: 装载期预打包(形状分流后)
  bool l3ds_panel_ready_ = false;
#endif
  Aff fuse34_;
  std::vector<Block> stages_[5];  // 1..4
  int c_after_[5]{};              // 各 stage 输出通道

  // scratch(引擎级共享)
  std::vector<float> cols_, tmp_, tmp2_, nxt_;
  std::vector<float> cur_, ds_, o_conv1_, o_fuse_, emb_;
  std::vector<std::vector<float>> o_layer_;
  std::vector<uint16_t> xh_, xh2_;    // fp16 激活暂存(AFF 转置量化复用)
  std::vector<uint16_t> cols16_;      // fp16 im2col 暂存(conv2d_f16)
  std::vector<uint16_t> cvt16_;       // T4d: 3x3 im2col 整张量批量转换暂存
#if defined(GSV_AMX_GEMM)
  std::vector<uint8_t> sv_act_scratch_;   // E12: AMX 激活 panel 缓冲(容量复用)
  // E14-SV/C3: 出核融合面板 —
  //   sv_pan_b_: convs[i] 出核直写的拼接槽位(conv3 输入, K=co1); conv3 直用
  //   零打包。conv1 出核经 epi_core 同遍直写其槽位 0。
  std::vector<uint8_t> sv_pan_b_;
  std::vector<float> ctile_;              // C3: 融合路径逐级 GEMM 数学结果
#endif

  // 静态 conv2d(fp32 权重, im2col → sgemm('N','T')); 仅 stem conv1(fp32-only 段)使用
  static void conv2d(const float* in, int c_in, int h, int w, const float* wt,
                     int c_out, int kh, int kw, int stride, int pad,
                     std::vector<float>& cols, std::vector<float>& out);
  // fp16 直读版: im2col 量化 fp16 + gemm_f16x_fmlal(FMLAL 融合 fp32 累加)
  void conv2d_f16(const float* in, int c_in, int h, int w, const uint16_t* w16,
                  int c_out, int kh, int kw, int stride, int pad,
                  std::vector<float>& cols_unused, std::vector<float>& out);
#if defined(GSV_AMX_GEMM)
  // E12: AMX 卷积(激活 panel 直写; 与 conv2d_f16 输出布局位级可对照)
  // E14-SV/C0: site 标签仅用于探针站点归因(默认 nullptr)。
  void conv2d_amx(const kern::AmxPanel& w_panel, const float* in,
                  int c_in, int h, int w, int kh, int kw, int stride,
                  int pad, int c_out, std::vector<float>& out,
                  const char* site = nullptr);
  // E14-SV/C3: 已就绪双面板直接派发(无打包/trace/dump; 供出核融合链用)
  void conv2d_amx_core(const kern::AmxPanel& w_panel, kern::AmxPanel& pb,
                       int c_out, std::vector<float>& out);
#endif
};

}  // namespace gsv::encoder
