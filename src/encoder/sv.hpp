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

#include <cstddef>
#include <vector>

namespace gsv::rt {
class GsvFile;
}

namespace gsv::encoder {

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
    std::vector<float> w1, w2;  // [inter,2C] / [C,inter] 1x1 卷积权重摊平
    std::vector<float> b1, b2;  // 两个 1x1 卷积的 bias(local_att Conv2d bias=True!)
    Bn bn1, bn2;
    int inter = 0, ch = 0;
    void apply(const float* x, const float* ds, float* out, int h, int w);
    // scratch / 对拍
    std::vector<float> cat_, att_, att_t_;
    std::vector<float> last_out;  // 最近一次 apply 的输出 xo
  };
  struct Block {
    bool aff = false;
    int width = 0, scale = 0, exp_planes = 0;
    int stride = 1;
    std::vector<float> conv1_w, conv3_w;          // 1x1 权重摊平 [Co,Ci]
    std::vector<std::vector<float>> convs_w;      // 3x3 权重摊平 [width,width*9]
    std::vector<Bn> bns;
    Bn bn1, bn3;
    std::vector<Aff> fuses;                       // scale-1 个(AFF 块才有)
    bool has_shortcut = false;
    std::vector<float> sc_w;
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
  std::vector<float> conv1_w_;  // [64, 9]
  Bn bn1_;
  std::vector<float> l3ds_w_;   // [2048, 1024*9]
  Aff fuse34_;
  std::vector<Block> stages_[5];  // 1..4
  int c_after_[5]{};              // 各 stage 输出通道

  // scratch(引擎级共享)
  std::vector<float> cols_, tmp_, tmp2_, nxt_;
  std::vector<float> cur_, ds_, o_conv1_, o_fuse_, emb_;
  std::vector<std::vector<float>> o_layer_;

  static void conv2d(const float* in, int c_in, int h, int w, const float* wt,
                     int c_out, int kh, int kw, int stride, int pad,
                     std::vector<float>& cols, std::vector<float>& out);
};

}  // namespace gsv::encoder
