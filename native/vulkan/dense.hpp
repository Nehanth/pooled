#pragma once
#include "vulkan.hpp"
#include "gguf.hpp"
#include <atomic>
namespace duga {
class Dense {
 public:
  explicit Dense(const std::string& path,uint32_t context=1024);
  const float* forward(uint32_t token);
  void reset(){pos_=0;cancelled=false;}
  void cancel(){cancelled=true;}
  uint32_t position()const{return pos_;}
  uint32_t vocab()const{return vocab_;}
  uint32_t context()const{return maxSeq_;}
  const char* deviceName()const{return gpu.properties().deviceName;}
 private:
  struct Weight {Buffer* data{};Buffer* scales{};bool quant{};uint32_t rows{},cols{};};
  struct Layer {Buffer* kc{};Buffer* vc{};std::vector<Operation> before,after;Operation scores,softmax,attention;};
  GGUF model;Vulkan gpu;
  uint32_t dim_{},kvDim_{},heads_{},kvHeads_{},headDim_{},inter_{},vocab_{},maxSeq_{},pos_=0;
  Buffer *config{},*frame{},*x{},*xn{},*q{},*k{},*v{},*g{},*up{},*attn{},*tmp{},*scores{},*logits{};
  std::vector<Layer> layers;
  std::vector<Operation> head;
  std::unordered_map<std::string,Weight> weights;
  std::atomic<bool> cancelled{false};
  Weight weight(const std::string& name,uint32_t rows,uint32_t cols=1);
  Buffer* uniform(uint32_t a,uint32_t b=0);
  Operation op(const std::string&,const std::vector<Binding>&,uint32_t groups);
  void mv(std::vector<Operation>&,const std::string&,Buffer*,Buffer*,uint32_t rows,uint32_t cols);
};
}
