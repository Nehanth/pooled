#include "dense.hpp"
#include <algorithm>
#include <cmath>
namespace duga {
Buffer* Dense::uniform(uint32_t a,uint32_t b){uint32_t u[4]={a,b,0,0};return gpu.buffer(sizeof(u),u);}
Operation Dense::op(const std::string& name,const std::vector<Binding>& b,uint32_t groups){return gpu.operation(name,config,frame,b,groups);}
Dense::Weight Dense::weight(const std::string& name,uint32_t rows,uint32_t cols){
  auto it=weights.find(name);if(it!=weights.end()){if(it->second.rows!=rows || it->second.cols!=cols)throw std::runtime_error("Reused weight shape mismatch");return it->second;}
  const auto& t=model.tensor(name);if(t.elements!=uint64_t(rows)*cols || t.dims[0]!=cols && cols!=1)throw std::runtime_error("Tensor shape mismatch: "+name);
  Weight w{};w.rows=rows;w.cols=cols;w.quant=t.type==8;
  if(!w.quant)w.data=gpu.buffer(t.bytes,t.data);
  else {
    w.data=gpu.buffer(t.elements);w.scales=gpu.buffer(t.elements/32*2);
    auto* qs=static_cast<uint8_t*>(w.data->data);auto* sc=static_cast<uint8_t*>(w.scales->data);
    for(uint64_t i=0;i<t.elements/32;i++){std::memcpy(qs+i*32,t.data+i*34+2,32);std::memcpy(sc+i*2,t.data+i*34,2);}
  }
  weights[name]=w;return w;
}
void Dense::mv(std::vector<Operation>& ops,const std::string& name,Buffer* in,Buffer* out,uint32_t rows,uint32_t cols){
  auto w=weight(name,rows,cols);uint64_t rowBytes=uint64_t(cols)*(w.quant?1:4),maxRows=gpu.properties().limits.maxStorageBufferRange/rowBytes;
  const uint32_t chunk=maxRows>=rows?rows:uint32_t(maxRows/256*256);if(!chunk)throw std::runtime_error("Matrix row exceeds Vulkan binding limit");
  for(uint32_t at=0;at<rows;at+=chunk){uint32_t n=std::min(chunk,rows-at);auto shape=uniform(n,cols);
    std::vector<Binding> b={{w.data,at*rowBytes,n*rowBytes}};if(w.quant)b.push_back({w.scales,uint64_t(at)*cols/32*2,uint64_t(n)*cols/32*2});
    b.push_back({in});b.push_back({out,uint64_t(at)*4,uint64_t(n)*4});b.push_back({shape});ops.push_back(op(w.quant?"matvec_q8_coop":"matvec_coop",b,(n+3)/4));
  }
}
Dense::Dense(const std::string& path,uint32_t context):model(path){
  auto integer=[&](const std::string& key,uint32_t fallback=0){double v=model.number(key,fallback);if(!std::isfinite(v)||v<1||v>1000000||std::floor(v)!=v)throw std::runtime_error("Invalid model metadata: "+key);return uint32_t(v);};
  dim_=integer("qwen3.embedding_length");inter_=integer("qwen3.feed_forward_length");heads_=integer("qwen3.attention.head_count");kvHeads_=integer("qwen3.attention.head_count_kv");headDim_=integer("qwen3.attention.key_length",dim_/heads_);uint32_t count=integer("qwen3.block_count");
  if(count>48||dim_>4096||inter_>16384||heads_>64||!kvHeads_||heads_%kvHeads_||headDim_>256||headDim_%2||dim_%32||inter_%32)throw std::runtime_error("Model dimensions exceed initial Vulkan port limits");
  if(integer("qwen3.attention.value_length",headDim_)!=headDim_)throw std::runtime_error("Different K/V head sizes are unsupported");
  const auto& embed=model.tensor("token_embd.weight");if(embed.dims.size()!=2||embed.dims[0]!=dim_||embed.dims[1]>300000)throw std::runtime_error("Invalid embedding matrix");vocab_=embed.dims[1];kvDim_=kvHeads_*headDim_;maxSeq_=context;
  if(context<16||context>2048)throw std::runtime_error("Native context must be 16..2048 tokens");
  uint64_t estimate=0;for(uint32_t i=0;i<count;i++)estimate+=uint64_t(context)*kvDim_*8;
  estimate+=uint64_t(vocab_)*dim_+(uint64_t(count)*(3ull*dim_*inter_+uint64_t(heads_*headDim_+2*kvDim_)*dim_+uint64_t(dim_)*heads_*headDim_));
  if(estimate>1400ull*1024*1024)throw std::runtime_error("Model and KV cache exceed the port's memory budget");
  uint32_t cfg[12]={dim_,kvDim_,heads_,kvHeads_,headDim_,inter_,vocab_,maxSeq_,0,0,heads_*headDim_,0};float eps=model.number("qwen3.attention.layer_norm_rms_epsilon",1e-6),theta=model.number("qwen3.rope.freq_base",1e6);if(!std::isfinite(eps)||eps<=0||!std::isfinite(theta)||theta<=0)throw std::runtime_error("Invalid normalization/RoPE metadata");std::memcpy(cfg+8,&eps,4);std::memcpy(cfg+9,&theta,4);config=gpu.buffer(sizeof(cfg),cfg);frame=uniform(0);
  auto floats=[&](uint64_t n){return gpu.buffer(n*4);};uint32_t qdim=heads_*headDim_;
  x=floats(dim_);xn=floats(dim_);q=floats(qdim);k=floats(kvDim_);v=floats(kvDim_);g=floats(inter_);up=floats(inter_);attn=floats(qdim);tmp=floats(dim_);scores=floats(uint64_t(heads_)*context);logits=floats(vocab_);
  auto ndim=uniform(dim_),nh=uniform(heads_),nkv=uniform(kvHeads_);
  auto norm=[&](const std::string& name){auto w=weight(name,dim_);if(w.quant)throw std::runtime_error("Quantized norm unsupported");return op("rmsnorm",{{x},{w.data},{xn},{ndim}},1);};
  for(uint32_t i=0;i<count;i++){
    Layer l;l.kc=floats(uint64_t(context)*kvDim_);l.vc=floats(uint64_t(context)*kvDim_);auto p="blk."+std::to_string(i)+".";
    l.before.push_back(norm(p+"attn_norm.weight"));mv(l.before,p+"attn_q.weight",xn,q,qdim,dim_);mv(l.before,p+"attn_k.weight",xn,k,kvDim_,dim_);mv(l.before,p+"attn_v.weight",xn,v,kvDim_,dim_);
    for(auto pair:{std::make_pair("attn_q_norm.weight",q),std::make_pair("attn_k_norm.weight",k)}){auto w=weight(p+pair.first,headDim_);if(w.quant)throw std::runtime_error("Quantized head norm unsupported");l.before.push_back(op("head_norm",{{pair.second},{w.data},{pair.second==q?nh:nkv}},((pair.second==q?heads_:kvHeads_)+31)/32));}
    l.before.push_back(op("rope",{{q},{nh}},(qdim/2+63)/64));l.before.push_back(op("rope",{{k},{nkv}},(kvDim_/2+63)/64));
    l.scores=op("attn_scores",{{q},{l.kc},{scores}},1);l.softmax=op("attn_softmax",{{scores}},heads_);l.attention=op("attn_out",{{scores},{l.vc},{attn}},(qdim+63)/64);
    mv(l.after,p+"attn_output.weight",attn,tmp,dim_,qdim);l.after.push_back(op("add_res",{{x},{tmp}},(dim_+63)/64));l.after.push_back(norm(p+"ffn_norm.weight"));
    mv(l.after,p+"ffn_gate.weight",xn,g,inter_,dim_);mv(l.after,p+"ffn_up.weight",xn,up,inter_,dim_);l.after.push_back(op("silu_mul",{{g},{up}},(inter_+63)/64));mv(l.after,p+"ffn_down.weight",g,tmp,dim_,inter_);l.after.push_back(op("add_res",{{x},{tmp}},(dim_+63)/64));layers.push_back(std::move(l));
  }
  head.push_back(norm("output_norm.weight"));mv(head,model.has("output.weight")?"output.weight":"token_embd.weight",xn,logits,vocab_,dim_);
}
const float* Dense::forward(uint32_t token){
  if(cancelled)throw std::runtime_error("Native inference cancelled");if(token>=vocab_||pos_>=maxSeq_)throw std::runtime_error("Token or context outside model bounds");
  const auto& e=model.tensor("token_embd.weight");auto dst=static_cast<float*>(x->data);
  if(e.type==0)std::memcpy(dst,e.data+uint64_t(token)*dim_*4,dim_*4);else{auto row=e.data+uint64_t(token)*(dim_/32)*34;for(uint32_t b=0;b<dim_/32;b++){uint16_t h;std::memcpy(&h,row+b*34,2);float scale=halfFloat(h);for(uint32_t j=0;j<32;j++)dst[b*32+j]=int8_t(row[b*34+2+j])*scale;}}
  uint32_t f[4]={pos_,pos_+1,0,0};std::memcpy(frame->data,f,sizeof(f));gpu.begin();
  for(auto& l:layers){for(auto& op:l.before)gpu.dispatch(op);gpu.copy(k,l.kc,kvDim_*4,0,uint64_t(pos_)*kvDim_*4);gpu.copy(v,l.vc,kvDim_*4,0,uint64_t(pos_)*kvDim_*4);gpu.dispatch(l.scores,(heads_*(pos_+1)+63)/64);gpu.dispatch(l.softmax);gpu.dispatch(l.attention);for(auto& op:l.after)gpu.dispatch(op);}
  for(auto& op:head)gpu.dispatch(op);gpu.finish();pos_++;if(cancelled)throw std::runtime_error("Native inference cancelled");return static_cast<const float*>(logits->data);
}
}
