#include "vulkan.hpp"
#include "shaders.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <set>
#include <algorithm>
using namespace duga;
namespace {
struct Reader {
  std::ifstream f;
  explicit Reader(const char* path):f(path,std::ios::binary){if(!f)throw std::runtime_error("Cannot open oracle trace");}
  uint32_t u(){uint32_t v;f.read(reinterpret_cast<char*>(&v),4);if(!f)throw std::runtime_error("Truncated oracle trace");return v;}
  std::vector<uint8_t> bytes(){
    const auto n=u();if(n>128*1024*1024)throw std::runtime_error("Trace buffer too large");
    std::vector<uint8_t> b(n);f.read(reinterpret_cast<char*>(b.data()),n);if(!f)throw std::runtime_error("Truncated buffer");return b;
  }
  std::string str(){auto b=bytes();return {b.begin(),b.end()};}
};
}
// 0: native against captured WebGPU bytes; 1: exported Tint against captured
// bytes; 2: native against independently compiled original shaders on this GPU.
// Modes 0/1 calibrate the portable reference in CI. Android uses mode 2 so that
// another GPU's exp/sin/reduction results are never mistaken for this GPU's oracle.
int verifyOracle(const char* path,int mode){try{
  if(mode<0||mode>2)throw std::runtime_error("Unknown oracle mode");
  Vulkan vk;const auto& p=vk.properties();
  std::cout<<"Vulkan oracle GPU: "<<p.deviceName<<" driver="<<p.driverVersion<<" api="<<p.apiVersion<<" mode="<<mode<<"\n";
  Reader r(path);if(r.u()!=0x314b5644)throw std::runtime_error("Bad trace magic");const auto count=r.u();
  if(count!=sizeof(SHADERS)/sizeof(SHADERS[0]))throw std::runtime_error("Oracle must contain every compiled kernel exactly once");
  uint32_t failed=0;std::set<std::string> seen;
  for(uint32_t c=0;c<count;c++){
    const auto name=r.str();
    if(!seen.insert(name).second)throw std::runtime_error("Duplicate oracle kernel");
    bool known=false;for(const auto& shader:SHADERS)if(name==shader.name)known=true;
    if(!known)throw std::runtime_error("Unknown oracle kernel");
    const uint32_t x=r.u(),y=r.u(),z=r.u();auto cfg=r.bytes(),frame=r.bytes();const auto n=r.u();
    if(n>8)throw std::runtime_error("Too many bindings");
    std::vector<std::vector<uint8_t>> inputs;
    std::vector<Buffer*> bufs;std::vector<Binding> binds;
    for(uint32_t i=0;i<n;i++){
      inputs.push_back(r.bytes());const auto& data=inputs.back();auto b=vk.buffer(data.size(),data.data());bufs.push_back(b);binds.push_back({b});
    }
    const auto out=r.u();auto expected=r.bytes();
    if(out>=bufs.size()||expected.empty()||expected.size()%4||expected.size()!=inputs[out].size())throw std::runtime_error("Invalid oracle output");
    auto config=vk.buffer(cfg.size(),cfg.data()),position=vk.buffer(frame.size(),frame.data());
    if(mode==2){
      const auto reference=vk.operation("oracle:"+name,config,position,binds);
      vk.begin();vk.dispatch(reference,x,y,z);vk.finish();
      std::memcpy(expected.data(),bufs[out]->data,expected.size());
      if(expected==inputs[out])throw std::runtime_error("Original-shader reference did not change its output");
      for(size_t i=0;i<expected.size();i+=4){float value;std::memcpy(&value,expected.data()+i,4);if(!std::isfinite(value))throw std::runtime_error("Non-finite original-shader reference");}
      // Reference and candidate start with identical bytes, including in-place buffers.
      for(size_t i=0;i<bufs.size();i++)std::memcpy(bufs[i]->data,inputs[i].data(),inputs[i].size());
    }
    const auto op=vk.operation((mode==1?"oracle:":"")+name,config,position,binds);
    vk.begin();vk.dispatch(op,x,y,z);vk.finish();const auto actual=static_cast<const uint8_t*>(bufs[out]->data);
    size_t diffs=0;double maxAbs=0,maxRel=0;
    for(size_t i=0;i<expected.size();i+=4){
      uint32_t a,b;float af,bf;std::memcpy(&a,actual+i,4);std::memcpy(&b,expected.data()+i,4);if(a!=b)diffs++;
      std::memcpy(&af,&a,4);std::memcpy(&bf,&b,4);
      if(!std::isfinite(af)||!std::isfinite(bf)){if(a!=b)maxAbs=std::numeric_limits<double>::infinity();}
      else {const auto d=std::abs(double(af)-bf);maxAbs=std::max(maxAbs,d);maxRel=std::max(maxRel,d/std::max(1e-30,std::abs(double(bf))));}
    }
    std::cout<<(diffs?"FAIL ":"PASS ")<<name<<" bytes="<<expected.size()<<" bitMismatch="<<diffs<<" maxAbs="<<maxAbs<<" maxRel="<<maxRel<<"\n";
    if(diffs)failed++;
  }
  std::cout<<count-failed<<"/"<<count<<" oracle cases bit-exact\n";return failed?2:0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

#ifndef DUGA_NO_ORACLE_MAIN
int main(int argc,char** argv){
  if(argc<2){try{Vulkan vk;std::cout<<vk.properties().deviceName<<"\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
  int mode=0;
  if(argc>2){const std::string flag=argv[2];if(flag=="--reference")mode=1;else if(flag=="--on-device")mode=2;else {std::cerr<<"Unknown oracle option\n";return 1;}}
  return verifyOracle(argv[1],mode);
}
#endif
