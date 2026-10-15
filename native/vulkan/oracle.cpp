#include "vulkan.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
using namespace duga;
struct Reader {std::ifstream f;explicit Reader(const char* p):f(p,std::ios::binary){if(!f)throw std::runtime_error("Cannot open oracle trace");}uint32_t u(){uint32_t v;f.read(reinterpret_cast<char*>(&v),4);if(!f)throw std::runtime_error("Truncated oracle trace");return v;}std::vector<uint8_t> bytes(){const auto n=u();if(n>128*1024*1024)throw std::runtime_error("Trace buffer too large");std::vector<uint8_t> b(n);f.read(reinterpret_cast<char*>(b.data()),n);if(!f)throw std::runtime_error("Truncated buffer");return b;}std::string str(){auto b=bytes();return {b.begin(),b.end()};}};
int verifyOracle(const char* path){try{
  Vulkan vk;auto p=vk.properties();std::cout<<"Vulkan oracle GPU: "<<p.deviceName<<" driver="<<p.driverVersion<<" api="<<p.apiVersion<<"\n";
  Reader r(path);if(r.u()!=0x314b5644)throw std::runtime_error("Bad trace magic");auto count=r.u();if(count>512)throw std::runtime_error("Too many oracle cases");uint32_t failed=0;
  for(uint32_t c=0;c<count;c++){
    auto name=r.str();uint32_t x=r.u(),y=r.u(),z=r.u();auto cfg=r.bytes(),frame=r.bytes();auto n=r.u();if(n>8)throw std::runtime_error("Too many bindings");
    std::vector<Buffer*> bufs;std::vector<Binding> binds;for(uint32_t i=0;i<n;i++){auto data=r.bytes();auto b=vk.buffer(data.size(),data.data());bufs.push_back(b);binds.push_back({b});}
    auto out=r.u();auto expected=r.bytes();if(out>=bufs.size() || expected.size()>bufs[out]->size)throw std::runtime_error("Invalid oracle output");
    auto op=vk.operation(name,vk.buffer(cfg.size(),cfg.data()),vk.buffer(frame.size(),frame.data()),binds);
    vk.begin();vk.dispatch(op,x,y,z);vk.finish();auto actual=static_cast<const uint8_t*>(bufs[out]->data);
    size_t diffs=0;double maxAbs=0,maxRel=0;for(size_t i=0;i<expected.size()/4;i++){uint32_t a,b;float af,bf;std::memcpy(&a,actual+i*4,4);std::memcpy(&b,expected.data()+i*4,4);if(a!=b)diffs++;std::memcpy(&af,&a,4);std::memcpy(&bf,&b,4);if(!std::isfinite(af)||!std::isfinite(bf)){if(a!=b)maxAbs=std::numeric_limits<double>::infinity();}else{auto d=std::abs(double(af)-bf);maxAbs=std::max(maxAbs,d);maxRel=std::max(maxRel,d/std::max(1e-30,std::abs(double(bf))));}}
    std::cout<<(diffs?"FAIL ":"PASS ")<<name<<" bytes="<<expected.size()<<" bitMismatch="<<diffs<<" maxAbs="<<maxAbs<<" maxRel="<<maxRel<<"\n";if(diffs)failed++;
  }
  std::cout<<count-failed<<"/"<<count<<" oracle cases bit-exact\n";return failed?2:0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

#ifndef DUGA_NO_ORACLE_MAIN
int main(int argc,char** argv){if(argc<2){try{Vulkan vk;std::cout<<vk.properties().deviceName<<"\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}return verifyOracle(argv[1]);}
#endif
