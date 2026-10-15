#include "gguf.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <limits>
namespace duga {
template<class T>T GGUF::read(){if(at>size || sizeof(T)>size-at)throw std::runtime_error("Truncated GGUF");T v;std::memcpy(&v,map+at,sizeof(T));at+=sizeof(T);return v;}
std::string GGUF::str(){auto n=read<uint64_t>();if(n>16*1024*1024 || at>size || n>size-at)throw std::runtime_error("Invalid GGUF string");std::string s(reinterpret_cast<char*>(map+at),n);at+=n;return s;}
double GGUF::value(uint32_t t,bool store,const std::string& key,unsigned depth){
  if(depth>2)throw std::runtime_error("GGUF metadata nesting limit");
  switch(t){case 0:return read<uint8_t>();case 1:return read<int8_t>();case 2:return read<uint16_t>();case 3:return read<int16_t>();case 4:return read<uint32_t>();case 5:return read<int32_t>();case 6:return read<float>();case 7:return read<uint8_t>();case 8:{auto s=str();if(store)strings[key]=s;return 0;}case 9:{auto e=read<uint32_t>();auto n=read<uint64_t>();if(n>10000000)throw std::runtime_error("GGUF metadata array too large");for(uint64_t i=0;i<n;i++)value(e,false,"",depth+1);return 0;}case 10:return double(read<uint64_t>());case 11:return double(read<int64_t>());case 12:return read<double>();default:throw std::runtime_error("Unsupported GGUF metadata type");}
}
GGUF::GGUF(const std::string& path){fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);if(fd<0)throw std::runtime_error("Cannot open local model");struct stat st{};if(fstat(fd,&st)!=0 || st.st_size<24 || uint64_t(st.st_size)>2ull*1024*1024*1024){close(fd);fd=-1;throw std::runtime_error("Model exceeds the initial Dense port's 2 GiB file limit");}size=st.st_size;map=static_cast<uint8_t*>(mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0));if(map==MAP_FAILED){map=nullptr;close(fd);fd=-1;throw std::runtime_error("Cannot map model file");}try{parse();}catch(...){munmap(map,size);close(fd);map=nullptr;fd=-1;throw;}}
GGUF::~GGUF(){if(map)munmap(map,size);if(fd>=0)close(fd);}
void GGUF::parse(){
  if(read<uint32_t>()!=0x46554747 || read<uint32_t>()!=3)throw std::runtime_error("Expected GGUF v3");auto nt=read<uint64_t>(),nm=read<uint64_t>();if(nt>4096 || nm>65536)throw std::runtime_error("GGUF table too large");
  for(uint64_t i=0;i<nm;i++){auto k=str();auto t=read<uint32_t>();auto n=value(t,true,k);if(t!=8 && t!=9)numbers[k]=n;}
  if(text("general.architecture")!="qwen3")throw std::runtime_error("Native Vulkan currently supports Qwen3 Dense only");
  for(uint64_t i=0;i<nt;i++){
    auto name=str();Tensor t;auto nd=read<uint32_t>();if(!nd || nd>4)throw std::runtime_error("Invalid tensor rank");t.elements=1;
    for(uint32_t j=0;j<nd;j++){auto d=read<uint64_t>();if(!d || d>1000000 || t.elements>UINT64_MAX/d)throw std::runtime_error("Invalid tensor dimensions");t.dims.push_back(d);t.elements*=d;}
    t.type=read<uint32_t>();t.offset=read<uint64_t>();if(t.type==0){if(t.elements>UINT64_MAX/4)throw std::runtime_error("Tensor overflow");t.bytes=t.elements*4;}else if(t.type==8 && t.elements%32==0){t.bytes=t.elements/32*34;}else throw std::runtime_error("Native Vulkan accepts F32 and Q8_0 tensors only");
    if(!tensors.emplace(name,t).second)throw std::runtime_error("Duplicate tensor name");
  }
  auto align=uint64_t(number("general.alignment",32));if(!align || align>4096 || (align&(align-1)))throw std::runtime_error("Invalid GGUF alignment");auto base=(at+align-1)&~(align-1);
  for(auto& [name,t]:tensors){if(base>size || t.offset>size-base || t.bytes>size-base-t.offset)throw std::runtime_error("Tensor outside GGUF file");t.data=map+base+t.offset;}
}
const Tensor& GGUF::tensor(const std::string& name)const{auto i=tensors.find(name);if(i==tensors.end())throw std::runtime_error("Missing tensor: "+name);return i->second;}
double GGUF::number(const std::string& key,double fallback)const{auto i=numbers.find(key);return i==numbers.end()?fallback:i->second;}
std::string GGUF::text(const std::string& key)const{auto i=strings.find(key);return i==strings.end()?"":i->second;}
float halfFloat(uint16_t h){uint32_t s=uint32_t(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,b;if(e==0){if(!m)b=s;else{int k=-14;while(!(m&1024)){m<<=1;k--;}b=s|uint32_t(k+127)<<23|(m&1023)<<13;}}else b=s|((e==31?255:e+112)<<23)|(m<<13);float f;std::memcpy(&f,&b,4);return f;}
}
