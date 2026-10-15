#include "dense.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
int main(int argc,char** argv){try{if(argc<4)throw std::runtime_error("Usage: duga-vk-model model.gguf token-ids.txt logits.bin");duga::Dense engine(argv[1],1024);std::ifstream input(argv[2]);if(!input)throw std::runtime_error("Cannot read token input");std::ofstream out(argv[3],std::ios::binary);if(!out)throw std::runtime_error("Cannot write output");uint32_t id;while(input>>id){auto logits=engine.forward(id);out.write(reinterpret_cast<const char*>(logits),engine.vocab()*4);}std::cout<<"Vulkan model: "<<engine.deviceName()<<" positions="<<engine.position()<<" vocab="<<engine.vocab()<<"\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
