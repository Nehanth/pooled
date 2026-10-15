#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
namespace duga {
struct Tensor {uint32_t type{};std::vector<uint64_t> dims;uint64_t offset{},elements{},bytes{};const uint8_t* data{};};
class GGUF {
 public:
  explicit GGUF(const std::string& path);
  ~GGUF();
  GGUF(const GGUF&)=delete;
  const Tensor& tensor(const std::string& name) const;
  bool has(const std::string& name)const{return tensors.count(name);}
  double number(const std::string& key,double fallback)const;
  std::string text(const std::string& key)const;
 private:
  void parse();
  int fd=-1;uint8_t* map=nullptr;size_t size=0,at=0;
  std::unordered_map<std::string,Tensor> tensors;
  std::unordered_map<std::string,double> numbers;
  std::unordered_map<std::string,std::string> strings;
  template<class T>T read();
  std::string str();
  double value(uint32_t type,bool store,const std::string& key="",unsigned depth=0);
};
float halfFloat(uint16_t value);
}
