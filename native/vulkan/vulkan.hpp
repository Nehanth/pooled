#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstring>

namespace duga {
struct Buffer { VkBuffer handle{}; VkDeviceMemory memory{}; void* data{}; size_t size{}; };
struct Binding { Buffer* buffer; VkDeviceSize offset = 0; VkDeviceSize size = VK_WHOLE_SIZE; };
struct Operation { VkPipeline pipeline{}; VkPipelineLayout layout{}; VkDescriptorSet sets[2]{}; uint32_t groups = 1; };
class Vulkan {
 public:
  Vulkan();
  ~Vulkan();
  Vulkan(const Vulkan&) = delete;
  Vulkan& operator=(const Vulkan&) = delete;
  Buffer* buffer(size_t bytes,const void* src=nullptr);
  Operation operation(const std::string& name,Buffer* config,Buffer* frame,const std::vector<Binding>& bindings,uint32_t groups=1);
  void begin();
  void dispatch(const Operation&,uint32_t x=0,uint32_t y=1,uint32_t z=1);
  void copy(Buffer* src,Buffer* dst,size_t bytes,size_t srcOffset=0,size_t dstOffset=0);
  void finish();
  const VkPhysicalDeviceProperties& properties() const {return props_;}
 private:
  struct Pipeline {VkPipeline pipeline{};VkPipelineLayout layout{};VkDescriptorSetLayout layouts[2]{};uint32_t bindings{},uniforms{};};
  void destroy();
  void barrier();
  Pipeline& pipeline(const std::string&);
  VkInstance instance_{};
  VkPhysicalDevice physical_{};
  VkDevice device_{};
  VkQueue queue_{};
  uint32_t family_{};
  VkPhysicalDeviceProperties props_{};
  VkPhysicalDeviceMemoryProperties memory_{};
  VkCommandPool commands_{};
  VkCommandBuffer command_{};
  VkDescriptorPool descriptors_{};
  VkFence fence_{};
  bool recording_ = false;
  std::vector<std::unique_ptr<Buffer>> buffers_;
  std::unordered_map<std::string,Pipeline> pipelines_;
};
inline void vkCheck(VkResult result,const char* what) {if(result!=VK_SUCCESS)throw std::runtime_error(std::string(what)+" failed ("+std::to_string(result)+")");}
}
