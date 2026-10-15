#include "vulkan.hpp"
#include "shaders.hpp"
#include "oracle-shaders.hpp"
#include <algorithm>

namespace duga {
Vulkan::Vulkan() {
  try {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Duga";app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;
    vkCheck(vkCreateInstance(&ci,nullptr,&instance_),"vkCreateInstance");
    uint32_t count=0;vkCheck(vkEnumeratePhysicalDevices(instance_,&count,nullptr),"physical devices");
    if(!count)throw std::runtime_error("No Vulkan device");
    std::vector<VkPhysicalDevice> devices(count);vkCheck(vkEnumeratePhysicalDevices(instance_,&count,devices.data()),"physical devices");
    // Explicit selector for oracle runs. Default to the first compute-capable physical device.
    for(auto dev:devices) {
      VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(dev,&p);
#ifndef __ANDROID__
      const char* chosen=std::getenv("DUGA_VK_DEVICE");if(chosen && std::string(p.deviceName).find(chosen)==std::string::npos)continue;
#endif
      uint32_t n=0;vkGetPhysicalDeviceQueueFamilyProperties(dev,&n,nullptr);std::vector<VkQueueFamilyProperties> fs(n);vkGetPhysicalDeviceQueueFamilyProperties(dev,&n,fs.data());
      for(uint32_t i=0;i<n;i++)if(fs[i].queueCount && (fs[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){physical_=dev;family_=i;props_=p;break;}
      if(physical_)break;
    }
    if(!physical_)throw std::runtime_error("No selected Vulkan compute device");
    if(props_.limits.maxComputeWorkGroupInvocations<256 || props_.limits.maxComputeWorkGroupSize[0]<256)throw std::runtime_error("Vulkan compute workgroup limit is too small");
    vkGetPhysicalDeviceMemoryProperties(physical_,&memory_);
    float priority=1;VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qc.queueFamilyIndex=family_;qc.queueCount=1;qc.pQueuePriorities=&priority;
    VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qc;
    vkCheck(vkCreateDevice(physical_,&dc,nullptr,&device_),"vkCreateDevice");vkGetDeviceQueue(device_,family_,0,&queue_);
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pc.queueFamilyIndex=family_;pc.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCheck(vkCreateCommandPool(device_,&pc,nullptr,&commands_),"command pool");
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=commands_;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;
    vkCheck(vkAllocateCommandBuffers(device_,&ca,&command_),"command buffer");
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,8192},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,16384}};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=8192;dp.poolSizeCount=2;dp.pPoolSizes=sizes;
    vkCheck(vkCreateDescriptorPool(device_,&dp,nullptr,&descriptors_),"descriptor pool");
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};vkCheck(vkCreateFence(device_,&fc,nullptr,&fence_),"fence");
  } catch(...) {destroy();throw;}
}
Vulkan::~Vulkan(){destroy();}
void Vulkan::destroy(){
  if(device_){vkDeviceWaitIdle(device_);
    for(auto& [name,p]:pipelines_){if(p.pipeline)vkDestroyPipeline(device_,p.pipeline,nullptr);if(p.layout)vkDestroyPipelineLayout(device_,p.layout,nullptr);for(auto l:p.layouts)if(l)vkDestroyDescriptorSetLayout(device_,l,nullptr);}
    pipelines_.clear();
    if(descriptors_)vkDestroyDescriptorPool(device_,descriptors_,nullptr);
    for(auto& b:buffers_){if(b->data)vkUnmapMemory(device_,b->memory);if(b->handle)vkDestroyBuffer(device_,b->handle,nullptr);if(b->memory)vkFreeMemory(device_,b->memory,nullptr);}buffers_.clear();
    if(fence_)vkDestroyFence(device_,fence_,nullptr);if(commands_)vkDestroyCommandPool(device_,commands_,nullptr);vkDestroyDevice(device_,nullptr);device_=VK_NULL_HANDLE;
  }
  if(instance_){vkDestroyInstance(instance_,nullptr);instance_=VK_NULL_HANDLE;}
}
Buffer* Vulkan::buffer(size_t bytes,const void* src){
  if(bytes>1536ull*1024*1024)throw std::runtime_error("Vulkan buffer exceeds port limit");
  auto b=std::make_unique<Buffer>();b->size=std::max<size_t>(16,(bytes+3)&~size_t(3));
  VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bc.size=b->size;bc.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  vkCheck(vkCreateBuffer(device_,&bc,nullptr,&b->handle),"buffer");
  buffers_.push_back(std::move(b));auto* out=buffers_.back().get();
  VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device_,out->handle,&req);
  uint32_t type=UINT32_MAX;for(uint32_t i=0;i<memory_.memoryTypeCount;i++)if((req.memoryTypeBits&(1u<<i)) && (memory_.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
  if(type==UINT32_MAX)throw std::runtime_error("This Vulkan device has no coherent host-visible memory");
  VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ma.allocationSize=req.size;ma.memoryTypeIndex=type;
  vkCheck(vkAllocateMemory(device_,&ma,nullptr,&out->memory),"GPU memory (insufficient device capacity)");
  vkCheck(vkBindBufferMemory(device_,out->handle,out->memory,0),"buffer memory");
  vkCheck(vkMapMemory(device_,out->memory,0,out->size,0,&out->data),"map memory");
  std::memset(out->data,0,out->size);if(src && bytes)std::memcpy(out->data,src,bytes);return out;
}
Vulkan::Pipeline& Vulkan::pipeline(const std::string& name){
  auto it=pipelines_.find(name);if(it!=pipelines_.end())return it->second;
  const Shader* shader=nullptr;
  if(name.rfind("oracle:",0)==0){for(const auto& s:ORACLE_SHADERS)if(name.substr(7)==s.name){shader=&s;break;}}
  else {for(const auto& s:SHADERS)if(name==s.name){shader=&s;break;}}
  if(!shader)throw std::runtime_error("Unknown compiled kernel");
  auto& p=pipelines_[name];p.bindings=shader->bindings;p.uniforms=shader->uniforms;
  for(uint32_t group=0;group<2;group++){
    const uint32_t count=group==0?2:p.bindings;std::vector<VkDescriptorSetLayoutBinding> bs(count);
    for(uint32_t b=0;b<count;b++){bs[b].binding=b;bs[b].descriptorCount=1;bs[b].descriptorType=(group==0 || (p.uniforms&(1u<<b)))?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bs[b].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=count;dl.pBindings=bs.data();vkCheck(vkCreateDescriptorSetLayout(device_,&dl,nullptr,&p.layouts[group]),"descriptor layout");
  }
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=2;pl.pSetLayouts=p.layouts;vkCheck(vkCreatePipelineLayout(device_,&pl,nullptr,&p.layout),"pipeline layout");
  VkShaderModule module{};VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=shader->bytes;sm.pCode=shader->code;vkCheck(vkCreateShaderModule(device_,&sm,nullptr,&module),"SPIR-V module");
  VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.layout=p.layout;cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=module;cp.stage.pName=shader->entryPoint?shader->entryPoint:shader->name;
  auto result=vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&cp,nullptr,&p.pipeline);vkDestroyShaderModule(device_,module,nullptr);vkCheck(result,"compute pipeline");return p;
}
Operation Vulkan::operation(const std::string& name,Buffer* config,Buffer* frame,const std::vector<Binding>& bindings,uint32_t groups){
  auto& p=pipeline(name);if(bindings.size()!=p.bindings)throw std::runtime_error("Kernel binding count mismatch");
  Operation out{};out.pipeline=p.pipeline;out.layout=p.layout;out.groups=groups;
  VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=descriptors_;da.descriptorSetCount=2;da.pSetLayouts=p.layouts;vkCheck(vkAllocateDescriptorSets(device_,&da,out.sets),"descriptors");
  std::vector<VkDescriptorBufferInfo> infos(2+bindings.size());std::vector<VkWriteDescriptorSet> writes(infos.size());
  for(size_t i=0;i<infos.size();i++){
    Binding b=i==0?Binding{config}:i==1?Binding{frame}:bindings[i-2];
    if(!b.buffer || b.offset>=b.buffer->size)throw std::runtime_error("Invalid buffer binding");
    const bool uniform=i<2 || (p.uniforms&(1u<<(i-2)));
    const size_t range=b.size==VK_WHOLE_SIZE?b.buffer->size-b.offset:b.size;
    if(!range || range>b.buffer->size-b.offset || range>(uniform?props_.limits.maxUniformBufferRange:props_.limits.maxStorageBufferRange))throw std::runtime_error("Vulkan descriptor range exceeds device limit");
    infos[i]={b.buffer->handle,b.offset,range};auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=out.sets[i<2?0:1];w.dstBinding=i<2?i:i-2;w.descriptorCount=1;w.descriptorType=uniform?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&infos[i];
  }
  vkUpdateDescriptorSets(device_,writes.size(),writes.data(),0,nullptr);return out;
}
void Vulkan::begin(){if(recording_)throw std::runtime_error("Vulkan command already active");vkCheck(vkResetCommandBuffer(command_,0),"reset command");VkCommandBufferBeginInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;vkCheck(vkBeginCommandBuffer(command_,&cb),"begin command");recording_=true;}
void Vulkan::barrier(){VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};m.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;m.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;vkCmdPipelineBarrier(command_,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&m,0,nullptr,0,nullptr);}
void Vulkan::dispatch(const Operation& op,uint32_t x,uint32_t y,uint32_t z){if(!recording_)throw std::runtime_error("No active command");x=x?x:op.groups;if(!x || x>props_.limits.maxComputeWorkGroupCount[0] || y>props_.limits.maxComputeWorkGroupCount[1] || z>props_.limits.maxComputeWorkGroupCount[2])throw std::runtime_error("Invalid dispatch size");barrier();vkCmdBindPipeline(command_,VK_PIPELINE_BIND_POINT_COMPUTE,op.pipeline);vkCmdBindDescriptorSets(command_,VK_PIPELINE_BIND_POINT_COMPUTE,op.layout,0,2,op.sets,0,nullptr);vkCmdDispatch(command_,x,y,z);}
void Vulkan::copy(Buffer* src,Buffer* dst,size_t bytes,size_t from,size_t to){if(!recording_ || from>src->size || to>dst->size || bytes>src->size-from || bytes>dst->size-to)throw std::runtime_error("Invalid copy range");barrier();VkBufferCopy c{from,to,bytes};vkCmdCopyBuffer(command_,src->handle,dst->handle,1,&c);}
void Vulkan::finish(){if(!recording_)throw std::runtime_error("No command to finish");barrier();vkCheck(vkEndCommandBuffer(command_),"end command");recording_=false;vkCheck(vkResetFences(device_,1,&fence_),"reset fence");VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO};s.commandBufferCount=1;s.pCommandBuffers=&command_;vkCheck(vkQueueSubmit(queue_,1,&s,fence_),"submit");vkCheck(vkWaitForFences(device_,1,&fence_,VK_TRUE,120000000000ull),"GPU timeout");}
}
