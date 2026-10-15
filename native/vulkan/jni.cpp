#include "dense.hpp"
#include <jni.h>
#include <memory>
#include <algorithm>
#include <vector>
#include <string>
extern int verifyOracle(const char* path,int mode);
static duga::Dense* engine(jlong h){if(!h)throw std::runtime_error("Native model is not loaded");return reinterpret_cast<duga::Dense*>(h);}
static std::string text(JNIEnv* env,jstring s){if(!s)throw std::runtime_error("Missing model path");const char* p=env->GetStringUTFChars(s,nullptr);std::string out(p);env->ReleaseStringUTFChars(s,p);return out;}
static void error(JNIEnv* e,const std::exception& x){e->ThrowNew(e->FindClass("java/lang/IllegalStateException"),x.what());}
extern "C" JNIEXPORT jlong JNICALL Java_app_duga_android_NativeVulkan_open(JNIEnv* e,jclass,jstring p){try{return reinterpret_cast<jlong>(new duga::Dense(text(e,p),1024));}catch(const std::exception& x){error(e,x);return 0;}}
extern "C" JNIEXPORT jint JNICALL Java_app_duga_android_NativeVulkan_evaluate(JNIEnv* e,jclass,jlong h,jintArray ids,jboolean reset){try{auto* m=engine(h);if(reset)m->reset();auto count=e->GetArrayLength(ids);if(count<1||count>1024)throw std::runtime_error("Token batch outside native bounds");std::vector<jint> input(count);e->GetIntArrayRegion(ids,0,count,input.data());const float* logits=nullptr;for(auto id:input){if(id<0)throw std::runtime_error("Invalid token");logits=m->forward(id);}return std::max_element(logits,logits+m->vocab())-logits;}catch(const std::exception& x){error(e,x);return -1;}}
extern "C" JNIEXPORT void JNICALL Java_app_duga_android_NativeVulkan_cancel(JNIEnv* e,jclass,jlong h){try{if(h)engine(h)->cancel();}catch(const std::exception& x){error(e,x);}}
extern "C" JNIEXPORT void JNICALL Java_app_duga_android_NativeVulkan_close(JNIEnv*,jclass,jlong h){delete reinterpret_cast<duga::Dense*>(h);}
extern "C" JNIEXPORT jstring JNICALL Java_app_duga_android_NativeVulkan_device(JNIEnv* e,jclass,jlong h){try{return e->NewStringUTF(engine(h)->deviceName());}catch(const std::exception& x){error(e,x);return nullptr;}}
extern "C" JNIEXPORT jint JNICALL Java_app_duga_android_NativeVulkan_oracle(JNIEnv* e,jclass,jstring p){try{return verifyOracle(text(e,p).c_str(),2);}catch(const std::exception& x){error(e,x);return -1;}}
