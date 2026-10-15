package app.duga.android;
final class NativeVulkan {
  static { System.loadLibrary("duga_vulkan"); }
  static native long open(String path);
  static native int evaluate(long handle, int[] tokens, boolean reset);
  static native void cancel(long handle);
  static native void close(long handle);
  static native String device(long handle);
  static native int oracle(String tracePath);
  private NativeVulkan() {}
}
