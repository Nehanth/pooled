package app.duga.android;
import org.junit.Test;
import static org.junit.Assert.*;
import org.json.JSONException;

public class PolicyTest {
  @Test public void invitationPreservesOnlyFragmentSecret() {
    String secret="abcdefghijklmnopqrstuv";
    assertEquals(UrlPolicy.ORIGIN+"/r/ABC234#k="+secret,UrlPolicy.invite("https://duga.app/r/ABC234#k="+secret));
    assertEquals(UrlPolicy.ORIGIN+"/r/ABC234",UrlPolicy.invite("duga://join/ABC234"));
    for(String bad:new String[]{"https://evil.example/r/ABC234","https://duga.app.evil/r/ABC234","https://user@duga.app/r/ABC234","https://duga.app/r/ABC234?k="+secret,"duga://join/ABC234/extra","file:///r/ABC234","intent://join/ABC234","https://duga.app/r/ABCDE"})assertNull(bad,UrlPolicy.invite(bad));
  }
  @Test public void assetRoutingRejectsPrivateFilesAndTraversal() {
    assertEquals("index.html",UrlPolicy.asset("/"));assertEquals("p2p.html",UrlPolicy.asset("/r/ABC234"));assertEquals("p2p.html",UrlPolicy.asset("/room"));
    assertEquals("room.js",UrlPolicy.asset("/room.js"));
    for(String path:new String[]{"/../local.properties","/app/../../models/secret","/app/%2e%2e/private","/android/local.properties","/native/vulkan/jni.cpp","/models/qwen3.gguf","/app//test.js"})assertNull(path,UrlPolicy.asset(path));
    assertFalse(UrlPolicy.isApp("https://appassets.androidplatform.net.evil/"));assertFalse(UrlPolicy.isApp("http://appassets.androidplatform.net/"));
    assertFalse(UrlPolicy.external("intent://run"));assertFalse(UrlPolicy.external("file:///secret"));assertTrue(UrlPolicy.external("https://duga.app/docs"));
  }
  @Test public void nativeRequestsBoundTokensAndExportCapabilities() throws Exception {
    assertArrayEquals(new int[]{1,2},new BridgeRequest("{\"id\":\"a\",\"type\":\"evaluate\",\"ids\":[1,2]}").tokens());
    assertThrows(JSONException.class,()->new BridgeRequest("{\"id\":\"a\",\"type\":\"shell\"}"));
    assertThrows(JSONException.class,()->new BridgeRequest("{\"id\":\"a\",\"type\":\"evaluate\",\"ids\":[-1]}").tokens());
    assertThrows(JSONException.class,()->new BridgeRequest("{\"id\":\"a\",\"type\":\"evaluate\",\"ids\":[1.2]}").tokens());
    assertThrows(JSONException.class,()->new BridgeRequest("{\"id\":\"a\",\"type\":\"export\",\"filename\":\"../secret\"}").filename());
    assertThrows(JSONException.class,()->new BridgeRequest("{\"id\":\"a\",\"type\":\"export\",\"mime\":\"text/html\"}").mime());
    assertThrows(JSONException.class,()->new BridgeRequest("{\"id\":\"a\",\"type\":\"export\",\"base64\":\"!bad\"}").exportBytes());
  }
}
