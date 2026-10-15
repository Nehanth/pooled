package app.duga.android;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import android.webkit.WebView;
import android.view.ViewGroup;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

@RunWith(AndroidJUnit4.class)
public class PortSmokeTest {
  private String js(ActivityScenario<MainActivity> scenario,String script)throws Exception {
    CountDownLatch latch=new CountDownLatch(1);AtomicReference<String> value=new AtomicReference<>();
    scenario.onActivity(a->{ViewGroup content=a.findViewById(android.R.id.content);ViewGroup root=(ViewGroup)content.getChildAt(0);WebView web=(WebView)root.getChildAt(0);web.evaluateJavascript(script,r->{value.set(r);latch.countDown();});});
    assertTrue("WebView callback timed out",latch.await(10,TimeUnit.SECONDS));return value.get();
  }
  private void ready(ActivityScenario<MainActivity> scenario)throws Exception {
    long end=System.currentTimeMillis()+30000;
    while(System.currentTimeMillis()<end){if("true".equals(js(scenario,"!!window.DugaNativeRuntime && !!document.querySelector('#prompt') && document.readyState === 'complete'")))return;Thread.sleep(100);}
    fail("Bundled Duga UI did not become ready");
  }
  @Test public void bundledAssistantStartsWithoutWebsiteAndUsesScopedNativeBridge()throws Exception {
    try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){
      ready(scenario);
      assertEquals("true",js(scenario,"location.origin === 'https://appassets.androidplatform.net'"));
      assertEquals("true",js(scenario,"document.querySelector('.model-name').textContent.includes('Vulkan')"));
      assertEquals("0",js(scenario,"document.querySelectorAll('iframe[src]').length"));
      assertEquals("false",js(scenario,"!!document.querySelector('input[type=password]')"));
      js(scenario,"document.querySelector('[data-view=memories]').click();document.querySelector('#memory-text').value='Android explicit note';document.querySelector('#memory-form').requestSubmit();");
      assertEquals("true",js(scenario,"document.querySelector('.memory-item').textContent.includes('Android explicit note')"));
      assertEquals("true",js(scenario,"JSON.parse(localStorage.getItem('duga.memories.v1'))[0].text === 'Android explicit note'"));
      js(scenario,"localStorage.removeItem('duga.memories.v1');");
    }
  }
  private int oracle()throws Exception {
    android.content.Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
    java.io.File trace=new java.io.File(c.getCacheDir(),"instrumentation-oracle.dvk");
    try(java.io.InputStream in=c.getAssets().open("oracle.dvk");java.io.OutputStream out=new java.io.FileOutputStream(trace)){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);}
    int result=NativeVulkan.oracle(trace.getAbsolutePath());
    System.out.println("Android native Vulkan oracle status: "+result);
    return result;
  }
  @Test public void nativeLibraryLoadsAndEnforcesExactOraclePolicy()throws Exception {
    int result=oracle();
    // 0 = exact; 2 = rejected numerical mismatch; 1 = unavailable driver. None enables an approximate backend.
    assertTrue("Unexpected oracle status",result==0||result==1||result==2);
    if(Boolean.parseBoolean(InstrumentationRegistry.getArguments().getString("requireExactVulkan","false")))assertEquals("CI software device must pass every oracle case",0,result);
    android.content.Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
    assertThrows(IllegalStateException.class,()->NativeVulkan.open(new java.io.File(c.getFilesDir(),"missing.gguf").getAbsolutePath()));
  }
  @Test public void nativeJniDecodesTiedAndUntiedModelsAndResetsAfterCancellation()throws Exception {
    assumeTrue("This device has no bit-exact oracle validation",oracle()==0);
    android.content.Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
    android.content.res.AssetManager assets=InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    for(String name:new String[]{"dense-tied","dense-untied"}){
      java.io.File model=new java.io.File(c.getCacheDir(),name+".gguf");
      try(java.io.InputStream in=assets.open(name+".gguf");java.io.OutputStream out=new java.io.FileOutputStream(model)){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);}
      org.json.JSONObject fixture;
      try(java.io.InputStream in=assets.open(name+".json");java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);fixture=new org.json.JSONObject(out.toString("UTF-8"));}
      org.json.JSONArray tokens=fixture.getJSONArray("tokens"),greedy=fixture.getJSONArray("greedy");
      long handle=NativeVulkan.open(model.getAbsolutePath());
      try {
        for(int i=0;i<tokens.length();i++)assertEquals(name+" position "+i,greedy.getInt(i),NativeVulkan.evaluate(handle,new int[]{tokens.getInt(i)},i==0));
        NativeVulkan.cancel(handle);
        assertThrows(IllegalStateException.class,()->NativeVulkan.evaluate(handle,new int[]{33},false));
        assertEquals("Reset clears both KV position and cancellation",greedy.getInt(0),NativeVulkan.evaluate(handle,new int[]{tokens.getInt(0)},true));
      }finally{NativeVulkan.close(handle);model.delete();}
    }
  }
  @Test public void oracleRejectsAnEmptyCaseSetInsteadOfPassingVacuously()throws Exception {
    android.content.Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
    java.io.File trace=new java.io.File(c.getCacheDir(),"empty-oracle.dvk");
    try(java.io.OutputStream out=new java.io.FileOutputStream(trace)){out.write(new byte[]{0x44,0x56,0x4b,0x31,0,0,0,0});}
    assertEquals("Every kernel must actually be checked",1,NativeVulkan.oracle(trace.getAbsolutePath()));
    trace.delete();
  }
}
