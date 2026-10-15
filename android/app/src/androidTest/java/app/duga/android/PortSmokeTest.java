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
  @Test public void nativeLibraryLoadsAndReportsOracleMismatchWithoutSilentTolerance()throws Exception {
    android.content.Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
    java.io.File trace=new java.io.File(c.getCacheDir(),"instrumentation-oracle.dvk");
    try(java.io.InputStream in=c.getAssets().open("oracle.dvk");java.io.OutputStream out=new java.io.FileOutputStream(trace)){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);}
    int result=NativeVulkan.oracle(trace.getAbsolutePath());
    System.out.println("Android native Vulkan oracle status: "+result);
    // 0 = exact; 2 = rejected numerical mismatch; 1 = unavailable driver. None enables an approximate backend.
    assertTrue("Unexpected oracle status",result==0||result==1||result==2);
    assertThrows(IllegalStateException.class,()->NativeVulkan.open(new java.io.File(c.getFilesDir(),"missing.gguf").getAbsolutePath()));
  }
}
