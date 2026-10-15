package app.duga.android;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import android.webkit.WebView;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.os.SystemClock;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.inspector.WindowInspector;
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
  private void awaitJs(ActivityScenario<MainActivity> scenario,String script)throws Exception {
    long end=System.currentTimeMillis()+10000;
    while(System.currentTimeMillis()<end){if("true".equals(js(scenario,script)))return;Thread.sleep(100);}
    fail("WebView condition timed out: "+script);
  }
  private void tapWeb(ActivityScenario<MainActivity> scenario,String selector)throws Exception {
    org.json.JSONArray point=new org.json.JSONArray(js(scenario,"(()=>{const e=document.querySelector("+org.json.JSONObject.quote(selector)+");e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2,innerWidth];})()"));
    float x=(float)point.getDouble(0),y=(float)point.getDouble(1),width=(float)point.getDouble(2);
    scenario.onActivity(a->{
      ViewGroup content=a.findViewById(android.R.id.content);WebView web=(WebView)((ViewGroup)content.getChildAt(0)).getChildAt(0);
      float scale=web.getWidth()/width;long now=SystemClock.uptimeMillis();
      MotionEvent down=MotionEvent.obtain(now,now,MotionEvent.ACTION_DOWN,x*scale,y*scale,0),up=MotionEvent.obtain(now,now+50,MotionEvent.ACTION_UP,x*scale,y*scale,0);
      web.dispatchTouchEvent(down);web.dispatchTouchEvent(up);down.recycle();up.recycle();
    });
  }
  private View confirmationButton() {
    for(View root:WindowInspector.getGlobalWindowViews()){
      View button=root.findViewById(android.R.id.button2);if(button!=null&&button.isShown())return button;
    }
    return null;
  }
  private View awaitConfirmation(ActivityScenario<MainActivity> scenario)throws Exception {
    long end=System.currentTimeMillis()+10000;AtomicReference<View> button=new AtomicReference<>();
    while(System.currentTimeMillis()<end){scenario.onActivity(a->button.set(confirmationButton()));if(button.get()!=null)return button.get();Thread.sleep(100);}
    fail("Native confirmation did not appear");return null;
  }
  private void cancelAndroidDialog(ActivityScenario<MainActivity> scenario)throws Exception {
    long end=System.currentTimeMillis()+10000;
    while(scenario.getState()==androidx.lifecycle.Lifecycle.State.RESUMED && System.currentTimeMillis()<end)Thread.sleep(100);
    assertNotEquals("Android dialog did not open",androidx.lifecycle.Lifecycle.State.RESUMED,scenario.getState());
    // A chooser may still be animating in, or the document picker may consume
    // the first Back to hide its keyboard. Wait for the actual Activity return.
    for(int attempt=0;attempt<4 && scenario.getState()!=androidx.lifecycle.Lifecycle.State.RESUMED;attempt++){
      Thread.sleep(500);
      InstrumentationRegistry.getInstrumentation().sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);
      end=System.currentTimeMillis()+2000;
      while(scenario.getState()!=androidx.lifecycle.Lifecycle.State.RESUMED && System.currentTimeMillis()<end)Thread.sleep(100);
    }
    assertEquals("Android dialog did not return",androidx.lifecycle.Lifecycle.State.RESUMED,scenario.getState());
  }
  @Test public void cancellingNativeConfirmationSettlesLoadAndAllowsRetry()throws Exception {
    try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){
      ready(scenario);
      for(String dismissal:new String[]{"back","outside","button"}){
        js(scenario,"window.loadResult='pending';window.DugaNativeRuntime.call('load').then(()=>window.loadResult='loaded',e=>window.loadResult=e.message);");
        long end=System.currentTimeMillis()+10000;AtomicReference<View> button=new AtomicReference<>();
        while(System.currentTimeMillis()<end){scenario.onActivity(a->button.set(confirmationButton()));if(button.get()!=null)break;Thread.sleep(100);}
        assertNotNull("Native confirmation did not appear",button.get());
        if("back".equals(dismissal))InstrumentationRegistry.getInstrumentation().sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);
        else if("outside".equals(dismissal))scenario.onActivity(a->{
          long now=SystemClock.uptimeMillis();MotionEvent outside=MotionEvent.obtain(now,now,MotionEvent.ACTION_OUTSIDE,-1,-1,0);
          button.get().getRootView().dispatchTouchEvent(outside);outside.recycle();
        });
        else scenario.onActivity(a->button.get().performClick());
        awaitJs(scenario,"window.loadResult === 'Native Vulkan preparation cancelled.'");
      }
    }
  }
  @Test public void codePopupOpensBundledRoomAndInitializesItsModule()throws Exception {
    try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){
      ready(scenario);
      js(scenario,"document.querySelector('[data-view=code]').click();");
      tapWeb(scenario,".workspace-intro a[target=_blank]");
      awaitJs(scenario,"location.pathname === '/room' && !!document.querySelector('#join-gpu') && !document.querySelector('#join-gpu').hidden");
      assertEquals("true",js(scenario,"!!window.DugaNativeRuntime && !navigator.gpu"));
      js(scenario,"window.roomSession='still open';");
      InstrumentationRegistry.getInstrumentation().sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);
      View stay=awaitConfirmation(scenario);
      scenario.onActivity(a->stay.performClick());
      assertEquals("true",js(scenario,"location.pathname === '/room' && window.roomSession === 'still open'"));
      InstrumentationRegistry.getInstrumentation().sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);
      View leave=awaitConfirmation(scenario);
      scenario.onActivity(a->leave.getRootView().findViewById(android.R.id.button1).performClick());
      ready(scenario);
      assertEquals("true",js(scenario,"location.pathname === '/'"));
    }
  }
  @Test public void rotationRetainsLiveAssistantAndDraft()throws Exception {
    try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){
      ready(scenario);
      AtomicReference<MainActivity> original=new AtomicReference<>();
      scenario.onActivity(original::set);
      js(scenario,"window.sessionTokens=[42,17,99];document.querySelector('#prompt').value='Unsent draft';");
      for(int orientation:new int[]{ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE,ActivityInfo.SCREEN_ORIENTATION_PORTRAIT}){
        scenario.onActivity(a->a.setRequestedOrientation(orientation));
        int expected=orientation==ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE?Configuration.ORIENTATION_LANDSCAPE:Configuration.ORIENTATION_PORTRAIT;
        long end=System.currentTimeMillis()+10000;AtomicReference<Integer> actual=new AtomicReference<>();
        do{scenario.onActivity(a->{assertSame("Rotation must retain the Activity and its live WebView",original.get(),a);actual.set(a.getResources().getConfiguration().orientation);});if(actual.get()==expected)break;Thread.sleep(100);}while(System.currentTimeMillis()<end);
        assertEquals(expected,(int)actual.get());
        assertEquals("true",js(scenario,"JSON.stringify(window.sessionTokens)==='[42,17,99]' && document.querySelector('#prompt').value==='Unsent draft'"));
      }
    }
  }
  @Test public void bridgeValidationFailuresRejectImmediatelyAndAllowRetry()throws Exception {
    try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){
      ready(scenario);
      for(String request:new String[]{"'copy',{text:42}","'export',{filename:'bad/name',mime:'text/plain',base64:'eA=='}","'export',{filename:'ok.txt',mime:'invalid/type',base64:'eA=='}","'export',{filename:'ok.txt',mime:'text/plain',base64:'!bad'}","'share',{text:'',url:'file:///private'}","'evaluate',{ids:[-1]}","'unsupported'"}){
        js(scenario,"window.bridgeResult='pending';window.DugaNativeRuntime.call("+request+").then(()=>window.bridgeResult='unexpected success',e=>window.bridgeResult=e.message);");
        awaitJs(scenario,"window.bridgeResult!=='pending'");
        assertEquals("false",js(scenario,"window.bridgeResult==='unexpected success'"));
      }
      js(scenario,"window.bridgeResult='pending';window.DugaNativeRuntime.call('copy',{text:'Valid retry'}).then(()=>window.bridgeResult='ok');");
      awaitJs(scenario,"window.bridgeResult==='ok'");
    }
  }
  @Test public void cancellingExportAndSharePreservesEmbeddedAndTopLevelRooms()throws Exception {
    try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){
      ready(scenario);
      js(scenario,"document.querySelector('[data-view=code]').click();document.querySelector('#workspace-open').click();");
      awaitJs(scenario,"document.querySelector('#workspace').contentDocument?.querySelector('#join-gpu') != null");
      for(boolean topLevel:new boolean[]{false,true}){
        if(topLevel){tapWeb(scenario,".workspace-intro a[target=_blank]");awaitJs(scenario,"location.pathname==='/room' && !!document.querySelector('#join-gpu')");}
        js(scenario,"window.roomSession='unchanged';window.pauseCount=0;addEventListener('duga:android-pause',()=>pauseCount++);"+(topLevel?"":"document.querySelector('#workspace').contentWindow.roomSession='embedded';"));
        for(String request:new String[]{"'export',{filename:'duga.html',mime:'text/html',base64:'PGgxPkR1Z2E8L2gxPg=='}","'share',{text:'Duga',url:''}"}){
          js(scenario,"window.dialogResult='pending';window.DugaNativeRuntime.call("+request+").then(()=>window.dialogResult='ok',e=>window.dialogResult=e.message);");
          cancelAndroidDialog(scenario);
          awaitJs(scenario,"window.dialogResult!=='pending'");
          assertEquals("true",js(scenario,"window.roomSession==='unchanged' && window.pauseCount===0"+(topLevel?" && location.pathname==='/room'":" && document.querySelector('#workspace').contentWindow.roomSession==='embedded'")));
        }
        // Once the dialog returns, ordinary backgrounding must still disconnect rooms.
        scenario.moveToState(androidx.lifecycle.Lifecycle.State.CREATED);
        scenario.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED);
        if(topLevel)ready(scenario);
        else awaitJs(scenario,"!document.querySelector('#workspace').hasAttribute('src')");
      }
    }
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
