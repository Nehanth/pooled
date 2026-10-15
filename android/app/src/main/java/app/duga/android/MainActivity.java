package app.duga.android;

import android.app.*;
import android.annotation.SuppressLint;
import android.content.*;
import android.graphics.Color;
import android.net.Uri;
import android.os.*;
import android.view.*;
import android.webkit.*;
import android.widget.*;
import androidx.webkit.*;
import org.json.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public final class MainActivity extends Activity {
  private WebView web;
  private final ExecutorService worker=Executors.newSingleThreadExecutor();
  private final Object modelLock=new Object();
  private long modelHandle;
  private ModelFiles models;
  private boolean backgrounded,approved,destroyed;
  private byte[] exportData;
  private String exportId;
  private JavaScriptReplyProxy exportReply;
  private TextView loadStatus;
  private static final int SAVE_DOCUMENT=30;

  @Override public void onCreate(Bundle state) {
    super.onCreate(state);
    models=new ModelFiles(this);
    FrameLayout root=new FrameLayout(this);root.setBackgroundColor(Color.rgb(246,245,241));setContentView(root);
    web=new WebView(this);web.setBackgroundColor(Color.rgb(246,245,241));root.addView(web,new FrameLayout.LayoutParams(-1,-1));
    loadStatus=new TextView(this);loadStatus.setText(R.string.loading);loadStatus.setGravity(Gravity.CENTER);loadStatus.setTextColor(Color.rgb(20,22,29));root.addView(loadStatus,new FrameLayout.LayoutParams(-1,80));
    root.setOnApplyWindowInsetsListener((v,insets)->{
      if(Build.VERSION.SDK_INT>=30){android.graphics.Insets i=insets.getInsets(WindowInsets.Type.systemBars()|WindowInsets.Type.displayCutout()|WindowInsets.Type.ime());v.setPadding(i.left,i.top,i.right,i.bottom);}
      else v.setPadding(insets.getSystemWindowInsetLeft(),insets.getSystemWindowInsetTop(),insets.getSystemWindowInsetRight(),insets.getSystemWindowInsetBottom());return insets;
    });
    WebSettings s=web.getSettings();
    s.setJavaScriptEnabled(true);s.setDomStorageEnabled(true);s.setAllowFileAccess(false);s.setAllowContentAccess(false);s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);s.setMediaPlaybackRequiresUserGesture(true);s.setSupportMultipleWindows(true);s.setJavaScriptCanOpenWindowsAutomatically(false);
    CookieManager.getInstance().setAcceptThirdPartyCookies(web,false);
    WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
    WebViewAssetLoader loader=new WebViewAssetLoader.Builder().addPathHandler("/",path->{
      try {
        if("models/tokenizer.json".equals(path))return new WebResourceResponse("application/json","UTF-8",new FileInputStream(models.tokenizer()));
        String asset=UrlPolicy.asset("/"+path);if(asset==null)return missing();return new WebResourceResponse(mime(asset),"UTF-8",getAssets().open(asset));
      }catch(IOException e){return missing();}
    }).build();
    web.setWebViewClient(new WebViewClientCompat(){
      @Override public WebResourceResponse shouldInterceptRequest(WebView view,WebResourceRequest request){if(UrlPolicy.isApp(request.getUrl().toString())){WebResourceResponse r=loader.shouldInterceptRequest(request.getUrl());return r==null?missing():r;}return null;}
      @Override public boolean shouldOverrideUrlLoading(WebView view,WebResourceRequest request){
        Uri u=request.getUrl();String url=u.toString();
        if(UrlPolicy.isApp(url)){
          if(u.getPath()!=null&&u.getPath().startsWith("/docs")){openExternal("https://duga.app"+u.getPath());return true;}return false;
        }
        if(!request.isForMainFrame()&&Set.of("blob","data","about").contains(u.getScheme()))return false;
        if(request.isForMainFrame()&&request.hasGesture())openExternal(url);return true;
      }
      @Override public void onPageFinished(WebView view,String url){loadStatus.setVisibility(View.GONE);}
      @Override public void onReceivedError(WebView view,WebResourceRequest request,WebResourceErrorCompat error){if(request.isForMainFrame()){loadStatus.setText("Duga could not open this page. Close and reopen the app to retry.");loadStatus.setVisibility(View.VISIBLE);}}
      @Override public boolean onRenderProcessGone(WebView view,RenderProcessGoneDetail detail){cancelNative();view.destroy();new AlertDialog.Builder(MainActivity.this).setTitle("Duga stopped").setMessage("Android closed the app’s rendering process. Saved notes remain; the active conversation may be lost.").setPositiveButton("Restart",(d,w)->recreate()).setNegativeButton("Close",(d,w)->finish()).show();return true;}
    });
    web.setWebChromeClient(new WebChromeClient(){
      @Override public void onPermissionRequest(PermissionRequest request){request.deny();}
      @Override public boolean onCreateWindow(WebView view,boolean dialog,boolean gesture,Message result){
        if(!gesture)return false;
        WebView popup=new WebView(MainActivity.this);popup.setWebViewClient(new WebViewClient(){@Override public boolean shouldOverrideUrlLoading(WebView v,WebResourceRequest r){String u=r.getUrl().toString();if(UrlPolicy.isApp(u)&&r.getUrl().getPath().startsWith("/docs"))u="https://duga.app"+r.getUrl().getPath();openExternal(u);v.destroy();return true;}});
        ((WebView.WebViewTransport)result.obj).setWebView(popup);result.sendToTarget();return true;
      }
    });
    if(WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)){
      WebViewCompat.addWebMessageListener(web,"DugaAndroid",Collections.singleton(UrlPolicy.ORIGIN),(view,message,origin,mainFrame,reply)->{
        if(!UrlPolicy.isApp(origin.toString())||message.getType()!=WebMessageCompat.TYPE_STRING)return;
        try {BridgeRequest r=new BridgeRequest(message.getData());
          if(!mainFrame&&!Set.of("copy","share","export").contains(r.type)){respond(reply,r.id,null,"Native inference is available only to the main assistant page.");return;}
          handle(r,reply);
        }catch(Exception e){Toast.makeText(this,"Invalid Android request",Toast.LENGTH_SHORT).show();}
      });
    }else loadStatus.setText("Update Android System WebView to enable the native Vulkan bridge.");
    String launch=readInvite(getIntent());web.loadUrl(launch==null?UrlPolicy.ORIGIN+"/":launch);
    if(Build.VERSION.SDK_INT>=33)getOnBackInvokedDispatcher().registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT,this::goBack);
  }
  private String readInvite(Intent intent){String uri=intent==null||intent.getData()==null?null:intent.getData().toString();if(intent!=null)intent.setData(null);return uri==null?null:UrlPolicy.invite(uri);}
  @Override protected void onNewIntent(Intent intent){super.onNewIntent(intent);String target=readInvite(intent);setIntent(new Intent(this,MainActivity.class));if(target!=null)new AlertDialog.Builder(this).setTitle("Open room invitation?").setMessage("This replaces the current workspace. Room members are trusted with inference data. A link does not prove device ownership.").setPositiveButton("Open room",(d,w)->{cancelNative();web.loadUrl(target);}).setNegativeButton("Cancel",null).show();}
  private void handle(BridgeRequest r,JavaScriptReplyProxy reply)throws Exception {
    switch(r.type){
      case "cancel":cancelNative();respond(reply,r.id,new JSONObject(),null);return;
      case "close":cancelNative();worker.execute(this::closeNative);respond(reply,r.id,new JSONObject(),null);return;
      case "copy":((ClipboardManager)getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("Duga",r.text("text",65536)));respond(reply,r.id,new JSONObject(),null);return;
      case "share":String text=r.text("text",65536),url=r.text("url",4096);if(!url.isEmpty()&&!UrlPolicy.isApp(url)&&!UrlPolicy.external(url))throw new JSONException("Unsupported share URL");if(UrlPolicy.isApp(url))url="https://duga.app"+Uri.parse(url).getEncodedPath()+(Uri.parse(url).getEncodedFragment()==null?"":"#"+Uri.parse(url).getEncodedFragment());Intent share=new Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT,text+(url.isEmpty()?"":"\n"+url));startActivity(Intent.createChooser(share,getString(R.string.share_prompt)));respond(reply,r.id,new JSONObject(),null);return;
      case "export":if(exportReply!=null){respond(reply,r.id,null,"Finish the current export first.");return;}String name=r.filename(),type=r.mime();exportData=r.exportBytes();exportId=r.id;exportReply=reply;startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType(type).putExtra(Intent.EXTRA_TITLE,name),SAVE_DOCUMENT);return;
      case "load":if(!approved){new AlertDialog.Builder(this).setTitle("Enable experimental native Vulkan?").setMessage("Original shaders and native shaders must match exactly on this GPU before loading. This kernel check does not guarantee every phone or conversation; the backend remains experimental. First use downloads about 640 MB of Qwen3 weights to app-private storage. No hosted inference fallback.").setPositiveButton("Validate & download",(d,w)->{approved=true;load(r,reply);}).setNegativeButton("Cancel",(d,w)->respond(reply,r.id,null,"Native Vulkan preparation cancelled.")).show();}else load(r,reply);return;
      case "evaluate":int[] ids=r.tokens();boolean reset=r.json.optBoolean("reset",false);worker.execute(()->{try{long h; synchronized(modelLock){h=modelHandle;}if(h==0)throw new IllegalStateException("Native model is not loaded");int token=NativeVulkan.evaluate(h,ids,reset);respond(reply,r.id,new JSONObject().put("token",token),null);}catch(Exception e){respond(reply,r.id,null,e.getMessage());}});return;
      default:throw new JSONException("Unsupported request");
    }
  }
  private void load(BridgeRequest r,JavaScriptReplyProxy reply){models.cancelled=false;worker.execute(()->{
    try{
      closeNative();progress(reply,r.id,"Checking native Vulkan against kernel oracle…",0,0);
      File trace=new File(getCacheDir(),"oracle.dvk");try(InputStream in=getAssets().open("oracle.dvk");OutputStream out=new FileOutputStream(trace)){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);}
      if(NativeVulkan.oracle(trace.getAbsolutePath())!=0)throw new IllegalStateException("The native Vulkan kernel oracle did not match on this GPU. Local stays unavailable; no fallback was attempted.");
      progress(reply,r.id,"Downloading validated Qwen3 weights…",0,639446688);
      models.prepare((d,t)->progress(reply,r.id,"Downloading native model…",d,t));
      if(models.cancelled)throw new IOException("Native preparation cancelled");
      long h=NativeVulkan.open(models.weights().getAbsolutePath());synchronized(modelLock){modelHandle=h;}
      respond(reply,r.id,new JSONObject().put("context",1024).put("backend","Vulkan").put("device",NativeVulkan.device(h)),null);
    }catch(Exception|LinkageError e){closeNative();respond(reply,r.id,null,e.getMessage());}
  });}
  private void cancelNative(){if(models!=null)models.cancelled=true;synchronized(modelLock){if(modelHandle!=0)NativeVulkan.cancel(modelHandle);}}
  private void closeNative(){synchronized(modelLock){if(modelHandle!=0){NativeVulkan.close(modelHandle);modelHandle=0;}}}
  private void progress(JavaScriptReplyProxy reply,String id,String status,long done,long total){try{JSONObject m=new JSONObject().put("id",id).put("progress",new JSONObject().put("status",status).put("done",done).put("total",total));runOnUiThread(()->{if(!destroyed&&WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER))reply.postMessage(m.toString());});}catch(JSONException ignored){}}
  private void respond(JavaScriptReplyProxy reply,String id,JSONObject result,String error){try{JSONObject m=new JSONObject().put("id",id);if(error!=null)m.put("error",error);else m.put("result",result==null?new JSONObject():result);runOnUiThread(()->{if(!destroyed&&WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER))reply.postMessage(m.toString());});}catch(JSONException ignored){}}
  @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request!=SAVE_DOCUMENT||exportReply==null)return;byte[] bytes=exportData;String id=exportId;JavaScriptReplyProxy reply=exportReply;exportReply=null;exportData=null;exportId=null;if(result!=RESULT_OK||data==null||data.getData()==null){respond(reply,id,null,"Export cancelled.");return;}Uri target=data.getData();worker.execute(()->{try(OutputStream out=getContentResolver().openOutputStream(target)){if(out==null)throw new IOException("Cannot open selected document");out.write(bytes);respond(reply,id,new JSONObject(),null);}catch(Exception e){respond(reply,id,null,"Could not write the selected document.");}});}
  private void openExternal(String url){if(!UrlPolicy.external(url))return;try{startActivity(new Intent(Intent.ACTION_VIEW,Uri.parse(url)).addCategory(Intent.CATEGORY_BROWSABLE));}catch(ActivityNotFoundException e){Toast.makeText(this,"No browser is available",Toast.LENGTH_SHORT).show();}}
  private void goBack(){if(web.canGoBack())web.goBack();else new AlertDialog.Builder(this).setTitle("Close Duga?").setMessage("Active work and room connections will stop. Saved notes remain on this device.").setPositiveButton("Close",(d,w)->finish()).setNegativeButton("Stay",null).show();}
  // API 33+ uses the native OnBackInvokedCallback registered in onCreate; this override is only the legacy fallback.
  @SuppressLint("GestureBackNavigation")
  @Override public void onBackPressed(){if(Build.VERSION.SDK_INT<33)goBack();else super.onBackPressed();}
  @Override protected void onStop(){super.onStop();backgrounded=true;cancelNative();worker.execute(this::closeNative);if(web!=null){web.evaluateJavascript("window.dispatchEvent(new Event('duga:android-pause'))",null);web.onPause();web.pauseTimers();if(web.getUrl()!=null&&!web.getUrl().startsWith(UrlPolicy.ORIGIN+"/#")&&!web.getUrl().equals(UrlPolicy.ORIGIN+"/"))web.loadUrl(UrlPolicy.ORIGIN+"/");}}
  @Override protected void onResume(){super.onResume();if(web!=null){web.resumeTimers();web.onResume();if(backgrounded){Toast.makeText(this,R.string.background_notice,Toast.LENGTH_LONG).show();backgrounded=false;}}}
  @Override protected void onDestroy(){destroyed=true;cancelNative();worker.execute(this::closeNative);worker.shutdown();if(web!=null){web.stopLoading();web.destroy();web=null;}super.onDestroy();}
  private static String mime(String p){if(p.endsWith(".html"))return "text/html";if(p.endsWith(".js"))return "application/javascript";if(p.endsWith(".css"))return "text/css";if(p.endsWith(".svg"))return "image/svg+xml";if(p.endsWith(".png"))return "image/png";if(p.endsWith(".woff2"))return "font/woff2";return "application/json";}
  private static WebResourceResponse missing(){return new WebResourceResponse("text/plain","UTF-8",404,"Not Found",Collections.emptyMap(),new ByteArrayInputStream(new byte[0]));}
}
