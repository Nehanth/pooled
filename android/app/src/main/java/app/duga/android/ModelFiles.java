package app.duga.android;
import android.content.Context;
import java.io.*;
import java.net.*;
import java.security.MessageDigest;
import java.util.function.BiConsumer;

final class ModelFiles {
  private static final String TOKENIZER_URL="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json";
  private static final String WEIGHTS_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf";
  private final File dir;
  volatile boolean cancelled;
  ModelFiles(Context c) {dir=new File(c.getFilesDir(),"models");}
  File tokenizer(){return new File(dir,"tokenizer.json");}
  File weights(){return new File(dir,"qwen3-0.6b.gguf");}
  void prepare(BiConsumer<Long,Long> progress) throws Exception {
    if(cancelled)throw new IOException("Native preparation cancelled");
    if(!dir.isDirectory()&&!dir.mkdirs())throw new IOException("Could not create private model storage");
    download(WEIGHTS_URL,weights(),"9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",639446688,progress);
    download(TOKENIZER_URL,tokenizer(),"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",16*1024*1024,(d,t)->{});
  }
  private static String digest(File f) throws Exception {MessageDigest sha=MessageDigest.getInstance("SHA-256");try(InputStream in=new FileInputStream(f)){byte[] b=new byte[256*1024];int n;while((n=in.read(b))>=0)sha.update(b,0,n);}StringBuilder s=new StringBuilder();for(byte x:sha.digest())s.append(String.format("%02x",x&255));return s.toString();}
  private void download(String source,File target,String expected,long max,BiConsumer<Long,Long> progress) throws Exception {
    if(target.isFile()&&digest(target).equals(expected))return;
    URL url=new URL(source);HttpURLConnection con=null;
    for(int i=0;i<6;i++){
      if(cancelled)throw new IOException("Model preparation cancelled");
      if(!"https".equals(url.getProtocol()))throw new IOException("Insecure model redirect refused");
      con=(HttpURLConnection)url.openConnection();con.setInstanceFollowRedirects(false);con.setConnectTimeout(20000);con.setReadTimeout(20000);int status=con.getResponseCode();
      if(status>=300&&status<400){String next=con.getHeaderField("Location");con.disconnect();if(next==null)throw new IOException("Model redirect missing");url=new URL(url,next);continue;}
      if(status!=200){con.disconnect();throw new IOException("Model download failed (HTTP "+status+")");}break;
    }
    if(con==null||con.getResponseCode()!=200)throw new IOException("Too many model redirects");
    File part=new File(dir,target.getName()+".part");long total=con.getContentLengthLong(),done=0,last=0;
    if(total>max){con.disconnect();throw new IOException("Model download exceeds size limit");}
    try(InputStream in=con.getInputStream();OutputStream out=new FileOutputStream(part)){
      byte[] b=new byte[256*1024];int n;while((n=in.read(b))!=-1){if(cancelled)throw new IOException("Model preparation cancelled");done+=n;if(done>max)throw new IOException("Download exceeds model limit");out.write(b,0,n);long now=System.currentTimeMillis();if(now-last>250){progress.accept(done,total);last=now;}}
    } finally {con.disconnect();}
    if(!digest(part).equals(expected)){part.delete();throw new IOException("Model checksum mismatch; the model was not loaded");}
    if(!part.renameTo(target))throw new IOException("Could not save validated model");
  }
}
