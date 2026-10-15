package app.duga.android;
import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;
import java.util.Set;
import java.util.Base64;

final class BridgeRequest {
  final String id,type;
  final JSONObject json;
  static final int MAX_EXPORT = 8*1024*1024;
  BridgeRequest(String message) throws JSONException {
    if (message==null || message.length()>12*1024*1024) throw new JSONException("Android message too large");
    json=new JSONObject(message);id=string("id");type=string("type");
    if (!id.matches("[A-Za-z0-9-]{1,64}") || !Set.of("load","evaluate","cancel","close","copy","share","export").contains(type)) throw new JSONException("Unsupported Android request");
  }
  String string(String key) throws JSONException {Object v=json.get(key);if(!(v instanceof String))throw new JSONException("Expected text: "+key);return (String)v;}
  String text(String key,int max) throws JSONException {String v=string(key);if(v.length()>max)throw new JSONException("Text exceeds limit");return v;}
  int[] tokens() throws JSONException {
    JSONArray a=json.getJSONArray("ids");if(a.length()<1||a.length()>1024)throw new JSONException("Token batch exceeds context");
    int[] out=new int[a.length()];for(int i=0;i<out.length;i++){Object v=a.get(i);if(!(v instanceof Integer)||((Integer)v)<0||((Integer)v)>300000)throw new JSONException("Invalid token ID");out[i]=(Integer)v;}return out;
  }
  String filename() throws JSONException {String value=text("filename",120);if(!value.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,119}")||value.contains(".."))throw new JSONException("Invalid export filename");return value;}
  String mime() throws JSONException {String m=text("mime",80).split(";")[0];if(!Set.of("text/plain","text/markdown","application/json","image/png","application/zip","application/octet-stream").contains(m))throw new JSONException("Unsupported export type");return m;}
  byte[] exportBytes() throws JSONException {try{byte[] b=Base64.getDecoder().decode(string("base64"));if(b.length>MAX_EXPORT)throw new JSONException("Export exceeds 8 MiB");return b;}catch(IllegalArgumentException e){throw new JSONException("Invalid export encoding");}}
}
