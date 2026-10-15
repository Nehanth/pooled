package app.duga.android;
import java.net.URI;
import java.util.regex.Pattern;

final class UrlPolicy {
  static final String ORIGIN = "https://appassets.androidplatform.net";
  private static final Pattern CODE = Pattern.compile("[A-HJKMNP-TV-Z2-9]{4}|[A-HJKMNP-TV-Z2-9]{6}");
  private static final Pattern KEY = Pattern.compile("[A-Za-z0-9_-]{22,64}");
  static boolean isApp(String value) {
    try { URI u = URI.create(value); return "https".equals(u.getScheme()) && "appassets.androidplatform.net".equals(u.getHost()) && u.getPort() == -1 && u.getRawUserInfo() == null; }
    catch (RuntimeException e) { return false; }
  }
  static String invite(String value) {
    try {
      URI u = URI.create(value);
      if (u.getRawUserInfo() != null || u.getPort() != -1 || u.getRawQuery() != null) return null;
      String code;
      if ("https".equals(u.getScheme()) && "duga.app".equals(u.getHost()) && u.getPath().startsWith("/r/")) code=u.getPath().substring(3);
      else if ("duga".equals(u.getScheme()) && "join".equals(u.getHost())) code=u.getPath().substring(1);
      else return null;
      if (!CODE.matcher(code).matches()) return null;
      String fragment=u.getRawFragment();
      if (fragment!=null && (!fragment.startsWith("k=") || !KEY.matcher(fragment.substring(2)).matches())) return null;
      return ORIGIN+"/r/"+code+(fragment==null?"":"#"+fragment);
    } catch (RuntimeException e) { return null; }
  }
  static String asset(String requestPath) {
    if (requestPath==null || requestPath.contains("..") || requestPath.contains("\\") || requestPath.contains("%") || requestPath.contains("//")) return null;
    if ("/".equals(requestPath)) return "index.html";
    if ("/room".equals(requestPath) || "/p2p.html".equals(requestPath)) return "p2p.html";
    if (requestPath.startsWith("/r/") && CODE.matcher(requestPath.substring(3)).matches()) return "p2p.html";
    for (String dir:new String[]{"app/","engine/","room/","harness/","site/"}) if (requestPath.startsWith("/"+dir)) return requestPath.substring(1);
    if ("/index.html".equals(requestPath) || "/room.js".equals(requestPath) || "/favicon.svg".equals(requestPath) || "/site.webmanifest".equals(requestPath) || "/android-host.js".equals(requestPath) || "/LICENSE".equals(requestPath)) return requestPath.substring(1);
    return null;
  }
  static boolean external(String value) { try {URI u=URI.create(value); return "https".equals(u.getScheme())&&u.getHost()!=null&&u.getRawUserInfo()==null&&!isApp(value);}catch(RuntimeException e){return false;} }
  private UrlPolicy() {}
}
