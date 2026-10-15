import fs from 'node:fs';
export function androidVersion(tag,run) {
  const match=/^android-v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z][0-9A-Za-z.-]*)?$/.exec(tag);
  if(!match)throw Error('Release tag must be android-vMAJOR.MINOR.PATCH[-prerelease].');
  const code=Number(run);if(!Number.isSafeInteger(code)||code<1||code>2100000000)throw Error('Invalid Android version code.');
  return {name:tag.slice('android-v'.length),code,prerelease:!!match[4]};
}
if(process.argv[1]===new URL(import.meta.url).pathname) {
  const v=androidVersion(process.env.RELEASE_TAG,process.env.GITHUB_RUN_NUMBER);
  const lines=`DUGA_VERSION_NAME=${v.name}\nDUGA_VERSION_CODE=${v.code}\nDUGA_PRERELEASE=${v.prerelease}\n`;
  if(!process.env.GITHUB_ENV)throw Error('GITHUB_ENV is required.');fs.appendFileSync(process.env.GITHUB_ENV,lines);
}
