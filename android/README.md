# Duga Android preview

An installable Android port with the Duga UI bundled in the APK and **direct native Vulkan inference** through JNI. The UI is a WebView; Local computation does not use WebView WebGPU. Android 10/API 29 or later, arm64-v8a and x86_64. This initial backend supports Qwen3 Dense Q8_0/F32 GGUF only; no MoE, DeltaNet, native distributed peers or speculative decoding.

The first Local task asks to enable the experimental backend and runs an exact kernel-oracle gate. A mismatch or unavailable driver leaves Local unavailable; there is no tolerance-based acceptance or hosted fallback. First successful setup downloads about 640 MB of pinned, SHA-256-verified Qwen3 0.6B weights to app-private storage. Context is 1,024 tokens and responses are capped at 384 tokens. Device memory, speed and real-phone quality are not guaranteed. Trusted rooms remain a separate compute choice. APK rooms are chat clients of another trusted device; WebGPU contribution is disabled in the APK until native distributed Vulkan workers are implemented.

## Build locally

Install JDK 17, Node 22, Rust, Android SDK 36, build-tools 35.0.0, NDK 27.3.13750724 and CMake 3.22.1. Set `JAVA_HOME` and `ANDROID_HOME` normally. From the repository root:

```sh
cargo install naga-cli --version 30.0.1 --locked
npm ci --prefix native/vulkan/oracle
node native/vulkan/shaders.mjs
DUGA_WEBGPU_MODULE=native/vulkan/oracle/node_modules/webgpu/index.js node native/vulkan/capture-oracle.mjs
android/gradlew -p android testDebugUnitTest lintDebug assembleDebug bundleDebug
```

Capturing the oracle requires a working WebGPU/Vulkan adapter. CI checks Mesa Lavapipe and the Android emulator's SwiftShader separately, then packages the SwiftShader reference for Android. A reference captured on a different GPU can legitimately fail: floating-point results are not universally bit-identical across drivers. The gate still rejects every mismatch. `syncWebAssets` automatically copies the allowlisted runtime assets, native Android shim and captured oracle; it never copies model weights or repository metadata.

Outputs:

- `app/build/outputs/apk/debug/app-debug.apk` — installable development-signed APK.
- `app/build/outputs/bundle/debug/app-debug.aab` — development bundle, not a production/Play Store upload.

For a SwiftShader emulator, capture its matching software reference and check the native Android binaries:

```sh
export VK_ICD_FILENAMES="$ANDROID_HOME/emulator/lib64/vulkan/vk_swiftshader_icd.json"
export DUGA_WEBGPU_MODULE=native/vulkan/oracle/node_modules/webgpu/index.js
node native/vulkan/capture-oracle.mjs
DUGA_ANDROID_SERIAL=emulator-5554 node scripts/android-native-check.mjs
android/gradlew -p android connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.requireExactVulkan=true
```

The script builds Android command-line runners with the pinned NDK, replays all kernel traces on the device and compares every tiny-model logit with original WebGPU execution. It produces test-only GGUF/greedy fixtures for instrumented JNI tests. Those fixtures are included only in the test APK. Instrumentation covers bundled UI, explicit notes, scoped bridge presence, tied/untied JNI decoding, cancellation/reset and missing-model rejection. CI requires oracle status 0; unknown real devices may reject the reference and leave Local unavailable. Android 15/x86_64 SwiftShader passed all twelve kernels, both tiny models and a separate real Qwen3 0.6B three-position comparison (455,808 exact logits). This does not establish physical-phone performance or universal mobile GPU equivalence.

## GitHub Actions and tag delivery

[Android workflow](../.github/workflows/android.yml) runs on pull requests, `main`, `feat/duga-android`, manual dispatch, and `android-v*` tags. It:

1. Compiles the original WGSL to SPIR-V using pinned Naga 30.0.1.
2. Executes the original WGSL on Dawn WebGPU, then requires bit-exact Vulkan parity for twelve kernels and all logits of tied/untied tiny Dense models, separately on Lavapipe and SwiftShader.
3. Runs Node packaging/version tests, Java policy tests and Android lint. Boots an Android 15/x86_64 SwiftShader emulator and requires exact device kernel/logit parity plus instrumented JNI/UI tests before delivery.
4. Builds APK and AAB for arm64/x86_64, checks ZIP alignment for 16 KB pages, and uploads them with SHA256SUMS as the `duga-android` Actions artifact.
5. For a distribution tag, signs release APK/AAB, verifies their signatures, then publishes the files to GitHub Releases. Tags with a prerelease suffix create prereleases. Browser/Vercel deployment is unchanged.

Configure these repository Secrets before distribution:

| Secret | Value |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | Base64-encoded existing release JKS/PKCS12 keystore |
| `ANDROID_KEYSTORE_PASSWORD` | Keystore password |
| `ANDROID_KEY_ALIAS` | Signing alias |
| `ANDROID_KEY_PASSWORD` | Key password |

A missing signing secret **fails tag delivery**; no unsigned artifact is published as a release. The key is decoded only into runner temporary storage and deleted afterwards. Retain the real signing key outside CI: future updates require the same identity. No production key is generated or committed by this port.

Once the workflow is on the remote branch and signing Secrets are configured:

```sh
git tag -a android-v0.1.0-preview.1 -m 'Duga Android Vulkan preview'
git push origin android-v0.1.0-preview.1
```

Tag format is validated before signing. `versionName` comes from the tag, and `versionCode` from the workflow run number (monotonic within this workflow). AAB delivery is to GitHub Releases; this does not automatically publish to Google Play. A version tag is not pushed automatically by the coding agent.

## Invitations, storage and lifecycle

- Existing room codes retain host approval by default; invite links retain bearer access. `duga://join/ABC234#k=…` opens the app. HTTPS `duga.app/r/…` links are declared **unverified**; Android may leave them in the browser until domain association and signing identity are deliberately configured. Links can always be pasted into Devices. Native asset origins are rewritten to `https://duga.app` for shared room links and QR codes.
- Native calls are origin-scoped to bundled app pages. Inference calls are main-frame only; opaque sandbox previews have no native bridge. External HTTPS pages open in a browser. File/content/intent navigation, cleartext traffic, automatic camera/microphone permissions and app backups are disabled.
- Notes use the app's private WebView profile, separate from browser storage. No account or sync is added. Uninstalling/clearing app data removes notes and downloaded weights. Exports use Android's document picker; copy/share expose no file-read or arbitrary tool capability.
- Backgrounding cancels native compute and disconnects the room workspace. This is a foreground app, not an always-on worker. Chat is ephemeral; room work may need reopening after an interruption. Disconnecting is not credential revocation.

See [the design and validation record](../docs/design/duga-android.md) for scope and limitations.
