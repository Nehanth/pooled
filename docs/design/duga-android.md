# Duga Android APK port

Status: initial port in progress. Native Android packaging and a direct Vulkan Dense inference backend; no public network.

## Architecture and boundary

Bundle the existing Duga root UI, room, engine and harness into an Android APK. Serve only an explicit asset allowlist at the reserved HTTPS origin `https://appassets.androidplatform.net` using AndroidX WebKit's WebViewAssetLoader. Rewrite `/room` and `/r/:code` to the existing room document. The app must launch without a deployed duga.app site; model downloads and room networking still need network access. Preserve browser behavior and all upstream wire/storage identifiers.

This adds an isolated `android/` Gradle build, AndroidX WebKit and Android test dependencies; it does not introduce a website build step. The Android UI uses the system WebView, while Local inference is being ported to a C++ Vulkan runtime through JNI. Local must not depend on WebView WebGPU. The first supported model format is Qwen3 dense Q8_0 GGUF. Existing WGSL entry points are compiled ahead of time to SPIR-V with a pinned Naga build; native Vulkan owns devices, buffers, descriptors, dispatch/barriers and KV state. Shared shader source reduces transcription drift, but does not establish numerical equivalence: the original WebGPU execution is the oracle. No hosted inference route or ownership-enforced Private Net is added. APK room pages disable WebGPU and act as chat clients; distributed native Vulkan workers are not implemented.

## Android behavior

- Receive supported Duga invitation links, preserving the key in the fragment. Never log invite URLs or save them in instance-state bundles. Do not claim verified app links without deployed domain association.
- Keep navigation to bundled content inside the app. Open documentation and other HTTPS links in a browser, with no native bridge on external pages. Deny non-HTTPS external navigation and untrusted file/content/intent schemes.
- Disable arbitrary file/content access, mixed content, cleartext networking, automatic permission grants and app backups. Never bypass TLS failures. Do not request microphone, camera or storage permissions for this port.
- A narrowly scoped message bridge may save user-exported files through Android's document picker or copy/share user-requested text. It grants no file reads, general tool execution or arbitrary native method access. Preview frames with opaque origins must not access it. Validate source origin, message shape and size.
- Respect system bars, keyboard resizing and back navigation. Show a confirmation before leaving active work. Backgrounding pauses the WebView; it does not turn the app into an always-on inference worker. Show interruption state on return rather than claiming background availability.
- Browser and APK storage are separate. Notes stay in the app's WebView profile; uninstalling/deleting app data removes them. Chat remains ephemeral. Avoid claiming credential revocation when disconnecting.

## Validation and delivery

Create an installable debug APK (development signing, not a Play Store/release identity), a reproducible Gradle wrapper, asset-sync checks, URL/message policy tests, Android lint and emulator smoke coverage when the environment supports it. Record skipped checks and system-WebView/GPU limits. Do not commit signing keys, SDK installations, generated web assets, models or build outputs. A production signing identity, app-link domain association, native inference migration and foreground compute service remain future decisions.

## References

- Android local content: https://developer.android.com/develop/ui/views/layout/webapps/load-local-content
- Scoped WebView messaging: https://developer.android.com/reference/androidx/webkit/WebViewCompat#addWebMessageListener(android.webkit.WebView,java.lang.String,java.util.Set,androidx.webkit.WebViewCompat.WebMessageListener)
- Build compatibility: https://developer.android.com/build/releases/agp-8-13-0-release-notes

## Oracle-driven Vulkan port

Use deterministic per-kernel traces (exact input bytes, uniform layout, dispatch size and expected output bytes) captured by the existing WGSL/WebGPU implementation. Replay those traces on the direct Vulkan runner on the same physical device; fail on any bit mismatch by default. Also compare logits and greedy token IDs through a small dense synthetic model, then the target Qwen3 model. Record backend, GPU, driver, source commit and compiler versions. Tolerances are diagnostic only and must not silently replace the repository’s bit-exact requirement. Any backend/device combination that has not passed equivalence remains explicitly experimental and default-off; no universal equivalence claim from one device.

The initial native backend excludes Q4/K-quants, hybrid DeltaNet, MoE, distributed Vulkan peers, speculative decoding, batching and native tools. Unsupported tensor types and model architectures fail before inference. Resource limits, cancellation, context bounds and teardown are mandatory; loading an arbitrary worker-provided library or shader at runtime is excluded. Shipping the APK does not settle those later features.

## Validation record — 2026-10-15

- Source baseline: `9d40300` plus this port; shader-source hashes are written by `shaders.mjs`/`capture-oracle.mjs`. Naga CLI 30.0.1 and Dawn Node `webgpu` 0.6.2.
- NVIDIA GeForce GTX 1080, driver 580.159.04: all 12 original WGSL kernel traces replayed bit-exactly through the direct Vulkan runner. Tiny tied/untied Qwen3 Dense models: eight positions each, all 3,584 logits each bit-exact.
- Real Qwen3 0.6B Q8_0 (639,446,688 bytes, SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`): all 455,808 logits at three fixed input positions matched exactly. This is limited decode correctness evidence, not a quality, speed, long-context or arbitrary-device claim.
- Mesa Lavapipe/LLVM 19.1.7: all 12 kernel cases and both tiny model logit sequences matched bit-exactly. The same pipeline is used in CI with the runner's Mesa driver; its own results remain authoritative for that driver.
- JDK 17, Gradle 8.13, AGP 8.13.2, SDK 36, NDK 27.3.13750724: debug APK/AAB built, Java policy tests and Android lint passed. Release APK/AAB were built with an ephemeral **test** signing key; this does not create or configure a production identity.
- Android 15/API 35 x86_64 emulator, system WebView 124.0.6367.219: bundled UI, explicit note save, scoped bridge presence, JNI loading and oracle failure-path tests passed (two instrumented tests). No physical Android handset or conversational native-model run has been verified yet. Native oracle status is reported separately; a rejected/unavailable status does not count as numerical parity.
- Existing Duga desktop/mobile UI browser suite passed after the native integration hook. New Node checks cover release-tag injection/format bounds and asset allowlisting; Java tests cover invite secrets, routing/traversal, message/token limits and export payloads. CI configuration is checked with actionlint 1.7.7.

## Distribution configuration

`.github/workflows/android.yml` builds preview APK/AAB for branches/PRs. `android-vMAJOR.MINOR.PATCH[-prerelease]` tags trigger signed delivery to GitHub Releases after the oracle and build jobs pass. Four Android signing Secrets are required; missing configuration fails before an unsigned release can be published. No Google Play upload, hosted-domain verification or automatic tag creation is included. See `android/README.md` for exact setup and commands.
