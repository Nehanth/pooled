# Duga Android APK port

Status: initial port in progress. Native Android packaging and a direct Vulkan Dense inference backend; no public network.

## Architecture and boundary

Bundle the existing Duga root UI, room, engine and harness into an Android APK. Serve only an explicit asset allowlist at the reserved HTTPS origin `https://appassets.androidplatform.net` using AndroidX WebKit's WebViewAssetLoader. Rewrite `/room` and `/r/:code` to the existing room document. The app must launch without a deployed duga.app site; model downloads and room networking still need network access. Preserve browser behavior and all upstream wire/storage identifiers.

This adds an isolated `android/` Gradle build, AndroidX WebKit and Android test dependencies; it does not introduce a website build step. The Android UI uses the system WebView, while Local inference uses a C++ Vulkan runtime through JNI. Local must not depend on WebView WebGPU. The first supported model format is Qwen3 dense Q8_0 GGUF. Existing WGSL entry points are compiled ahead of time to SPIR-V with a pinned Naga build; native Vulkan owns devices, buffers, descriptors, dispatch/barriers and KV state. Shared shader source reduces transcription drift, but does not establish numerical equivalence: the original WebGPU execution is the oracle. No hosted inference route or ownership-enforced Private Net is added. APK room pages disable WebGPU and act as chat clients; distributed native Vulkan workers are not implemented.

## Android behavior

- Receive supported Duga invitation links, preserving the key in the fragment. Never log invite URLs or save them in instance-state bundles. Do not claim verified app links without deployed domain association.
- Keep navigation to bundled content inside the app. Open documentation and other HTTPS links in a browser, with no native bridge on external pages. Deny non-HTTPS external navigation and untrusted file/content/intent schemes.
- Disable arbitrary file/content access, mixed content, cleartext networking, automatic permission grants and app backups. Never bypass TLS failures. Do not request microphone, camera or storage permissions for this port.
- A narrowly scoped message bridge may save user-exported files through Android's document picker or copy/share user-requested text. It grants no file reads, general tool execution or arbitrary native method access. Preview frames with opaque origins must not access it. Validate source origin, message shape and size.
- Respect system bars, keyboard resizing and back navigation. Show a confirmation before leaving active work. Backgrounding pauses the WebView; it does not turn the app into an always-on inference worker. Show interruption state on return rather than claiming background availability.
- Browser and APK storage are separate. Notes stay in the app's WebView profile; uninstalling/deleting app data removes them. Chat remains ephemeral. Avoid claiming credential revocation when disconnecting.

## Validation and delivery

Create an installable debug APK (development signing, not a Play Store/release identity), a reproducible Gradle wrapper, asset-sync checks, URL/message policy tests, Android lint and emulator smoke coverage when the environment supports it. Record skipped checks and system-WebView/GPU limits. Do not commit signing keys, SDK installations, generated web assets, models or build outputs. A production signing identity, app-link domain association and foreground compute service remain future decisions.

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
- Android 15/API 35 x86_64 emulator, SwiftShader 5.0.0 and system WebView 124.0.6367.219: all twelve native kernel cases and tied/untied eight-position model sequences matched original WebGPU execution on the host SwiftShader implementation exactly. Real Qwen3 0.6B Q8_0: all 455,808 logits at three positions matched exactly through the Android executable. The earlier status-2 rejection used a Lavapipe golden against SwiftShader; changing the reference GPU resolved it without changing kernels or accepting tolerances. No physical Android handset is validated by this result.
- Existing Duga desktop/mobile UI browser suite passed after the native integration hook. New Node checks cover release-tag injection/format bounds and asset allowlisting; Java tests cover invite secrets, routing/traversal, message/token limits and export payloads. CI configuration is checked with actionlint 1.7.7.

## Distribution configuration

`.github/workflows/android.yml` builds preview APK/AAB for branches/PRs. `android-vMAJOR.MINOR.PATCH[-prerelease]` tags trigger signed delivery to GitHub Releases after the oracle and build jobs pass. Four Android signing Secrets are required; missing configuration fails before an unsigned release can be published. No Google Play upload, hosted-domain verification or automatic tag creation is included. See `android/README.md` for exact setup and commands.

## Android correctness in delivery

CI first validates Lavapipe, then captures and validates SwiftShader with the emulator driver. The packaged fixture contains the SwiftShader capture’s synthetic inputs and calibration bytes. At runtime, Android obtains expected outputs by executing independently compiled original Tint shaders on the current GPU, then requires bit-exact production Naga output on that same GPU. An Android 15/x86_64 emulator must replay all twelve cases exactly, compare every tiny-model logit against original WebGPU execution and pass instrumented JNI/UI tests before APK/AAB delivery. The new emulator-runner action is pinned to a full commit SHA; it adds a CI dependency and KVM requirement, not an app dependency. Synthetic GGUF fixtures and expected tokens are generated only into test assets and never included in the distributed app.

Captured bytes remain a driver-scoped calibration baseline. Runtime cross-compiler validation avoids applying a foreign GPU’s numerical results to a new device. Unknown mobile GPUs remain experimental and reject shader/capability mismatches; a kernel self-check does not establish every full-model or browser-driver behavior. Original WebGPU remains development-only; production inference calls direct Vulkan. Prefill is submitted one token per bridge request so slow devices report progress and cancellation instead of timing out a whole prompt batch.

## Portable original-shader oracle

`export-reference.mjs` runs original WGSL through pinned Dawn/Tint and exports the programs actually used during that WebGPU execution. It requests core Vulkan 1.1/SPIR-V 1.3 (GLSL450 memory model, without optional Vulkan memory-model, SPIR-V 1.4 or maximal-reconvergence capabilities), assembles them with SPIR-V Tools and validates the result. These profile choices must still reproduce ordinary WebGPU capture bytes exactly during calibration. Only the precompiled SPIR-V is embedded; no Dawn, browser WebGPU implementation, runtime compiler or shader download is included in the APK. This adds build-only SPIR-V Tools and a second generated shader header.

The native gate executes the exported original program and Naga candidate with identical fixture bytes on the same GPU, restores inputs between executions and rejects every bit difference. All twelve kernels must appear exactly once; reference outputs must be finite, complete and changed from their initial state. This prevents empty/no-op checks from enabling the backend. Captured WebGPU comparisons and full-model tests remain independent CI checks, because sharing the Vulkan wrapper during self-validation alone cannot prove every orchestration or driver behavior.

Validation: the Lavapipe-exported reference reproduced ordinary captured output on NVIDIA GTX 1080/580.159.04, Mesa Lavapipe/LLVM 19.1.7 and SwiftShader 5.0.0; all twelve cases passed in each of the captured/native, captured/reference and runtime cross-compiler modes. Android with host GTX 1080 also matched all 13,826,176 logits over 91 positions for the complete Duga chat prompt and answer `OK.`. These are correctness checks, not throughput or physical-phone claims.

## Foreground session recovery validation

The Android adapter returns exact generated IDs to the assistant controller, which retains them with each in-memory chat turn while the native GPU model is released on backgrounding. A reloaded model prefers these IDs over re-tokenized display text. No new persistent store or account identity is introduced. Context bounds reject prompts with no remaining response budget before prefill. Node tests exercise cancellation, zero-budget rejection and noncanonical tokenization across model reload; the browser integration test checks controller ID retention.

Android 15/x86_64 with host RTX 2070 SUPER/580.159.04: four instrumented tests passed, including strict device oracle status 0, tied/untied JNI decoding, cancellation/reset, invalid empty-oracle rejection and bundled UI/notes. With WebView WebGPU disabled, the real Qwen3 0.6B UI produced `OK.`, then after background/resume produced `OK.` for a follow-up, both matching original WebGPU. All 13,826,176 logits over the first 91 positions and all 17,168,768 logits over the follow-up’s 113 positions matched exactly through the Android executable. The initial host-GPU emulator used GTX for Vulkan and RTX for GLES and terminated during an activity-lifecycle test; selecting RTX for both allowed the instrumented and two-turn UI checks to complete. CI continues using SwiftShader; no physical phone performance claim follows from these emulators.
