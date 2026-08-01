# Vendored LiteRT C API headers

Pinned to **v2.1.6** (`VERSION`).

Source: [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT) tag `v2.1.6`, licensed Apache-2.0 (see `LICENSE`).

These headers are the bindgen input for `src/litert_ffi.rs` via `wrapper_litert.h` and `../update.sh`. They are **not** linked at build time; LiteRT symbols are resolved at runtime from the same shared library as TensorFlow Lite when present.

## Layout

| Path | Origin |
|------|--------|
| `c/**/*.h` | Verbatim from upstream, then `patches/` applied |
| `LICENSE` | Verbatim from upstream |
| `patches/*.patch` | Local modifications, applied in filename order |
| `build_common/build_config.h` | Hand-written stub — upstream generates this via CMake |

## Re-vendor

```sh
./vendor.sh            # re-fetch the tag in VERSION, reapply patches
./vendor.sh v2.2.0     # move to a new tag and update VERSION
cd .. && ./update.sh   # regenerate src/litert_ffi.rs
```

`vendor.sh` reapplies everything under `patches/`, so local modifications survive an upgrade instead of being silently lost. Never edit `c/**/*.h` in place — add or amend a patch instead, so the next re-vendor keeps the change and reports a conflict loudly if upstream moves underneath it.

## Local patches

- **`0001-litert_gl_types-c-casts.patch`** — replaces `static_cast<T>(0)` with a C cast in the `LITE_RT_EGL_NO_CONTEXT` / `LITE_RT_EGL_NO_DISPLAY` macros. Upstream compiles these headers as C++; bindgen parses them as C, where `static_cast` is not valid syntax.

Every other file under `c/` is byte-identical to upstream v2.1.6.

## Header/runtime skew

`update.sh` regenerates the full bindgen table, and `src/litert.rs` contains a compile-time cross-check asserting that every hand-written `LiteRtFunctions` signature still matches the generated one. If an upgrade changes an argument or return type, the crate stops compiling rather than producing a silently wrong ABI. Treat that failure as a review prompt, not a lint to suppress.
