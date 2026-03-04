# edgefirst-tflite-sys

Low-level FFI bindings for the [TensorFlow Lite](https://www.tensorflow.org/lite)
C API with runtime symbol loading via `libloading`.

**Most users should depend on [`edgefirst-tflite`](https://crates.io/crates/edgefirst-tflite) instead.**

## Overview

This crate provides:

- `bindgen`-generated function pointer struct (`tensorflowlite_c`) with 164
  TFLite C API functions loaded at runtime.
- Library version probing (`discovery` module).
- `VxDelegate` DMA-BUF and `CameraAdaptor` function pointer structs
  (`vx_ffi` module) -- loaded at runtime from the delegate shared library
  for NPU acceleration, zero-copy inference, and camera preprocessing.

## License

Apache-2.0
