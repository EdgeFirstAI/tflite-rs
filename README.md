# edgefirst-tflite

Ergonomic Rust bindings for the [TensorFlow Lite](https://www.tensorflow.org/lite) C API
with runtime symbol loading, DMA-BUF zero-copy inference, and NPU-accelerated
preprocessing.

## Crates

| Crate | Description |
|-------|-------------|
| [`edgefirst-tflite`](crates/tflite/) | Safe, idiomatic Rust API |
| [`edgefirst-tflite-sys`](crates/tflite-sys/) | Low-level FFI bindings |

Most users should depend on **`edgefirst-tflite`** only.

## Quick Start

```toml
[dependencies]
edgefirst-tflite = "0.1"
```

```rust,no_run
use edgefirst_tflite::{Library, Model, Interpreter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = Library::new()?;
    let model = Model::from_file(&lib, "model.tflite")?;

    let mut interpreter = Interpreter::builder(&lib)?
        .num_threads(4)
        .build(&model)?;

    interpreter.invoke()?;

    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        println!("output[{i}]: {tensor}");
    }
    Ok(())
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `dmabuf` | DMA-BUF zero-copy inference via `VxDelegate` |
| `camera_adaptor` | NPU-accelerated format conversion via `VxDelegate` |
| `metadata` | TFLite model metadata extraction (FlatBuffers) |
| `full` | Enables all optional features |

## Library Discovery

The `TFLite` shared library is loaded at runtime via `libloading`. The
discovery process tries paths in this order:

1. Versioned: `libtensorflow-lite.so.2.{49..1}.{9..0}`
2. Unversioned: `libtensorflowlite_c.so`
3. Fallback: `libtensorflow-lite.so`

Use `Library::from_path()` to load from a specific path.

## VxDelegate Extensions

When using the i.MX NPU delegate (`libvx_delegate.so`), two optional APIs
are available:

### DMA-BUF Zero-Copy

Bind DMA-BUF file descriptors (from V4L2, DRM, etc.) directly to TFLite
tensors, avoiding CPU-side memory copies. Enable with `features = ["dmabuf"]`.

### CameraAdaptor

Configure the delegate to inject format conversion (e.g., RGBA to RGB),
resize, and letterbox operations into the TIM-VX graph, running them on the
NPU instead of the CPU. Enable with `features = ["camera_adaptor"]`.

## Examples

| Example | Description | Features |
|---------|-------------|----------|
| [`basic_inference`](examples/basic_inference/) | Load a model and run inference | default |
| [`dmabuf_zero_copy`](examples/dmabuf_zero_copy/) | DMA-BUF zero-copy with VxDelegate | `dmabuf` |
| [`quantized_inference`](examples/quantized_inference/) | Quantized model I/O with dequantization | default |
| [`error_handling`](examples/error_handling/) | Error classification and graceful fallbacks | `dmabuf` |
| [`metadata_extraction`](examples/metadata_extraction/) | Extract model metadata | `metadata` |
| [`delegate_options`](examples/delegate_options/) | Delegate configuration and feature probing | `dmabuf`, `camera_adaptor` |
| [`camera_preprocessing`](examples/camera_preprocessing/) | NPU-accelerated format conversion | `camera_adaptor` |

## Building

```sh
# Default features
cargo build --workspace

# All features
cargo build --workspace --features full

# Cross-compile for i.MX devices
cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu
```

## Testing

```sh
# Run all tests (unit tests always pass; integration tests require TFLite)
cargo test --workspace --all-features

# Set library path for integration tests if TFLite is not in default search path
TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so cargo test --workspace --all-features
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

The vendored `vx_delegate_dmabuf.h` header is MIT-licensed.
