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

## TFLite Library

`edgefirst-tflite` does not link against TFLite at build time. The shared
library (`libtensorflowlite_c.so` / `.dylib` / `.dll`) is loaded at runtime
via `libloading`, so the same binary works across different TFLite versions
without recompilation.

### Library Discovery

`Library::new()` searches for the TFLite shared library in this order:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | `TFLITE_LIBRARY_PATH` | Explicit path override (env var) |
| 2 | Vendored | Library downloaded by `build.rs` (when `vendored` feature is enabled) |
| 3 | Versioned system | `libtensorflow-lite.so.2.{49..1}.{9..0}` |
| 4 | Unversioned system | `libtensorflowlite_c.so`, then `libtensorflow-lite.so` |

To override discovery at runtime:

```sh
TFLITE_LIBRARY_PATH=/usr/local/lib/libtensorflowlite_c.so ./my-app
```

To load from a specific path in code:

```rust,no_run
let lib = edgefirst_tflite::Library::from_path("/usr/local/lib/libtensorflowlite_c.so")?;
```

### Installing TFLite Manually

Build from source using the official CMake build (TFLite 2.14+):

```sh
git clone --depth 1 --branch v2.19.0 https://github.com/tensorflow/tensorflow
cmake -S tensorflow/lite/c -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_C_BUILD_SHARED_LIBS=ON \
  -DTFLITE_ENABLE_XNNPACK=ON
cmake --build build --parallel
sudo cp build/libtensorflowlite_c.so /usr/local/lib/
sudo ldconfig
```

On i.MX devices, TFLite is typically pre-installed by the BSP at
`/usr/lib/libtensorflowlite_c.so`.

### Vendored Feature (Rust)

The `vendored` feature on `edgefirst-tflite-sys` downloads a pre-built
`libtensorflowlite_c` from the project's GitHub Releases during `cargo build`.
This is useful for CI/CD pipelines and development machines without a
system-installed TFLite.

```sh
# Build with bundled TFLite (downloads during build)
cargo build -p edgefirst-tflite-sys --features vendored

# Override the TFLite version to download
TFLITE_VERSION=2.19.0 cargo build --features vendored
```

The library is downloaded to Cargo's `OUT_DIR`. For deployed binaries, copy
the library alongside the binary — `build.rs` prints a `cargo:warning=` with
the exact path. This follows the same model as `openssl-sys`'s `vendored`
feature.

### Python: edgefirst-tflite-library

For Python users, a companion package ships the pre-built library as a
platform wheel:

```sh
pip install edgefirst-tflite-library
```

Pass the library path to the interpreter:

```python
from edgefirst_tflite import Interpreter
from edgefirst_tflite_library import library_path

interp = Interpreter(model_path="model.tflite", library_path=library_path())
```

The package version matches the TFLite version it ships (e.g.,
`edgefirst-tflite-library==2.19.0` ships TFLite 2.19.0).

## Platform Support

| Platform | Architecture | Notes |
|----------|-------------|-------|
| Linux | x86_64 | Development and CI |
| Linux | aarch64 | i.MX devices, Raspberry Pi |
| macOS | arm64 / x86_64 | Universal binary via vendored feature |
| Windows | x86_64 | Via vendored feature |

### i.MX Device Support

| Device | NPU | Delegate | DMA-BUF | CameraAdaptor |
|--------|-----|----------|---------|---------------|
| i.MX 8M Plus | Vivante GC | `libvx_delegate.so` | Yes | Yes |
| i.MX 93 | Ethos-U65 | `libethosu_delegate.so` | No | No |
| i.MX 95 | Neutron | `libneutron_delegate.so` | No | No |

DMA-BUF zero-copy and CameraAdaptor preprocessing require the Vivante VxDelegate
(`features = ["dmabuf"]` and `features = ["camera_adaptor"]`). On i.MX 93 and
i.MX 95, inference uses standard tensor I/O.

## VxDelegate Extensions

When using the i.MX 8M Plus NPU delegate (`libvx_delegate.so`), two optional APIs
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

See [TESTING.md](TESTING.md) for the full testing guide including on-device
integration tests and the vendored feature test procedure.

```sh
# Run all tests (unit tests always pass; integration tests require TFLite)
cargo test --workspace --all-features

# Set library path for integration tests if TFLite is not in default search path
TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so cargo test --workspace --all-features
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

The vendored `vx_delegate_dmabuf.h` header is MIT-licensed.
