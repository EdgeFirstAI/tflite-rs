# Testing Guide

This guide describes how to test `edgefirst-tflite` on NXP FRDM development
boards. Unit tests run on any host machine; integration tests and examples
require on-target hardware with a TensorFlow Lite runtime.

## Prerequisites

**TFLite library** — Integration tests require `libtensorflowlite_c.so`.
On i.MX devices it is provided by the BSP. On a development host, either build
it from source (see [README.md](README.md#installing-tflite-manually)) or use
the `vendored` feature.

**Test models** — The `testdata/` directory at the repo root contains:

| File | Purpose |
|------|---------|
| `testdata/minimal.tflite` | Minimal valid TFLite flatbuffer for unit tests |
| `testdata/yolov8n-t-b86_quant-u8-i8_*.tflite` | YOLOv8n detection, u8 input / i8 output |
| `testdata/yolov8n-seg-t-b7f_quant-u8-i8_*.tflite` | YOLOv8n segmentation, u8 input / i8 output |

The YOLOv8 fixtures follow `<model>_<layout>[.imx95].tflite`:

- **`_combined` / `_logical` / `_smart`** — the three YOLO output layouts
  (fused, logical-split, per-scale FPN-split). The `yolov8` example configures
  its decoder from the model's embedded `edgefirst.json`, so all three should
  produce equivalent detections.
- **`.imx95.tflite`** — converted for the i.MX 95 Neutron NPU; run these with
  `--delegate /usr/lib/libneutron_delegate.so`. They fail with `UnresolvedOps`
  on any other runtime, which is expected.
- **plain `.tflite`** — portable; runs on any host with a TFLite or LiteRT
  runtime, including CPU-only development machines.

These fixtures are stored in [Git LFS](https://git-lfs.com). Clone with
`git lfs install` configured, or run `git lfs pull` after cloning, otherwise
they arrive as pointer stubs. `minimal.tflite` is intentionally *not* in LFS —
it is compiled into the test binaries with `include_bytes!`.

## Host Unit Tests

Unit tests exercise pure Rust logic (type conversions, error handling, metadata
parsing) without requiring a TFLite shared library.

```sh
cargo test --workspace --all-features
```

## Testing with the Vendored Feature

The `vendored` feature downloads a pre-built `libtensorflowlite_c` at build
time. Use this on CI or development machines without a system-installed TFLite.

```sh
# Build and test with vendored TFLite (downloads ~50 MB during first build)
cargo test -p edgefirst-tflite-sys --features vendored

# Override the TFLite version
TFLITE_VERSION=2.19.0 cargo test --workspace --features vendored
```

The downloaded library is placed in Cargo's `OUT_DIR`. Subsequent builds use
the cached copy — `build.rs` skips the download if the library already exists.

To verify which library was loaded:

```sh
RUST_LOG=debug cargo test --features vendored 2>&1 | grep "Found TFLite"
# debug: Found TFLite library: /path/to/target/.../out/.../libtensorflowlite_c.so
```

### Custom delegate tests

`crates/tflite/tests/custom_delegate.rs` needs a runtime that exports the opaque delegate API. The vendored library does not (upstream's CMake build of `libtensorflowlite_c` leaves `c_api_opaque.cc` out), so on a host those tests print `SKIPPED` and pass. Run them on a board, where the BSP's `libtensorflow-lite.so` exports the API. Two of them stack the test delegate with the board's NPU delegate, named by `TFLITE_TEST_NPU_DELEGATE`, in both orders: after Neutron as on i.MX 95, and before VX as on i.MX 8M Plus.

### EdgeFirst delegate tests

`crates/tflite-delegate/tests/edgefirst_delegate.rs` runs models with and without the EdgeFirst delegate and compares every output. Like `custom_delegate`, these tests need the opaque delegate API and skip on a host.

| Test | Model | Expectation |
|------|-------|-------------|
| `int8_region_*`, `int16_softmax_*` | `testdata/softmax_int8.tflite`, `testdata/softmax_int16.tflite` (from `testdata/generate_softmax.py`) | Scalar tier bit for bit with `TFLite`; NEON int8 within 1 LSB |
| `yolov8_cpu_*` | the YOLOv8n `combined` export | Scalar tier bit for bit; NEON within 2 LSB (the softmax's 1 LSB, widened by the decode layers after it) |
| `yolov8_after_npu_matches_npu_alone` | `yolov8n-det-int8.imx95.tflite` on Neutron, the `combined` export on VX | Bit for bit with the NPU delegate alone, except the boxes a fused DFL head feeds (within 8 LSB; up to 5 seen); on Neutron the delegate claims the decode tail |
| `attention_after_neutron_matches_neutron_alone` | `yolo11n-det-int8.imx95.tflite`, `yolo26n-det-int8.imx95.tflite` | One and two fused attention blocks (and YOLO11's fused DFL head); bit for bit on the scalar tier and within 4 LSB on NEON, the DFL boxes aside |
| `attention_before_npu_is_fused` | `yolo11n-det-int8.tflite` | With `Scope::Attention` before the NPU delegate, the delegate claims exactly the four attention nodes, fused |

The last three need `TFLITE_TEST_NPU_DELEGATE`. The YOLO models are the model zoo's exports; the `.imx95` ones are compiled with Neutron 3.2.3. They are AGPL-3.0, so they are not in the repository. Download them before running the tests by hand. CI pins the same revisions and checks their SHA-256 (see `.github/workflows/test.yml`):

```sh
hf=https://huggingface.co/EdgeFirst
curl -fsSL -o testdata/yolov8n-det-int8.imx95.tflite \
    $hf/yolov8-det/resolve/7fb4413528af0e22c489e7b85839df6e21fa221f/imx95/yolov8n-det-int8.imx95.tflite
curl -fsSL -o testdata/yolo11n-det-int8.imx95.tflite \
    $hf/yolo11-det/resolve/6a09aa9c4b5e98370b84cacc97e79b2a9a7ae5cb/imx95/yolo11n-det-int8.imx95.tflite
curl -fsSL -o testdata/yolo11n-det-int8.tflite \
    $hf/yolo11-det/resolve/6a09aa9c4b5e98370b84cacc97e79b2a9a7ae5cb/tflite/yolo11n-det-int8.tflite
curl -fsSL -o testdata/yolo26n-det-int8.imx95.tflite \
    $hf/yolo26-det/resolve/1f764994a8805b5531c04c34dd4d7ded8a3bc864/imx95/yolo26n-det-int8.imx95.tflite
```

A model converted for a different Neutron driver still runs, but its output changes from one invoke to the next. The after-NPU tests therefore also check that Neutron alone repeats its results. The `testdata/*.imx95.tflite` exports in the repository were converted for Neutron 3.0.1 and fail that check on the current BSP.

The delegate's unit tests (`cargo test -p edgefirst-tflite-delegate --lib`) need no runtime. They plan and run partitions on plain buffers, check the worker pool, and compare the fused attention and DFL steps with the op-by-op reference.

### Board CI lane

CI runs `custom_delegate`, `edgefirst_delegate` and `integration` on an i.MX 8M Plus EVK (VX) and an i.MX 95 EVK (Neutron) through `.github/scripts/board-tests.sh`. A skipped test passes, so the script sets `TFLITE_TEST_NO_SKIP=1`, which turns the runtime-capability skips into failures: no library, no opaque delegate API, no telemetry profiler, no NPU delegate. It also fails the run if either delegate test binary prints any skip, if `integration` could not load the library, or if any binary passes fewer tests than expected. Set `TFLITE_TEST_NO_SKIP=1` yourself when running these tests by hand on a board. The script runs a payload like this:

```sh
cargo zigbuild -p edgefirst-tflite --target aarch64-unknown-linux-gnu --test custom_delegate --test integration
cargo zigbuild -p edgefirst-tflite-delegate --target aarch64-unknown-linux-gnu --test edgefirst_delegate
mkdir -p board-payload/bin board-payload/testdata
cp target/aarch64-unknown-linux-gnu/debug/deps/custom_delegate-<hash> board-payload/bin/custom_delegate
cp target/aarch64-unknown-linux-gnu/debug/deps/edgefirst_delegate-<hash> board-payload/bin/edgefirst_delegate
cp target/aarch64-unknown-linux-gnu/debug/deps/integration-<hash> board-payload/bin/integration
cp testdata/minimal.tflite testdata/minimal_offset_buffers.tflite testdata/softmax_int8.tflite \
    testdata/softmax_int16.tflite testdata/yolov8n-t-b86_quant-u8-i8_combined.tflite \
    testdata/yolov8n-det-int8.imx95.tflite testdata/yolo11n-det-int8.imx95.tflite \
    testdata/yolo11n-det-int8.tflite testdata/yolo26n-det-int8.imx95.tflite board-payload/testdata/
cp .github/scripts/board-tests.sh board-payload/
scp -r board-payload <board>:
ssh <board> bash board-payload/board-tests.sh board-payload /usr/lib/libneutron_delegate.so  # or libvx_delegate.so
```

## Pre-Built Models

Pre-converted YOLOv8n models are available for download:

| Model | Target | URL |
|-------|--------|-----|
| YOLOv8n 640×640 (TFLite) | i.MX 8M Plus | <https://repo.edgefirst.ai/models/yolov8/yolov8n_640x640.tflite> |
| YOLOv8n 640×640 (Neutron) | i.MX 95 | <https://repo.edgefirst.ai/models/yolov8/yolov8n_640x640.imx95.tflite> |

Download both to `/opt/edgefirst/` on the target device (or any path you
prefer).

## Cross-Compilation

All workspace members except `yolov8-live` are cross-compiled for
`aarch64-unknown-linux-gnu` using `cargo-zigbuild`:

```sh
# Install zigbuild (once)
cargo install cargo-zigbuild

# Build everything
cargo zigbuild --workspace --exclude yolov8-live --all-features --target aarch64-unknown-linux-gnu

# Build test binaries
cargo zigbuild --workspace --exclude yolov8-live --all-features --target aarch64-unknown-linux-gnu --tests
```

Binaries are located in `target/aarch64-unknown-linux-gnu/debug/`.

### Cross-compiling `yolov8-live`

`yolov8-live` links libcamera through a small C++ shim compiled at build time, so it needs two things zig cannot provide:

- **A GCC cross toolchain.** The target's `libcamera.so` is built against libstdc++ and passes `std::string`/`std::vector` across its API; zig's libc++ is not ABI-compatible with it. Use the target's Yocto SDK (`aarch64-poky-linux-g++`).
- **libcamera headers and libraries matching the target**, version >= 0.4. The Yocto SDK sysroot may carry a different libcamera than the board image (the i.MX95 6.12 SDK ships 0.5.2 while the 6.18 image runs 0.7.1); in that case point pkg-config at the board's `libcamera.so`/`libcamera-base.so` plus headers of the same minor version.

```sh
source /opt/<yocto-sdk>/environment-setup-armv8a-poky-linux
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH=<libcamera-target>/lib/pkgconfig
export CC_aarch64_unknown_linux_gnu="$CC" CXX_aarch64_unknown_linux_gnu="$CXX"
export BINDGEN_EXTRA_CLANG_ARGS_aarch64_unknown_linux_gnu="--sysroot=$SDKTARGETSYSROOT"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-poky-linux-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=--sysroot=$SDKTARGETSYSROOT"
unset CC CXX LD AR LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
cargo build --release -p yolov8-live --target aarch64-unknown-linux-gnu
```

A failed zigbuild shows up as `ld.lld: error: undefined symbol: libcamera::BoundMethodBase::activatePack(std::__1::shared_ptr<...>)`; the `std::__1` is zig's libc++ meeting libcamera's libstdc++. Nothing short of using a GCC toolchain fixes it.

#### Without an installed SDK

An EdgeFirst Yocto build already contains the GCC 15 cross compiler and the target libraries its libcamera was built with, so the example can be built against exactly the libcamera the image runs. Assemble a sysroot from the build's `tmp/sysroots-components`: merge `glibc`, `linux-libc-headers`, `gcc-runtime` and `libgcc` from the tune directory (for example `armv8-2a`) with `libcamera` from the SoC directory (for example `armv8-2a-mx95`), or extract the image's `libcamera` and `libcamera-dev` packages into it. A bare cross toolchain, unlike an SDK environment, needs three more things:

- `-B <dir>` with plain `as`, `ld`, `ar`, `nm`, `objcopy`, `strip` and `ranlib` symlinks to the `aarch64-poky-linux-*` binutils, otherwise gcc runs the host assembler (`as: unrecognized option '-EL'`).
- GCC's own include directory (`aarch64-poky-linux-gcc -print-file-name=include`) and the sysroot's `usr/include/c++/<version>` directories in `BINDGEN_EXTRA_CLANG_ARGS`, otherwise bindgen cannot find `stdbool.h` or the C++ headers.
- `-Wl,--allow-shlib-undefined`, because `libcamera.so` pulls in libudev and gnutls, which are not in the partial sysroot; the board has them.

```sh
C=<build-dir>/tmp/sysroots-components
T=$C/x86_64/gcc-cross-aarch64/usr/bin/aarch64-poky-linux
BU=<dir with the plain binutils symlinks>
Y=<merged sysroot>
export PATH=$T:$C/x86_64/binutils-cross-aarch64/usr/bin/aarch64-poky-linux:$PATH
export PKG_CONFIG_ALLOW_CROSS=1 PKG_CONFIG_SYSROOT_DIR=$Y PKG_CONFIG_LIBDIR=$Y/usr/lib/pkgconfig
export CC_aarch64_unknown_linux_gnu="aarch64-poky-linux-gcc -B$BU --sysroot=$Y"
export CXX_aarch64_unknown_linux_gnu="aarch64-poky-linux-g++ -B$BU --sysroot=$Y"
export AR_aarch64_unknown_linux_gnu=$BU/ar
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-poky-linux-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=--sysroot=$Y -C link-arg=-Wl,--allow-shlib-undefined -C link-arg=-B$BU"
GI=$(aarch64-poky-linux-gcc -B$BU --sysroot=$Y -print-file-name=include)
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=$Y -I$Y/usr/include -I$GI -I$Y/usr/include/c++/<gcc-version> -I$Y/usr/include/c++/<gcc-version>/aarch64-poky-linux"
cargo build --release -p yolov8-live --target aarch64-unknown-linux-gnu
```

The binary links `libcamera.so.0.7`, `libstdc++.so.6` and `libgcc_s.so.1` from the board image.

## On-Target Testing: NXP FRDM i.MX 8M Plus

The i.MX 8M Plus uses the **VxDelegate** with the Vivante NPU. It supports
DMA-BUF zero-copy inference and CameraAdaptor RGBA→RGB preprocessing.

### Prerequisites

- FRDM i.MX 8M Plus board running an EdgeFirst or NXP BSP image
- `/usr/lib/libtensorflowlite_c.so` and `/usr/lib/libvx_delegate.so` installed
- A test image (JPEG or PNG) copied to the device

### Deploy and Run

```sh
# Copy binary and test image to the device
scp target/aarch64-unknown-linux-gnu/debug/yolov8 root@<imx8mp-ip>:/tmp/
scp test_image.jpg root@<imx8mp-ip>:/tmp/

# Run with VxDelegate (uses DMA-BUF + CameraAdaptor automatically)
ssh root@<imx8mp-ip> '/tmp/yolov8 \
    /opt/edgefirst/yolov8n_640x640.tflite \
    /tmp/test_image.jpg \
    --delegate /usr/lib/libvx_delegate.so \
    --save'
```

The `--save` flag writes a `test_image_overlay.jpg` with bounding box overlays.

### Integration Tests

```sh
# Copy test binary and testdata to the device
scp target/aarch64-unknown-linux-gnu/debug/deps/edgefirst_tflite-* root@<imx8mp-ip>:/tmp/
scp -r testdata root@<imx8mp-ip>:/tmp/

# Run integration tests
ssh root@<imx8mp-ip> 'cd /tmp && ./edgefirst_tflite-*'
```

Set `TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so` if the library is not in
the default search paths.

## On-Target Testing: NXP FRDM i.MX 95

The i.MX 95 uses the **Neutron NPU** delegate. DMA-BUF zero-copy input is
available when the installed delegate exports the HAL DMA-BUF API
(`hal_dmabuf_*`, present in the EdgeFirst delegate build); a delegate without
it, such as a stock NXP build, falls back to standard tensor I/O
automatically, and the examples print which path they took. CameraAdaptor is
not supported.

### Prerequisites

- FRDM i.MX 95 board running an EdgeFirst or NXP BSP image
- `/usr/lib/libtensorflowlite_c.so` and `/usr/lib/libneutron_delegate.so`
  installed
- A test image (JPEG or PNG) copied to the device

### Model Conversion for Neutron NPU

Standard TFLite models must be converted for the Neutron NPU using the
**eIQ Neutron Converter** tool from NXP. The converter is part of the
[eIQ Toolkit for End-to-End Model Development and Deployment][eiq-toolkit].

Pre-converted models are available at the URLs listed above. To convert your
own models, refer to the eIQ Toolkit documentation.

[eiq-toolkit]: https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT

### Deploy and Run

```sh
# Copy binary and test image to the device
scp target/aarch64-unknown-linux-gnu/debug/yolov8 root@<imx95-ip>:/tmp/
scp test_image.jpg root@<imx95-ip>:/tmp/

# Run with Neutron delegate
ssh root@<imx95-ip> '/tmp/yolov8 \
    /opt/edgefirst/yolov8n_640x640.imx95.tflite \
    /tmp/test_image.jpg \
    --delegate /usr/lib/libneutron_delegate.so \
    --save'
```

### Live Camera

Requires the os08a20 camera routed through the NeoISP
(`imx95-19x19-evk-os08a20-isp-adv7535.dtb`) and Weston running on the HDMI
output. Cross-compile `yolov8-live` as described above.

```sh
scp target/aarch64-unknown-linux-gnu/release/yolov8-live root@<imx95-ip>:/tmp/

ssh root@<imx95-ip> 'cd /tmp && \
    XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-0 \
    LIBCAMERA_PIPELINES_MATCH_LIST=nxp/neo \
    ./yolov8-live /opt/edgefirst/yolov8n-seg-int8-smart.imx95.tflite \
        --delegate /usr/lib/libneutron_delegate.so'
```

Expect an `Input:` line naming the path taken (`delegate DMA-BUF` or `staging copy`) and a running `FPS:` line with per-stage timings; the window on the display shows the camera with masks and boxes. Without `LIBCAMERA_PIPELINES_MATCH_LIST=nxp/neo` the camera is claimed by the `imx8-isi` handler, which offers only raw Bayer, and configuration fails.

For the Python version, create a venv with `--system-site-packages` (for the libcamera bindings), `pip install pywayland "edgefirst-tflite>=0.10.3" "edgefirst-image>=0.38.1" "edgefirst-decoder>=0.38.1" "edgefirst-tensor>=0.38.1"`, and run `examples/yolov8_live/python/yolov8_live.py` with the same environment; it takes the model, capture and threshold arguments.

### Integration Tests

```sh
# Copy test binary and testdata to the device
scp target/aarch64-unknown-linux-gnu/debug/deps/edgefirst_tflite-* root@<imx95-ip>:/tmp/
scp -r testdata root@<imx95-ip>:/tmp/

# Run integration tests
ssh root@<imx95-ip> 'cd /tmp && ./edgefirst_tflite-*'
```

## Testing the Python Package

The `crates/python/` package is built with [maturin](https://maturin.rs).

### Prerequisites

```sh
pip install maturin
# A TFLite library must be discoverable — either via TFLITE_LIBRARY_PATH,
# system path, or by using the vendored feature (see below).
```

### Build and Install in Development Mode

```sh
cd crates/python
maturin develop
```

This installs the package into the current Python environment. Verify the
install:

```python
from edgefirst_tflite import Interpreter
interp = Interpreter(model_path="../../testdata/minimal.tflite")
print(interp.get_input_details())
```

### Testing with edgefirst-tflite-library

If you have published (or locally built) the `edgefirst-tflite-library` wheel:

```sh
pip install edgefirst-tflite-library
python - <<'EOF'
from edgefirst_tflite import Interpreter
from edgefirst_tflite_library import library_path

interp = Interpreter(
    model_path="testdata/minimal.tflite",
    library_path=library_path(),
)
print("Library loaded:", library_path())
print("Input details:", interp.get_input_details())
EOF
```

To test the library package locally before publishing:

```sh
cd crates/tflite-library
# Place libtensorflowlite_c.so in edgefirst_tflite_library/ first
pip install -e .
python -c "from edgefirst_tflite_library import library_path; print(library_path())"
```

### Running YOLOv8 Inference via Python

Use the test models in `testdata/` to validate end-to-end inference:

```python
import numpy as np
from edgefirst_tflite import Interpreter

interp = Interpreter(model_path="testdata/yolov8n-int8.tflite", num_threads=4)
print(interp.get_input_details())
print(interp.get_output_details())

# Create a dummy input matching the model shape (e.g., 1×640×640×3 uint8)
details = interp.get_input_details()[0]
dummy = np.zeros(details["shape"], dtype=np.uint8)
interp.set_tensor(0, dummy)
interp.invoke()

output = interp.get_output_tensor(0)
print("Output shape:", output.shape)
```

## Expected Output

A successful YOLOv8 run prints detected objects with class, confidence, and
bounding box coordinates:

```
Model: /opt/edgefirst/yolov8n_640x640.tflite
Input: 640x640, UInt8, 3 channels
Outputs: 1

Timing:
  Model load:      12.3 ms
  Preprocessing:    5.1 ms
  Inference:        8.7 ms
  Post-processing:  1.2 ms

Detections (threshold=0.25, IoU=0.45):
  dog (16): 77.2% [491, 448, 991, 1074]
  dog (16): 56.1% [959, 418, 1443, 1079]
```

Detection coordinates are in the original image pixel space (not the 640×640
model input space).
