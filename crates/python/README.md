# edgefirst-tflite

Python API for TensorFlow Lite inference with EdgeFirst extensions for
DMA-BUF zero-copy, NPU-accelerated camera preprocessing, and model metadata
extraction.

Built on the [edgefirst-tflite](https://github.com/EdgeFirstAI/tflite-rs)
Rust crate with native performance via [PyO3](https://pyo3.rs).

## Installation

```bash
pip install edgefirst-tflite
```

Requires Python 3.9+ and NumPy 1.24+. The package ships as a native wheel
with the TFLite runtime loaded dynamically at startup — no separate TFLite
installation is needed as long as `libtensorflowlite_c.so` is available on
the system library path.

To specify a custom library path:

```python
interp = Interpreter(model_path="model.tflite", library_path="/usr/lib/libtensorflowlite_c.so")
```

## Quick Start

```python
import numpy as np
from edgefirst_tflite import Interpreter

# Load model and inspect tensors
interp = Interpreter(model_path="model.tflite", num_threads=4)
print(interp.get_input_details())
print(interp.get_output_details())

# Run inference
input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
interp.set_tensor(0, input_data)
interp.invoke()
output = interp.get_output_tensor(0)
print(output)
```

## TFLite API Compatibility

The `Interpreter` class is designed to be familiar to users of
`tflite_runtime.interpreter.Interpreter`. The core inference path is
compatible:

| Method | Description |
|--------|-------------|
| `Interpreter(model_path=, model_content=, num_threads=, experimental_delegates=)` | Load a model |
| `allocate_tensors()` | Re-allocate tensors (required after `resize_tensor_input`) |
| `resize_tensor_input(input_index, tensor_size)` | Resize an input tensor |
| `invoke()` | Run inference |
| `get_input_details()` / `get_output_details()` | Tensor metadata dicts |
| `get_input_tensor(index)` / `get_output_tensor(index)` | Copy tensor data to NumPy |
| `set_tensor(input_index, value)` | Copy NumPy data into an input tensor |
| `tensor(index)` | Zero-copy NumPy view (callable returning array) |

**Note on tensor indices:** `get_input_tensor`, `get_output_tensor`, and
`set_tensor` use 0-based indices relative to the input or output tensor
lists. The `"index"` field returned by `get_input_details()` /
`get_output_details()` matches these relative indices.

## Hardware Acceleration with Delegates

Delegates provide hardware acceleration (e.g., NPU offload via VxDelegate
on NXP i.MX platforms):

```python
from edgefirst_tflite import Interpreter, load_delegate

delegate = load_delegate("libvx_delegate.so", options={
    "cache_file_path": "/tmp/vx_cache",
})

interp = Interpreter(
    model_path="model.tflite",
    experimental_delegates=[delegate],
)
interp.invoke()
```

## EdgeFirst Extensions

### DMA-BUF Zero-Copy Inference

DMA-BUF enables zero-copy data transfer between camera, CPU, and NPU
by binding DMA-BUF file descriptors directly to TFLite tensors. This
eliminates memory copies in the inference pipeline.

**Import mode** — register an externally-allocated DMA-BUF (e.g., from
V4L2 camera capture):

```python
from edgefirst_tflite import Interpreter, load_delegate

delegate = load_delegate("libvx_delegate.so")
interp = Interpreter(model_path="model.tflite", experimental_delegates=[delegate])

dmabuf = interp.dmabuf()
if dmabuf and dmabuf.is_supported():
    # Register a DMA-BUF fd from the camera driver
    handle = dmabuf.register(camera_fd, buffer_size, sync_mode="none")
    dmabuf.bind_to_tensor(handle, tensor_index=0)

    # Run inference — data flows camera → NPU with zero CPU copies
    interp.invoke()
    output = interp.get_output_tensor(0)

    # Cleanup
    dmabuf.unregister(handle)
```

**Export mode** — let the delegate allocate DMA-BUF buffers:

```python
dmabuf = interp.dmabuf()
handle, desc = dmabuf.request(tensor_index=0, ownership="delegate")
print(f"Allocated buffer: fd={desc['fd']}, size={desc['size']}")

dmabuf.bind_to_tensor(handle, tensor_index=0)
interp.invoke()

dmabuf.release(handle)
```

**Buffer cycling** for multi-buffer pipelines (e.g., triple-buffering
with V4L2):

```python
handles = [dmabuf.register(fd, size) for fd, size in camera_buffers]

for frame in camera_stream:
    dmabuf.set_active(tensor_index=0, handle=handles[frame.index])
    interp.invoke()
    result = interp.get_output_tensor(0)
```

**Cache synchronization** for coherent CPU access:

```python
dmabuf.begin_cpu_access(handle, mode="read")
# ... read tensor data on CPU ...
dmabuf.end_cpu_access(handle, mode="read")

dmabuf.sync_for_device(handle)  # Before NPU access
dmabuf.sync_for_cpu(handle)     # Before CPU access
```

### CameraAdaptor — NPU-Accelerated Preprocessing

CameraAdaptor offloads camera format conversion (e.g., RGBA → RGB,
YUV → RGB) to the NPU, eliminating CPU-side preprocessing. The
conversion is injected directly into the TIM-VX inference graph.

```python
delegate = load_delegate("libvx_delegate.so")

# Configure BEFORE building the interpreter — CameraAdaptor modifies
# the delegate's graph compilation
adaptor = delegate.camera_adaptor
if adaptor:
    # Simple format conversion: camera sends RGBA, model expects RGB
    adaptor.set_format(tensor_index=0, format="rgba")
```

**Format conversion with resize and letterboxing:**

```python
adaptor.set_format_ex(
    tensor_index=0,
    format="rgba",
    width=1920,
    height=1080,
    letterbox=True,
    letterbox_color=0,
)
```

**Explicit camera and model format specification:**

```python
adaptor.set_formats(
    tensor_index=0,
    camera_format="rgba",
    model_format="rgb",
)
```

**Query format capabilities:**

```python
adaptor.is_supported("rgba")          # True
adaptor.input_channels("rgba")        # 4
adaptor.output_channels("rgba")       # 3
adaptor.fourcc("rgba")                # "RGBP" (V4L2 FourCC)
adaptor.from_fourcc("NV12")           # "nv12"
```

### Model Metadata

Extract metadata embedded in TFLite model files:

```python
interp = Interpreter(model_path="model.tflite")
meta = interp.get_metadata()
if meta:
    print(f"Model: {meta.name}")
    print(f"Version: {meta.version}")
    print(f"Author: {meta.author}")
    print(f"License: {meta.license}")
    print(f"Description: {meta.description}")
```

## Zero-Copy Tensor Views

The `tensor()` method returns a callable that produces a NumPy array
sharing memory with the TFLite C-allocated buffer:

```python
# Get a zero-copy accessor for output tensor 1
# (index = input_count + output_offset)
accessor = interp.tensor(interp.input_count + 0)

interp.invoke()
view = accessor()  # Zero-copy NumPy view of the output
print(view)        # Reflects the latest inference results

interp.invoke()
view = accessor()  # Updated in-place — no copy needed
```

The accessor is invalidated by `allocate_tensors()` or
`resize_tensor_input()`. Call `tensor()` again to get a fresh one.

## Error Handling

```python
from edgefirst_tflite import (
    TfLiteError,           # Base exception
    LibraryError,          # TFLite library not found
    DelegateError,         # Delegate error status
    InvalidArgumentError,  # Bad arguments (index out of range, etc.)
)

try:
    interp = Interpreter(model_path="missing.tflite")
except InvalidArgumentError as e:
    print(f"Bad argument: {e}")
except LibraryError as e:
    print(f"Library not found: {e}")
except TfLiteError as e:
    print(f"TFLite error: {e}")
```

## Platform Support

| Platform | Architecture | Wheel |
|----------|-------------|-------|
| Linux | x86_64 | manylinux2014 |
| Linux | aarch64 | manylinux2014 |
| macOS | arm64 | native |
| Windows | x86_64 | native |

Primary target: NXP i.MX8M Plus, i.MX93, and i.MX95 (aarch64 Linux with
VxDelegate for NPU acceleration).

## License

Apache-2.0. See [LICENSE](https://github.com/EdgeFirstAI/tflite-rs/blob/main/LICENSE).
