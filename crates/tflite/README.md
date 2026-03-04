# edgefirst-tflite

Ergonomic Rust API for [TensorFlow Lite](https://www.tensorflow.org/lite)
inference with DMA-BUF zero-copy and NPU-accelerated preprocessing.

## Usage

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

## API Tour

The main entry points are:

- **`Library`** -- Load the TFLite shared library (auto-discovery or explicit path)
- **`Model`** -- Load a model from a file or byte buffer
- **`Interpreter`** -- Run inference via a builder pattern
- **`Tensor` / `TensorMut`** -- Type-safe tensor access with shape and quantization info
- **`Delegate`** -- Hardware acceleration via external delegates
- **`DmaBuf`** -- Zero-copy DMA-BUF operations (feature: `dmabuf`)
- **`CameraAdaptor`** -- NPU preprocessing configuration (feature: `camera_adaptor`)
- **`Metadata`** -- Model metadata extraction (feature: `metadata`)

## Feature Flags

| Feature | Description |
|---------|-------------|
| `dmabuf` | DMA-BUF zero-copy inference via `VxDelegate` |
| `camera_adaptor` | NPU-accelerated format conversion |
| `metadata` | TFLite model metadata extraction |
| `full` | Enables all optional features |

## License

Apache-2.0
